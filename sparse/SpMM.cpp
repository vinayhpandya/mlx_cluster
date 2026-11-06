#include <algorithm>
#include <limits>
#include <vector>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

#include "sparse/SpMM.h"

namespace mlx_sparse {

namespace {

std::string binary_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
      throw std::runtime_error("[mlx_cluster] Unable to locate binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return dir;
}

void check(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::invalid_argument("[spmm] " + msg);
  }
}

// Make inputs kernel-friendly: int64 indices, float32 features, row-contiguous.
mx::array as_index(const mx::array& a, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(a, mx::int64, s), false, s);
}
mx::array as_float(const mx::array& a, mx::StreamOrDevice s) {
  return mx::contiguous(mx::astype(a, mx::float32, s), false, s);
}

// Threadgroup shape for a (D, rows) grid: x covers features, y covers rows.
// D is on the fast axis so neighbouring threads read neighbouring floats.
std::pair<size_t, size_t> tg_shape(size_t D) {
  size_t tx = std::min<size_t>(std::max<size_t>(D, 1), 32);
  size_t ty = std::max<size_t>(256 / tx, 1);
  return {tx, ty};
}

} // namespace

// ---------------------------------------------------------------------------
// SpMM : CPU
// ---------------------------------------------------------------------------

void SpMM::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& enc = mx::cpu::get_command_encoder(stream());
  for (auto& in : inputs) {
    enc.set_input_array(in);
  }
  enc.set_output_array(out);

  const int64_t D = inputs[2].shape(1);
  const int64_t E = inputs[1].size();
  const int64_t N = num_rows_;
  const size_t li = long_offset(); // first long-row input

  enc.dispatch([idx = inputs[0],
                col = inputs[1],
                x = inputs[2],
                val = has_values_ ? std::optional<mx::array>(inputs[3])
                                  : std::nullopt,
                lng = has_long_ ? std::vector<mx::array>(
                                      inputs.begin() + li, inputs.begin() + li + 4)
                                : std::vector<mx::array>{},
                threshold = static_cast<int64_t>(long_threshold_),
                out_ptr = out.data<float>(),
                fmt = format_,
                red = reduce_,
                D,
                E,
                N]() {
    const int64_t* idx_p = idx.data<int64_t>();
    const int64_t* col_p = col.data<int64_t>();
    const float* x_p = x.data<float>();
    const float* v_p = val ? val->data<float>() : nullptr;
    const bool has_long = !lng.empty();

    // identity element and combine for the reduction
    const float ident = red == SpMMReduce::Sum
        ? 0.0f
        : (red == SpMMReduce::Max ? -std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::infinity());
    auto combine = [red](float a, float b) {
      return red == SpMMReduce::Sum ? a + b
          : red == SpMMReduce::Max  ? std::max(a, b)
                                    : std::min(a, b);
    };
    // o[f] = combine(o[f], w_e * x[col_e, f]) over edges [e0, e1)
    auto accumulate = [&](float* o, int64_t e0, int64_t e1) {
      for (int64_t e = e0; e < e1; ++e) {
        const float* xr = x_p + col_p[e] * D;
        for (int64_t f = 0; f < D; ++f) {
          float m = xr[f];
          if (v_p) {
            m *= v_p[e]; // same operation order as the Metal kernels
          }
          o[f] = combine(o[f], m);
        }
      }
    };

    if (fmt == SparseFormat::COO) {
      std::fill(out_ptr, out_ptr + N * D, ident);
      std::vector<char> nonempty(red == SpMMReduce::Sum ? 0 : N, 0);
      for (int64_t e = 0; e < E; ++e) {
        accumulate(out_ptr + idx_p[e] * D, e, e + 1);
        if (red != SpMMReduce::Sum) {
          nonempty[idx_p[e]] = 1;
        }
      }
      if (red != SpMMReduce::Sum) { // empty rows -> 0, like mlx-graphs
        for (int64_t r = 0; r < N; ++r) {
          if (!nonempty[r]) {
            std::fill(out_ptr + r * D, out_ptr + (r + 1) * D, 0.0f);
          }
        }
      }
    } else {
      // short rows (all rows when there is no long-row metadata)
      for (int64_t r = 0; r < N; ++r) {
        if (has_long && idx_p[r + 1] - idx_p[r] > threshold) {
          continue; // written by the long-row path below
        }
        float* o = out_ptr + r * D;
        std::fill(o, o + D, ident);
        accumulate(o, idx_p[r], idx_p[r + 1]);
        if (red != SpMMReduce::Sum && idx_p[r + 1] == idx_p[r]) {
          std::fill(o, o + D, 0.0f); // empty row
        }
      }
      // long rows: same two phases as the GPU (per-chunk partials, then reduce)
      if (has_long) {
        const int64_t* rows = lng[0].data<int64_t>();
        const int64_t* cptr = lng[1].data<int64_t>();
        const int64_t* cs = lng[2].data<int64_t>();
        const int64_t* ce = lng[3].data<int64_t>();
        const int64_t L = lng[0].size();
        const int64_t K = lng[2].size();
        std::vector<float> partials(K * D, ident);
        for (int64_t k = 0; k < K; ++k) {
          accumulate(partials.data() + k * D, cs[k], ce[k]);
        }
        for (int64_t l = 0; l < L; ++l) {
          float* o = out_ptr + rows[l] * D;
          std::fill(o, o + D, ident);
          for (int64_t k = cptr[l]; k < cptr[l + 1]; ++k) {
            for (int64_t f = 0; f < D; ++f) {
              o[f] = combine(o[f], partials[k * D + f]);
            }
          }
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// SpMM : GPU (Metal)
// ---------------------------------------------------------------------------

#ifdef _METAL_
void SpMM::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& idx = inputs[0];
  auto& col = inputs[1];
  auto& x = inputs[2];
  // When there are no values we bind x in the values slot as a dummy so every
  // buffer slot is bound; the kernel never reads it.
  const mx::array& val = has_values_ ? inputs[3] : inputs[2];

  const uint32_t D = static_cast<uint32_t>(x.shape(1));
  const uint32_t E = static_cast<uint32_t>(col.size());
  const uint32_t N = static_cast<uint32_t>(num_rows_);

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("mlx_cluster", binary_dir());
  auto& enc = mx::metal::get_command_encoder(s);
  const std::string suffix = has_values_ ? "_val" : "";
  const std::string op = reduce_ == SpMMReduce::Sum ? ""
      : reduce_ == SpMMReduce::Max                  ? "_max"
                                                    : "_min";
  auto [tx, ty] = tg_shape(D);

  if (format_ == SparseFormat::COO && reduce_ != SpMMReduce::Sum) {
    // Max/Min with COO. Metal has no float atomic max/min, so every float is
    // mapped to an order-preserving uint key and the uint atomics are used,
    // working in place in `out`:
    //   1) fill keys with an "empty" sentinel (below/above every float key)
    //   2) atomic max/min of the message keys
    //   3) decode keys back to floats; rows still holding the sentinel -> 0
    const uint32_t n = static_cast<uint32_t>(out.size());
    const uint32_t sentinel = reduce_ == SpMMReduce::Max ? 0u : 0xFFFFFFFFu;
    auto fk = d.get_kernel("spmm_fill_u32", lib);
    enc.set_compute_pipeline_state(fk);
    enc.set_output_array(out, 0);
    enc.set_bytes(sentinel, 1);
    enc.set_bytes(n, 2);
    enc.dispatch_threads(
        MTL::Size(n, 1, 1),
        MTL::Size(std::min<size_t>(n, fk->maxTotalThreadsPerThreadgroup()), 1, 1));

    if (E > 0) {
      auto k = d.get_kernel("spmm_coo" + op + suffix, lib);
      enc.set_compute_pipeline_state(k);
      enc.set_input_array(idx, 0);
      enc.set_input_array(col, 1);
      enc.set_input_array(val, 2);
      enc.set_input_array(x, 3);
      enc.set_output_array(out, 4);
      enc.set_bytes(D, 5);
      enc.set_bytes(E, 6);
      enc.dispatch_threads(MTL::Size(D, E, 1), MTL::Size(tx, ty, 1));
    }

    auto dk = d.get_kernel("spmm_decode" + op, lib);
    enc.set_compute_pipeline_state(dk);
    enc.set_output_array(out, 0);
    enc.set_bytes(n, 1);
    enc.dispatch_threads(
        MTL::Size(n, 1, 1),
        MTL::Size(std::min<size_t>(n, dk->maxTotalThreadsPerThreadgroup()), 1, 1));
  } else if (format_ == SparseFormat::COO) {
    // 1) zero the output (the COO kernel accumulates with atomics)
    auto zk = d.get_kernel("spmm_zero", lib);
    const uint32_t n = static_cast<uint32_t>(out.size());
    enc.set_compute_pipeline_state(zk);
    enc.set_output_array(out, 0);
    enc.set_bytes(n, 1);
    enc.dispatch_threads(
        MTL::Size(n, 1, 1),
        MTL::Size(std::min<size_t>(n, zk->maxTotalThreadsPerThreadgroup()), 1, 1));
    if (E == 0) {
      return;
    }

    // 2) one thread per (feature, edge), atomic add into out[row[e]]
    auto k = d.get_kernel("spmm_coo" + suffix, lib);
    enc.set_compute_pipeline_state(k);
    enc.set_input_array(idx, 0);
    enc.set_input_array(col, 1);
    enc.set_input_array(val, 2);
    enc.set_input_array(x, 3);
    enc.set_output_array(out, 4); // barrier vs. the zero-fill is inserted here
    enc.set_bytes(D, 5);
    enc.set_bytes(E, 6);
    enc.dispatch_threads(MTL::Size(D, E, 1), MTL::Size(tx, ty, 1));
  } else {
    // 1) short rows: one thread per (feature, row), sequential loop over the
    //    row's edges. Rows with more than `threshold` edges are skipped here.
    const uint32_t threshold = has_long_
        ? static_cast<uint32_t>(long_threshold_)
        : std::numeric_limits<uint32_t>::max();
    auto k = d.get_kernel("spmm_csr" + op + suffix, lib);
    enc.set_compute_pipeline_state(k);
    enc.set_input_array(idx, 0);
    enc.set_input_array(col, 1);
    enc.set_input_array(val, 2);
    enc.set_input_array(x, 3);
    enc.set_output_array(out, 4);
    enc.set_bytes(D, 5);
    enc.set_bytes(N, 6);
    enc.set_bytes(threshold, 7);
    enc.dispatch_threads(MTL::Size(D, N, 1), MTL::Size(tx, ty, 1));

    if (!has_long_) {
      return;
    }
    const size_t li = long_offset();
    const auto& rows = inputs[li];
    const auto& chunk_ptr = inputs[li + 1];
    const auto& chunk_start = inputs[li + 2];
    const auto& chunk_end = inputs[li + 3];
    const uint32_t L = static_cast<uint32_t>(rows.size());
    const uint32_t K = static_cast<uint32_t>(chunk_start.size());
    if (L == 0 || K == 0) {
      return;
    }

    // 2) long rows, phase 1: one threadgroup per (feature tile, chunk).
    //    Threads are (FX features) x (EY edge lanes); the EY partial sums are
    //    combined in threadgroup memory and written to partials[chunk, :].
    mx::array partials({static_cast<int>(K), static_cast<int>(D)}, mx::float32, nullptr, {});
    partials.set_data(mx::allocator::malloc(partials.nbytes()));
    enc.add_temporary(partials);

    size_t fx = 1;
    while (fx < D && fx < 32) {
      fx <<= 1;
    }
    const size_t ey = 256 / fx;
    auto pk = d.get_kernel("spmm_csr_long_partial" + op + suffix, lib);
    enc.set_compute_pipeline_state(pk);
    enc.set_input_array(chunk_start, 0);
    enc.set_input_array(chunk_end, 1);
    enc.set_input_array(col, 2);
    enc.set_input_array(val, 3);
    enc.set_input_array(x, 4);
    enc.set_output_array(partials, 5);
    enc.set_bytes(D, 6);
    enc.dispatch_threadgroups(
        MTL::Size((D + fx - 1) / fx, K, 1), MTL::Size(fx, ey, 1));

    // 3) long rows, phase 2: one thread per (feature, long row) adds that
    //    row's chunk sums in order and writes the output row.
    auto rk = d.get_kernel("spmm_csr_long_reduce" + op, lib);
    enc.set_compute_pipeline_state(rk);
    enc.set_input_array(rows, 0);
    enc.set_input_array(chunk_ptr, 1);
    enc.set_input_array(partials, 2); // barrier vs. phase 1 inserted here
    enc.set_output_array(out, 3);
    enc.set_bytes(D, 4);
    enc.set_bytes(L, 5);
    enc.dispatch_threads(MTL::Size(D, L, 1), MTL::Size(tx, ty, 1));
  }
}
#else
void SpMM::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[spmm] This build has no Metal backend.");
}
#endif

// ---------------------------------------------------------------------------
// SpMM : autodiff
// ---------------------------------------------------------------------------

// Per-edge row ids, whatever the format.
static mx::array edge_rows(
    SparseFormat fmt,
    const mx::array& idx,
    int64_t E,
    mx::StreamOrDevice s) {
  return fmt == SparseFormat::COO ? idx : csr_to_row(idx, E, s);
}

std::vector<mx::array> SpMM::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
  const auto& idx = primals[0];
  const auto& col = primals[1];
  const auto& x = primals[2];
  const auto& g = cotangents[0]; // [num_rows, D]
  const int64_t E = col.size();
  auto s = stream();

  std::optional<mx::array> vals;
  if (has_values_) {
    vals = primals[3];
  }

  std::vector<mx::array> grads;

  if (reduce_ != SpMMReduce::Sum) {
    // Max/Min: the gradient goes to every edge whose message equals the
    // output (ties included), recomputed from x/values and compared with the
    // saved output. No argmax needs to be stored during the forward.
    std::optional<mx::array> rows;
    for (int a : argnums) {
      if (a == 2 && format_ == SparseFormat::CSR && has_transpose_) {
        // On the cached transpose: no atomics, deterministic, load-balanced.
        const size_t t = transpose_offset();
        std::optional<mx::array> vals_t;
        if (vals) {
          vals_t = mx::take(*vals, primals[t + 2], 0, s);
        }
        std::optional<LongRows> lng_t;
        if (has_long_t_) {
          lng_t = LongRows{
              primals[t + 3],
              primals[t + 4],
              primals[t + 5],
              primals[t + 6],
              long_threshold_t_};
        }
        grads.push_back(spmm_select_grad_t(
            primals[t], primals[t + 1], x, outputs[0], g, vals_t, lng_t, s));
      } else if (a == 2 || (a == 3 && has_values_)) {
        if (!rows) {
          rows = edge_rows(format_, idx, E, s);
        }
        grads.push_back(spmm_select_grad(
            a == 2 ? SpMMSelectGrad::Target::X : SpMMSelectGrad::Target::Values,
            *rows, col, x, outputs[0], g, vals, s));
      } else {
        grads.push_back(mx::zeros_like(primals[a], s));
      }
    }
    return grads;
  }

  for (int a : argnums) {
    if (a == 2 && format_ == SparseFormat::CSR && has_transpose_) {
      // dL/dx = A^T @ g with the CSR kernel on the cached transpose:
      // no atomics, deterministic, load-balanced for high out-degree nodes.
      const size_t t = transpose_offset();
      std::optional<mx::array> vals_t;
      if (vals) {
        vals_t = mx::take(*vals, primals[t + 2], 0, s); // into transposed order
      }
      std::optional<LongRows> lng_t;
      if (has_long_t_) {
        lng_t = LongRows{
            primals[t + 3],
            primals[t + 4],
            primals[t + 5],
            primals[t + 6],
            long_threshold_t_};
      }
      grads.push_back(spmm_csr(
          primals[t],
          primals[t + 1],
          g,
          vals_t,
          lng_t,
          std::nullopt,
          SpMMReduce::Sum,
          s));
    } else if (a == 2) {
      // dL/dx = A^T @ g : same op with row and col swapped (COO kernel).
      // CSR without a cached transpose also lands here.
      auto rows = edge_rows(format_, idx, E, s);
      grads.push_back(
          spmm_coo(col, rows, g, vals, x.shape(0), SpMMReduce::Sum, s));
    } else if (a == 3 && has_values_) {
      // dL/dvalue[e] = <g[row[e]], x[col[e]]>
      auto rows = edge_rows(format_, idx, E, s);
      grads.push_back(sddmm(rows, col, g, x, s));
    } else {
      // integer index / long-row metadata inputs: no gradient
      grads.push_back(mx::zeros_like(primals[a], s));
    }
  }
  return grads;
}

std::vector<mx::array> SpMM::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
  if (reduce_ != SpMMReduce::Sum) {
    throw std::runtime_error(
        "[spmm] jvp is not supported for reduce='max'/'min' (use vjp / mx.grad).");
  }
  // SpMM is bilinear in (x, values), so the JVP is a sum of SpMMs.
  auto s = stream();
  const auto& idx = primals[0];
  const auto& col = primals[1];
  std::optional<mx::array> vals;
  if (has_values_) {
    vals = primals[3];
  }

  std::optional<LongRows> lng;
  if (has_long_) {
    const size_t li = long_offset();
    lng = LongRows{
        primals[li],
        primals[li + 1],
        primals[li + 2],
        primals[li + 3],
        long_threshold_};
  }
  std::optional<CSRTranspose> tr;
  if (has_transpose_) {
    const size_t t = transpose_offset();
    std::optional<LongRows> lng_t;
    if (has_long_t_) {
      lng_t = LongRows{
          primals[t + 3],
          primals[t + 4],
          primals[t + 5],
          primals[t + 6],
          long_threshold_t_};
    }
    tr = CSRTranspose{primals[t], primals[t + 1], primals[t + 2], lng_t};
  }
  auto apply = [&](const mx::array& x, const std::optional<mx::array>& v) {
    return format_ == SparseFormat::COO
        ? spmm_coo(idx, col, x, v, num_rows_, SpMMReduce::Sum, s)
        : spmm_csr(idx, col, x, v, lng, tr, SpMMReduce::Sum, s);
  };

  std::optional<mx::array> out;
  for (size_t i = 0; i < argnums.size(); ++i) {
    std::optional<mx::array> term;
    if (argnums[i] == 2) {
      term = apply(tangents[i], vals);
    } else if (argnums[i] == 3 && has_values_) {
      term = apply(primals[2], tangents[i]);
    }
    if (term) {
      out = out ? mx::add(*out, *term, s) : *term;
    }
  }
  if (!out) {
    out = mx::zeros({num_rows_, primals[2].shape(1)}, mx::float32, s);
  }
  return {*out};
}

std::pair<std::vector<mx::array>, std::vector<int>> SpMM::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[spmm] vmap is not supported.");
}

bool SpMM::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const SpMM*>(&other);
  return o && o->format_ == format_ && o->has_values_ == has_values_ &&
      o->num_rows_ == num_rows_ && o->has_long_ == has_long_ &&
      o->long_threshold_ == long_threshold_ &&
      o->has_transpose_ == has_transpose_ && o->has_long_t_ == has_long_t_ &&
      o->long_threshold_t_ == long_threshold_t_ && o->reduce_ == reduce_;
}

// ---------------------------------------------------------------------------
// SpMMSelectGrad: backward of Max/Min SpMM
// Inputs: row, col, x, out, g, (values). An edge "won" output element
// (row_e, f) iff its message m = x[col_e, f] * w_e equals out[row_e, f].
// ---------------------------------------------------------------------------

void SpMMSelectGrad::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& res = outputs[0];
  res.set_data(mx::allocator::malloc(res.nbytes()));

  auto& enc = mx::cpu::get_command_encoder(stream());
  for (auto& in : inputs) {
    enc.set_input_array(in);
  }
  enc.set_output_array(res);

  const int64_t D = inputs[2].shape(1);
  const int64_t E = inputs[0].size();
  enc.dispatch([row = inputs[0],
                col = inputs[1],
                x = inputs[2],
                out = inputs[3],
                g = inputs[4],
                val = has_values_ ? std::optional<mx::array>(inputs[5])
                                  : std::nullopt,
                res_ptr = res.data<float>(),
                res_size = res.size(),
                target = target_,
                D,
                E]() {
    const int64_t* r = row.data<int64_t>();
    const int64_t* c = col.data<int64_t>();
    const float* xp = x.data<float>();
    const float* op = out.data<float>();
    const float* gp = g.data<float>();
    const float* vp = val ? val->data<float>() : nullptr;
    std::fill(res_ptr, res_ptr + res_size, 0.0f);
    for (int64_t e = 0; e < E; ++e) {
      const float w = vp ? vp[e] : 1.0f;
      const float* xr = xp + c[e] * D;
      const float* orow = op + r[e] * D;
      const float* grow = gp + r[e] * D;
      float acc = 0.0f;
      for (int64_t f = 0; f < D; ++f) {
        float m = xr[f];
        if (vp) {
          m *= w; // must match the forward's operation exactly
        }
        if (m == orow[f]) {
          if (target == SpMMSelectGrad::Target::X) {
            res_ptr[c[e] * D + f] += w * grow[f];
          } else {
            acc += xr[f] * grow[f];
          }
        }
      }
      if (target == SpMMSelectGrad::Target::Values) {
        res_ptr[e] = acc;
      }
    }
  });
}

#ifdef _METAL_
void SpMMSelectGrad::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& res = outputs[0];
  res.set_data(mx::allocator::malloc(res.nbytes()));
  if (res.size() == 0) {
    return;
  }
  const auto& x = inputs[2];
  const mx::array& val = has_values_ ? inputs[5] : inputs[2]; // dummy if none
  const uint32_t D = static_cast<uint32_t>(x.shape(1));
  const uint32_t E = static_cast<uint32_t>(inputs[0].size());

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("mlx_cluster", binary_dir());
  auto& enc = mx::metal::get_command_encoder(s);

  if (target_ == Target::X) {
    // zero, then one thread per (feature, edge) with atomic adds
    auto zk = d.get_kernel("spmm_zero", lib);
    const uint32_t n = static_cast<uint32_t>(res.size());
    enc.set_compute_pipeline_state(zk);
    enc.set_output_array(res, 0);
    enc.set_bytes(n, 1);
    enc.dispatch_threads(
        MTL::Size(n, 1, 1),
        MTL::Size(std::min<size_t>(n, zk->maxTotalThreadsPerThreadgroup()), 1, 1));
    if (E == 0) {
      return;
    }
    auto k = d.get_kernel(
        std::string("spmm_select_grad_x") + (has_values_ ? "_val" : ""), lib);
    enc.set_compute_pipeline_state(k);
    for (int i = 0; i < 5; ++i) {
      enc.set_input_array(inputs[i], i); // row, col, x, out, g
    }
    enc.set_input_array(val, 5);
    enc.set_output_array(res, 6);
    enc.set_bytes(D, 7);
    enc.set_bytes(E, 8);
    auto [tx, ty] = tg_shape(D);
    enc.dispatch_threads(MTL::Size(D, E, 1), MTL::Size(tx, ty, 1));
  } else {
    // one SIMD group per edge, reduce over features (deterministic)
    auto k = d.get_kernel("spmm_select_grad_w", lib);
    enc.set_compute_pipeline_state(k);
    for (int i = 0; i < 5; ++i) {
      enc.set_input_array(inputs[i], i);
    }
    enc.set_input_array(val, 5);
    enc.set_output_array(res, 6);
    enc.set_bytes(D, 7);
    enc.set_bytes(E, 8);
    enc.dispatch_threads(MTL::Size(32, E, 1), MTL::Size(32, 8, 1));
  }
}
#else
void SpMMSelectGrad::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[spmm] This build has no Metal backend.");
}
#endif

std::vector<mx::array> SpMMSelectGrad::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[spmm] max/min backward is not differentiable.");
}

std::vector<mx::array> SpMMSelectGrad::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "[spmm] second-order gradients of reduce='max'/'min' are not supported.");
}

std::pair<std::vector<mx::array>, std::vector<int>> SpMMSelectGrad::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[spmm] vmap is not supported.");
}

mx::array spmm_select_grad(
    SpMMSelectGrad::Target target,
    const mx::array& row,
    const mx::array& col,
    const mx::array& x,
    const mx::array& out,
    const mx::array& g,
    const std::optional<mx::array>& values,
    mx::StreamOrDevice s) {
  check(
      target == SpMMSelectGrad::Target::X || values.has_value(),
      "values are required for the gradient w.r.t. values.");
  std::vector<mx::array> inputs = {
      as_index(row, s),
      as_index(col, s),
      as_float(x, s),
      as_float(out, s),
      as_float(g, s)};
  if (values) {
    inputs.push_back(as_float(*values, s));
  }
  auto shape = target == SpMMSelectGrad::Target::X
      ? mx::Shape{x.shape(0), x.shape(1)}
      : mx::Shape{static_cast<int>(row.size())};
  return mx::array(
      std::move(shape),
      mx::float32,
      std::make_shared<SpMMSelectGrad>(
          mx::to_stream(s), target, values.has_value()),
      std::move(inputs));
}

// ---------------------------------------------------------------------------
// SpMMSelectGradT: Max/Min gradient w.r.t. x on the transposed CSR.
// Inputs: t_rowptr, t_col, x, out, g, (values_t), (4 long-row arrays).
// ---------------------------------------------------------------------------

void SpMMSelectGradT::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& res = outputs[0];
  res.set_data(mx::allocator::malloc(res.nbytes()));

  auto& enc = mx::cpu::get_command_encoder(stream());
  for (auto& in : inputs) {
    enc.set_input_array(in);
  }
  enc.set_output_array(res);

  const int64_t Nx = inputs[2].shape(0);
  const int64_t D = inputs[2].shape(1);
  // The CPU path ignores the long-row metadata: same result, simpler loop.
  enc.dispatch([rp = inputs[0],
                tc = inputs[1],
                x = inputs[2],
                out = inputs[3],
                g = inputs[4],
                val = has_values_ ? std::optional<mx::array>(inputs[5])
                                  : std::nullopt,
                res_ptr = res.data<float>(),
                Nx,
                D]() {
    const int64_t* rpp = rp.data<int64_t>();
    const int64_t* tcp = tc.data<int64_t>();
    const float* xp = x.data<float>();
    const float* op = out.data<float>();
    const float* gp = g.data<float>();
    const float* vp = val ? val->data<float>() : nullptr;
    for (int64_t c = 0; c < Nx; ++c) {
      float* o = res_ptr + c * D;
      std::fill(o, o + D, 0.0f);
      const float* xr = xp + c * D;
      for (int64_t i = rpp[c]; i < rpp[c + 1]; ++i) {
        const float w = vp ? vp[i] : 1.0f;
        const float* orow = op + tcp[i] * D;
        const float* grow = gp + tcp[i] * D;
        for (int64_t f = 0; f < D; ++f) {
          float m = xr[f];
          if (vp) {
            m *= w; // must match the forward's operation exactly
          }
          if (m == orow[f]) {
            o[f] += w * grow[f];
          }
        }
      }
    }
  });
}

#ifdef _METAL_
void SpMMSelectGradT::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& res = outputs[0];
  res.set_data(mx::allocator::malloc(res.nbytes()));
  if (res.size() == 0) {
    return;
  }
  const auto& x = inputs[2];
  const mx::array& val = has_values_ ? inputs[5] : inputs[2]; // dummy if none
  const uint32_t D = static_cast<uint32_t>(x.shape(1));
  const uint32_t Nx = static_cast<uint32_t>(x.shape(0));
  const std::string suffix = has_values_ ? "_val" : "";

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("mlx_cluster", binary_dir());
  auto& enc = mx::metal::get_command_encoder(s);
  auto [tx, ty] = tg_shape(D);

  // 1) short rows: one thread per (feature, source node)
  const uint32_t threshold = has_long_ ? static_cast<uint32_t>(long_threshold_)
                                       : std::numeric_limits<uint32_t>::max();
  auto k = d.get_kernel("spmm_select_grad_xt" + suffix, lib);
  enc.set_compute_pipeline_state(k);
  for (int i = 0; i < 5; ++i) {
    enc.set_input_array(inputs[i], i); // t_rowptr, t_col, x, out, g
  }
  enc.set_input_array(val, 5);
  enc.set_output_array(res, 6);
  enc.set_bytes(D, 7);
  enc.set_bytes(Nx, 8);
  enc.set_bytes(threshold, 9);
  enc.dispatch_threads(MTL::Size(D, Nx, 1), MTL::Size(tx, ty, 1));

  if (!has_long_) {
    return;
  }
  const size_t li = has_values_ ? 6 : 5;
  const auto& rows = inputs[li];
  const auto& chunk_ptr = inputs[li + 1];
  const auto& chunk_start = inputs[li + 2];
  const auto& chunk_end = inputs[li + 3];
  const uint32_t L = static_cast<uint32_t>(rows.size());
  const uint32_t K = static_cast<uint32_t>(chunk_start.size());
  if (L == 0 || K == 0) {
    return;
  }

  // 2) long rows, phase 1: one threadgroup per (feature tile, chunk)
  mx::array partials({static_cast<int>(K), static_cast<int>(D)}, mx::float32, nullptr, {});
  partials.set_data(mx::allocator::malloc(partials.nbytes()));
  enc.add_temporary(partials);
  size_t fx = 1;
  while (fx < D && fx < 32) {
    fx <<= 1;
  }
  const size_t ey = 256 / fx;
  auto pk = d.get_kernel("spmm_select_grad_xt_long_partial" + suffix, lib);
  enc.set_compute_pipeline_state(pk);
  enc.set_input_array(rows, 0);
  enc.set_input_array(chunk_ptr, 1);
  enc.set_input_array(chunk_start, 2);
  enc.set_input_array(chunk_end, 3);
  enc.set_input_array(inputs[1], 4); // t_col
  enc.set_input_array(x, 5);
  enc.set_input_array(inputs[3], 6); // out
  enc.set_input_array(inputs[4], 7); // g
  enc.set_input_array(val, 8);
  enc.set_output_array(partials, 9);
  enc.set_bytes(D, 10);
  enc.set_bytes(L, 11);
  enc.dispatch_threadgroups(
      MTL::Size((D + fx - 1) / fx, K, 1), MTL::Size(fx, ey, 1));

  // 3) long rows, phase 2: add each row's chunk sums in order (sum reduce)
  auto rk = d.get_kernel("spmm_csr_long_reduce", lib);
  enc.set_compute_pipeline_state(rk);
  enc.set_input_array(rows, 0);
  enc.set_input_array(chunk_ptr, 1);
  enc.set_input_array(partials, 2);
  enc.set_output_array(res, 3);
  enc.set_bytes(D, 4);
  enc.set_bytes(L, 5);
  enc.dispatch_threads(MTL::Size(D, L, 1), MTL::Size(tx, ty, 1));
}
#else
void SpMMSelectGradT::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[spmm] This build has no Metal backend.");
}
#endif

std::vector<mx::array> SpMMSelectGradT::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[spmm] max/min backward is not differentiable.");
}

std::vector<mx::array> SpMMSelectGradT::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "[spmm] second-order gradients of reduce='max'/'min' are not supported.");
}

std::pair<std::vector<mx::array>, std::vector<int>> SpMMSelectGradT::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[spmm] vmap is not supported.");
}

mx::array spmm_select_grad_t(
    const mx::array& t_rowptr,
    const mx::array& t_col,
    const mx::array& x,
    const mx::array& out,
    const mx::array& g,
    const std::optional<mx::array>& values_t,
    const std::optional<LongRows>& long_rows_t,
    mx::StreamOrDevice s) {
  check(
      t_rowptr.size() == static_cast<size_t>(x.shape(0)) + 1,
      "transpose rowptr must have x.shape[0] + 1 entries.");
  std::vector<mx::array> inputs = {
      as_index(t_rowptr, s),
      as_index(t_col, s),
      as_float(x, s),
      as_float(out, s),
      as_float(g, s)};
  if (values_t) {
    inputs.push_back(as_float(*values_t, s));
  }
  if (long_rows_t) {
    inputs.push_back(as_index(long_rows_t->rows, s));
    inputs.push_back(as_index(long_rows_t->chunk_ptr, s));
    inputs.push_back(as_index(long_rows_t->chunk_start, s));
    inputs.push_back(as_index(long_rows_t->chunk_end, s));
  }
  return mx::array(
      {x.shape(0), x.shape(1)},
      mx::float32,
      std::make_shared<SpMMSelectGradT>(
          mx::to_stream(s),
          values_t.has_value(),
          long_rows_t.has_value(),
          long_rows_t ? long_rows_t->threshold : 0),
      std::move(inputs));
}

// ---------------------------------------------------------------------------
// SDDMM
// ---------------------------------------------------------------------------

void SDDMM::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& enc = mx::cpu::get_command_encoder(stream());
  for (auto& in : inputs) {
    enc.set_input_array(in);
  }
  enc.set_output_array(out);

  const int64_t D = inputs[2].shape(1);
  const int64_t E = inputs[0].size();

  enc.dispatch([row = inputs[0],
                col = inputs[1],
                a = inputs[2],
                b = inputs[3],
                out_ptr = out.data<float>(),
                D,
                E]() {
    const int64_t* r = row.data<int64_t>();
    const int64_t* c = col.data<int64_t>();
    const float* ap = a.data<float>();
    const float* bp = b.data<float>();
    for (int64_t e = 0; e < E; ++e) {
      const float* ar = ap + r[e] * D;
      const float* br = bp + c[e] * D;
      float acc = 0.0f;
      for (int64_t f = 0; f < D; ++f) {
        acc += ar[f] * br[f];
      }
      out_ptr[e] = acc;
    }
  });
}

#ifdef _METAL_
void SDDMM::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  const uint32_t D = static_cast<uint32_t>(inputs[2].shape(1));
  const uint32_t E = static_cast<uint32_t>(inputs[0].size());

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto lib = d.get_library("mlx_cluster", binary_dir());
  auto k = d.get_kernel("sddmm", lib);
  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(k);
  enc.set_input_array(inputs[0], 0);
  enc.set_input_array(inputs[1], 1);
  enc.set_input_array(inputs[2], 2);
  enc.set_input_array(inputs[3], 3);
  enc.set_output_array(out, 4);
  enc.set_bytes(D, 5);
  enc.set_bytes(E, 6);
  // one SIMD group (32 lanes) per edge, 8 edges per threadgroup
  enc.dispatch_threads(MTL::Size(32, E, 1), MTL::Size(32, 8, 1));
}
#else
void SDDMM::eval_gpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("[sddmm] This build has no Metal backend.");
}
#endif

std::vector<mx::array> SDDMM::jvp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[sddmm] jvp is not supported.");
}

std::vector<mx::array> SDDMM::vjp(
    const std::vector<mx::array>&,
    const std::vector<mx::array>&,
    const std::vector<int>&,
    const std::vector<mx::array>&) {
  throw std::runtime_error(
      "[sddmm] vjp is not supported (second-order gradients of spmm).");
}

std::pair<std::vector<mx::array>, std::vector<int>> SDDMM::vmap(
    const std::vector<mx::array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[sddmm] vmap is not supported.");
}

// ---------------------------------------------------------------------------
// Public ops
// ---------------------------------------------------------------------------

mx::array spmm_coo(
    const mx::array& row,
    const mx::array& col,
    const mx::array& x,
    const std::optional<mx::array>& values,
    int num_rows,
    SpMMReduce reduce,
    mx::StreamOrDevice s) {
  check(row.ndim() == 1 && col.ndim() == 1, "row and col must be 1-D.");
  check(row.size() == col.size(), "row and col must have the same length.");
  check(x.ndim() == 2, "x must be 2-D [num_nodes, features].");
  check(num_rows >= 0, "num_rows must be non-negative.");
  if (values) {
    check(
        values->ndim() == 1 && values->size() == col.size(),
        "values must be 1-D with one entry per edge.");
  }

  std::vector<mx::array> inputs = {
      as_index(row, s), as_index(col, s), as_float(x, s)};
  if (values) {
    inputs.push_back(as_float(*values, s));
  }
  return mx::array(
      {num_rows, x.shape(1)},
      mx::float32,
      std::make_shared<SpMM>(
          mx::to_stream(s),
          SparseFormat::COO,
          values.has_value(),
          num_rows,
          false,
          0,
          false,
          false,
          0,
          reduce),
      std::move(inputs));
}

mx::array spmm_csr(
    const mx::array& rowptr,
    const mx::array& col,
    const mx::array& x,
    const std::optional<mx::array>& values,
    const std::optional<LongRows>& long_rows,
    const std::optional<CSRTranspose>& transpose,
    SpMMReduce reduce,
    mx::StreamOrDevice s) {
  check(rowptr.ndim() == 1 && rowptr.size() >= 1, "rowptr must be 1-D, len >= 1.");
  check(col.ndim() == 1, "col must be 1-D.");
  check(x.ndim() == 2, "x must be 2-D [num_nodes, features].");
  if (values) {
    check(
        values->ndim() == 1 && values->size() == col.size(),
        "values must be 1-D with one entry per edge.");
  }
  const int num_rows = static_cast<int>(rowptr.size()) - 1;

  std::vector<mx::array> inputs = {
      as_index(rowptr, s), as_index(col, s), as_float(x, s)};
  if (values) {
    inputs.push_back(as_float(*values, s));
  }
  auto push_long = [&](const LongRows& lr) {
    check(lr.threshold >= 0, "long_row_threshold must be >= 0.");
    check(
        lr.chunk_ptr.size() == lr.rows.size() + 1,
        "chunk_ptr must have len(long_rows) + 1 entries.");
    check(
        lr.chunk_start.size() == lr.chunk_end.size(),
        "chunk_start and chunk_end must have the same length.");
    inputs.push_back(as_index(lr.rows, s));
    inputs.push_back(as_index(lr.chunk_ptr, s));
    inputs.push_back(as_index(lr.chunk_start, s));
    inputs.push_back(as_index(lr.chunk_end, s));
  };
  if (long_rows) {
    push_long(*long_rows);
  }
  if (transpose) {
    check(
        transpose->rowptr.ndim() == 1 &&
            transpose->rowptr.size() == static_cast<size_t>(x.shape(0)) + 1,
        "transpose.rowptr must have x.shape[0] + 1 entries.");
    check(
        transpose->col.size() == col.size() && transpose->perm.size() == col.size(),
        "transpose.col and transpose.perm must have one entry per edge.");
    inputs.push_back(as_index(transpose->rowptr, s));
    inputs.push_back(as_index(transpose->col, s));
    inputs.push_back(as_index(transpose->perm, s));
    if (transpose->long_rows) {
      push_long(*transpose->long_rows);
    }
  }
  return mx::array(
      {num_rows, x.shape(1)},
      mx::float32,
      std::make_shared<SpMM>(
          mx::to_stream(s),
          SparseFormat::CSR,
          values.has_value(),
          num_rows,
          long_rows.has_value(),
          long_rows ? long_rows->threshold : 0,
          transpose.has_value(),
          transpose && transpose->long_rows.has_value(),
          (transpose && transpose->long_rows) ? transpose->long_rows->threshold
                                              : 0,
          reduce),
      std::move(inputs));
}

mx::array sddmm(
    const mx::array& row,
    const mx::array& col,
    const mx::array& a,
    const mx::array& b,
    mx::StreamOrDevice s) {
  check(row.ndim() == 1 && row.size() == col.size(), "row/col mismatch.");
  check(a.ndim() == 2 && b.ndim() == 2, "a and b must be 2-D.");
  check(a.shape(1) == b.shape(1), "a and b must have the same feature dim.");
  return mx::array(
      {static_cast<int>(row.size())},
      mx::float32,
      std::make_shared<SDDMM>(mx::to_stream(s)),
      {as_index(row, s), as_index(col, s), as_float(a, s), as_float(b, s)});
}

mx::array csr_to_row(
    const mx::array& rowptr,
    int64_t num_edges,
    mx::StreamOrDevice s) {
  // Mark each row boundary with +1, then a cumulative sum gives the row id
  // of every edge. Duplicate boundaries (empty rows) add up correctly.
  const int N = static_cast<int>(rowptr.size()) - 1;
  const int E = static_cast<int>(num_edges);
  if (E == 0) {
    return mx::zeros({0}, mx::int64, s);
  }
  // int32 marks: MLX's GPU scatter does not support int64 inputs/updates.
  auto marks = mx::zeros({E + 1}, mx::int32, s);
  if (N > 1) {
    auto bounds = mx::slice(as_index(rowptr, s), {1}, {N}, s); // rowptr[1:N]
    auto ones = mx::ones({N - 1, 1}, mx::int32, s);
    marks = mx::scatter_add(marks, bounds, ones, 0, s);
  }
  auto rows = mx::cumsum(marks, 0, false, true, s);
  return mx::astype(mx::slice(rows, {0}, {E}, s), mx::int64, s);
}

} // namespace mlx_sparse

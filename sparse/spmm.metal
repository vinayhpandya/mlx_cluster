// Sparse x dense matmul kernels for message passing.
//
//   out[r, :] = sum_{e : row(e) == r} value[e] * x[col[e], :]
//
// Grid layout for both SpMM kernels is (features, rows-or-edges): the feature
// index is the fast axis, so the 32 lanes of a SIMD group read 32 contiguous
// floats of x and write 32 contiguous floats of out (coalesced).

#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Reductions. OP: 0 = sum, 1 = max, 2 = min.
// ---------------------------------------------------------------------------
template <int OP>
inline float reduce_identity() {
  return OP == 0 ? 0.0f : (OP == 1 ? -INFINITY : INFINITY);
}

template <int OP>
inline float reduce_combine(float a, float b) {
  return OP == 0 ? a + b : (OP == 1 ? max(a, b) : min(a, b));
}

// Order-preserving float <-> uint mapping: a < b  <=>  key(a) < key(b).
// Lets max/min use the uint atomics (Metal has no float atomic max/min).
inline uint float_to_key(float f) {
  const uint b = as_type<uint>(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

inline float key_to_float(uint k) {
  const uint b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return as_type<float>(b);
}

// ---------------------------------------------------------------------------
// Zero fill (used before the COO kernel, which accumulates atomically)
// ---------------------------------------------------------------------------
[[kernel]] void spmm_zero(
    device float* out [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid < n) {
    out[tid] = 0.0f;
  }
}

// ---------------------------------------------------------------------------
// Fill a buffer with a uint pattern (the "empty" sentinel for COO max/min).
// ---------------------------------------------------------------------------
[[kernel]] void spmm_fill_u32(
    device uint* out [[buffer(0)]],
    constant uint& value [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid < n) {
    out[tid] = value;
  }
}

// ---------------------------------------------------------------------------
// COO max/min: one thread per (feature, edge), uint atomic max/min of the
// message's order-preserving key. `out` holds keys until spmm_decode runs.
// ---------------------------------------------------------------------------
template <bool HAS_VALUES, bool IS_MAX>
[[kernel]] void spmm_coo_maxmin(
    const device long* row [[buffer(0)]],
    const device long* col [[buffer(1)]],
    const device float* values [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device atomic_uint* out [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& E [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x;
  const uint e = gid.y;
  if (f >= D || e >= E) {
    return;
  }
  const ulong r = static_cast<ulong>(row[e]);
  const ulong c = static_cast<ulong>(col[e]);
  float v = x[c * D + f];
  if (HAS_VALUES) {
    v *= values[e];
  }
  const uint key = float_to_key(v);
  if (IS_MAX) {
    atomic_fetch_max_explicit(&out[r * D + f], key, memory_order_relaxed);
  } else {
    atomic_fetch_min_explicit(&out[r * D + f], key, memory_order_relaxed);
  }
}

// In place: keys -> floats. Elements still holding the sentinel belong to
// rows without edges and become 0 (same as mlx-graphs' scatter max/min).
template <bool IS_MAX>
[[kernel]] void spmm_decode(
    device uint* buf [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= n) {
    return;
  }
  const uint k = buf[tid];
  const uint sentinel = IS_MAX ? 0u : 0xFFFFFFFFu;
  const float v = (k == sentinel) ? 0.0f : key_to_float(k);
  buf[tid] = as_type<uint>(v);
}

// ---------------------------------------------------------------------------
// COO: one thread per (feature, edge), atomic add into the destination row.
// Simple and works on an unsorted edge list, but contended for high-degree
// rows and float sums are not bit-for-bit deterministic across runs.
// ---------------------------------------------------------------------------
template <bool HAS_VALUES>
[[kernel]] void spmm_coo(
    const device long* row [[buffer(0)]],
    const device long* col [[buffer(1)]],
    const device float* values [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device atomic<float>* out [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& E [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x;
  const uint e = gid.y;
  if (f >= D || e >= E) {
    return;
  }
  const ulong r = static_cast<ulong>(row[e]);
  const ulong c = static_cast<ulong>(col[e]);
  float v = x[c * D + f];
  if (HAS_VALUES) {
    v *= values[e];
  }
  atomic_fetch_add_explicit(&out[r * D + f], v, memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// CSR: one thread per (feature, row), sequential loop over the row's edges.
// No atomics, deterministic, each output element written exactly once.
// Rows with more than `long_threshold` edges are skipped here and handled by
// spmm_csr_long_partial + spmm_csr_long_reduce below (the host passes
// UINT_MAX when there is no long-row metadata, so nothing is skipped).
// ---------------------------------------------------------------------------
template <bool HAS_VALUES, int OP = 0>
[[kernel]] void spmm_csr(
    const device long* rowptr [[buffer(0)]],
    const device long* col [[buffer(1)]],
    const device float* values [[buffer(2)]],
    const device float* x [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& long_threshold [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x;
  const uint r = gid.y;
  if (f >= D || r >= N) {
    return;
  }
  const long start = rowptr[r];
  const long end = rowptr[r + 1];
  if (end - start > static_cast<long>(long_threshold)) {
    return; // long row: written by spmm_csr_long_reduce
  }
  float acc = reduce_identity<OP>();
  for (long e = start; e < end; ++e) {
    const ulong c = static_cast<ulong>(col[e]);
    float v = x[c * D + f];
    if (HAS_VALUES) {
      v *= values[e];
    }
    acc = reduce_combine<OP>(acc, v);
  }
  if (OP != 0 && end == start) {
    acc = 0.0f; // max/min of an empty row is 0, like mlx-graphs
  }
  out[static_cast<ulong>(r) * D + f] = acc;
}

// ---------------------------------------------------------------------------
// CSR long rows, phase 1: one threadgroup per (feature tile, chunk), where a
// chunk is a run of consecutive edges of one long row (host: coo_to_csr).
// Threads are laid out as (FX features) x (EY edge lanes), FX * EY = 256.
// Each thread sums every EY-th edge of the chunk for one feature, then the EY
// partial sums are combined in threadgroup memory -> partials[chunk, f].
// ---------------------------------------------------------------------------
template <bool HAS_VALUES, int OP = 0>
[[kernel]] void spmm_csr_long_partial(
    const device long* chunk_start [[buffer(0)]],
    const device long* chunk_end [[buffer(1)]],
    const device long* col [[buffer(2)]],
    const device float* values [[buffer(3)]],
    const device float* x [[buffer(4)]],
    device float* partials [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  threadgroup float scratch[256];
  const uint f = tg.x * tsize.x + lid.x;
  const uint k = tg.y;
  float acc = reduce_identity<OP>();
  if (f < D) {
    const long e1 = chunk_end[k];
    for (long e = chunk_start[k] + lid.y; e < e1; e += tsize.y) {
      const ulong c = static_cast<ulong>(col[e]);
      float v = x[c * D + f];
      if (HAS_VALUES) {
        v *= values[e];
      }
      acc = reduce_combine<OP>(acc, v);
    }
  }
  // no early return above: every thread must reach the barrier
  scratch[lid.y * tsize.x + lid.x] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0 && f < D) {
    float total = reduce_identity<OP>();
    for (uint j = 0; j < tsize.y; ++j) {
      total = reduce_combine<OP>(total, scratch[j * tsize.x + lid.x]);
    }
    partials[static_cast<ulong>(k) * D + f] = total;
  }
}

// ---------------------------------------------------------------------------
// CSR long rows, phase 2: one thread per (feature, long row). Adds the row's
// chunk sums in a fixed order (deterministic) and writes the output row.
// ---------------------------------------------------------------------------
template <int OP = 0>
[[kernel]] void spmm_csr_long_reduce(
    const device long* rows [[buffer(0)]],
    const device long* chunk_ptr [[buffer(1)]],
    const device float* partials [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& L [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x;
  const uint l = gid.y;
  if (f >= D || l >= L) {
    return;
  }
  float total = reduce_identity<OP>();
  for (long k = chunk_ptr[l]; k < chunk_ptr[l + 1]; ++k) {
    total = reduce_combine<OP>(total, partials[static_cast<ulong>(k) * D + f]);
  }
  out[static_cast<ulong>(rows[l]) * D + f] = total;
}

// ---------------------------------------------------------------------------
// Backward of max/min ("select" gradient). An edge won output element
// (row_e, f) iff its message x[col_e, f] * w_e equals out[row_e, f]; ties all
// win, matching MLX's scatter max/min VJP. Recomputing the message with the
// exact same operations as the forward makes the comparison bit-exact.
// ---------------------------------------------------------------------------
template <bool HAS_VALUES>
[[kernel]] void spmm_select_grad_x(
    const device long* row [[buffer(0)]],
    const device long* col [[buffer(1)]],
    const device float* x [[buffer(2)]],
    const device float* out [[buffer(3)]],
    const device float* g [[buffer(4)]],
    const device float* values [[buffer(5)]],
    device atomic<float>* gx [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    constant uint& E [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x;
  const uint e = gid.y;
  if (f >= D || e >= E) {
    return;
  }
  const ulong r = static_cast<ulong>(row[e]);
  const ulong c = static_cast<ulong>(col[e]);
  float m = x[c * D + f];
  float w = 1.0f;
  if (HAS_VALUES) {
    w = values[e];
    m *= w;
  }
  if (m == out[r * D + f]) {
    atomic_fetch_add_explicit(&gx[c * D + f], w * g[r * D + f], memory_order_relaxed);
  }
}

// One SIMD group per edge: gw[e] = sum over won features of x * g.
[[kernel]] void spmm_select_grad_w(
    const device long* row [[buffer(0)]],
    const device long* col [[buffer(1)]],
    const device float* x [[buffer(2)]],
    const device float* out [[buffer(3)]],
    const device float* g [[buffer(4)]],
    const device float* values [[buffer(5)]],
    device float* gw [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    constant uint& E [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint e = gid.y;
  if (e >= E) {
    return;
  }
  const ulong r = static_cast<ulong>(row[e]);
  const ulong c = static_cast<ulong>(col[e]);
  const float w = values[e];
  float acc = 0.0f;
  for (uint f = lane; f < D; f += 32) {
    const float xv = x[c * D + f];
    float m = xv;
    m *= w;
    if (m == out[r * D + f]) {
      acc += xv * g[r * D + f];
    }
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    gw[e] = acc;
  }
}

// ---------------------------------------------------------------------------
// Max/min gradient w.r.t. x on the TRANSPOSED CSR (rows = source nodes).
// One thread per (feature, source node c) loops over c's outgoing edges and
// adds w * g[dst] for every edge that won: no atomics, deterministic. Rows
// with more than `long_threshold` edges are skipped and handled by the
// chunked kernel below (+ spmm_csr_long_reduce).
// ---------------------------------------------------------------------------
template <bool HAS_VALUES>
[[kernel]] void spmm_select_grad_xt(
    const device long* t_rowptr [[buffer(0)]],
    const device long* t_col [[buffer(1)]],
    const device float* x [[buffer(2)]],
    const device float* out [[buffer(3)]],
    const device float* g [[buffer(4)]],
    const device float* values [[buffer(5)]],
    device float* gx [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    constant uint& Nx [[buffer(8)]],
    constant uint& long_threshold [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint f = gid.x;
  const uint c = gid.y;
  if (f >= D || c >= Nx) {
    return;
  }
  const long start = t_rowptr[c];
  const long end = t_rowptr[c + 1];
  if (end - start > static_cast<long>(long_threshold)) {
    return; // long row: written by spmm_csr_long_reduce
  }
  const float xv = x[static_cast<ulong>(c) * D + f];
  float acc = 0.0f;
  for (long i = start; i < end; ++i) {
    const ulong r = static_cast<ulong>(t_col[i]);
    float m = xv;
    float w = 1.0f;
    if (HAS_VALUES) {
      w = values[i];
      m *= w; // same operation as the forward
    }
    if (m == out[r * D + f]) {
      acc += w * g[r * D + f];
    }
  }
  gx[static_cast<ulong>(c) * D + f] = acc;
}

// Long rows of the transpose, phase 1: one threadgroup per (feature tile,
// chunk); (FX features) x (EY edge lanes) threads, partial sums combined in
// threadgroup memory -> partials[chunk, f]. The chunk's source node is found
// by binary search in chunk_ptr.
template <bool HAS_VALUES>
[[kernel]] void spmm_select_grad_xt_long_partial(
    const device long* rows [[buffer(0)]],
    const device long* chunk_ptr [[buffer(1)]],
    const device long* chunk_start [[buffer(2)]],
    const device long* chunk_end [[buffer(3)]],
    const device long* t_col [[buffer(4)]],
    const device float* x [[buffer(5)]],
    const device float* out [[buffer(6)]],
    const device float* g [[buffer(7)]],
    const device float* values [[buffer(8)]],
    device float* partials [[buffer(9)]],
    constant uint& D [[buffer(10)]],
    constant uint& L [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tsize [[threads_per_threadgroup]]) {
  threadgroup float scratch[256];
  const uint f = tg.x * tsize.x + lid.x;
  const uint k = tg.y;

  // long row owning chunk k: chunk_ptr[l] <= k < chunk_ptr[l + 1]
  uint lo = 0;
  uint hi = L;
  while (lo + 1 < hi) {
    const uint mid = (lo + hi) / 2;
    if (chunk_ptr[mid] <= static_cast<long>(k)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const ulong c = static_cast<ulong>(rows[lo]);

  float acc = 0.0f;
  if (f < D) {
    const float xv = x[c * D + f];
    const long e1 = chunk_end[k];
    for (long i = chunk_start[k] + lid.y; i < e1; i += tsize.y) {
      const ulong r = static_cast<ulong>(t_col[i]);
      float m = xv;
      float w = 1.0f;
      if (HAS_VALUES) {
        w = values[i];
        m *= w;
      }
      if (m == out[r * D + f]) {
        acc += w * g[r * D + f];
      }
    }
  }
  // no early return above: every thread must reach the barrier
  scratch[lid.y * tsize.x + lid.x] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0 && f < D) {
    float total = 0.0f;
    for (uint j = 0; j < tsize.y; ++j) {
      total += scratch[j * tsize.x + lid.x];
    }
    partials[static_cast<ulong>(k) * D + f] = total;
  }
}

// Same instantiation idiom as MLX's own kernels (mlx/backend/metal/kernels).
#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) \
      func<__VA_ARGS__>;

instantiate_kernel("spmm_coo", spmm_coo, false)
instantiate_kernel("spmm_coo_val", spmm_coo, true)
instantiate_kernel("spmm_csr", spmm_csr, false)
instantiate_kernel("spmm_csr_val", spmm_csr, true)
instantiate_kernel("spmm_csr_long_partial", spmm_csr_long_partial, false)
instantiate_kernel("spmm_csr_long_partial_val", spmm_csr_long_partial, true)
instantiate_kernel("spmm_csr_long_reduce", spmm_csr_long_reduce, 0)

// max / min
instantiate_kernel("spmm_coo_max", spmm_coo_maxmin, false, true)
instantiate_kernel("spmm_coo_max_val", spmm_coo_maxmin, true, true)
instantiate_kernel("spmm_coo_min", spmm_coo_maxmin, false, false)
instantiate_kernel("spmm_coo_min_val", spmm_coo_maxmin, true, false)
instantiate_kernel("spmm_decode_max", spmm_decode, true)
instantiate_kernel("spmm_decode_min", spmm_decode, false)
instantiate_kernel("spmm_csr_max", spmm_csr, false, 1)
instantiate_kernel("spmm_csr_max_val", spmm_csr, true, 1)
instantiate_kernel("spmm_csr_min", spmm_csr, false, 2)
instantiate_kernel("spmm_csr_min_val", spmm_csr, true, 2)
instantiate_kernel("spmm_csr_long_partial_max", spmm_csr_long_partial, false, 1)
instantiate_kernel("spmm_csr_long_partial_max_val", spmm_csr_long_partial, true, 1)
instantiate_kernel("spmm_csr_long_partial_min", spmm_csr_long_partial, false, 2)
instantiate_kernel("spmm_csr_long_partial_min_val", spmm_csr_long_partial, true, 2)
instantiate_kernel("spmm_csr_long_reduce_max", spmm_csr_long_reduce, 1)
instantiate_kernel("spmm_csr_long_reduce_min", spmm_csr_long_reduce, 2)
instantiate_kernel("spmm_select_grad_x", spmm_select_grad_x, false)
instantiate_kernel("spmm_select_grad_x_val", spmm_select_grad_x, true)
instantiate_kernel("spmm_select_grad_xt", spmm_select_grad_xt, false)
instantiate_kernel("spmm_select_grad_xt_val", spmm_select_grad_xt, true)
instantiate_kernel(
    "spmm_select_grad_xt_long_partial", spmm_select_grad_xt_long_partial, false)
instantiate_kernel(
    "spmm_select_grad_xt_long_partial_val", spmm_select_grad_xt_long_partial, true)

// ---------------------------------------------------------------------------
// SDDMM: out[e] = <a[row[e]], b[col[e]]>. One SIMD group per edge; lanes
// stride over the feature dim, then reduce with simd_sum.
// The host dispatches a (32, E) grid with (32, 8) threadgroups, so all 32
// lanes of a SIMD group share the same edge and exit together.
// ---------------------------------------------------------------------------
[[kernel]] void sddmm(
    const device long* row [[buffer(0)]],
    const device long* col [[buffer(1)]],
    const device float* a [[buffer(2)]],
    const device float* b [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& D [[buffer(5)]],
    constant uint& E [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint e = gid.y;
  if (e >= E) {
    return;
  }
  const ulong r = static_cast<ulong>(row[e]);
  const ulong c = static_cast<ulong>(col[e]);
  float acc = 0.0f;
  for (uint f = lane; f < D; f += 32) {
    acc += a[r * D + f] * b[c * D + f];
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    out[e] = acc;
  }
}

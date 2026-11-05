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
template <bool HAS_VALUES>
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
  float acc = 0.0f;
  for (long e = start; e < end; ++e) {
    const ulong c = static_cast<ulong>(col[e]);
    float v = x[c * D + f];
    if (HAS_VALUES) {
      v *= values[e];
    }
    acc += v;
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
template <bool HAS_VALUES>
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
  float acc = 0.0f;
  if (f < D) {
    const long e1 = chunk_end[k];
    for (long e = chunk_start[k] + lid.y; e < e1; e += tsize.y) {
      const ulong c = static_cast<ulong>(col[e]);
      float v = x[c * D + f];
      if (HAS_VALUES) {
        v *= values[e];
      }
      acc += v;
    }
  }
  // no early return above: every thread must reach the barrier
  scratch[lid.y * tsize.x + lid.x] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0 && f < D) {
    float sum = 0.0f;
    for (uint j = 0; j < tsize.y; ++j) {
      sum += scratch[j * tsize.x + lid.x];
    }
    partials[static_cast<ulong>(k) * D + f] = sum;
  }
}

// ---------------------------------------------------------------------------
// CSR long rows, phase 2: one thread per (feature, long row). Adds the row's
// chunk sums in a fixed order (deterministic) and writes the output row.
// ---------------------------------------------------------------------------
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
  float sum = 0.0f;
  for (long k = chunk_ptr[l]; k < chunk_ptr[l + 1]; ++k) {
    sum += partials[static_cast<ulong>(k) * D + f];
  }
  out[static_cast<ulong>(rows[l]) * D + f] = sum;
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

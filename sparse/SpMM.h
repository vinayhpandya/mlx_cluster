#pragma once

#include <optional>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/primitives.h>

namespace mx = mlx::core;

namespace mlx_sparse {

/**
 * Storage format of the sparse matrix A in out = A @ X.
 *
 *  COO: inputs are (row, col), one entry per edge, any order.
 *       GPU kernel: one thread per (edge, feature), atomic adds into out.
 *  CSR: inputs are (rowptr, col), edges grouped by row.
 *       GPU kernel: one thread per (row, feature), no atomics, deterministic.
 */
enum class SparseFormat { COO = 0, CSR = 1 };

/**
 * Sparse x dense matmul with sum reduction:
 *
 *   out[r, :] = sum_{e : row(e) == r} value[e] * x[col[e], :]
 *
 * Inputs (in order):
 *   0: row (COO, int64 [E]) or rowptr (CSR, int64 [num_rows + 1])
 *   1: col   int64   [E]
 *   2: x     float32 [num_cols, D]
 *   3: value float32 [E]   (only if has_values)
 *   then, only for CSR with has_long (see LongRows):
 *      long_rows int64 [L], chunk_ptr int64 [L + 1],
 *      chunk_start int64 [K], chunk_end int64 [K]
 *
 * Output: float32 [num_rows, D]
 *
 * In message-passing terms: row = destination node, col = source node,
 * x = node features, value = edge weight.
 */
class SpMM : public mx::Primitive {
 public:
  SpMM(
      mx::Stream stream,
      SparseFormat format,
      bool has_values,
      int num_rows,
      bool has_long = false,
      int long_threshold = 0)
      : mx::Primitive(stream),
        format_(format),
        has_values_(has_values),
        num_rows_(num_rows),
        has_long_(has_long),
        long_threshold_(long_threshold) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return format_ == SparseFormat::COO ? "SpMMCOO" : "SpMMCSR";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  SparseFormat format_;
  bool has_values_;
  int num_rows_;
  bool has_long_;
  int long_threshold_;
};

/**
 * Sampled dense-dense matmul (one dot product per edge):
 *
 *   out[e] = sum_d a[row[e], d] * b[col[e], d]
 *
 * Inputs: row int64 [E], col int64 [E], a float32 [Na, D], b float32 [Nb, D]
 * Output: float32 [E]
 *
 * Used as the gradient of SpMM w.r.t. the edge values, so the backward
 * pass never materialises an [E, D] array either. Not differentiable itself.
 */
class SDDMM : public mx::Primitive {
 public:
  explicit SDDMM(mx::Stream stream) : mx::Primitive(stream) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> jvp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const override {
    return "SDDMM";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    return dynamic_cast<const SDDMM*>(&other) != nullptr;
  }
};

/** out[row[e]] += value[e] * x[col[e]]   (COO, sum). */
mx::array spmm_coo(
    const mx::array& row,
    const mx::array& col,
    const mx::array& x,
    const std::optional<mx::array>& values,
    int num_rows,
    mx::StreamOrDevice s = {});

/**
 * Load-balancing metadata for CSR rows with many edges ("long rows").
 *
 * Rows with more than `threshold` edges are skipped by the regular
 * one-thread-per-(row, feature) CSR kernel. Instead each long row is cut into
 * chunks of consecutive edges; every chunk is summed by its own threadgroup,
 * and a final pass adds each row's chunk sums in a fixed order (so results
 * stay deterministic). Built once per graph by mlx_cluster.coo_to_csr.
 *
 *   rows        int64 [L]      ids of the long rows
 *   chunk_ptr   int64 [L + 1]  row l owns chunks chunk_ptr[l]:chunk_ptr[l+1]
 *   chunk_start int64 [K]      first edge of each chunk
 *   chunk_end   int64 [K]      one past the last edge of each chunk
 */
struct LongRows {
  mx::array rows;
  mx::array chunk_ptr;
  mx::array chunk_start;
  mx::array chunk_end;
  int threshold;
};

/** Same as spmm_coo but A given in CSR (rowptr, col). num_rows = |rowptr|-1. */
mx::array spmm_csr(
    const mx::array& rowptr,
    const mx::array& col,
    const mx::array& x,
    const std::optional<mx::array>& values,
    const std::optional<LongRows>& long_rows = std::nullopt,
    mx::StreamOrDevice s = {});

/** out[e] = <a[row[e]], b[col[e]]>. */
mx::array sddmm(
    const mx::array& row,
    const mx::array& col,
    const mx::array& a,
    const mx::array& b,
    mx::StreamOrDevice s = {});

/** Expand a CSR rowptr [N+1] into a per-edge row array [E] (int64). */
mx::array csr_to_row(
    const mx::array& rowptr,
    int64_t num_edges,
    mx::StreamOrDevice s = {});

} // namespace mlx_sparse

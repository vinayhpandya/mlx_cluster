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
 * How messages arriving at a row are combined.
 *   Sum: out[r] = sum_e  w_e * x[col_e]
 *   Max: out[r] = max_e  w_e * x[col_e]   (per feature; 0 for empty rows)
 *   Min: out[r] = min_e  w_e * x[col_e]   (per feature; 0 for empty rows)
 * ("mean" is Sum divided by the degree, done by the Python wrapper.)
 *
 * Max/Min gradients follow MLX's scatter max/min: every edge whose message
 * equals the output (ties included) receives the full output gradient.
 */
enum class SpMMReduce { Sum = 0, Max = 1, Min = 2 };

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
 *   then, only for CSR with has_transpose (see CSRTranspose):
 *      t_rowptr int64 [num_cols + 1], t_col int64 [E], t_perm int64 [E],
 *      and, with has_long_t, the 4 LongRows arrays of the transpose.
 *   The transpose inputs are not used by the forward pass; they are carried
 *   so the backward (A^T @ g) can use the CSR kernel instead of COO atomics.
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
      int long_threshold = 0,
      bool has_transpose = false,
      bool has_long_t = false,
      int long_threshold_t = 0,
      SpMMReduce reduce = SpMMReduce::Sum)
      : mx::Primitive(stream),
        format_(format),
        has_values_(has_values),
        num_rows_(num_rows),
        has_long_(has_long),
        long_threshold_(long_threshold),
        has_transpose_(has_transpose),
        has_long_t_(has_long_t),
        long_threshold_t_(long_threshold_t),
        reduce_(reduce) {}

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
    static const char* names[2][3] = {
        {"SpMMCOO", "SpMMCOOMax", "SpMMCOOMin"},
        {"SpMMCSR", "SpMMCSRMax", "SpMMCSRMin"}};
    return names[static_cast<int>(format_)][static_cast<int>(reduce_)];
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  SparseFormat format_;
  bool has_values_;
  int num_rows_;
  bool has_long_;
  int long_threshold_;
  bool has_transpose_;
  bool has_long_t_;
  int long_threshold_t_;
  SpMMReduce reduce_;

  // offsets of the optional input groups
  size_t long_offset() const {
    return 3 + (has_values_ ? 1 : 0);
  }
  size_t transpose_offset() const {
    return long_offset() + (has_long_ ? 4 : 0);
  }
};

/**
 * Backward of SpMM with Max/Min reduction ("select" gradient). With
 * m_e = w_e * x[col_e] and out = spmm(..., Max|Min), the edges that "won"
 * are those with m_e == out[row_e] (per feature, ties included):
 *
 *   target X:      gx[c, f] = sum_{e: col_e = c, won} w_e * g[row_e, f]
 *   target Values: gw[e]    = sum_{f: won}            x[col_e, f] * g[row_e, f]
 *
 * Inputs: row int64 [E], col int64 [E], x float32 [Nx, D],
 *         out float32 [N, D], g float32 [N, D], (values float32 [E])
 * Output: X -> float32 [Nx, D] ; Values -> float32 [E]
 * Not differentiable itself.
 */
class SpMMSelectGrad : public mx::Primitive {
 public:
  enum class Target { X = 0, Values = 1 };

  SpMMSelectGrad(mx::Stream stream, Target target, bool has_values)
      : mx::Primitive(stream), target_(target), has_values_(has_values) {}

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
    return target_ == Target::X ? "SpMMSelectGradX" : "SpMMSelectGradValues";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const SpMMSelectGrad*>(&other);
    return o && o->target_ == target_ && o->has_values_ == has_values_;
  }

 private:
  Target target_;
  bool has_values_;
};

/**
 * Max/Min gradient w.r.t. x computed on the transposed CSR (edges grouped by
 * source node): one thread per (source node c, feature f),
 *
 *   gx[c, f] = sum_{i in rows_t(c), won} w_i * g[t_col[i], f]
 *
 * where w_i are the edge values in transposed order and "won" means
 * x[c, f] * w_i == out[t_col[i], f]. No atomics, deterministic.
 *
 * Inputs: t_rowptr int64 [Nx + 1], t_col int64 [E], x float32 [Nx, D],
 *         out float32 [N, D], g float32 [N, D], (values_t float32 [E]),
 *         (long-row metadata of the transpose: 4 int64 arrays)
 * Output: float32 [Nx, D]. Not differentiable itself.
 */
class SpMMSelectGradT : public mx::Primitive {
 public:
  SpMMSelectGradT(
      mx::Stream stream,
      bool has_values,
      bool has_long,
      int long_threshold)
      : mx::Primitive(stream),
        has_values_(has_values),
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
    return "SpMMSelectGradXTransposed";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const SpMMSelectGradT*>(&other);
    return o && o->has_values_ == has_values_ && o->has_long_ == has_long_ &&
        o->long_threshold_ == long_threshold_;
  }

 private:
  bool has_values_;
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

/** out[row[e]] (+|max|min)= value[e] * x[col[e]]   (COO). */
mx::array spmm_coo(
    const mx::array& row,
    const mx::array& col,
    const mx::array& x,
    const std::optional<mx::array>& values,
    int num_rows,
    SpMMReduce reduce = SpMMReduce::Sum,
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

/**
 * The transpose of a CSR matrix (edges grouped by source node), used to run
 * the backward pass A^T @ g with the CSR kernel. Built once per graph by
 * mlx_cluster.coo_to_csr.
 *
 *   rowptr     int64 [num_cols + 1]
 *   col        int64 [E]   destination (original row) of each edge
 *   perm       int64 [E]   transposed edge i is CSR edge perm[i]
 *   long_rows  optional LongRows for high out-degree nodes
 */
struct CSRTranspose {
  mx::array rowptr;
  mx::array col;
  mx::array perm;
  std::optional<LongRows> long_rows;
};

/** Same as spmm_coo but A given in CSR (rowptr, col). num_rows = |rowptr|-1. */
mx::array spmm_csr(
    const mx::array& rowptr,
    const mx::array& col,
    const mx::array& x,
    const std::optional<mx::array>& values,
    const std::optional<LongRows>& long_rows = std::nullopt,
    const std::optional<CSRTranspose>& transpose = std::nullopt,
    SpMMReduce reduce = SpMMReduce::Sum,
    mx::StreamOrDevice s = {});

/** Max/Min gradient w.r.t. x on the transposed CSR (see SpMMSelectGradT). */
mx::array spmm_select_grad_t(
    const mx::array& t_rowptr,
    const mx::array& t_col,
    const mx::array& x,
    const mx::array& out,
    const mx::array& g,
    const std::optional<mx::array>& values_t,
    const std::optional<LongRows>& long_rows_t,
    mx::StreamOrDevice s = {});

/** Gradient of a Max/Min SpMM w.r.t. x or the edge values (see SpMMSelectGrad). */
mx::array spmm_select_grad(
    SpMMSelectGrad::Target target,
    const mx::array& row,
    const mx::array& col,
    const mx::array& x,
    const mx::array& out,
    const mx::array& g,
    const std::optional<mx::array>& values,
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

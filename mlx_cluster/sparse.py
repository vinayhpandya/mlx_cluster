"""Sparse x dense matrix multiplication (SpMM) for message passing.

These ops compute ``out = A @ x`` where ``A`` is a sparse adjacency matrix
and ``x`` is a dense node-feature matrix, without materialising the
``[num_edges, features]`` array that gather-scatter message passing creates.

Conventions (match ``mlx_graphs`` ``edge_index = [src, dst]``):

* ``row`` = destination node of each edge (``edge_index[1]``)
* ``col`` = source node of each edge      (``edge_index[0]``)

so ``spmm_coo(edge_index[1], edge_index[0], x)`` sums each node's incoming
neighbour features, i.e. the same as
``scatter(x[edge_index[0]], edge_index[1], num_nodes, "add")``.

Both ops are differentiable w.r.t. ``x`` and ``values``.
"""

from typing import Literal, NamedTuple, Optional

import mlx.core as mx
import numpy as np

from ._ext import _spmm_coo, _spmm_csr, sddmm

SpMMReduce = Literal["sum", "add", "mean"]

__all__ = [
    "CSR",
    "LongRows",
    "coo_to_csr",
    "sddmm",
    "spmm_coo",
    "spmm_csr",
]


class LongRows(NamedTuple):
    """Load-balancing metadata for CSR rows with many edges.

    Rows with more than ``threshold`` edges are not handled by the regular
    one-thread-per-(row, feature) CSR kernel, where a single thread would
    loop over every edge of a hub node. Instead each such row is cut into
    chunks of at most ``chunk_size`` consecutive edges, each chunk is summed
    by its own GPU threadgroup, and the chunk sums are added in a fixed
    order (results stay deterministic). Built by :func:`coo_to_csr`.

    Attributes:
        rows: ``int64 [L]`` ids of the long rows.
        chunk_ptr: ``int64 [L + 1]``; long row ``l`` owns chunks
            ``chunk_ptr[l]:chunk_ptr[l + 1]``.
        chunk_start: ``int64 [K]`` first edge of each chunk.
        chunk_end: ``int64 [K]`` one past the last edge of each chunk.
        threshold: rows with more edges than this are "long".
    """

    rows: mx.array
    chunk_ptr: mx.array
    chunk_start: mx.array
    chunk_end: mx.array
    threshold: int


def build_long_rows(
    rowptr: mx.array, threshold: int = 256, chunk_size: int = 2048
) -> Optional[LongRows]:
    """Finds rows with more than ``threshold`` edges and splits them into
    chunks of at most ``chunk_size`` edges. Returns ``None`` if there are none.

    Runs once per graph (it reads ``rowptr`` back to the host).
    """
    if threshold < 0 or chunk_size < 1:
        raise ValueError("threshold must be >= 0 and chunk_size >= 1")
    rp = np.array(rowptr, dtype=np.int64)
    deg = np.diff(rp)
    rows = np.nonzero(deg > threshold)[0]
    if rows.size == 0:
        return None
    n_chunks = (deg[rows] + chunk_size - 1) // chunk_size
    chunk_ptr = np.concatenate([[0], np.cumsum(n_chunks)]).astype(np.int64)
    owner = np.repeat(rows, n_chunks)  # long row of each chunk
    local = np.arange(chunk_ptr[-1]) - np.repeat(chunk_ptr[:-1], n_chunks)
    start = rp[owner] + local * chunk_size
    end = np.minimum(start + chunk_size, rp[owner + 1])
    return LongRows(
        rows=mx.array(rows.astype(np.int64)),
        chunk_ptr=mx.array(chunk_ptr),
        chunk_start=mx.array(start.astype(np.int64)),
        chunk_end=mx.array(end.astype(np.int64)),
        threshold=int(threshold),
    )


class CSR(NamedTuple):
    """A sparse matrix in CSR format.

    Attributes:
        rowptr: ``int64 [num_rows + 1]``; row ``r`` owns edges
            ``rowptr[r]:rowptr[r + 1]``.
        col: ``int64 [num_edges]`` column (source node) of each edge.
        values: optional ``[num_edges]`` edge weights, in CSR order.
        perm: ``[num_edges]`` permutation such that ``col == input_col[perm]``.
            Use it to reorder any other per-edge array into CSR order.
        long_rows: optional :class:`LongRows` metadata for high-degree rows.
    """

    rowptr: mx.array
    col: mx.array
    values: Optional[mx.array]
    perm: mx.array
    long_rows: Optional[LongRows] = None

    @property
    def num_rows(self) -> int:
        return self.rowptr.size - 1


def _check_reduce(reduce: str) -> str:
    if reduce in ("sum", "add"):
        return "sum"
    if reduce == "mean":
        return "mean"
    raise ValueError(f"reduce must be 'sum', 'add' or 'mean', got {reduce!r}")


def coo_to_csr(
    row: mx.array,
    col: mx.array,
    num_rows: int,
    values: Optional[mx.array] = None,
    long_row_threshold: Optional[int] = 256,
    chunk_size: int = 2048,
) -> CSR:
    """Converts a COO edge list into CSR (sorted by ``row``).

    This needs a sort, so compute it once per graph and reuse it.

    Args:
        long_row_threshold: rows with more edges than this get the
            load-balanced long-row path in :func:`spmm_csr` (see
            :class:`LongRows`). ``None`` disables it and keeps the conversion
            fully lazy (no host read-back), e.g. inside ``mx.compile``.
        chunk_size: maximum number of edges per long-row chunk.
    """
    row = row.astype(mx.int64)
    perm = mx.argsort(row)
    # int32 counts: MLX's GPU scatter does not support int64 inputs/updates.
    counts = mx.zeros((num_rows,), dtype=mx.int32).at[row].add(1)
    rowptr = mx.concatenate(
        [mx.zeros((1,), dtype=mx.int64), mx.cumsum(counts.astype(mx.int64))]
    )
    long_rows = (
        None
        if long_row_threshold is None
        else build_long_rows(rowptr, long_row_threshold, chunk_size)
    )
    return CSR(
        rowptr=rowptr,
        col=col.astype(mx.int64)[perm],
        values=None if values is None else values[perm],
        perm=perm,
        long_rows=long_rows,
    )


def spmm_coo(
    row: mx.array,
    col: mx.array,
    x: mx.array,
    values: Optional[mx.array] = None,
    num_rows: Optional[int] = None,
    reduce: SpMMReduce = "sum",
    *,
    stream=None,
) -> mx.array:
    """SpMM with ``A`` given as a COO edge list (any edge order).

    ``out[r] = reduce_{e : row[e] == r} values[e] * x[col[e]]``

    GPU kernel: one thread per (edge, feature) with atomic adds.

    Args:
        row: ``[num_edges]`` destination node of each edge.
        col: ``[num_edges]`` source node of each edge.
        x: ``[num_nodes, features]`` node features.
        values: optional ``[num_edges]`` edge weights.
        num_rows: number of output rows. Defaults to ``x.shape[0]``.
        reduce: ``"sum"`` (or ``"add"``) or ``"mean"``.

    Returns:
        ``[num_rows, features]`` array with the dtype of ``x``.
    """
    reduce = _check_reduce(reduce)
    n = x.shape[0] if num_rows is None else num_rows
    out = _spmm_coo(row, col, x, n, values, stream=stream)
    if reduce == "mean":
        deg = mx.zeros((n,), dtype=mx.float32).at[row].add(1.0)
        out = out / mx.maximum(deg, 1.0)[:, None]
    return out.astype(x.dtype)


def spmm_csr(
    rowptr: mx.array,
    col: mx.array,
    x: mx.array,
    values: Optional[mx.array] = None,
    reduce: SpMMReduce = "sum",
    *,
    long_rows: Optional[LongRows] = None,
    stream=None,
) -> mx.array:
    """SpMM with ``A`` given in CSR format (see :func:`coo_to_csr`).

    GPU kernel: one thread per (row, feature), no atomics, deterministic.

    Args:
        rowptr: ``[num_rows + 1]`` row pointer.
        col: ``[num_edges]`` source node of each edge, in CSR order.
        x: ``[num_nodes, features]`` node features.
        values: optional ``[num_edges]`` edge weights, in CSR order.
        reduce: ``"sum"`` (or ``"add"``) or ``"mean"``.
        long_rows: optional :class:`LongRows` (``CSR.long_rows`` from
            :func:`coo_to_csr`). Strongly recommended for graphs with hub
            nodes: without it a single thread loops over each hub's edges.

    Returns:
        ``[rowptr.size - 1, features]`` array with the dtype of ``x``.
    """
    reduce = _check_reduce(reduce)
    if long_rows is None:
        out = _spmm_csr(rowptr, col, x, values, stream=stream)
    else:
        out = _spmm_csr(
            rowptr,
            col,
            x,
            values,
            long_rows=long_rows.rows,
            chunk_ptr=long_rows.chunk_ptr,
            chunk_start=long_rows.chunk_start,
            chunk_end=long_rows.chunk_end,
            long_row_threshold=long_rows.threshold,
            stream=stream,
        )
    if reduce == "mean":
        deg = (rowptr[1:] - rowptr[:-1]).astype(mx.float32)
        out = out / mx.maximum(deg, 1.0)[:, None]
    return out.astype(x.dtype)

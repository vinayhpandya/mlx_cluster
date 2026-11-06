"""Correctness tests for SpMM / SDDMM.

Every case is checked against the gather-scatter reference that mlx-graphs
uses today (``x[col]`` then ``.at[row].add``), on CPU and, when available, GPU.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_cluster import build_long_rows, coo_to_csr, sddmm, spmm_coo, spmm_csr

DEVICES = [mx.cpu] + ([mx.gpu] if mx.is_available(mx.gpu) else [])
ATOL = 1e-4
RTOL = 1e-4


@pytest.fixture(params=DEVICES, ids=lambda d: str(d))
def device(request):
    old = mx.default_device()
    mx.set_default_device(request.param)
    yield request.param
    mx.set_default_device(old)


def random_graph(num_nodes, num_edges, seed=0, power_law=False):
    rng = np.random.default_rng(seed)
    if power_law:
        # a few destination nodes receive most edges
        p = 1.0 / np.arange(1, num_nodes + 1) ** 1.2
        dst = rng.choice(num_nodes, size=num_edges, p=p / p.sum())
    else:
        dst = rng.integers(0, num_nodes, num_edges)
    src = rng.integers(0, num_nodes, num_edges)
    return mx.array(dst), mx.array(src)  # row, col


def reference(row, col, x, values=None, num_rows=None, reduce="sum"):
    n = x.shape[0] if num_rows is None else num_rows
    msg = x[col]
    if values is not None:
        msg = msg * values[:, None]
    out = mx.zeros((n, x.shape[1]), dtype=x.dtype).at[row].add(msg)
    if reduce == "mean":
        deg = mx.zeros((n,)).at[row].add(1.0)
        out = out / mx.maximum(deg, 1.0)[:, None]
    return out


def both_formats(row, col, x, values=None, num_rows=None, reduce="sum"):
    n = x.shape[0] if num_rows is None else num_rows
    coo = spmm_coo(row, col, x, values, num_rows=n, reduce=reduce)
    csr = coo_to_csr(row, col, n, values, num_cols=x.shape[0])
    out_csr = spmm_csr(
        csr.rowptr,
        csr.col,
        x,
        csr.values,
        reduce=reduce,
        long_rows=csr.long_rows,
        transpose=csr.transpose,
    )
    return coo, out_csr


def assert_close(a, b):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("D", [1, 3, 16, 33, 128])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("power_law", [False, True])
def test_forward(device, D, weighted, reduce, power_law):
    row, col = random_graph(200, 1500, power_law=power_law)
    x = mx.random.normal((200, D), key=mx.random.key(1))
    values = mx.random.uniform(shape=(1500,), key=mx.random.key(2)) if weighted else None
    ref = reference(row, col, x, values, reduce=reduce)
    coo, csr = both_formats(row, col, x, values, reduce=reduce)
    assert_close(coo, ref)
    assert_close(csr, ref)


def test_matches_dense_matmul(device):
    row, col = random_graph(50, 300)
    x = mx.random.normal((50, 8), key=mx.random.key(0))
    A = np.zeros((50, 50), dtype=np.float32)
    np.add.at(A, (np.array(row), np.array(col)), 1.0)
    coo, csr = both_formats(row, col, x)
    assert_close(coo, A @ np.array(x))
    assert_close(csr, A @ np.array(x))


def test_empty_rows_and_rectangular(device):
    # rows 0, 3 and the trailing rows 5..9 receive nothing; x has fewer rows
    row = mx.array([1, 1, 2, 4])
    col = mx.array([0, 2, 1, 2])
    x = mx.random.normal((3, 4), key=mx.random.key(0))
    ref = reference(row, col, x, num_rows=10)
    coo, csr = both_formats(row, col, x, num_rows=10)
    assert coo.shape == (10, 4) and csr.shape == (10, 4)
    assert_close(coo, ref)
    assert_close(csr, ref)


def test_no_edges(device):
    row = mx.array([], dtype=mx.int64)
    col = mx.array([], dtype=mx.int64)
    x = mx.ones((4, 5))
    coo, csr = both_formats(row, col, x)
    assert_close(coo, mx.zeros((4, 5)))
    assert_close(csr, mx.zeros((4, 5)))


def test_dtype_and_index_types(device):
    row, col = random_graph(30, 100)
    x = mx.random.normal((30, 8), key=mx.random.key(0)).astype(mx.float16)
    out = spmm_coo(row.astype(mx.int32), col.astype(mx.uint32), x)
    assert out.dtype == mx.float16
    ref = reference(row, col, x.astype(mx.float32))
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), np.array(ref), atol=5e-2)


@pytest.mark.parametrize("fmt", ["coo", "csr", "csr_t"])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_gradients(device, fmt, reduce):
    row, col = random_graph(60, 400, seed=3)
    x = mx.random.normal((60, 12), key=mx.random.key(4))
    values = mx.random.uniform(shape=(400,), key=mx.random.key(5))
    target = mx.random.normal((60, 12), key=mx.random.key(6))
    csr = coo_to_csr(row, col, 60, values)

    def loss_fused(x, v):
        if fmt == "coo":
            out = spmm_coo(row, col, x, v, reduce=reduce)
        elif fmt == "csr":  # backward on the COO kernel
            out = spmm_csr(csr.rowptr, csr.col, x, v[csr.perm], reduce=reduce)
        else:  # "csr_t": backward on the CSR kernel via the cached transpose
            out = spmm_csr(
                csr.rowptr,
                csr.col,
                x,
                v[csr.perm],
                reduce=reduce,
                long_rows=csr.long_rows,
                transpose=csr.transpose,
            )
        return ((out - target) ** 2).sum()

    def loss_ref(x, v):
        return ((reference(row, col, x, v, reduce=reduce) - target) ** 2).sum()

    gx, gv = mx.grad(loss_fused, argnums=(0, 1))(x, values)
    rx, rv = mx.grad(loss_ref, argnums=(0, 1))(x, values)
    assert_close(gx, rx)
    assert_close(gv, rv)


def test_jvp(device):
    row, col = random_graph(40, 200)
    x = mx.random.normal((40, 6), key=mx.random.key(0))
    v = mx.random.uniform(shape=(200,), key=mx.random.key(1))
    tx = mx.random.normal((40, 6), key=mx.random.key(2))
    tv = mx.random.normal((200,), key=mx.random.key(3))
    _, (t_fused,) = mx.jvp(lambda a, b: spmm_coo(row, col, a, b), [x, v], [tx, tv])
    # Reference: dense A(v) @ x. (MLX's own scatter has no JVP.)
    R = mx.array(np.eye(40, dtype=np.float32)[np.array(row)])  # [E, N] one-hot
    C = mx.array(np.eye(40, dtype=np.float32)[np.array(col)])  # [E, N]
    dense = lambda a, b: (R.T * b) @ (C @ a)
    _, (t_ref,) = mx.jvp(dense, [x, v], [tx, tv])
    assert_close(t_fused, t_ref)


def test_sddmm(device):
    row, col = random_graph(30, 150)
    a = mx.random.normal((30, 37), key=mx.random.key(0))
    b = mx.random.normal((30, 37), key=mx.random.key(1))
    ref = (a[row] * b[col]).sum(axis=1)
    assert_close(sddmm(row, col, a, b), ref)


def test_compile(device):
    row, col = random_graph(80, 500)
    x = mx.random.normal((80, 16), key=mx.random.key(0))
    f = mx.compile(lambda x: spmm_coo(row, col, x, reduce="mean"))
    assert_close(f(x), reference(row, col, x, reduce="mean"))


# ---------------------------------------------------------------------------
# CSR long-row (load-balanced) path
# ---------------------------------------------------------------------------


def hub_graph(num_nodes=300, hub_edges=5000, other_edges=1000, seed=0):
    """Node 0 receives `hub_edges` edges, everything else is sparse."""
    rng = np.random.default_rng(seed)
    dst = np.concatenate(
        [np.zeros(hub_edges, dtype=np.int64), rng.integers(1, num_nodes, other_edges)]
    )
    src = rng.integers(0, num_nodes, hub_edges + other_edges)
    return mx.array(dst), mx.array(src), num_nodes


def assert_close_to_exact(out, row, col, x, values, reduce):
    """Compare with exact float64 maths. The hub row adds thousands of terms,
    so after cancellation float32 rounding is relative to the size of the
    terms, not of the (possibly small) result: scale the tolerance by
    sum_e |w_e * x_e| per output element, i.e. a few float32 ulps of that."""
    R, C = np.array(row), np.array(col)
    X = np.array(x).astype(np.float64)
    W = np.ones(R.size) if values is None else np.array(values).astype(np.float64)
    n = X.shape[0]
    exact = np.zeros_like(X)
    mag = np.zeros_like(X)
    np.add.at(exact, R, W[:, None] * X[C])
    np.add.at(mag, R, np.abs(W[:, None] * X[C]))
    if reduce == "mean":
        deg = np.maximum(np.bincount(R, minlength=n), 1)[:, None]
        exact, mag = exact / deg, mag / deg
    err = np.abs(np.array(out).astype(np.float64) - exact)
    bound = 1e-5 + 1e-5 * mag  # ~80 float32 ulps of the summed magnitude
    worst = np.unravel_index(np.argmax(err - bound), err.shape)
    assert np.all(err <= bound), (
        f"max excess at {worst}: err={err[worst]:.3g} bound={bound[worst]:.3g}"
    )


def test_long_rows_metadata():
    row, col, n = hub_graph()
    csr = coo_to_csr(row, col, n, long_row_threshold=100, chunk_size=64)
    lr = csr.long_rows
    assert lr is not None and lr.threshold == 100
    rp = np.array(csr.rowptr)
    deg = np.diff(rp)
    rows = np.array(lr.rows)
    np.testing.assert_array_equal(rows, np.nonzero(deg > 100)[0])
    cptr, cs, ce = np.array(lr.chunk_ptr), np.array(lr.chunk_start), np.array(lr.chunk_end)
    for i, r in enumerate(rows):
        starts, ends = cs[cptr[i] : cptr[i + 1]], ce[cptr[i] : cptr[i + 1]]
        # chunks tile the row exactly, in order, each <= chunk_size edges
        assert starts[0] == rp[r] and ends[-1] == rp[r + 1]
        np.testing.assert_array_equal(starts[1:], ends[:-1])
        assert np.all(ends - starts <= 64) and np.all(ends > starts)


def test_no_long_rows_gives_none():
    row, col = random_graph(50, 200)
    assert coo_to_csr(row, col, 50, long_row_threshold=10_000).long_rows is None
    assert coo_to_csr(row, col, 50, long_row_threshold=None).long_rows is None


@pytest.mark.parametrize("D", [1, 16, 33, 64])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("threshold,chunk", [(0, 1), (8, 16), (100, 64), (100, 5000)])
def test_long_rows_forward(device, D, weighted, threshold, chunk):
    row, col, n = hub_graph()
    x = mx.random.normal((n, D), key=mx.random.key(0))
    values = (
        mx.random.uniform(shape=(row.size,), key=mx.random.key(1)) if weighted else None
    )
    csr = coo_to_csr(row, col, n, values, long_row_threshold=threshold, chunk_size=chunk)
    assert csr.long_rows is not None
    for reduce in ("sum", "mean"):
        out = spmm_csr(
            csr.rowptr, csr.col, x, csr.values, reduce=reduce, long_rows=csr.long_rows
        )
        assert_close_to_exact(out, row, col, x, values, reduce)


def test_long_rows_deterministic(device):
    row, col, n = hub_graph(hub_edges=20000)
    x = mx.random.normal((n, 32), key=mx.random.key(0))
    csr = coo_to_csr(row, col, n, long_row_threshold=64, chunk_size=256)
    outs = [
        np.array(spmm_csr(csr.rowptr, csr.col, x, long_rows=csr.long_rows))
        for _ in range(3)
    ]
    np.testing.assert_array_equal(outs[0], outs[1])
    np.testing.assert_array_equal(outs[0], outs[2])


def test_long_rows_gradients(device):
    row, col, n = hub_graph()
    x = mx.random.normal((n, 12), key=mx.random.key(4))
    values = mx.random.uniform(shape=(row.size,), key=mx.random.key(5))
    target = mx.random.normal((n, 12), key=mx.random.key(6))
    csr = coo_to_csr(row, col, n, values, long_row_threshold=32, chunk_size=100)

    def loss_fused(x, v):
        out = spmm_csr(csr.rowptr, csr.col, x, v[csr.perm], long_rows=csr.long_rows)
        return ((out - target) ** 2).sum()

    gx, gv = mx.grad(loss_fused, argnums=(0, 1))(x, values)

    # Exact float64 reference. The hub row sums 5000 terms, so a float32
    # gather-scatter reference itself carries ~4e-3 relative error here
    # (the chunked CSR sum is ~20x more accurate than that).
    R, C = np.array(row), np.array(col)
    X, V, T = (np.array(a).astype(np.float64) for a in (x, values, target))
    out = np.zeros((n, 12))
    np.add.at(out, R, V[:, None] * X[C])
    g = 2 * (out - T)
    rx = np.zeros_like(X)
    np.add.at(rx, C, V[:, None] * g[R])
    rv = (g[R] * X[C]).sum(1)
    np.testing.assert_allclose(np.array(gx), rx, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.array(gv), rv, rtol=1e-3, atol=1e-3)


def test_long_rows_jvp(device):
    row, col, n = hub_graph()
    x = mx.random.normal((n, 6), key=mx.random.key(0))
    tx = mx.random.normal((n, 6), key=mx.random.key(1))
    csr = coo_to_csr(row, col, n, long_row_threshold=16, chunk_size=50)
    f = lambda a: spmm_csr(csr.rowptr, csr.col, a, long_rows=csr.long_rows)  # noqa
    _, (t,) = mx.jvp(f, [x], [tx])
    assert_close(t, reference(row, col, tx))


def test_build_long_rows_validation():
    with pytest.raises(ValueError):
        build_long_rows(mx.array([0, 1]), threshold=-1)
    with pytest.raises(ValueError):
        build_long_rows(mx.array([0, 1]), chunk_size=0)


# ---------------------------------------------------------------------------
# Transposed CSR (backward pass on the CSR kernel)
# ---------------------------------------------------------------------------


def test_transpose_structure():
    row, col = random_graph(40, 300)
    csr = coo_to_csr(row, col, 40)
    t = csr.transpose
    assert t is not None
    A = np.zeros((40, 40))
    np.add.at(A, (np.array(row), np.array(col)), 1.0)
    # rebuild A^T from the transposed CSR
    At = np.zeros((40, 40))
    rp, c = np.array(t.rowptr), np.array(t.col)
    for r in range(40):
        np.add.at(At, (np.full(rp[r + 1] - rp[r], r), c[rp[r] : rp[r + 1]]), 1.0)
    np.testing.assert_array_equal(At, A.T)
    # perm maps transposed edges to the matching CSR edges
    np.testing.assert_array_equal(np.array(csr.col)[np.array(t.perm)], np.repeat(
        np.arange(40), np.diff(rp)))


def test_transpose_optional():
    row, col = random_graph(30, 100)
    assert coo_to_csr(row, col, 30, transpose=False).transpose is None


def out_hub_graph(num_nodes=300, hub_edges=5000, other_edges=1000, seed=1):
    """Node 0 SENDS `hub_edges` edges: a long row of the transpose."""
    dst, src, n = hub_graph(num_nodes, hub_edges, other_edges, seed)
    return src, dst, n  # swap direction


@pytest.mark.parametrize("graph", ["in_hub", "out_hub"])
@pytest.mark.parametrize("weighted", [False, True])
def test_transpose_backward_matches_exact(device, graph, weighted):
    row, col, n = hub_graph() if graph == "in_hub" else out_hub_graph()
    x = mx.random.normal((n, 16), key=mx.random.key(0))
    values = mx.random.uniform(shape=(row.size,), key=mx.random.key(1))
    g_out = mx.random.normal((n, 16), key=mx.random.key(2))
    csr = coo_to_csr(row, col, n, values if weighted else None,
                     long_row_threshold=32, chunk_size=128)
    lt = csr.transpose.long_rows
    if graph == "out_hub":
        assert lt is not None and 0 in np.array(lt.rows).tolist()

    def f(x):
        out = spmm_csr(csr.rowptr, csr.col, x, csr.values,
                       long_rows=csr.long_rows, transpose=csr.transpose)
        return (out * g_out).sum()

    gx = mx.grad(f)(x)  # = A^T @ g_out
    # exact float64 A^T @ g_out
    R, C = np.array(row), np.array(col)
    W = np.array(values).astype(np.float64) if weighted else np.ones(R.size)
    G = np.array(g_out).astype(np.float64)
    exact = np.zeros((n, 16)); mag = np.zeros((n, 16))
    np.add.at(exact, C, W[:, None] * G[R]); np.add.at(mag, C, np.abs(W[:, None] * G[R]))
    err = np.abs(np.array(gx).astype(np.float64) - exact)
    assert np.all(err <= 1e-5 + 1e-5 * mag)


def test_transpose_backward_deterministic(device):
    row, col, n = out_hub_graph(hub_edges=20000)
    x = mx.random.normal((n, 32), key=mx.random.key(0))
    csr = coo_to_csr(row, col, n, long_row_threshold=64, chunk_size=256)
    f = lambda x: spmm_csr(csr.rowptr, csr.col, x, long_rows=csr.long_rows,  # noqa
                           transpose=csr.transpose).sum()
    grads = [np.array(mx.grad(f)(x)) for _ in range(3)]
    np.testing.assert_array_equal(grads[0], grads[1])
    np.testing.assert_array_equal(grads[0], grads[2])


def test_transpose_rectangular(device):
    # 7 destination rows, 4 source nodes (x has 4 rows)
    row = mx.array([0, 2, 2, 6, 6, 6])
    col = mx.array([1, 0, 3, 1, 2, 3])
    x = mx.random.normal((4, 5), key=mx.random.key(0))
    csr = coo_to_csr(row, col, 7, num_cols=4)
    f = lambda x: spmm_csr(csr.rowptr, csr.col, x, transpose=csr.transpose).sum()  # noqa
    ref = mx.grad(lambda x: reference(row, col, x, num_rows=7).sum())(x)
    assert_close(mx.grad(f)(x), ref)


# ---------------------------------------------------------------------------
# max / min reductions
# ---------------------------------------------------------------------------


def reference_maxmin(row, col, x, values=None, num_rows=None, reduce="max"):
    """mlx-graphs' scatter max/min semantics: true max/min, 0 for empty rows.
    Built from MLX ops, so its gradient is MLX's own (ties all get it)."""
    n = x.shape[0] if num_rows is None else num_rows
    msg = x[col]
    if values is not None:
        msg = msg * values[:, None]
    fill = -float("inf") if reduce == "max" else float("inf")
    base = mx.full((n, x.shape[1]), fill)
    out = base.at[row].maximum(msg) if reduce == "max" else base.at[row].minimum(msg)
    count = mx.zeros((n,)).at[row].add(1.0)[:, None]
    return mx.where(count > 0, out, mx.zeros_like(out))


def run_maxmin(fmt, row, col, x, values, n, reduce, **csr_kwargs):
    if fmt == "coo":
        return spmm_coo(row, col, x, values, num_rows=n, reduce=reduce)
    csr = coo_to_csr(row, col, n, values, num_cols=x.shape[0], **csr_kwargs)
    return spmm_csr(
        csr.rowptr,
        csr.col,
        x,
        csr.values,
        reduce=reduce,
        long_rows=csr.long_rows,
        transpose=csr.transpose,
    )


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("reduce", ["max", "min"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("D", [1, 16, 33])
def test_maxmin_forward(device, fmt, reduce, weighted, D):
    row, col = random_graph(120, 900, power_law=True)
    x = mx.random.normal((120, D), key=mx.random.key(1))  # mixed signs
    values = (
        mx.random.normal((900,), key=mx.random.key(2)) if weighted else None
    )  # negative weights flip which message is the max
    out = run_maxmin(fmt, row, col, x, values, 120, reduce)
    ref = reference_maxmin(row, col, x, values, reduce=reduce)
    # max/min pick one of the messages: results must be bit-identical
    np.testing.assert_array_equal(np.array(out), np.array(ref))


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("reduce", ["max", "min"])
def test_maxmin_empty_rows_and_negatives(device, fmt, reduce):
    # row 0: all-negative messages (true max is negative, not clamped at 0)
    # rows 1, 3: no edges -> 0 ; row 2: a single edge ; rows 4..6: none
    row = mx.array([0, 0, 0, 2])
    col = mx.array([0, 1, 2, 1])
    x = mx.array([[-3.0, 5.0], [-1.0, -7.0], [-2.0, 0.5]])
    out = run_maxmin(fmt, row, col, x, None, 7, reduce)
    ref = reference_maxmin(row, col, x, num_rows=7, reduce=reduce)
    np.testing.assert_array_equal(np.array(out), np.array(ref))
    if reduce == "max":
        assert np.array(out)[0, 0] == -1.0


@pytest.mark.parametrize("reduce", ["max", "min"])
@pytest.mark.parametrize("threshold,chunk", [(0, 1), (8, 16), (100, 64)])
def test_maxmin_long_rows(device, reduce, threshold, chunk):
    row, col, n = hub_graph()
    x = mx.random.normal((n, 24), key=mx.random.key(0))
    values = mx.random.normal((row.size,), key=mx.random.key(1))
    out = run_maxmin(
        "csr", row, col, x, values, n, reduce,
        long_row_threshold=threshold, chunk_size=chunk,
    )
    ref = reference_maxmin(row, col, x, values, reduce=reduce)
    np.testing.assert_array_equal(np.array(out), np.array(ref))


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("reduce", ["max", "min"])
@pytest.mark.parametrize("weighted", [False, True])
def test_maxmin_gradients(device, fmt, reduce, weighted):
    row, col = random_graph(80, 600, seed=3)
    x = mx.random.normal((80, 10), key=mx.random.key(4))
    values = mx.random.normal((600,), key=mx.random.key(5))
    target = mx.random.normal((80, 10), key=mx.random.key(6))
    csr = coo_to_csr(row, col, 80, values, long_row_threshold=4, chunk_size=8)

    def loss_fused(x, v):
        w = v if weighted else None
        if fmt == "coo":
            out = spmm_coo(row, col, x, w, reduce=reduce)
        else:
            out = spmm_csr(
                csr.rowptr, csr.col, x, None if w is None else w[csr.perm],
                reduce=reduce, long_rows=csr.long_rows, transpose=csr.transpose,
            )
        return ((out - target) ** 2).sum()

    def loss_ref(x, v):
        out = reference_maxmin(row, col, x, v if weighted else None, reduce=reduce)
        return ((out - target) ** 2).sum()

    gx, gv = mx.grad(loss_fused, argnums=(0, 1))(x, values)
    rx, rv = mx.grad(loss_ref, argnums=(0, 1))(x, values)
    assert_close(gx, rx)
    assert_close(gv, rv)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("reduce", ["max", "min"])
def test_maxmin_ties_get_full_gradient(device, fmt, reduce):
    # duplicate edges (0 -> 1 twice) and equal features create exact ties;
    # MLX gives the full gradient to every tied entry, and so must we
    row = mx.array([1, 1, 1, 0])
    col = mx.array([0, 0, 2, 2])
    x = mx.array([[2.0, 1.0], [9.0, 9.0], [2.0, -4.0]])
    g = mx.array([[1.0, 1.0], [10.0, 100.0]])

    def f(x):
        return (run_maxmin(fmt, row, col, x, None, 2, reduce) * g).sum()

    ref = mx.grad(lambda x: (reference_maxmin(row, col, x, num_rows=2, reduce=reduce) * g).sum())(x)
    np.testing.assert_array_equal(np.array(mx.grad(f)(x)), np.array(ref))


def test_maxmin_deterministic(device):
    row, col, n = hub_graph(hub_edges=20000)
    x = mx.random.normal((n, 32), key=mx.random.key(0))
    csr = coo_to_csr(row, col, n, long_row_threshold=64, chunk_size=256)
    outs = [
        np.array(spmm_csr(csr.rowptr, csr.col, x, reduce="max", long_rows=csr.long_rows))
        for _ in range(3)
    ] + [np.array(spmm_coo(row, col, x, reduce="max")) for _ in range(3)]
    for o in outs[1:]:
        np.testing.assert_array_equal(outs[0], o)  # max is order-independent


def test_maxmin_jvp_not_supported(device):
    row, col = random_graph(10, 30)
    x = mx.random.normal((10, 4), key=mx.random.key(0))
    with pytest.raises(RuntimeError, match="jvp"):
        mx.jvp(lambda a: spmm_coo(row, col, a, reduce="max"), [x], [x])


def test_bad_reduce():
    row, col = random_graph(10, 30)
    with pytest.raises(ValueError):
        spmm_coo(row, col, mx.ones((10, 2)), reduce="median")


# ---------------------------------------------------------------------------
# max / min backward on the transposed CSR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduce", ["max", "min"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("graph", ["in_hub", "out_hub"])
def test_maxmin_transpose_backward_matches_mlx(device, reduce, weighted, graph):
    row, col, n = hub_graph() if graph == "in_hub" else out_hub_graph()
    x = mx.random.normal((n, 12), key=mx.random.key(0))
    values = mx.random.normal((row.size,), key=mx.random.key(1))
    g_out = mx.random.normal((n, 12), key=mx.random.key(2))
    csr = coo_to_csr(row, col, n, values if weighted else None,
                     long_row_threshold=32, chunk_size=128)
    if graph == "out_hub":
        assert csr.transpose.long_rows is not None  # chunked backward path

    def f(x):
        out = spmm_csr(csr.rowptr, csr.col, x, csr.values, reduce=reduce,
                       long_rows=csr.long_rows, transpose=csr.transpose)
        return (out * g_out).sum()

    def f_ref(x):
        out = reference_maxmin(row, col, x, values if weighted else None,
                               reduce=reduce)
        return (out * g_out).sum()

    assert_close(mx.grad(f)(x), mx.grad(f_ref)(x))


@pytest.mark.parametrize("reduce", ["max", "min"])
def test_maxmin_transpose_vs_atomic_backward(device, reduce):
    row, col, n = out_hub_graph()
    x = mx.random.normal((n, 20), key=mx.random.key(0))
    values = mx.random.normal((row.size,), key=mx.random.key(1))
    csr = coo_to_csr(row, col, n, values, long_row_threshold=16, chunk_size=64)

    def grads(transpose):
        def f(x, v):
            return spmm_csr(csr.rowptr, csr.col, x, v, reduce=reduce,
                            long_rows=csr.long_rows, transpose=transpose).sum()
        return mx.grad(f, argnums=(0, 1))(x, csr.values)

    gx_t, gv_t = grads(csr.transpose)  # transposed CSR kernel
    gx_a, gv_a = grads(None)           # atomic COO-style kernel
    assert_close(gx_t, gx_a)
    np.testing.assert_array_equal(np.array(gv_t), np.array(gv_a))  # same kernel


def test_maxmin_transpose_backward_deterministic(device):
    row, col, n = out_hub_graph(hub_edges=20000)
    x = mx.random.normal((n, 32), key=mx.random.key(0))
    # many ties: duplicated feature values make several edges win at once
    x = mx.round(x * 2) / 2
    csr = coo_to_csr(row, col, n, long_row_threshold=64, chunk_size=256)
    f = lambda x: spmm_csr(csr.rowptr, csr.col, x, reduce="max",  # noqa: E731
                           long_rows=csr.long_rows, transpose=csr.transpose).sum()
    grads = [np.array(mx.grad(f)(x)) for _ in range(3)]
    np.testing.assert_array_equal(grads[0], grads[1])
    np.testing.assert_array_equal(grads[0], grads[2])


def test_maxmin_transpose_rectangular(device):
    row = mx.array([0, 2, 2, 6, 6, 6])
    col = mx.array([1, 0, 3, 1, 2, 3])
    x = mx.random.normal((4, 5), key=mx.random.key(0))
    csr = coo_to_csr(row, col, 7, num_cols=4)
    f = lambda x: spmm_csr(csr.rowptr, csr.col, x, reduce="max",  # noqa: E731
                           transpose=csr.transpose).sum()
    ref = mx.grad(lambda x: reference_maxmin(row, col, x, num_rows=7).sum())(x)
    assert_close(mx.grad(f)(x), ref)

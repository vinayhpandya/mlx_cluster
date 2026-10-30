"""Tests for mlx_cluster.hetero_neighbor_sample.

Self-contained: only numpy + mlx, no dataset downloads.
"""

from collections import Counter

import mlx.core as mx
import numpy as np
import pytest

from mlx_cluster import hetero_neighbor_sample, neighbor_sample

WRITES = ("author", "writes", "paper")
CITES = ("paper", "cites", "paper")
PUB = ("paper", "published_in", "venue")


# ----------------------------------------------------------------- helpers
def to_csc(src, dst, num_dst):
    """(src, dst) edge lists -> (colptr, row) int64 mx arrays, CSC by dst."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    order = np.argsort(dst, kind="stable")
    colptr = np.zeros(num_dst + 1, dtype=np.int64)
    np.cumsum(np.bincount(dst, minlength=num_dst), out=colptr[1:])
    return mx.array(colptr), mx.array(src[order])


def build_graph(edge_lists, num_nodes):
    """edge_lists: {edge_type: (src_list, dst_list)} -> colptr/row dicts."""
    colptr_dict, row_dict = {}, {}
    for et, (s, d) in edge_lists.items():
        colptr_dict[et], row_dict[et] = to_csc(s, d, num_nodes[et[2]])
    return colptr_dict, row_dict


def global_edges(out, et):
    """Set of sampled edges of type `et` as global (src, dst) pairs."""
    src_ids = np.array(out.node_id[et[0]])
    dst_ids = np.array(out.node_id[et[2]])
    rows, cols = np.array(out.row[et]), np.array(out.col[et])
    return set(zip(src_ids[rows].tolist(), dst_ids[cols].tolist()))


def ids(*xs):
    return mx.array(list(xs), dtype=mx.int64)


@pytest.fixture
def toy():
    """4 authors, 5 papers, 2 venues."""
    num_nodes = {"author": 4, "paper": 5, "venue": 2}
    edges = {
        WRITES: ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 3, 4]),
        CITES: ([1, 2, 3, 4], [0, 0, 1, 2]),
        PUB: ([0, 1, 2, 3, 4], [0, 0, 1, 1, 1]),
    }
    colptr_dict, row_dict = build_graph(edges, num_nodes)
    return colptr_dict, row_dict, num_nodes, edges


def random_hetero_graph(rng, num_nodes, num_edges):
    edges = {}
    for et, m in num_edges.items():
        edges[et] = (
            rng.integers(0, num_nodes[et[0]], m),
            rng.integers(0, num_nodes[et[2]], m),
        )
    return edges


@pytest.fixture
def random_graph():
    rng = np.random.default_rng(0)
    num_nodes = {"author": 300, "paper": 500, "venue": 20}
    num_edges = {WRITES: 2000, CITES: 3000, ("paper", "rev_writes", "author"): 2000}
    edges = random_hetero_graph(rng, num_nodes, num_edges)
    # rev_writes must mirror writes to be a real reverse relation
    edges[("paper", "rev_writes", "author")] = (edges[WRITES][1], edges[WRITES][0])
    colptr_dict, row_dict = build_graph(edges, num_nodes)
    return colptr_dict, row_dict, num_nodes


# ------------------------------------------------------- exact (fanout -1)
def test_exact_full_neighborhood(toy):
    colptr_dict, row_dict, num_nodes, _ = toy
    cp = {k: colptr_dict[k] for k in (WRITES, CITES)}
    rw = {k: row_dict[k] for k in (WRITES, CITES)}

    out = hetero_neighbor_sample(cp, rw, {"paper": ids(0)}, [-1, -1])

    # hop 0: p0 <- a0 (writes), p0 <- p1, p2 (cites)
    # hop 1: p1 <- a0, a1; p2 <- a2; p1 <- p3; p2 <- p4
    assert out.node_id["paper"].tolist() == [0, 1, 2, 3, 4]
    assert out.node_id["author"].tolist() == [0, 1, 2]
    assert global_edges(out, WRITES) == {(0, 0), (0, 1), (1, 1), (2, 2)}
    assert global_edges(out, CITES) == {(1, 0), (2, 0), (3, 1), (4, 2)}
    assert out.num_sampled_nodes == {"author": [0, 1, 2], "paper": [1, 2, 2]}
    assert out.num_sampled_edges == {WRITES: [1, 3], CITES: [2, 2]}


def test_edge_id_points_into_csc(toy):
    colptr_dict, row_dict, *_ = toy
    out = hetero_neighbor_sample(colptr_dict, row_dict, {"venue": ids(1)}, [-1, -1])
    for et in colptr_dict:
        eid = np.array(out.edge_id[et])
        src_global = np.array(out.node_id[et[0]])[np.array(out.row[et])]
        assert np.array_equal(np.array(row_dict[et])[eid], src_global)


def test_per_edge_type_fanout_and_padding(toy):
    colptr_dict, row_dict, *_ = toy
    out = hetero_neighbor_sample(
        colptr_dict,
        row_dict,
        {"paper": ids(0)},
        {WRITES: [-1], CITES: [-1, -1], PUB: [0]},  # WRITES padded to [-1, 0]
    )
    assert out.num_sampled_edges[WRITES] == [1, 0]
    assert out.num_sampled_edges[CITES] == [2, 2]
    assert out.num_sampled_edges[PUB] == [0, 0]
    assert out.node_id["author"].tolist() == [0]


def test_src_only_and_unseeded_types_present(toy):
    colptr_dict, row_dict, *_ = toy
    out = hetero_neighbor_sample(
        {WRITES: colptr_dict[WRITES]}, {WRITES: row_dict[WRITES]},
        {"paper": ids(4)}, [2],
    )
    assert out.node_id["paper"].tolist() == [4]
    assert out.node_id["author"].tolist() == [3]
    assert out.num_sampled_nodes["author"] == [0, 1]


def test_isolated_seed_gives_empty_edges(toy):
    colptr_dict, row_dict, *_ = toy
    # paper 3 and 4 are never cited
    out = hetero_neighbor_sample(
        {CITES: colptr_dict[CITES]}, {CITES: row_dict[CITES]},
        {"paper": ids(4)}, [5, 5],
    )
    assert out.node_id["paper"].tolist() == [4]
    for arr in (out.row[CITES], out.col[CITES], out.edge_id[CITES]):
        assert arr.shape == (0,) and arr.dtype == mx.int64


# ------------------------------------------------- invariants (random)
def check_invariants(out, colptr_dict, row_dict, seed_dict, fanout, replace):
    num_sampled = {nt: len(v) for nt, v in out.node_id.items()}
    for nt, s in seed_dict.items():
        assert out.node_id[nt][: len(s)].tolist() == s.tolist(), "seeds first"
    for nt, ids_ in out.node_id.items():
        assert len(set(ids_.tolist())) == len(ids_), f"duplicate {nt} nodes"
        assert sum(out.num_sampled_nodes[nt]) == num_sampled[nt]

    for et in colptr_dict:
        src, _, dst = et
        rows, cols = np.array(out.row[et]), np.array(out.col[et])
        eid = np.array(out.edge_id[et])
        assert len(rows) == len(cols) == len(eid)
        assert sum(out.num_sampled_edges[et]) == len(rows)
        if len(rows) == 0:
            continue
        assert rows.min() >= 0 and rows.max() < num_sampled[src]
        assert cols.min() >= 0 and cols.max() < num_sampled[dst]

        colptr = np.array(colptr_dict[et])
        row = np.array(row_dict[et])
        src_g = np.array(out.node_id[src])[rows]
        dst_g = np.array(out.node_id[dst])[cols]
        # every sampled edge exists in the graph
        assert np.array_equal(row[eid], src_g)
        assert np.all(colptr[dst_g] <= eid) and np.all(eid < colptr[dst_g + 1])

        if not replace:
            assert len(set(eid.tolist())) == len(eid), "duplicate edges"

    # Exact fan-out per destination node. A node with local id in
    # [cum[h], cum[h+1]) was discovered at hop h and is expanded at hop h
    # with fanout[h] (nodes discovered at the last hop are not expanded).
    for et in colptr_dict:
        dst = et[2]
        colptr = np.array(colptr_dict[et])
        per_dst = Counter(np.array(out.col[et]).tolist())
        cum = np.cumsum([0] + out.num_sampled_nodes[dst])
        dst_ids = np.array(out.node_id[dst])
        for hop, k in enumerate(fanout):
            for local in range(cum[hop], cum[hop + 1]):
                g = dst_ids[local]
                deg = colptr[g + 1] - colptr[g]
                if deg == 0:
                    expected = 0
                elif replace:
                    expected = k
                else:
                    expected = min(k, deg)
                assert per_dst.get(local, 0) == expected, (et, hop, local)
        # nodes discovered at the final hop have no incoming sampled edges
        for local in range(cum[len(fanout)], cum[-1]):
            assert per_dst.get(local, 0) == 0


@pytest.mark.parametrize("replace", [False, True])
def test_invariants_random_graph(random_graph, replace):
    colptr_dict, row_dict, num_nodes = random_graph
    seed_dict = {"paper": ids(*range(0, 50)), "author": ids(5, 7, 11)}
    fanout = [4, 3]
    out = hetero_neighbor_sample(
        colptr_dict, row_dict, seed_dict, fanout,
        num_nodes_dict=num_nodes, replace=replace, seed=123,
    )
    check_invariants(out, colptr_dict, row_dict, seed_dict, fanout, replace)


def test_without_replacement_takes_min_of_k_and_degree(random_graph):
    colptr_dict, row_dict, num_nodes = random_graph
    et = CITES
    seeds = ids(*range(100))
    out = hetero_neighbor_sample(
        {et: colptr_dict[et]}, {et: row_dict[et]}, {"paper": seeds}, [3], seed=1
    )
    colptr = np.array(colptr_dict[et])
    deg = colptr[1:] - colptr[:-1]
    per_dst = Counter(np.array(out.col[et]).tolist())
    for i, p in enumerate(seeds.tolist()):
        assert per_dst.get(i, 0) == min(3, deg[p])


def test_hash_mapper_path_matches_dense(random_graph):
    """Without num_nodes_dict, 'author' only known via rev_writes; dropping
    that edge type forces the hash-map path for authors."""
    colptr_dict, row_dict, num_nodes = random_graph
    cp = {k: v for k, v in colptr_dict.items() if k[2] != "author"}
    rw = {k: v for k, v in row_dict.items() if k[2] != "author"}
    seeds = {"paper": ids(*range(30))}
    a = hetero_neighbor_sample(cp, rw, seeds, [5, 5], seed=7)
    b = hetero_neighbor_sample(cp, rw, seeds, [5, 5], num_nodes_dict=num_nodes, seed=7)
    for nt in a.node_id:
        assert a.node_id[nt].tolist() == b.node_id[nt].tolist()
    for et in cp:
        assert a.row[et].tolist() == b.row[et].tolist()


# --------------------------------------------------- reproducibility / RNG
def test_same_seed_same_output(random_graph):
    colptr_dict, row_dict, _ = random_graph
    kw = dict(seed_dict={"paper": ids(*range(40))}, num_neighbors=[3, 2])
    a = hetero_neighbor_sample(colptr_dict, row_dict, seed=42, **kw)
    b = hetero_neighbor_sample(colptr_dict, row_dict, seed=42, **kw)
    c = hetero_neighbor_sample(colptr_dict, row_dict, seed=43, **kw)
    assert all(a.node_id[k].tolist() == b.node_id[k].tolist() for k in a.node_id)
    assert all(a.edge_id[k].tolist() == b.edge_id[k].tolist() for k in a.edge_id)
    assert any(a.edge_id[k].tolist() != c.edge_id[k].tolist() for k in a.edge_id)


def test_mx_random_seed_controls_default_seed(random_graph):
    colptr_dict, row_dict, _ = random_graph
    kw = dict(seed_dict={"paper": ids(*range(40))}, num_neighbors=[3, 2])
    mx.random.seed(0)
    a = hetero_neighbor_sample(colptr_dict, row_dict, **kw)
    mx.random.seed(0)
    b = hetero_neighbor_sample(colptr_dict, row_dict, **kw)
    assert all(a.edge_id[k].tolist() == b.edge_id[k].tolist() for k in a.edge_id)


# ------------------------------------------ agreement with homogeneous op
def test_single_type_matches_homogeneous_neighbor_sample():
    rng = np.random.default_rng(3)
    n = 200
    src, dst = rng.integers(0, n, 1500), rng.integers(0, n, 1500)
    colptr, row = to_csc(src, dst, n)
    seeds = ids(*range(0, 20, 2))
    et = ("node", "to", "node")

    het = hetero_neighbor_sample({et: colptr}, {et: row}, {"node": seeds}, [-1, -1])
    samples, rows, cols, edges = neighbor_sample(
        colptr, row, seeds, [-1, -1], False, True
    )
    assert het.node_id["node"].tolist() == samples.tolist()
    assert het.edge_id[et].tolist() == edges.tolist()
    assert het.row[et].tolist() == rows.tolist()
    assert het.col[et].tolist() == cols.tolist()


# ----------------------------------------------------------------- errors
def test_undirected_not_supported(toy):
    colptr_dict, row_dict, *_ = toy
    with pytest.raises(ValueError, match="directed=False"):
        hetero_neighbor_sample(colptr_dict, row_dict, {"paper": ids(0)}, [2],
                               directed=False)


def test_mismatched_keys(toy):
    colptr_dict, row_dict, *_ = toy
    with pytest.raises(ValueError, match="same edge types"):
        hetero_neighbor_sample(colptr_dict, {WRITES: row_dict[WRITES]},
                               {"paper": ids(0)}, [2])


def test_num_neighbors_dict_must_cover_all_edge_types(toy):
    colptr_dict, row_dict, *_ = toy
    with pytest.raises(ValueError, match="missing edge types"):
        hetero_neighbor_sample(colptr_dict, row_dict, {"paper": ids(0)},
                               {WRITES: [2]})


def test_wrong_dtype(toy):
    colptr_dict, row_dict, *_ = toy
    bad = dict(row_dict)
    bad[WRITES] = bad[WRITES].astype(mx.int32)
    with pytest.raises(ValueError, match="int64"):
        hetero_neighbor_sample(colptr_dict, bad, {"paper": ids(0)}, [2])


def test_inconsistent_colptr_row(toy):
    colptr_dict, row_dict, *_ = toy
    bad = dict(row_dict)
    bad[WRITES] = bad[WRITES][:-1]
    with pytest.raises(ValueError, match="does not match"):
        hetero_neighbor_sample(colptr_dict, bad, {"paper": ids(0)}, [2])


def test_seed_out_of_range(toy):
    colptr_dict, row_dict, *_ = toy
    with pytest.raises(IndexError):
        hetero_neighbor_sample(colptr_dict, row_dict, {"paper": ids(99)}, [2])


def test_duplicate_seeds(toy):
    colptr_dict, row_dict, *_ = toy
    with pytest.raises(ValueError, match="duplicate"):
        hetero_neighbor_sample(colptr_dict, row_dict, {"paper": ids(1, 1)}, [2])


def test_delimiter_in_names_rejected(toy):
    colptr_dict, row_dict, *_ = toy
    et = ("pa__per", "x", "paper")
    with pytest.raises(ValueError, match="__"):
        hetero_neighbor_sample({et: colptr_dict[CITES]}, {et: row_dict[CITES]},
                               {"paper": ids(0)}, [2])
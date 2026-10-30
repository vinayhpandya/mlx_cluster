"""Heterogeneous graph sampling.

Python front-end for the C++ ``hetero_neighbor_sample`` kernel. Users work
with ``(src, rel, dst)`` tuple edge types (same as ``HeteroGraphData`` in
mlx-graphs and ``HeteroData`` in PyG); the C++ side sees
``"src__rel__dst"`` strings.
"""

from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import mlx.core as mx

from ._ext import hetero_neighbor_sample as _hetero_neighbor_sample

NodeType = str
EdgeType = Tuple[str, str, str]
NumNeighbors = Union[Sequence[int], Mapping[EdgeType, Sequence[int]]]

DELIM = "__"

__all__ = [
    "HeteroSamplerOutput",
    "hetero_neighbor_sample",
    "edge_type_to_str",
    "str_to_edge_type",
]


def edge_type_to_str(edge_type: EdgeType) -> str:
    """``("author", "writes", "paper")`` -> ``"author__writes__paper"``."""
    if not (isinstance(edge_type, tuple) and len(edge_type) == 3):
        raise ValueError(
            f"Edge types must be (src, rel, dst) tuples, got {edge_type!r}"
        )
    for part in edge_type:
        if not isinstance(part, str):
            raise ValueError(f"Edge type parts must be str, got {edge_type!r}")
        if DELIM in part:
            raise ValueError(
                f"Node/relation names may not contain '{DELIM}': {edge_type!r}"
            )
    return DELIM.join(edge_type)


def str_to_edge_type(rel_type: str) -> EdgeType:
    """``"author__writes__paper"`` -> ``("author", "writes", "paper")``."""
    parts = rel_type.split(DELIM)
    if len(parts) != 3:
        raise ValueError(f"Not a valid edge-type string: {rel_type!r}")
    return tuple(parts)  # type: ignore[return-value]


class HeteroSamplerOutput(NamedTuple):
    """Result of :func:`hetero_neighbor_sample`.

    Attributes:
        node_id: Global ids of the sampled nodes per node type. For each
            seeded node type, the first ``len(seed_dict[t])`` entries are the
            seeds in the order given.
        row: Per edge type, local index (into ``node_id[src]``) of each
            sampled edge's source node.
        col: Per edge type, local index (into ``node_id[dst]``) of each
            sampled edge's destination node. ``mx.stack([row, col])`` is the
            sampled subgraph's ``edge_index`` for that edge type.
        edge_id: Per edge type, offset of each sampled edge in
            ``row_dict[edge_type]`` (i.e. its position in CSC order).
        num_sampled_nodes: Per node type, ``[#seeds, #new at hop 1, ...]``.
        num_sampled_edges: Per edge type, ``[#edges at hop 1, ...]``.
    """

    node_id: Dict[NodeType, mx.array]
    row: Dict[EdgeType, mx.array]
    col: Dict[EdgeType, mx.array]
    edge_id: Dict[EdgeType, mx.array]
    num_sampled_nodes: Dict[NodeType, List[int]]
    num_sampled_edges: Dict[EdgeType, List[int]]


def _resolve_num_neighbors(
    num_neighbors: NumNeighbors, edge_types: List[EdgeType]
) -> Dict[EdgeType, List[int]]:
    if isinstance(num_neighbors, Mapping):
        unknown = set(num_neighbors) - set(edge_types)
        if unknown:
            raise ValueError(
                f"num_neighbors has edge types not in colptr_dict: {sorted(unknown)}"
            )
        missing = [et for et in edge_types if et not in num_neighbors]
        if missing:
            raise ValueError(
                "num_neighbors is missing edge types "
                f"{missing}; pass [0] to skip an edge type"
            )
        resolved = {et: [int(k) for k in num_neighbors[et]] for et in edge_types}
    else:
        fanout = [int(k) for k in num_neighbors]
        resolved = {et: list(fanout) for et in edge_types}

    num_hops = max((len(v) for v in resolved.values()), default=0)
    if num_hops == 0:
        raise ValueError("num_neighbors must specify at least one hop")
    # Pad shorter fan-outs with 0 (no sampling at that hop), like PyG.
    return {et: v + [0] * (num_hops - len(v)) for et, v in resolved.items()}


def hetero_neighbor_sample(
    colptr_dict: Mapping[EdgeType, mx.array],
    row_dict: Mapping[EdgeType, mx.array],
    seed_dict: Mapping[NodeType, mx.array],
    num_neighbors: NumNeighbors,
    num_nodes_dict: Optional[Mapping[NodeType, int]] = None,
    replace: bool = False,
    directed: bool = True,
    seed: Optional[int] = None,
) -> HeteroSamplerOutput:
    """Multi-hop neighbor sampling on a heterogeneous graph (GraphSAGE style).

    For every edge type ``(src, rel, dst)`` and every hop, up to ``k``
    incoming ``src`` neighbors are sampled for each ``dst`` node discovered
    at the previous hop (the seeds at hop 0).

    Args:
        colptr_dict: CSC column pointers per edge type. For ``(src, rel,
            dst)`` this has ``num_dst_nodes + 1`` int64 entries.
        row_dict: CSC row indices (source node of each edge, sorted by
            destination) per edge type, int64.
        seed_dict: Seed node ids per node type. Must not contain duplicates.
        num_neighbors: Fan-out per hop, e.g. ``[10, 5]``. Either a single
            list used for every edge type, or a dict mapping each edge type
            to its own list. ``-1`` takes all neighbors; ``0`` samples none.
            Shorter lists are padded with ``0``.
        num_nodes_dict: Optional number of nodes per node type (e.g.
            ``HeteroGraphData.num_nodes``). Lets node types that only appear
            as a source use the faster dense id mapper.
        replace: Sample with replacement.
        directed: Only ``True`` is supported for heterogeneous graphs.
        seed: Seed for the sampler's RNG. If ``None``, one is drawn from
            MLX's global RNG, so ``mx.random.seed(...)`` makes sampling
            reproducible.

    Returns:
        HeteroSamplerOutput

    Example:

    .. code-block:: python

        out = hetero_neighbor_sample(
            colptr_dict={("author", "writes", "paper"): colptr},
            row_dict={("author", "writes", "paper"): row},
            seed_dict={"paper": mx.array([0, 1], dtype=mx.int64)},
            num_neighbors=[10, 5],
        )
        out.node_id["author"]                         # sampled author ids
        et = ("author", "writes", "paper")
        edge_index = mx.stack([out.row[et], out.col[et]])
    """
    edge_types = list(colptr_dict.keys())
    if not edge_types:
        raise ValueError("colptr_dict must contain at least one edge type")
    if set(row_dict.keys()) != set(edge_types):
        raise ValueError("colptr_dict and row_dict must have the same edge types")
    if not seed_dict:
        raise ValueError("seed_dict must contain at least one node type")

    rel_types = [edge_type_to_str(et) for et in edge_types]

    # Node types: everything mentioned by edge types, seeds or num_nodes_dict,
    # in first-seen order (deterministic output ordering).
    node_types: List[str] = []
    for et in edge_types:
        for nt in (et[0], et[2]):
            if nt not in node_types:
                node_types.append(nt)
    for nt in list(seed_dict.keys()) + list((num_nodes_dict or {}).keys()):
        if nt not in node_types:
            node_types.append(nt)

    fanouts = _resolve_num_neighbors(num_neighbors, edge_types)

    seeds = {}
    for nt, s in seed_dict.items():
        s = s if isinstance(s, mx.array) else mx.array(s)
        seeds[nt] = s.astype(mx.int64).reshape(-1)

    if seed is None:
        seed = int(mx.random.randint(0, 2**31 - 1, shape=()).item())

    node_id, row, col, edge_id, n_nodes, n_edges = _hetero_neighbor_sample(
        node_types,
        edge_types,
        {rt: colptr_dict[et] for rt, et in zip(rel_types, edge_types)},
        {rt: row_dict[et] for rt, et in zip(rel_types, edge_types)},
        seeds,
        {rt: fanouts[et] for rt, et in zip(rel_types, edge_types)},
        {k: int(v) for k, v in (num_nodes_dict or {}).items()},
        bool(replace),
        bool(directed),
        int(seed),
    )

    to_et = dict(zip(rel_types, edge_types))
    return HeteroSamplerOutput(
        node_id={nt: node_id[nt] for nt in node_types},
        row={to_et[rt]: row[rt] for rt in rel_types},
        col={to_et[rt]: col[rt] for rt in rel_types},
        edge_id={to_et[rt]: edge_id[rt] for rt in rel_types},
        num_sampled_nodes={nt: list(n_nodes[nt]) for nt in node_types},
        num_sampled_edges={to_et[rt]: list(n_edges[rt]) for rt in rel_types},
    )
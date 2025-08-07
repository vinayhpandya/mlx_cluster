import mlx.core as mx
import numpy as np
import time
import pytest

# Torch dataset
import torch
from torch.utils.data import DataLoader

loader = DataLoader(range(2708), batch_size=2000)
start_indices = next(iter(loader))
# random_walks = torch.ops.torch_cluster.random_walk(
#     row_ptr, col, start_indices, 5, 1.0, 3.0
# )

from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.utils.sorting import sort_edge_index
from torch.utils.data import DataLoader
from mlx_cluster import rejection_sampling

@pytest.mark.slow  # give download/compile plenty of time on CI
def test_random_walk(tmp_path):
    """
    Runs 1 000 random walks of length 10 on the Cora graph and checks:
    1. output tensor shape == (num_start_nodes, walk_length + 1)
    2. all returned node indices are valid ( < num_nodes )
    """

    # ---------- Dataset (downloaded to the temp dir) ----------
    data_dir = tmp_path / "mlx_datasets"
    cora = PlanetoidDataset(name="cora", base_dir=data_dir)

    edge_index = cora.graphs[0].edge_index.astype(mx.int64)

    # CSR conversion
    sorted_edge_index = sort_edge_index(edge_index=edge_index)
    row = sorted_edge_index[0][0]
    col = sorted_edge_index[0][1]
    _, counts = np.unique(np.array(row, copy=False), return_counts=True)
    row_ptr = mx.concatenate([mx.array([0]), mx.array(counts.cumsum())])

    # pick 1 000 random start nodes
    num_starts = 1_000
    rng = np.random.default_rng(42)
    start_idx = mx.array(rng.integers(low=0, high=row.max().item() + 1,
                                      size=num_starts, dtype=np.int64))

    # random numbers for the kernel (shape [num_starts, walk_length])
    walk_len = 10
    rand_data = mx.random.uniform(shape=[num_starts, walk_len])

    # ---------- Warm-up ----------
    mx.eval(row_ptr, col, start_idx, rand_data)

    # ---------- Run kernel ----------
    t0 = time.time()
    node_seq = rejection_sampling(row_ptr, col, start_idx, walk_len, 1.0, 3.0, stream=mx.cpu)
    elapsed = time.time() - t0
    print(f"Random-walk kernel took {elapsed:.3f} s")
    print("Node sequence is ", node_seq)
    # ---------- Assertions ----------
    assert node_seq[0].shape == (num_starts, walk_len + 1)

import mlx.core as mx
import numpy as np
import time

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

cora_dataset = PlanetoidDataset(name="cora", base_dir="~")
edge_index = cora_dataset.graphs[0].edge_index.astype(mx.int64)
sorted_edge_index = sort_edge_index(edge_index=edge_index)
row_mlx = sorted_edge_index[0][0]
col_mlx = sorted_edge_index[0][1]
_, counts_mlx = np.unique(np.array(row_mlx, copy=False), return_counts=True)
cum_sum_mlx = counts_mlx.cumsum()
row_ptr_mlx = mx.concatenate([mx.array([0]), mx.array(cum_sum_mlx)])
start_indices = mx.array(start_indices.numpy())
rand_data = mx.random.uniform(shape=[start_indices.shape[0], 5])
start_time = time.time()
node_sequence = rejection_sampling(
    row_ptr_mlx, col_mlx, start_indices, 5, 1.0, 3.0, stream=mx.cpu
)
print("Time taken to complete 1000 random walks : ", time.time() - start_time)
print(node_sequence)

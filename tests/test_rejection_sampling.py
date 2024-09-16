import mlx.core as mx
import numpy as np
import time

# Torch dataset
import torch
import torch_geometric.datasets as pyg_datasets
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr
from torch.utils.data import DataLoader

torch_planetoid = pyg_datasets.Planetoid(root="data/Cora", name="Cora")
edge_index_torch = torch_planetoid.edge_index
num_nodes = maybe_num_nodes(edge_index=edge_index_torch)
row, col = sort_edge_index(edge_index=edge_index_torch, num_nodes=num_nodes)
row_ptr, col = index2ptr(row, num_nodes), col
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
print("row pointer datatype", row_ptr_mlx.dtype)
print("col datatype", col_mlx.dtype)
print("start pointer datatype", start_indices.dtype)
assert mx.array_equal(row_ptr_mlx, mx.array(row_ptr.numpy())), "Arrays not equal"
assert mx.array_equal(col_mlx, mx.array(col.numpy())), "Col arrays are not equal"
rand_data = mx.random.uniform(shape=[start_indices.shape[0], 5])
start_time = time.time()
node_sequence = rejection_sampling(
    row_ptr_mlx, col_mlx, start_indices, 5, 1.0, 3.0, stream=mx.cpu
)
# print("Time taken to complete 1000 random walks : ", time.time() - start_time)
print(node_sequence)

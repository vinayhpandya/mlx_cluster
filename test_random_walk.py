import mlx.core as mx
import numpy as np
import time
from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.utils.sorting import sort_edge_index
from mlx_graphs.loaders import Dataloader
from mlx_graphs_extension import random_walk


cora_dataset = PlanetoidDataset(name="cora", base_dir="~")
start = mx.arange(0, 1000)
start_time = time.time()
edge_index = cora_dataset.graphs[0].edge_index
num_nodes = cora_dataset.graphs[0].num_nodes
sorted_edge_index = sort_edge_index(edge_index=edge_index)
row_mlx = sorted_edge_index[0][0]
col_mlx = sorted_edge_index[0][1]
unique_vals, counts_mlx = np.unique(np.array(row_mlx, copy=False), return_counts=True)
cum_sum_mlx = counts_mlx.cumsum()
row_ptr_mlx = mx.concatenate([mx.array([0]), mx.array(cum_sum_mlx)])
random_walk(row_ptr_mlx, col_mlx, start, 1000, stream = mx.cpu)
print("Time taken by custom kernel is", time.time()-start_time)
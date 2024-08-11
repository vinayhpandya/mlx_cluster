import mlx.core as mx
import numpy as np
import time
from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.utils.sorting import sort_edge_index
from mlx_graphs.loaders import Dataloader
from mlx_cluster import random_walk


cora_dataset = PlanetoidDataset(name="cora", base_dir="~")
start = mx.arange(0, 1000)
edge_index = cora_dataset.graphs[0].edge_index
num_nodes = cora_dataset.graphs[0].num_nodes
sort_time = time.time()
sorted_edge_index = sort_edge_index(edge_index=edge_index)
print("Time take to sort edge_index", time.time() - sort_time)
row_mlx = sorted_edge_index[0][0]
col_mlx = sorted_edge_index[0][1]
row_ptr_time = time.time()
unique_vals, counts_mlx = np.unique(np.array(row_mlx, copy=False), return_counts=True)
cum_sum_mlx = counts_mlx.cumsum()
print("Time take to create row_ptr", time.time() - row_ptr_time)
row_ptr_mlx = mx.concatenate([mx.array([0]), mx.array(cum_sum_mlx)])
start_time = time.time()
random_walk(row_ptr_mlx, col_mlx, start, 100, stream=mx.gpu)
print("Time taken to complete 1000 random walks : ", time.time() - start_time)

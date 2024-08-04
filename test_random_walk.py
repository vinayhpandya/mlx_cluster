import mlx.core as mx
from mlx_graphs_extension import random_walk


rowptr = mx.array([0, 2, 4, 6, 8, 10, 12], dtype=mx.int64)
col = mx.array([1, 3, 0, 4, 1, 5, 0, 4, 1, 5, 2, 4], dtype=mx.int64)
start = mx.array([0, 2, 4], dtype=mx.int64)
walk_length = 4
print(random_walk(rowptr, col, start, walk_length, stream=mx.cpu))
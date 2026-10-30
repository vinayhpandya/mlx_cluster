import mlx.core as mx
from mlx_cluster import random_walk, neighbor_sample
print(mx.__version__)

rowptr = mx.array([0, 2, 3, 4], dtype=mx.int64)
col    = mx.array([1, 2, 0, 0], dtype=mx.int64)
start  = mx.array([0, 1], dtype=mx.int64)
rand   = mx.random.uniform(shape=(2 * 3,))
print(random_walk(rowptr, col, start, rand, 3, stream=mx.gpu)[0])
print(random_walk(rowptr, col, start, rand, 3, stream=mx.cpu)[0])

colptr = mx.array([0, 0, 0, 1, 3, 4, 5, 6, 7], dtype=mx.int64)
row    = mx.array([0, 0, 1, 1, 2, 3, 4], dtype=mx.int64)
print(neighbor_sample(colptr, row, mx.array([3], dtype=mx.int64), [2, 2]))
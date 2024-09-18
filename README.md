# mlx_cluster

A C++ extension for generating random walks for Homogeneous graphs using mlx

## Installation

To install the necessary dependencies:

Clone the repositories:
```bash
git clone https://github.com/vinayhpandya/mlx_cluster.git
```

After cloning the repository install library using 

```bash
python setup.py build_ext -j8 --inplace
```

You can also just install the library via pip

```bash
pip install mlx_cluster
```

for testing purposes you need to have `mlx-graphs`  and `torch_geometric` installed

## Usage


```
from mlx_graphs.utils.sorting import sort_edge_index
from mlx_graphs.loaders import Dataloader
from mlx_graphs_extension import random_walk


cora_dataset = PlanetoidDataset(name="cora", base_dir="~")
start = mx.arange(0, 1000)
start_time = time.time()
edge_index = cora_dataset.graphs[0].edge_index.astype(mx.int64)
num_nodes = cora_dataset.graphs[0].num_nodes
sorted_edge_index = sort_edge_index(edge_index=edge_index)
row_mlx = sorted_edge_index[0][0]
col_mlx = sorted_edge_index[0][1]
unique_vals, counts_mlx = np.unique(np.array(row_mlx, copy=False), return_counts=True)
cum_sum_mlx = counts_mlx.cumsum()
rand = mx.random.uniform(shape=[start.shape[0], 100])
row_ptr_mlx = mx.concatenate([mx.array([0]), mx.array(cum_sum_mlx)])
random_walk(row_ptr_mlx, col_mlx, start, rand, 1000, stream = mx.gpu)
```

## TODO

- [x] Add metal shaders to optimize the code
- [ ] Benchmark random walk against different frameworks
- [ ] Add more algorithms

## Credits:

torch_cluster random walk implementation : [random_walk](https://github.com/rusty1s/pytorch_cluster/blob/master/csrc/cpu/rw_cpu.cpp)

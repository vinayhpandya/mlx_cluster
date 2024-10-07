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
# Can also use mlx for generating starting indices
import torch
from torch.utils.data import DataLoader

loader = DataLoader(range(2708), batch_size=2000)
start_indices = next(iter(loader))


from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.utils.sorting import sort_edge_index
from torch.utils.data import DataLoader
from mlx_cluster import random_walk

cora_dataset = PlanetoidDataset(name="cora", base_dir="~")
# For some reason int_64t and int_32t are not compatible
edge_index = cora_dataset.graphs[0].edge_index.astype(mx.int64)

# Convert edge index into a CSR matrix
sorted_edge_index = sort_edge_index(edge_index=edge_index)
row_mlx = sorted_edge_index[0][0]
col_mlx = sorted_edge_index[0][1]
_, counts_mlx = np.unique(np.array(row_mlx, copy=False), return_counts=True)
cum_sum_mlx = counts_mlx.cumsum()
row_ptr_mlx = mx.concatenate([mx.array([0]), mx.array(cum_sum_mlx)])
start_indices = mx.array(start_indices.numpy())

rand_data = mx.random.uniform(shape=[start_indices.shape[0], 5])

node_sequence = random_walk(
    row_ptr_mlx, col_mlx, start_indices, rand_data, 5, stream=mx.cpu
)
```

## TODO

- [x] Add metal shaders to optimize the code
- [ ] Benchmark random walk against different frameworks
- [ ] Add more algorithms

## Credits:

torch_cluster random walk implementation : [random_walk](https://github.com/rusty1s/pytorch_cluster/blob/master/csrc/cpu/rw_cpu.cpp)

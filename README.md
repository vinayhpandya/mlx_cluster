# MLX-Cluster

High-performance graph algorithms optimized for Apple's MLX framework, featuring random walks, biased random walks, and neighbor sampling.

[![PyPI version](https://badge.fury.io/py/mlx-cluster.svg)](https://badge.fury.io/py/mlx-cluster)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**[Documentation](https://vinayhpandya.github.io/mlx_cluster/)** | **[Quickstart](https://vinayhpandya.github.io/mlx_cluster/)** |

## 🚀 Features

- **🔥 MLX Optimized**: Built specifically for Apple's MLX framework with GPU acceleration
- **⚡ High Performance**: Optimized C++ implementations with Metal shaders for Apple Silicon
- **🎯 Graph Algorithms**: 
  - Uniform random walks
  - Biased random walks (Node2Vec style with p/q parameters)
  - Multi-hop neighbor sampling (GraphSAGE style)

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install mlx-cluster
```

### From Source

```bash
git clone https://github.com/vinayhpandya/mlx_cluster.git
cd mlx_cluster
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/vinayhpandya/mlx_cluster.git
cd mlx_cluster
pip install -e . --verbose
```

### Dependencies

Required:
- Python 3.8+
- MLX framework
- NumPy

Optional (for examples and testing):
- MLX-Graphs
- PyTorch (for dataset utilities)
- pytest

## 🔧 Quick Start

### Random Walks

```python
import mlx.core as mx
import numpy as np
from mlx_cluster import random_walk
from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.utils.sorting import sort_edge_index

# Load dataset
cora = PlanetoidDataset(name="cora")
edge_index = cora.graphs[0].edge_index.astype(mx.int64)

# Convert to CSR format
sorted_edge_index = sort_edge_index(edge_index=edge_index)
row = sorted_edge_index[0][0]
col = sorted_edge_index[0][1]
_, counts = np.unique(np.array(row, copy=False), return_counts=True)
row_ptr = mx.concatenate([mx.array([0]), mx.array(counts.cumsum())])

# Generate random walks
num_walks = 1000
walk_length = 10
start_nodes = mx.array(np.random.randint(0, cora.graphs[0].num_nodes, num_walks))
rand_values = mx.random.uniform(shape=[num_walks, walk_length])

mx.eval(rowptr,col, start_nodes, rand_values)
# Perform walks
node_sequences, edge_sequences = random_walk(
    row_ptr, col, start_nodes, rand_values, walk_length, stream=mx.gpu
)

print(f"Generated {num_walks} walks of length {walk_length + 1}")
print(f"Shape: {node_sequences.shape}")
```

### Biased Random Walks (Node2Vec)

```python
from mlx_cluster import rejection_sampling

# Biased random walks with p/q parameters
node_sequences, edge_sequences = rejection_sampling(
    row_ptr, col, start_nodes, walk_length,
    p=1.0,  # Return parameter
    q=2.0,  # In-out parameter
    stream=mx.gpu
)
```

### Neighbor Sampling

```python
from mlx_cluster import neighbor_sample

# Convert to CSC format (required for neighbor sampling)
def create_csc_format(edge_index, num_nodes):
    sources, targets = edge_index[0].tolist(), edge_index[1].tolist()
    edges = sorted(zip(sources, targets), key=lambda x: x[1])
    
    colptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for _, target in edges:
        colptr[target + 1] += 1
    colptr = np.cumsum(colptr)
    
    sorted_sources = [source for source, _ in edges]
    return mx.array(colptr), mx.array(sorted_sources, dtype=mx.int64)

colptr, row = create_csc_format(edge_index, cora.graphs[0].num_nodes)

# Multi-hop neighbor sampling
input_nodes = mx.array([0, 1, 2], dtype=mx.int64)
num_neighbors = [10, 5]  # 10 neighbors in first hop, 5 in second
mx.eval(colptr, row, input_nodes)
samples, rows, cols, edges = neighbor_sample(
    colptr, row, input_nodes, num_neighbors,
    replace=True, directed=True
)

print(f"Sampled {len(samples)} nodes and {len(edges)} edges")
```

## 📚 Documentation

For comprehensive documentation, examples, and API reference, visit:
[Documentation]()

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest mlx-graphs torch

# Run tests
pytest -s -v
```

## ⚡ Performance

MLX-Cluster is optimized for Apple Silicon and shows significant speedups:

- **Apple M1/M2/M3**: 2-5x faster than CPU-only implementations
- **GPU Acceleration**: Automatic optimization for Metal Performance Shaders
- **Memory Efficient**: Optimized sparse graph representations
- **Batch Processing**: Efficient handling of thousands of concurrent walks

## 🤝 Contributing

We welcome contributions!
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new algorithm'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request
For installation and test intructions please visit the documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch Cluster](https://github.com/rusty1s/pytorch_cluster) for everything
- [MLX](https://github.com/ml-explore/mlx) for the foundational framework
- [MLX-Graphs](https://github.com/mlx-graphs/mlx-graphs) for graph utilities and datasets

## 📊 Citation

If you use MLX-Cluster in your research, please cite:

```bibtex
@software{mlx_cluster,
  author = {Vinay Pandya},
  title = {MLX-Cluster: High-Performance Graph Algorithms for Apple MLX},
  url = {https://github.com/vinayhpandya/mlx_cluster},
  version = {0.0.6},
  year = {2025}
}
```

## 🔗 Related Projects

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [MLX-Graphs](https://github.com/mlx-graphs/mlx-graphs) - Graph neural networks for MLX
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) - Graph deep learning library

---
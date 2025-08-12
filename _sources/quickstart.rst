Quick Start
===========

This guide will help you get started with mlx-cluster's graph algorithms in just a few minutes.

Random Walks
-------------

Here's a simple example of performing random walks on a graph:

.. code-block:: python

   import mlx.core as mx
   import mlx_cluster
   import numpy as np
   from mlx_graphs.datasets import PlanetoidDataset
   from mlx_graphs.utils.sorting import sort_edge_index

   # Load Cora dataset
   cora = PlanetoidDataset(name="cora")
   edge_index = cora.graphs[0].edge_index.astype(mx.int64)

   # Convert to CSR format
   sorted_edge_index = sort_edge_index(edge_index=edge_index)
   row = sorted_edge_index[0][0]
   col = sorted_edge_index[0][1]
   _, counts = np.unique(np.array(row, copy=False), return_counts=True)
   row_ptr = mx.concatenate([mx.array([0]), mx.array(counts.cumsum())])

   # Set up random walk parameters
   num_starts = 100
   walk_length = 10
   start_nodes = mx.array(np.random.randint(0, cora.graphs[0].num_nodes, num_starts))
   rand_values = mx.random.uniform(shape=[num_starts, walk_length])
   mx.eval(row_ptr, start_nodes, rand_values, col) #We need to call eval first since mlx uses lazy evaluation and the data won't be available
   # Perform random walk
   node_sequences, edge_sequences = mlx_cluster.random_walk(
       row_ptr, col, start_nodes, rand_values, walk_length
   )

   print(f"Generated {num_starts} walks of length {walk_length + 1}")
   print(f"Node sequences shape: {node_sequences.shape}")

Biased Random Walks (Node2Vec style)
-------------------------------------

For biased random walks with return parameter ``p`` and in-out parameter ``q``:

.. code-block:: python

   # Biased random walk (Node2Vec style)
   node_sequences, edge_sequences = mlx_cluster.rejection_sampling(
       row_ptr, col, start_nodes, walk_length, 
       p=1.0,  # Return parameter
       q=3.0   # In-out parameter
   )

   print(f"Generated biased walks with p={1.0}, q={3.0}")

Neighbor Sampling
-----------------

For GraphSAGE-style neighbor sampling:

.. code-block:: python

   # Convert to CSC format for neighbor sampling
   def create_csc_from_edge_index(edge_index, num_nodes):
       sources = edge_index[0].tolist()
       targets = edge_index[1].tolist()
       
       edges = list(zip(sources, targets))
       edges.sort(key=lambda x: x[1])  # Sort by target
       
       sorted_sources = [e[0] for e in edges]
       sorted_targets = [e[1] for e in edges]
       
       colptr = np.zeros(num_nodes + 1, dtype=np.int64)
       for target in sorted_targets:
           colptr[target + 1] += 1
       
       for i in range(1, num_nodes + 1):
           colptr[i] += colptr[i - 1]
       
       return mx.array(colptr), mx.array(sorted_sources, dtype=mx.int64)

   # Create CSC format
   colptr, row = create_csc_from_edge_index(edge_index, cora.graphs[0].num_nodes)

   # Sample neighbors
   input_nodes = mx.array([0, 1, 2], dtype=mx.int64)
   num_neighbors = [10, 5]  # 10 neighbors in first hop, 5 in second

   samples, rows, cols, edges = mlx_cluster.neighbor_sample(
       colptr, row, input_nodes, num_neighbors,
       replace=True, directed=True
   )

   print(f"Sampled {len(samples)} nodes and {len(edges)} edges")

Key Features
------------

* **MLX Optimized**: Built specifically for Apple's MLX framework
* **Fast Performance**: Optimized for Apple Silicon hardware with GPU support(Only random walks)
* **Graph Algorithms**: Random walks, biased random walks, and neighbor sampling
* **Flexible**: Support for both directed and undirected graphs
* **Memory Efficient**: Efficient CSR/CSC graph representations
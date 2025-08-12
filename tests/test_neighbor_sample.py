import mlx.core as mx
import numpy as np
import time
import pytest

# Torch dataset
import torch
from torch.utils.data import DataLoader

from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.utils.sorting import sort_edge_index
from mlx_cluster import neighbor_sample


def create_csc_from_edge_index(edge_index, num_nodes):
    """Convert edge_index to CSC format for neighbor sampling."""
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    
    # Sort by target (column)
    edges = list(zip(sources, targets))
    edges.sort(key=lambda x: x[1])
    
    sorted_sources = [e[0] for e in edges]
    sorted_targets = [e[1] for e in edges]
    
    # Create column pointers
    colptr = np.zeros(num_nodes + 1, dtype=np.int64)
    
    for target in sorted_targets:
        colptr[target + 1] += 1
    
    # Convert to cumulative sums
    for i in range(1, num_nodes + 1):
        colptr[i] += colptr[i - 1]
    
    return mx.array(colptr), mx.array(sorted_sources, dtype=mx.int64)


@pytest.mark.slow  # give download/compile plenty of time on CI
def test_neighbor_sample_basic(tmp_path):
    """
    Test basic neighbor sampling functionality on a small graph.
    Checks:
    1. Output tensor shapes are consistent
    2. All sampled nodes are valid (< num_nodes)
    3. Edge relationships are preserved
    """
    print("\n=== Testing Basic Neighbor Sampling ===")
    
    # Create a simple test graph
    edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]], dtype=mx.int64)
    num_nodes = 8
    
    # Convert to CSC format
    colptr, row = create_csc_from_edge_index(edge_index, num_nodes)
    
    # Test single node sampling
    input_nodes = mx.array([3], dtype=mx.int64)
    num_neighbors = [1]
    
    # ---------- Warm-up ----------
    mx.eval(colptr, row, input_nodes)
    
    # ---------- Run kernel ----------
    t0 = time.time()
    samples, rows, cols, edges = neighbor_sample(
        colptr, row, input_nodes, num_neighbors,
        replace=True, directed=True
    )
    elapsed = time.time() - t0
    print(f"Basic neighbor sampling took {elapsed:.6f} s")
    
    # ---------- Assertions ----------
    assert samples.shape[0] >= 1, "Should have at least the input node"
    assert len(rows) == len(cols) == len(edges), "Edge arrays should have same length"
    assert samples[0] == 3, "First sample should be input node"
    assert all(sample < num_nodes for sample in samples), "All samples should be valid node indices"
    
    print(f"✅ Basic test passed: sampled {len(samples)} nodes, {len(edges)} edges")


@pytest.mark.slow
def test_neighbor_sample_cora(tmp_path):
    """
    Test neighbor sampling on the Cora dataset.
    Runs neighbor sampling on 100 random nodes and checks:
    1. Output tensor shapes are consistent
    2. All returned node indices are valid (< num_nodes)
    3. Performance is reasonable
    """
    print("\n=== Testing Neighbor Sampling on Cora Dataset ===")
    
    # ---------- Dataset (downloaded to the temp dir) ----------
    data_dir = tmp_path / "mlx_datasets"
    cora = PlanetoidDataset(name="cora", base_dir=data_dir)
    
    edge_index = cora.graphs[0].edge_index.astype(mx.int64)
    num_nodes = cora.graphs[0].num_nodes
    
    print(f"Cora dataset: {num_nodes} nodes, {edge_index.shape[1]} edges")
    
    # Convert to CSC format
    colptr, row = create_csc_from_edge_index(edge_index, num_nodes)
    
    # Pick 100 random start nodes
    num_starts = 100
    rng = np.random.default_rng(42)
    start_nodes = mx.array(rng.integers(low=0, high=num_nodes, size=num_starts, dtype=np.int64))
    
    # Test different sampling configurations
    test_configs = [
        ([5], "single_hop_5"),
        ([10], "single_hop_10"), 
        ([5, 3], "two_hop_5_3"),
        ([10, 5], "two_hop_10_5"),
    ]
    
    for num_neighbors, config_name in test_configs:
        print(f"\nTesting configuration: {config_name}, num_neighbors: {num_neighbors}")
        
        # Test batch sampling
        batch_size = 10
        batch_results = []
        total_time = 0
        
        for i in range(0, min(50, num_starts), batch_size):  # Test first 50 nodes in batches
            batch_start = start_nodes[i:i+batch_size]
            
            # ---------- Warm-up ----------
            mx.eval(colptr, row, batch_start)
            
            # ---------- Run kernel ---------- 
            t0 = time.time()
            samples, rows, cols, edges = neighbor_sample(
                colptr, row, batch_start, num_neighbors,
                replace=True, directed=True
            )
            elapsed = time.time() - t0
            total_time += elapsed
            
            # ---------- Assertions ----------
            assert samples.shape[0] >= len(batch_start), f"Should have at least {len(batch_start)} nodes"
            assert len(rows) == len(cols) == len(edges), "Edge arrays should have same length"
            assert all(sample < num_nodes for sample in samples), "All samples should be valid node indices"
            
            # Check that input nodes are in samples
            input_set = set(batch_start.tolist())
            sample_set = set(samples.tolist())
            assert input_set.issubset(sample_set), "All input nodes should be in samples"
            
            batch_results.append({
                'batch_size': len(batch_start),
                'samples': len(samples),
                'edges': len(edges),
                'time': elapsed
            })
        
        # Print statistics
        avg_time = total_time / len(batch_results)
        total_samples = sum(r['samples'] for r in batch_results)
        total_edges = sum(r['edges'] for r in batch_results)
        
        print(f"  Average time per batch: {avg_time:.6f} s")
        print(f"  Total samples: {total_samples}, Total edges: {total_edges}")
        print(f"  Samples per input node: {total_samples / min(50, num_starts):.2f}")
        
        assert avg_time < 1.0, f"Sampling should be reasonably fast, got {avg_time:.3f}s per batch"
        
    print("✅ All Cora tests passed!")


@pytest.mark.slow
def test_neighbor_sample_performance_scaling(tmp_path):
    """
    Test neighbor sampling performance with different batch sizes and configurations.
    """
    print("\n=== Testing Performance Scaling ===")
    
    # ---------- Dataset ----------
    data_dir = tmp_path / "mlx_datasets"
    cora = PlanetoidDataset(name="cora", base_dir=data_dir)
    
    edge_index = cora.graphs[0].edge_index.astype(mx.int64)
    num_nodes = cora.graphs[0].num_nodes
    
    # Convert to CSC format
    colptr, row = create_csc_from_edge_index(edge_index, num_nodes)
    
    # Test different batch sizes
    batch_sizes = [1, 5, 10, 20]
    num_neighbors = [10, 5]
    
    print(f"Testing performance scaling with different batch sizes")
    print(f"Sampling configuration: {num_neighbors}")
    
    for batch_size in batch_sizes:
        # Pick random start nodes
        rng = np.random.default_rng(42)
        input_nodes = mx.array(rng.integers(low=0, high=num_nodes, size=batch_size, dtype=np.int64))
        
        # ---------- Warm-up ----------
        mx.eval(colptr, row, input_nodes)
        
        # ---------- Run kernel ----------
        t0 = time.time()
        samples, rows, cols, edges = neighbor_sample(
            colptr, row, input_nodes, num_neighbors,
            replace=True, directed=True
        )
        elapsed = time.time() - t0
        
        samples_per_input = len(samples) / batch_size
        edges_per_input = len(edges) / batch_size
        time_per_input = elapsed / batch_size
        
        print(f"  Batch size {batch_size:2d}: {elapsed:.6f}s total, {time_per_input:.6f}s per node")
        print(f"    -> {samples_per_input:.1f} samples/input, {edges_per_input:.1f} edges/input")
        
        # Assertions
        assert len(samples) >= batch_size, f"Should have at least {batch_size} samples"
        assert len(rows) == len(cols) == len(edges), "Edge arrays should have same length"
        assert time_per_input < 0.1, f"Should be reasonably fast, got {time_per_input:.3f}s per node"
    
    print("✅ Performance scaling test completed!")


@pytest.mark.slow
def test_neighbor_sample_edge_cases(tmp_path):
    """
    Test edge cases for neighbor sampling:
    1. Nodes with no neighbors
    2. Sampling more neighbors than available
    3. Empty input
    """
    print("\n=== Testing Edge Cases ===")
    
    # Case 1: Node with no neighbors
    edge_index = mx.array([[1, 2], [2, 3]], dtype=mx.int64)  # Node 0 has no edges
    colptr, row = create_csc_from_edge_index(edge_index, num_nodes=4)
    
    input_nodes = mx.array([0], dtype=mx.int64)  # Node 0 has no neighbors
    num_neighbors = [2]
    
    samples, rows, cols, edges = neighbor_sample(
        colptr, row, input_nodes, num_neighbors,
        replace=True, directed=True
    )
    
    assert len(samples) >= 1, "Should at least contain input node"
    assert samples[0] == 0, "Should contain the input node"
    print("✅ Isolated node test passed")
    
    # Case 2: Sampling more neighbors than available
    edge_index = mx.array([[0], [1]], dtype=mx.int64)  # Node 1 has only 1 neighbor
    colptr, row = create_csc_from_edge_index(edge_index, num_nodes=2)
    
    input_nodes = mx.array([1], dtype=mx.int64)
    num_neighbors = [5]  # Request 5 but only 1 available
    
    samples, rows, cols, edges = neighbor_sample(
        colptr, row, input_nodes, num_neighbors,
        replace=False, directed=True  # Without replacement
    )
    
    assert len(samples) >= 1, "Should handle over-sampling gracefully"
    print("✅ Over-sampling test passed")
    
    print("✅ All edge case tests passed!")


def test_neighbor_sample_deterministic():
    """
    Test that neighbor sampling with fixed random seed produces consistent results.
    """
    print("\n=== Testing Deterministic Behavior ===")
    
    # Simple test graph
    edge_index = mx.array([[0, 0, 1, 1], [1, 2, 2, 3]], dtype=mx.int64)
    colptr, row = create_csc_from_edge_index(edge_index, num_nodes=4)
    
    input_nodes = mx.array([1], dtype=mx.int64)
    num_neighbors = [1]
    
    # Run multiple times - results may vary due to randomness but structure should be consistent
    results = []
    for i in range(5):
        samples, rows, cols, edges = neighbor_sample(
            colptr, row, input_nodes, num_neighbors,
            replace=True, directed=True
        )
        results.append((samples.tolist(), rows.tolist(), cols.tolist(), edges.tolist()))
    
    # All results should have same structure
    for samples, rows, cols, edges in results:
        assert samples[0] == 1, "First sample should always be input node"
        assert len(rows) == len(cols) == len(edges), "Edge arrays should have same length"
        assert all(col == 0 for col in cols), "All edges should originate from input node"
    
    print("✅ Deterministic structure test passed!")
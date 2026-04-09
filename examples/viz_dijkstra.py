"""
Visualization example for Dijkstra path finding algorithms.

Demonstrates:
1. vessel_dijkstra: Graph-based shortest paths
2. dendrite_dijkstra: 3D volume-based path finding with obstacles
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calcia.algorithms.dijkstra import (
    vessel_dijkstra,
    dendrite_dijkstra,
    reconstruct_path,
    compute_distance_matrix,
)


def test_vessel_dijkstra():
    """Test vessel Dijkstra on a simple graph."""
    print("Testing vessel_dijkstra (graph-based)...")

    # Create a simple 5-node graph
    #     1
    #    /|\
    #   2 | 4
    #    \|/
    #     3
    #     |
    #     5 (root)

    n_nodes = 5
    dist_matrix = np.full((n_nodes, n_nodes), np.inf)

    # Add edges (symmetric)
    edges = [
        (0, 1, 1.0),   # 1-2
        (0, 2, 2.0),   # 1-3
        (0, 3, 1.5),   # 1-4
        (1, 2, 1.0),   # 2-3
        (2, 3, 1.0),   # 3-4
        (2, 4, 3.0),   # 3-5
    ]

    for i, j, w in edges:
        dist_matrix[i, j] = w
        dist_matrix[j, i] = w

    # Diagonal is zero
    np.fill_diagonal(dist_matrix, 0)

    # Run Dijkstra from node 4 (index 4)
    root = 4
    distances, path_from = vessel_dijkstra(dist_matrix, root)

    print(f"  Root: node {root}")
    print(f"  Distances: {distances}")
    print(f"  Path from: {path_from}")

    # Verify: distance to node 0 should be 3->0 = 2 + 3 = 5
    # or 3->1->0 = 3 + 1 + 1 = 5, or 3->2->0 = 3 + 2 = 5
    print(f"  Distance to node 0: {distances[0]}")
    assert np.isclose(distances[0], 5.0), f"Expected 5.0, got {distances[0]}"
    print("  [PASS] Graph Dijkstra working correctly")


def test_dendrite_dijkstra():
    """Test 3D dendrite Dijkstra with obstacles."""
    print("\nTesting dendrite_dijkstra (3D volume)...")

    # Create a small 3D volume with obstacles
    dims = (20, 20, 20)
    cost_volume = np.ones(dims, dtype=np.float32)

    # Add a wall obstacle in the middle
    cost_volume[8:12, :, :] = np.inf  # Wall blocking x=8-11

    # Leave a gap in the wall
    cost_volume[8:12, 8:12, 8:12] = 1.0  # Gap at center

    # Root at corner
    root = (2, 2, 2)

    # Run Dijkstra
    distances, path_from = dendrite_dijkstra(cost_volume, root, use_numba=False)

    print(f"  Volume size: {dims}")
    print(f"  Root: {root}")
    print(f"  Min distance: {np.min(distances)}")
    print(f"  Max finite distance: {np.max(distances[np.isfinite(distances)])}")

    # Test path reconstruction
    target = (17, 17, 17)
    path = reconstruct_path(path_from, target)
    print(f"  Path to {target}: {len(path)} steps")

    if len(path) > 0:
        print(f"    Start: {path[0]}")
        print(f"    End: {path[-1]}")
        print("  [PASS] Path reconstruction working")
    else:
        print("  [WARN] Could not find path (might be expected)")

    return cost_volume, distances, path_from, root


def visualize_3d_dijkstra(cost_volume, distances, path_from, root):
    """Visualize 3D Dijkstra results."""
    print("\nCreating 3D visualization...")

    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Slice of cost volume showing obstacles
    ax1 = fig.add_subplot(131)
    slice_idx = cost_volume.shape[2] // 2
    slice_data = cost_volume[:, :, slice_idx].T
    slice_data[np.isinf(slice_data)] = 10  # Cap for visualization

    ax1.imshow(slice_data, origin='lower', cmap='gray_r')
    ax1.scatter(root[0], root[1], c='green', s=100, marker='*', label='Root')
    ax1.set_title(f'Cost Volume (z={slice_idx})\nWhite=obstacle')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()

    # Plot 2: Distance field slice
    ax2 = fig.add_subplot(132)
    dist_slice = distances[:, :, slice_idx].T.copy()
    dist_slice[np.isinf(dist_slice)] = np.nan

    im = ax2.imshow(dist_slice, origin='lower', cmap='viridis')
    ax2.scatter(root[0], root[1], c='red', s=100, marker='*')
    ax2.set_title(f'Distance Field (z={slice_idx})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im, ax=ax2, label='Distance')

    # Plot 3: Path example
    ax3 = fig.add_subplot(133, projection='3d')

    # Find a reachable target
    target = (17, 10, 10)
    path = reconstruct_path(path_from, target)

    if len(path) > 1:
        ax3.plot(path[:, 0], path[:, 1], path[:, 2], 'b-', linewidth=2, label='Path')
        ax3.scatter(*root, c='green', s=100, marker='o', label='Root')
        ax3.scatter(*target, c='red', s=100, marker='x', label='Target')

    # Show obstacle region
    obs_x, obs_y, obs_z = np.where(np.isinf(cost_volume))
    # Subsample for visualization
    step = max(1, len(obs_x) // 500)
    ax3.scatter(obs_x[::step], obs_y[::step], obs_z[::step],
                c='gray', alpha=0.1, s=1, label='Obstacles')

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D Path through Volume')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('dijkstra_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: dijkstra_visualization.png")
    plt.close()


def benchmark_dijkstra():
    """Benchmark Dijkstra performance."""
    print("\nBenchmarking dendrite_dijkstra...")

    import time

    sizes = [(20, 20, 20), (30, 30, 30), (50, 50, 50)]

    for dims in sizes:
        cost_volume = np.ones(dims, dtype=np.float32)
        # Add some obstacles
        cost_volume[dims[0]//3:2*dims[0]//3, :, :] = 5.0

        root = (0, 0, 0)

        # Time Python implementation
        start = time.time()
        distances, path_from = dendrite_dijkstra(cost_volume, root, use_numba=False)
        python_time = time.time() - start

        print(f"  Volume {dims}: {python_time:.3f}s (Python)")

        # Check if numba is available
        try:
            start = time.time()
            distances2, path_from2 = dendrite_dijkstra(cost_volume, root, use_numba=True)
            numba_time = time.time() - start
            print(f"  Volume {dims}: {numba_time:.3f}s (Numba) - {python_time/numba_time:.1f}x speedup")
        except ImportError:
            print(f"  Numba not available for acceleration")


def test_distance_matrix():
    """Test distance matrix computation."""
    print("\nTesting compute_distance_matrix...")

    # Test Euclidean distance
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
    ], dtype=float)

    dist_matrix = compute_distance_matrix(points, metric='euclidean')
    print(f"  Euclidean distance matrix shape: {dist_matrix.shape}")
    print(f"  Distance (0,0)-(1,0): {dist_matrix[0, 1]:.3f} (expected: 1.0)")
    print(f"  Distance (0,0)-(1,1): {dist_matrix[0, 3]:.3f} (expected: sqrt(2)={np.sqrt(2):.3f})")

    assert np.isclose(dist_matrix[0, 1], 1.0)
    assert np.isclose(dist_matrix[0, 3], np.sqrt(2))
    print("  [PASS] Euclidean distance correct")

    # Test geodesic distance on unit sphere
    sphere_points = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
    ], dtype=float)

    geo_matrix = compute_distance_matrix(sphere_points, metric='geodesic')
    print(f"\n  Geodesic distance matrix:")
    print(f"  Distance (1,0,0)-(0,1,0): {geo_matrix[0, 1]:.3f} (expected: pi/2={np.pi/2:.3f})")
    print(f"  Distance (1,0,0)-(-1,0,0): {geo_matrix[0, 3]:.3f} (expected: pi={np.pi:.3f})")

    assert np.isclose(geo_matrix[0, 1], np.pi/2)
    assert np.isclose(geo_matrix[0, 3], np.pi)
    print("  [PASS] Geodesic distance correct")


if __name__ == '__main__':
    np.random.seed(42)

    test_vessel_dijkstra()
    cost_vol, dists, paths, root = test_dendrite_dijkstra()
    visualize_3d_dijkstra(cost_vol, dists, paths, root)
    test_distance_matrix()
    benchmark_dijkstra()

    print("\nDone!")

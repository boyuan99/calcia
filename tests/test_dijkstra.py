"""
Tests for Dijkstra path finding algorithms.
"""

import pytest
import numpy as np

from calcia.algorithms.dijkstra import (
    vessel_dijkstra,
    dendrite_dijkstra,
    reconstruct_path,
    compute_distance_matrix,
)


class TestVesselDijkstra:
    """Tests for vessel_dijkstra function."""

    def test_distance_to_self_is_zero(self):
        """Distance from root to itself should be zero."""
        dist_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ], dtype=float)

        distances, _ = vessel_dijkstra(dist_matrix, root=0)

        assert distances[0] == 0.0

    def test_direct_edges(self):
        """Direct edge distances should be correct."""
        dist_matrix = np.array([
            [0, 1, 3],
            [1, 0, 1],
            [3, 1, 0],
        ], dtype=float)

        distances, _ = vessel_dijkstra(dist_matrix, root=0)

        assert distances[1] == 1.0  # Direct edge 0->1

    def test_shortest_path(self):
        """Should find shortest path, not just any path."""
        # Graph where direct path is longer than indirect
        dist_matrix = np.array([
            [0, 1, 10],
            [1, 0, 1],
            [10, 1, 0],
        ], dtype=float)

        distances, path_from = vessel_dijkstra(dist_matrix, root=0)

        # Shortest path to node 2: 0->1->2 = 2, not 0->2 = 10
        assert distances[2] == 2.0
        assert path_from[2] == 1  # Came from node 1

    def test_unreachable_nodes(self):
        """Unreachable nodes should have infinite distance."""
        dist_matrix = np.array([
            [0, 1, np.inf],
            [1, 0, np.inf],
            [np.inf, np.inf, 0],
        ], dtype=float)

        distances, path_from = vessel_dijkstra(dist_matrix, root=0)

        assert np.isinf(distances[2])
        assert path_from[2] == -1

    def test_symmetric_graph(self):
        """Symmetric graph should work correctly."""
        n = 5
        dist_matrix = np.random.rand(n, n) + 0.1
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)

        distances, path_from = vessel_dijkstra(dist_matrix, root=0)

        # All nodes should be reachable
        assert np.all(np.isfinite(distances))

    def test_path_from_valid(self):
        """Path from should contain valid node indices."""
        dist_matrix = np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ], dtype=float)

        _, path_from = vessel_dijkstra(dist_matrix, root=0)

        # Root has no parent
        assert path_from[0] == -1

        # Other nodes should have valid parents
        for i in range(1, 4):
            assert 0 <= path_from[i] < 4


class TestDendriteDijkstra:
    """Tests for dendrite_dijkstra function."""

    def test_distance_at_root_is_zero(self):
        """Distance at root should be zero."""
        cost_volume = np.ones((10, 10, 10), dtype=np.float32)
        root = (5, 5, 5)

        distances, _ = dendrite_dijkstra(cost_volume, root, use_numba=False)

        assert distances[root] == 0.0

    def test_uniform_cost_manhattan_distance(self):
        """With uniform cost, distance should equal path length."""
        cost_volume = np.ones((10, 10, 10), dtype=np.float32)
        root = (0, 0, 0)

        distances, _ = dendrite_dijkstra(cost_volume, root, use_numba=False)

        # Distance to (5, 0, 0) should be 5 (5 steps of cost 1)
        assert distances[5, 0, 0] == 5.0

        # Distance to corner should be sum of coordinates
        assert distances[3, 4, 2] == 3 + 4 + 2

    def test_obstacles_block_path(self):
        """High cost obstacles should be avoided."""
        cost_volume = np.ones((10, 10, 10), dtype=np.float32)
        # Create wall
        cost_volume[5, :, :] = np.inf
        root = (0, 0, 0)

        distances, _ = dendrite_dijkstra(cost_volume, root, use_numba=False)

        # Points behind wall should be unreachable
        assert np.isinf(distances[6, 5, 5])

    def test_obstacle_with_gap(self):
        """Should find path through gap in obstacle."""
        cost_volume = np.ones((10, 10, 10), dtype=np.float32)
        # Wall with gap
        cost_volume[5, :, :] = np.inf
        cost_volume[5, 5, 5] = 1.0  # Gap
        root = (0, 0, 0)

        distances, path_from = dendrite_dijkstra(cost_volume, root, use_numba=False)

        # Point behind wall should be reachable through gap
        assert np.isfinite(distances[7, 5, 5])

    def test_path_reconstruction(self):
        """Reconstructed path should be valid."""
        cost_volume = np.ones((10, 10, 10), dtype=np.float32)
        root = (0, 0, 0)
        target = (5, 5, 5)

        distances, path_from = dendrite_dijkstra(cost_volume, root, use_numba=False)
        path = reconstruct_path(path_from, target)

        # Path should start at root and end at target
        assert len(path) > 0
        assert tuple(path[-1]) == target

        # Each step should be 6-connected
        for i in range(len(path) - 1):
            diff = np.abs(path[i+1] - path[i])
            assert np.sum(diff) == 1  # Only one coordinate changes by 1

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        dims = (8, 10, 12)
        cost_volume = np.ones(dims, dtype=np.float32)
        root = (2, 3, 4)

        distances, path_from = dendrite_dijkstra(cost_volume, root, use_numba=False)

        assert distances.shape == dims
        assert path_from.shape == (*dims, 3)


class TestReconstructPath:
    """Tests for reconstruct_path function."""

    def test_empty_path_for_unreachable(self):
        """Should return empty path for unreachable target."""
        path_from = np.full((5, 5, 5, 3), -1, dtype=np.int32)

        path = reconstruct_path(path_from, (3, 3, 3))

        assert len(path) == 0

    def test_path_to_root(self):
        """Path to root should be empty or single point."""
        cost_volume = np.ones((5, 5, 5), dtype=np.float32)
        root = (2, 2, 2)

        _, path_from = dendrite_dijkstra(cost_volume, root, use_numba=False)
        path = reconstruct_path(path_from, root)

        # Root has no parent, so path might be empty or just root
        assert len(path) <= 1


class TestComputeDistanceMatrix:
    """Tests for compute_distance_matrix function."""

    def test_euclidean_distance(self):
        """Euclidean distances should be correct."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float)

        dist = compute_distance_matrix(points, metric='euclidean')

        assert dist.shape == (3, 3)
        assert dist[0, 1] == 1.0
        assert dist[0, 2] == 1.0
        assert np.isclose(dist[1, 2], np.sqrt(2))

    def test_geodesic_distance_orthogonal(self):
        """Geodesic distance between orthogonal vectors should be pi/2."""
        points = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)

        dist = compute_distance_matrix(points, metric='geodesic')

        assert np.allclose(dist[0, 1], np.pi / 2)
        assert np.allclose(dist[0, 2], np.pi / 2)
        assert np.allclose(dist[1, 2], np.pi / 2)

    def test_geodesic_distance_antipodal(self):
        """Geodesic distance between antipodal points should be pi."""
        points = np.array([
            [1, 0, 0],
            [-1, 0, 0],
        ], dtype=float)

        dist = compute_distance_matrix(points, metric='geodesic')

        assert np.isclose(dist[0, 1], np.pi)

    def test_diagonal_is_zero(self):
        """Distance from point to itself should be zero."""
        points = np.random.randn(10, 3)

        dist_euc = compute_distance_matrix(points, metric='euclidean')
        dist_geo = compute_distance_matrix(points / np.linalg.norm(points, axis=1, keepdims=True),
                                           metric='geodesic')

        assert np.allclose(np.diag(dist_euc), 0)
        assert np.allclose(np.diag(dist_geo), 0)

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        points = np.random.randn(10, 3)

        dist = compute_distance_matrix(points, metric='euclidean')

        assert np.allclose(dist, dist.T)

    def test_invalid_metric(self):
        """Should raise error for invalid metric."""
        points = np.random.randn(5, 3)

        with pytest.raises(ValueError):
            compute_distance_matrix(points, metric='invalid')


class TestDijkstraNumba:
    """Tests for numba-accelerated Dijkstra."""

    def test_numba_matches_python(self):
        """Numba implementation should match pure Python."""
        cost_volume = np.ones((15, 15, 15), dtype=np.float32)
        cost_volume[7, :, :] = 5.0  # Add some variation
        root = (0, 0, 0)

        dist_python, path_python = dendrite_dijkstra(cost_volume, root, use_numba=False)

        try:
            dist_numba, path_numba = dendrite_dijkstra(cost_volume, root, use_numba=True)

            # Distances should match
            assert np.allclose(dist_python, dist_numba)
        except ImportError:
            pytest.skip("Numba not available")

"""
Statistical tests for sphere sampling module.

These tests verify the statistical properties of sphere sampling,
not exact numerical values (which would differ from MATLAB).
"""

import pytest
import numpy as np
from scipy.spatial import cKDTree, ConvexHull

from calcia.geometry.sphere_sampling import (
    spiral_sample_sphere,
    fibonacci_sphere,
    geodesic_distance_matrix,
    icosahedron_vertices,
    subdivide_sphere_mesh,
)


class TestSpiralSampleSphere:
    """Tests for spiral_sample_sphere function."""

    def test_points_on_unit_sphere(self):
        """All sampled points should lie on the unit sphere."""
        V, _ = spiral_sample_sphere(200)

        distances = np.linalg.norm(V, axis=1)
        assert np.allclose(distances, 1.0, atol=1e-10), \
            f"Points not on unit sphere: max deviation = {np.max(np.abs(distances - 1.0))}"

    def test_correct_number_of_points(self):
        """Should return exactly the requested number of points."""
        for n in [50, 100, 200, 500]:
            V, _ = spiral_sample_sphere(n)
            assert V.shape == (n, 3), f"Expected {n} points, got {V.shape[0]}"

    def test_uniformity_coefficient_of_variation(self):
        """
        Points should be uniformly distributed.

        Measured by coefficient of variation (CV) of nearest neighbor distances.
        CV should be small for uniform distributions.
        """
        V, _ = spiral_sample_sphere(500)

        tree = cKDTree(V)
        # Query 2 nearest neighbors (first is self with distance 0)
        nn_dists, _ = tree.query(V, k=2)
        nn_dists = nn_dists[:, 1]  # Exclude self

        cv = np.std(nn_dists) / np.mean(nn_dists)
        assert cv < 0.3, f"Poor uniformity: CV = {cv:.3f} (threshold: 0.3)"

    def test_triangulation_valid(self):
        """Triangulation should form a valid closed surface."""
        V, Tri = spiral_sample_sphere(200, return_triangulation=True)

        assert Tri is not None, "Triangulation should not be None"
        assert Tri.shape[1] == 3, "Faces should be triangles"

        # All vertex indices should be valid
        assert np.all(Tri >= 0), "Negative vertex indices"
        assert np.all(Tri < len(V)), "Vertex index out of bounds"

    def test_no_triangulation_option(self):
        """Should return None for Tri when triangulation not requested."""
        V, Tri = spiral_sample_sphere(100, return_triangulation=False)

        assert V is not None
        assert Tri is None

    def test_coverage_of_sphere(self):
        """
        Points should cover the entire sphere.

        Check that points span the full range of z-coordinates.
        """
        V, _ = spiral_sample_sphere(200)

        z_coords = V[:, 2]
        assert np.min(z_coords) < -0.95, "Missing points near south pole"
        assert np.max(z_coords) > 0.95, "Missing points near north pole"

    def test_different_sample_sizes(self):
        """Test with various sample sizes."""
        for n in [10, 50, 100, 500, 1000]:
            V, Tri = spiral_sample_sphere(n)

            # Basic validity
            assert V.shape == (n, 3)
            assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-10)


class TestFibonacciSphere:
    """Tests for fibonacci_sphere function."""

    def test_points_on_unit_sphere(self):
        """All points should lie on the unit sphere."""
        V = fibonacci_sphere(200)

        distances = np.linalg.norm(V, axis=1)
        assert np.allclose(distances, 1.0, atol=1e-10)

    def test_uniformity(self):
        """Points should be uniformly distributed."""
        V = fibonacci_sphere(500)

        tree = cKDTree(V)
        nn_dists, _ = tree.query(V, k=2)
        cv = np.std(nn_dists[:, 1]) / np.mean(nn_dists[:, 1])

        assert cv < 0.3, f"Poor uniformity: CV = {cv:.3f}"


class TestGeodesicDistanceMatrix:
    """Tests for geodesic distance computation."""

    def test_diagonal_is_zero(self):
        """Distance from a point to itself should be zero."""
        V, _ = spiral_sample_sphere(50)
        D = geodesic_distance_matrix(V)

        assert np.allclose(np.diag(D), 0.0, atol=1e-10)

    def test_symmetry(self):
        """Distance matrix should be symmetric."""
        V, _ = spiral_sample_sphere(50)
        D = geodesic_distance_matrix(V)

        assert np.allclose(D, D.T, atol=1e-10)

    def test_known_distances(self):
        """Test against known geodesic distances."""
        # Orthogonal unit vectors
        V = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
        ], dtype=float)

        D = geodesic_distance_matrix(V)

        # Distance between orthogonal vectors is pi/2
        assert np.isclose(D[0, 1], np.pi / 2, atol=1e-10)
        assert np.isclose(D[0, 2], np.pi / 2, atol=1e-10)
        assert np.isclose(D[1, 2], np.pi / 2, atol=1e-10)

        # Distance between opposite vectors is pi
        assert np.isclose(D[0, 3], np.pi, atol=1e-10)

    def test_triangle_inequality(self):
        """Geodesic distances should satisfy triangle inequality."""
        V, _ = spiral_sample_sphere(50)
        D = geodesic_distance_matrix(V)

        n = len(V)
        for i in range(min(n, 20)):  # Check subset for speed
            for j in range(i + 1, min(n, 20)):
                for k in range(j + 1, min(n, 20)):
                    assert D[i, j] <= D[i, k] + D[k, j] + 1e-10


class TestIcosahedronVertices:
    """Tests for icosahedron mesh generation."""

    def test_vertex_count(self):
        """Icosahedron should have exactly 12 vertices."""
        V, F = icosahedron_vertices()
        assert len(V) == 12

    def test_face_count(self):
        """Icosahedron should have exactly 20 faces."""
        V, F = icosahedron_vertices()
        assert len(F) == 20

    def test_vertices_on_sphere(self):
        """All vertices should be on unit sphere."""
        V, F = icosahedron_vertices()
        distances = np.linalg.norm(V, axis=1)
        assert np.allclose(distances, 1.0, atol=1e-10)


class TestSubdivideSphere:
    """Tests for sphere mesh subdivision."""

    def test_vertex_count_increases(self):
        """Subdivision should increase vertex count."""
        V, F = icosahedron_vertices()
        V_sub, F_sub = subdivide_sphere_mesh(V, F, n_subdivisions=1)

        assert len(V_sub) > len(V)
        assert len(F_sub) > len(F)

    def test_vertices_remain_on_sphere(self):
        """After subdivision, all vertices should still be on unit sphere."""
        V, F = icosahedron_vertices()
        V_sub, F_sub = subdivide_sphere_mesh(V, F, n_subdivisions=2)

        distances = np.linalg.norm(V_sub, axis=1)
        assert np.allclose(distances, 1.0, atol=1e-10)

    def test_face_count_quadruples(self):
        """Each subdivision should quadruple the face count."""
        V, F = icosahedron_vertices()
        n_faces_0 = len(F)

        V1, F1 = subdivide_sphere_mesh(V, F, n_subdivisions=1)
        assert len(F1) == 4 * n_faces_0

        V2, F2 = subdivide_sphere_mesh(V, F, n_subdivisions=2)
        assert len(F2) == 16 * n_faces_0

"""
Tests for Gaussian Process sampling module.

Tests statistical properties rather than exact values due to
randomness in GP sampling.
"""

import pytest
import numpy as np

from calcia.algorithms.gaussian_process import (
    sample_gp_sphere,
    ensure_psd,
    sample_multivariate_normal,
    compute_geodesic_covariance,
    teardrop_projection,
)
from calcia.geometry.sphere_sampling import spiral_sample_sphere


class TestSampleGpSphere:
    """Tests for sample_gp_sphere function."""

    def test_output_shape(self):
        """Output should have same length as input vertices."""
        V, _ = spiral_sample_sphere(100)
        sample = sample_gp_sphere(V)

        assert sample.shape == (100,)

    def test_different_samples(self):
        """Multiple samples should be different."""
        V, _ = spiral_sample_sphere(50)

        sample1 = sample_gp_sphere(V)
        sample2 = sample_gp_sphere(V)

        assert not np.allclose(sample1, sample2)

    def test_reproducible_with_seed(self):
        """Same seed should produce same results."""
        V, _ = spiral_sample_sphere(50)

        np.random.seed(42)
        sample1 = sample_gp_sphere(V)

        np.random.seed(42)
        sample2 = sample_gp_sphere(V)

        assert np.allclose(sample1, sample2)

    def test_mean_function(self):
        """Samples with mean should differ from zero-mean samples."""
        V, _ = spiral_sample_sphere(100)
        mean = np.ones(100) * 5.0

        np.random.seed(42)
        sample_zero_mean = sample_gp_sphere(V, mean=None)

        np.random.seed(42)
        sample_with_mean = sample_gp_sphere(V, mean=mean)

        # Sample with mean should be shifted by the mean
        assert np.allclose(sample_with_mean, sample_zero_mean + mean)

    def test_length_scale_effect(self):
        """Larger l_scale should produce smoother samples."""
        V, _ = spiral_sample_sphere(200)

        # Small length scale - more variation
        sample_rough = sample_gp_sphere(V, l_scale=10.0)

        # Large length scale - smoother
        sample_smooth = sample_gp_sphere(V, l_scale=200.0)

        # Compute local variation (difference between neighbors)
        # Smoother should have lower variation
        rough_var = np.std(np.diff(sample_rough))
        smooth_var = np.std(np.diff(sample_smooth))

        assert smooth_var < rough_var


class TestEnsurePsd:
    """Tests for ensure_psd function."""

    def test_already_psd(self):
        """Should not modify already PSD matrix."""
        # Create a valid covariance matrix
        A = np.random.randn(10, 5)
        cov = A @ A.T + 0.1 * np.eye(10)

        result = ensure_psd(cov)

        # Should be nearly unchanged
        assert np.allclose(result, cov, atol=0.01)

    def test_fixes_negative_eigenvalue(self):
        """Should fix matrix with negative eigenvalues."""
        # Create matrix with negative eigenvalue
        n = 10
        cov = np.random.randn(n, n)
        cov = (cov + cov.T) / 2  # Make symmetric

        result = ensure_psd(cov)

        # Check all eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.all(eigenvalues >= -1e-10)

    def test_symmetric_output(self):
        """Output should be symmetric."""
        cov = np.random.randn(10, 10)
        cov = (cov + cov.T) / 2

        result = ensure_psd(cov)

        assert np.allclose(result, result.T)


class TestSampleMultivariateNormal:
    """Tests for sample_multivariate_normal function."""

    def test_output_shape(self):
        """Output should match mean vector length."""
        mean = np.zeros(20)
        cov = np.eye(20)

        sample = sample_multivariate_normal(mean, cov)

        assert sample.shape == (20,)

    def test_mean_convergence(self):
        """Sample mean should converge to true mean."""
        mean = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)

        samples = [sample_multivariate_normal(mean, cov) for _ in range(1000)]
        sample_mean = np.mean(samples, axis=0)

        assert np.allclose(sample_mean, mean, atol=0.2)

    def test_cholesky_method(self):
        """Cholesky method should work for PSD matrix."""
        mean = np.zeros(5)
        cov = np.eye(5)

        sample = sample_multivariate_normal(mean, cov, method='cholesky')

        assert sample.shape == (5,)

    def test_eig_method(self):
        """Eigenvalue method should work."""
        mean = np.zeros(5)
        cov = np.eye(5)

        sample = sample_multivariate_normal(mean, cov, method='eig')

        assert sample.shape == (5,)


class TestComputeGeodesicCovariance:
    """Tests for compute_geodesic_covariance function."""

    def test_output_shape(self):
        """Output should be NxN matrix."""
        V, _ = spiral_sample_sphere(50)

        cov = compute_geodesic_covariance(V, l_scale=90.0, p_scale=1000.0)

        assert cov.shape == (50, 50)

    def test_symmetric(self):
        """Covariance matrix should be symmetric."""
        V, _ = spiral_sample_sphere(50)

        cov = compute_geodesic_covariance(V, l_scale=90.0, p_scale=1000.0)

        assert np.allclose(cov, cov.T)

    def test_diagonal_is_p_scale(self):
        """Diagonal should equal p_scale (distance to self is 0)."""
        V, _ = spiral_sample_sphere(50)
        p_scale = 1000.0

        cov = compute_geodesic_covariance(V, l_scale=90.0, p_scale=p_scale)

        assert np.allclose(np.diag(cov), p_scale)

    def test_positive_values(self):
        """All covariance values should be positive."""
        V, _ = spiral_sample_sphere(50)

        cov = compute_geodesic_covariance(V, l_scale=90.0, p_scale=1000.0)

        assert np.all(cov > 0)


class TestTeardropProjection:
    """Tests for teardrop_projection function."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        V, _ = spiral_sample_sphere(100)

        V_tear = teardrop_projection(V, p=1)

        assert V_tear.shape == V.shape

    def test_p1_teardrop_shape(self):
        """p=1 should create teardrop with pointed end."""
        V, _ = spiral_sample_sphere(200)

        V_tear = teardrop_projection(V, p=1)

        # Teardrop should have asymmetric z-extent
        z_max = np.max(V_tear[:, 2])
        z_min = np.min(V_tear[:, 2])

        # The teardrop goes from z=-1 to z=1 but is asymmetric in radii
        assert np.isclose(z_max, 1.0, atol=0.1)
        assert np.isclose(z_min, -1.0, atol=0.1)

    def test_p2_peanut_shape(self):
        """p=2 should create peanut/double-bulb shape."""
        V, _ = spiral_sample_sphere(200)

        V_peanut = teardrop_projection(V, p=2)

        # Should still have points
        assert len(V_peanut) == 200

    def test_no_nan_values(self):
        """Output should not contain NaN values."""
        V, _ = spiral_sample_sphere(100)

        V_tear = teardrop_projection(V, p=1)

        assert not np.any(np.isnan(V_tear))

    def test_preserves_z_range(self):
        """Z coordinates should span from -1 to 1."""
        V, _ = spiral_sample_sphere(500)

        V_tear = teardrop_projection(V, p=1)

        assert np.min(V_tear[:, 2]) < -0.9
        assert np.max(V_tear[:, 2]) > 0.9

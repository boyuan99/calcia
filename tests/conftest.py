"""
Pytest configuration and fixtures for calcia tests.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducibility within a test."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture
def default_vol_params():
    """Provide default volume parameters for testing."""
    from calcia.config import VolumeParams
    return VolumeParams(
        vol_sz=(100, 100, 50),
        vres=2,
        min_dist=16.0,
        verbose=0
    )


@pytest.fixture
def default_neur_params():
    """Provide default neuron parameters for testing."""
    from calcia.config import NeuronParams
    return NeuronParams(
        n_samps=200,
        avg_rad=5.9,
        l_scale=90.0,
        p_scale=1000.0
    )


@pytest.fixture
def small_vol_params():
    """Provide small volume parameters for fast testing."""
    from calcia.config import VolumeParams
    return VolumeParams(
        vol_sz=(50, 50, 30),
        vres=1,
        min_dist=10.0,
        verbose=0
    )


@pytest.fixture
def unit_sphere_points():
    """Generate test points on unit sphere."""
    from calcia.geometry.sphere_sampling import spiral_sample_sphere
    V, Tri = spiral_sample_sphere(100)
    return V, Tri


# Tolerance settings for numerical comparisons
ATOL = 1e-10  # Absolute tolerance
RTOL = 1e-5   # Relative tolerance

# Statistical thresholds
UNIFORMITY_CV_THRESHOLD = 0.3  # Coefficient of variation threshold for uniformity

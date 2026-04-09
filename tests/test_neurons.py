"""
Tests for neuron shape generation module.

Tests statistical properties of generated neuron shapes.
"""

import pytest
import numpy as np
from scipy.spatial import ConvexHull

from calcia.config.params import NeuronParams
from calcia.volume.neurons import (
    generate_neural_body,
    generate_multiple_neurons,
    sample_dense_neurons,
    compute_neuron_statistics,
)
from calcia.config.params import VolumeParams


class TestGenerateNeuralBody:
    """Tests for generate_neural_body function."""

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        params = NeuronParams(n_samps=100)

        Vcell, Vnuc, faces, angles = generate_neural_body(params)

        assert Vcell.shape == (100, 3)
        assert Vnuc.shape == (100, 3)
        assert faces.shape[1] == 3  # Triangular faces
        assert angles.shape == (3,)

    def test_soma_contains_nucleus(self):
        """Nucleus should be inside or very close to soma."""
        params = NeuronParams(n_samps=200)

        Vcell, Vnuc, _, _ = generate_neural_body(params)

        # Nucleus centroid should be inside soma bounds
        soma_center = np.mean(Vcell, axis=0)
        nuc_center = np.mean(Vnuc, axis=0)

        # Distance between centers should be small relative to cell size
        soma_radius = np.mean(np.linalg.norm(Vcell - soma_center, axis=1))
        center_dist = np.linalg.norm(soma_center - nuc_center)

        assert center_dist < soma_radius

    def test_nucleus_smaller_than_soma(self):
        """Nucleus should be smaller than soma."""
        params = NeuronParams(n_samps=200)

        Vcell, Vnuc, _, _ = generate_neural_body(params)

        try:
            soma_hull = ConvexHull(Vcell)
            nuc_hull = ConvexHull(Vnuc)

            assert nuc_hull.volume < soma_hull.volume
        except Exception:
            # If convex hull fails, compare average radii
            soma_radius = np.mean(np.linalg.norm(Vcell, axis=1))
            nuc_radius = np.mean(np.linalg.norm(Vnuc, axis=1))
            assert nuc_radius < soma_radius

    def test_pyramidal_neuron_shape(self):
        """Pyramidal neurons should have teardrop-like shape."""
        params = NeuronParams(n_samps=200, neur_type='pyr', max_ang=0)

        Vcell, _, _, _ = generate_neural_body(params)

        # Pyramidal neurons should be elongated in z
        z_range = np.max(Vcell[:, 2]) - np.min(Vcell[:, 2])
        x_range = np.max(Vcell[:, 0]) - np.min(Vcell[:, 0])

        # Z extent should be significant
        assert z_range > 0.5 * x_range

    def test_different_shapes_each_call(self):
        """Each call should produce different shapes."""
        params = NeuronParams(n_samps=100)

        Vcell1, _, _, _ = generate_neural_body(params)
        Vcell2, _, _, _ = generate_neural_body(params)

        assert not np.allclose(Vcell1, Vcell2)

    def test_reproducible_with_seed(self):
        """Same seed should produce same shape."""
        params = NeuronParams(n_samps=100)

        np.random.seed(42)
        Vcell1, Vnuc1, _, angles1 = generate_neural_body(params)

        np.random.seed(42)
        Vcell2, Vnuc2, _, angles2 = generate_neural_body(params)

        assert np.allclose(Vcell1, Vcell2)
        assert np.allclose(Vnuc1, Vnuc2)
        assert np.allclose(angles1, angles2)

    def test_rotation_angles_within_bounds(self):
        """Rotation angles should be within max_ang bounds."""
        max_ang = 20
        params = NeuronParams(n_samps=100, max_ang=max_ang)

        for _ in range(10):
            _, _, _, angles = generate_neural_body(params)
            assert np.all(np.abs(angles) <= max_ang)

    def test_custom_vertices_faces(self):
        """Should accept custom vertices and faces."""
        from calcia.geometry.sphere_sampling import spiral_sample_sphere

        V, F = spiral_sample_sphere(150)
        params = NeuronParams()

        Vcell, Vnuc, faces, _ = generate_neural_body(params, vertices=V, faces=F)

        assert Vcell.shape[0] == 150
        assert np.array_equal(faces, F)


class TestGenerateMultipleNeurons:
    """Tests for generate_multiple_neurons function."""

    def test_correct_number_of_neurons(self):
        """Should generate requested number of neurons."""
        neurons, angles, positions = generate_multiple_neurons(5)

        assert len(neurons) == 5
        assert len(angles) == 5
        assert positions.shape == (5, 3)

    def test_neurons_at_positions(self):
        """Neurons should be centered at specified positions."""
        positions = np.array([
            [0, 0, 0],
            [50, 0, 0],
            [0, 50, 0],
        ], dtype=float)

        neurons, _, _ = generate_multiple_neurons(3, positions=positions)

        for i, (Vcell, Vnuc, _) in enumerate(neurons):
            cell_center = np.mean(Vcell, axis=0)
            expected = positions[i]

            # Center should be close to specified position
            assert np.allclose(cell_center, expected, atol=15.0)

    def test_all_neurons_have_correct_structure(self):
        """Each neuron should have valid structure."""
        neurons, angles, _ = generate_multiple_neurons(3)

        for Vcell, Vnuc, faces in neurons:
            assert Vcell.ndim == 2
            assert Vcell.shape[1] == 3
            assert Vnuc.shape == Vcell.shape
            assert faces.shape[1] == 3


class TestComputeNeuronStatistics:
    """Tests for compute_neuron_statistics function."""

    def test_statistics_keys(self):
        """Should return expected statistics."""
        params = NeuronParams(n_samps=100)
        Vcell, Vnuc, _, _ = generate_neural_body(params)

        stats = compute_neuron_statistics(Vcell, Vnuc)

        assert 'avg_radius' in stats
        assert 'min_radius' in stats
        assert 'max_radius' in stats
        assert 'volume' in stats
        assert 'sphericity' in stats
        assert 'nuc_avg_radius' in stats
        assert 'nuc_volume' in stats

    def test_radius_values(self):
        """Radius values should be reasonable."""
        params = NeuronParams(n_samps=200, avg_rad=6.0)
        Vcell, _, _, _ = generate_neural_body(params)

        stats = compute_neuron_statistics(Vcell)

        # Average radius should be in reasonable range
        assert 3.0 < stats['avg_radius'] < 15.0
        assert stats['min_radius'] < stats['avg_radius']
        assert stats['max_radius'] > stats['avg_radius']

    def test_sphericity_range(self):
        """Sphericity should be between 0 and 1."""
        params = NeuronParams(n_samps=200)
        Vcell, _, _, _ = generate_neural_body(params)

        stats = compute_neuron_statistics(Vcell)

        if stats['sphericity'] is not None:
            assert 0 < stats['sphericity'] <= 1.0

    def test_volume_positive(self):
        """Volume should be positive."""
        params = NeuronParams(n_samps=200)
        Vcell, Vnuc, _, _ = generate_neural_body(params)

        stats = compute_neuron_statistics(Vcell, Vnuc)

        if stats['volume'] is not None:
            assert stats['volume'] > 0
        if stats['nuc_volume'] is not None:
            assert stats['nuc_volume'] > 0


class TestNeuronStatisticalProperties:
    """Statistical tests over many neurons."""

    def test_average_radius_distribution(self):
        """Average radius should be close to specified avg_rad."""
        params = NeuronParams(n_samps=200, avg_rad=5.9)

        radii = []
        for _ in range(30):
            Vcell, _, _, _ = generate_neural_body(params)
            stats = compute_neuron_statistics(Vcell)
            radii.append(stats['avg_radius'])

        mean_radius = np.mean(radii)

        # Should be within 30% of target (allowing for GP variation)
        assert abs(mean_radius - params.avg_rad) / params.avg_rad < 0.3

    def test_shape_variation(self):
        """Different neurons should have different shapes."""
        params = NeuronParams(n_samps=100)

        volumes = []
        for _ in range(20):
            Vcell, _, _, _ = generate_neural_body(params)
            stats = compute_neuron_statistics(Vcell)
            if stats['volume'] is not None:
                volumes.append(stats['volume'])

        # Should have non-zero standard deviation
        if len(volumes) > 1:
            assert np.std(volumes) > 0


class TestSampleDenseNeurons:
    """Tests for sample_dense_neurons function."""

    def test_basic_sampling(self):
        """Test basic neuron sampling without vessels."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(100, 100, 50), N_neur=10)

        neurons, angles, positions = sample_dense_neurons(
            vol_params=vol_params, verbose=0
        )

        assert len(neurons) == 10
        assert len(angles) == 10
        assert positions.shape == (10, 3)

    def test_minimum_distance_enforced(self):
        """Test that min_dist is enforced between neurons."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(100, 100, 50), N_neur=15, min_dist=20.0)

        _, _, positions = sample_dense_neurons(vol_params=vol_params, verbose=0)

        # Check all pairwise distances
        from scipy.spatial.distance import pdist

        if len(positions) > 1:
            distances = pdist(positions)
            # All distances should be >= min_dist (with small tolerance for rounding)
            assert np.all(distances >= vol_params.min_dist - 1.0)

    def test_vessel_avoidance(self):
        """Test that neurons avoid vessel mask."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(50, 50, 30), N_neur=10, vres=2)

        # Create vessel mask with a vessel in the center
        grid_shape = tuple(int(s * vol_params.vres) for s in vol_params.vol_sz)
        vessel_mask = np.zeros(grid_shape, dtype=np.uint8)
        # Block in center (in voxel coordinates)
        vessel_mask[40:60, 40:60, 20:40] = 1

        _, _, positions = sample_dense_neurons(
            vol_params=vol_params,
            vessel_mask=vessel_mask,
            verbose=0,
        )

        # Check neurons are not in vessel region
        for pos in positions:
            # Convert to voxel coords
            vx = int(pos[0] * vol_params.vres)
            vy = int(pos[1] * vol_params.vres)
            vz = int(pos[2] * vol_params.vres)

            # Clamp to valid range
            vx = min(max(vx, 0), grid_shape[0] - 1)
            vy = min(max(vy, 0), grid_shape[1] - 1)
            vz = min(max(vz, 0), grid_shape[2] - 1)

            # Should not be inside vessel
            assert vessel_mask[vx, vy, vz] == 0

    def test_saturation_handling(self):
        """Test graceful handling when volume saturates."""
        np.random.seed(42)
        # Small volume, many neurons requested
        vol_params = VolumeParams(vol_sz=(30, 30, 20), N_neur=100, min_dist=15.0)

        neurons, _, _ = sample_dense_neurons(vol_params=vol_params, verbose=0)

        # Should return fewer neurons than requested
        assert len(neurons) < 100
        assert len(neurons) > 0

    def test_positions_within_bounds(self):
        """Test all positions are within volume bounds."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(80, 80, 40), N_neur=15)

        _, _, positions = sample_dense_neurons(vol_params=vol_params, verbose=0)

        assert np.all(positions[:, 0] >= 0) and np.all(positions[:, 0] <= 80)
        assert np.all(positions[:, 1] >= 0) and np.all(positions[:, 1] <= 80)
        assert np.all(positions[:, 2] >= 0) and np.all(positions[:, 2] <= 40)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        vol_params = VolumeParams(vol_sz=(50, 50, 30), N_neur=5)

        np.random.seed(42)
        _, _, pos1 = sample_dense_neurons(vol_params=vol_params, verbose=0)

        np.random.seed(42)
        _, _, pos2 = sample_dense_neurons(vol_params=vol_params, verbose=0)

        assert np.allclose(pos1, pos2)

    def test_single_neuron_centered(self):
        """Test that single neuron is centered."""
        vol_params = VolumeParams(vol_sz=(100, 100, 50), N_neur=1)

        _, _, positions = sample_dense_neurons(vol_params=vol_params, verbose=0)

        expected_center = np.array([50, 50, 25])
        assert np.allclose(positions[0], expected_center, atol=1.0)

    def test_neuron_structure_valid(self):
        """Test that each neuron has valid structure."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(80, 80, 40), N_neur=5)

        neurons, angles, _ = sample_dense_neurons(vol_params=vol_params, verbose=0)

        for Vcell, Vnuc, faces in neurons:
            assert Vcell.ndim == 2
            assert Vcell.shape[1] == 3
            assert Vnuc.shape == Vcell.shape
            assert faces.shape[1] == 3

        for ang in angles:
            assert ang.shape == (3,)

    def test_empty_result_when_no_valid_positions(self):
        """Test that function returns empty when vessel blocks everything."""
        vol_params = VolumeParams(vol_sz=(20, 20, 10), N_neur=5, vres=2)

        # Create vessel mask that blocks everything
        grid_shape = tuple(int(s * vol_params.vres) for s in vol_params.vol_sz)
        vessel_mask = np.ones(grid_shape, dtype=np.uint8)

        neurons, angles, positions = sample_dense_neurons(
            vol_params=vol_params,
            vessel_mask=vessel_mask,
            verbose=0,
        )

        assert len(neurons) == 0
        assert len(angles) == 0
        assert positions.shape == (0, 3)

"""
Tests for neural volume voxelization module.

Tests the voxelization of neuron soma/nucleus meshes into 3D grids.
"""

import pytest
import numpy as np

from calcia.config.params import VolumeParams, NeuronParams
from calcia.volume.neurons import sample_dense_neurons
from calcia.volume.neural_volume import (
    generate_neural_volume,
    NeuralVolumeResult,
    NeuronVoxelData,
)


def _make_neurons(vol_sz=(50, 50, 30), n_neur=3, seed=42):
    """Helper to generate test neurons."""
    np.random.seed(seed)
    vol_params = VolumeParams(vol_sz=vol_sz, vres=2, N_neur=n_neur)
    neur_params = NeuronParams(n_samps=100)
    neurons, _, positions = sample_dense_neurons(
        vol_params, neur_params, verbose=0
    )
    return neurons, positions, vol_params, neur_params


class TestGenerateNeuralVolume:
    """Tests for generate_neural_volume function."""

    def test_output_types(self):
        """Result should have correct types and shapes."""
        neurons, positions, vol_params, neur_params = _make_neurons()

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )

        assert isinstance(result, NeuralVolumeResult)
        assert result.neur_soma.dtype == np.uint16
        assert result.neur_vol.dtype == np.float32
        expected_shape = (100, 100, 60)  # 50*2, 50*2, 30*2
        assert result.neur_soma.shape == expected_shape
        assert result.neur_vol.shape == expected_shape
        assert result.grid_shape == expected_shape
        assert result.voxel_resolution == 2

    def test_neuron_ids_correct(self):
        """neur_soma should contain correct 1-based neuron IDs."""
        neurons, positions, vol_params, neur_params = _make_neurons()

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )

        unique_ids = np.unique(result.neur_soma)
        assert 0 in unique_ids
        for i in range(1, len(neurons) + 1):
            assert i in unique_ids

    def test_no_overlap(self):
        """Each voxel should belong to at most one neuron (no soma overlap)."""
        neurons, positions, vol_params, neur_params = _make_neurons(
            vol_sz=(80, 80, 40), n_neur=10
        )

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )

        all_soma = []
        for indices in result.gp_soma:
            if len(indices) > 0:
                all_soma.append(indices)
        if all_soma:
            all_indices = np.concatenate(all_soma)
            assert len(all_indices) == len(np.unique(all_indices))

    def test_gp_nuc_structure(self):
        """gp_nuc should have correct structure."""
        neurons, positions, vol_params, _ = _make_neurons(n_neur=2)
        neur_params = NeuronParams(n_samps=100, nuc_fluorsc=0.3)

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )

        assert len(result.gp_nuc) == len(neurons)
        for indices, fluor in result.gp_nuc:
            assert indices.dtype == np.int32
            assert fluor == pytest.approx(0.3)

    def test_gp_soma_structure(self):
        """gp_soma should have correct structure."""
        neurons, positions, vol_params, neur_params = _make_neurons(n_neur=2)

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )

        assert len(result.gp_soma) == len(neurons)
        for indices in result.gp_soma:
            assert indices.dtype == np.int32
            assert len(indices) > 0

    def test_nucleus_fluorescence_in_neur_vol(self):
        """neur_vol should have fluorescence at nucleus locations."""
        neurons, positions, vol_params, _ = _make_neurons(n_neur=1)
        neur_params = NeuronParams(n_samps=100, nuc_fluorsc=0.5)

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )

        nonzero_vals = result.neur_vol[result.neur_vol > 0]
        assert len(nonzero_vals) > 0
        assert np.allclose(nonzero_vals, 0.5)

    def test_vessel_mask_exclusion(self):
        """Voxels occupied by vessels should not have soma."""
        neurons, positions, vol_params, neur_params = _make_neurons()
        grid_shape = (100, 100, 60)

        vessel_mask = np.zeros(grid_shape, dtype=np.uint8)
        vessel_mask[40:60, 40:60, 20:40] = 1

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params,
            vessel_mask=vessel_mask, verbose=0,
        )

        assert np.all(result.neur_soma[vessel_mask > 0] == 0)

    def test_vessel_mask_shape_mismatch_raises(self):
        """Should raise ValueError if vessel_mask shape is wrong."""
        neurons, positions, vol_params, neur_params = _make_neurons(n_neur=1)
        wrong_mask = np.zeros((10, 10, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="vessel_mask shape"):
            generate_neural_volume(
                neurons, positions, vol_params, neur_params,
                vessel_mask=wrong_mask, verbose=0,
            )

    def test_empty_neurons_list(self):
        """Should handle empty neurons list gracefully."""
        vol_params = VolumeParams(vol_sz=(50, 50, 30), vres=2)

        result = generate_neural_volume(
            [], np.zeros((0, 3)), vol_params, verbose=0
        )

        assert result.neur_soma.shape == (100, 100, 60)
        assert np.all(result.neur_soma == 0)
        assert np.all(result.neur_vol == 0)
        assert len(result.gp_nuc) == 0
        assert len(result.gp_soma) == 0

    def test_single_neuron_has_voxels(self):
        """A single centered neuron should produce nonzero voxels."""
        neurons, positions, vol_params, neur_params = _make_neurons(n_neur=1)

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )

        assert np.sum(result.neur_soma > 0) > 0
        assert len(result.gp_soma[0]) > 0

    def test_soma_excludes_nucleus(self):
        """gp_soma indices should not overlap with gp_nuc indices."""
        neurons, positions, vol_params, _ = _make_neurons(n_neur=1)
        neur_params = NeuronParams(n_samps=100, nuc_fluorsc=0.3)

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )

        soma_set = set(result.gp_soma[0].tolist())
        nuc_set = set(result.gp_nuc[0][0].tolist())
        assert len(soma_set & nuc_set) == 0

    def test_reproducibility(self):
        """Same inputs should produce identical results."""
        vol_params = VolumeParams(vol_sz=(50, 50, 30), vres=2, N_neur=3)
        neur_params = NeuronParams(n_samps=100)

        np.random.seed(42)
        n1, _, p1 = sample_dense_neurons(vol_params, neur_params, verbose=0)
        r1 = generate_neural_volume(n1, p1, vol_params, neur_params, verbose=0)

        np.random.seed(42)
        n2, _, p2 = sample_dense_neurons(vol_params, neur_params, verbose=0)
        r2 = generate_neural_volume(n2, p2, vol_params, neur_params, verbose=0)

        assert np.array_equal(r1.neur_soma, r2.neur_soma)
        assert np.array_equal(r1.neur_vol, r2.neur_vol)

    def test_neuron_data_list(self):
        """neuron_data should have one entry per neuron with correct IDs."""
        neurons, positions, vol_params, neur_params = _make_neurons()

        result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )

        assert len(result.neuron_data) == len(neurons)
        for i, nd in enumerate(result.neuron_data):
            assert nd.neuron_id == i + 1
            assert isinstance(nd, NeuronVoxelData)

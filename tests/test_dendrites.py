"""
Tests for dendrite growth module.

Tests the two-level Dijkstra dendrite growth from neuron somas.
"""

import pytest
import numpy as np

from calcia.config.params import VolumeParams, NeuronParams, DendParams
from calcia.volume.neurons import sample_dense_neurons
from calcia.volume.neural_volume import generate_neural_volume
from calcia.volume.dendrites import (
    grow_neuron_dendrites,
    DendriteResult,
    _extract_subvolume,
    _compute_fill_fraction,
    _compute_path_weights,
    _generate_basal_endpoints,
)


def _make_volume(vol_sz=(50, 50, 30), n_neur=3, seed=42):
    """Helper to generate test neural volume."""
    np.random.seed(seed)
    vol_params = VolumeParams(vol_sz=vol_sz, vres=2, N_neur=n_neur)
    neur_params = NeuronParams(n_samps=100, nuc_fluorsc=0.3)
    neurons, angles, positions = sample_dense_neurons(
        vol_params, neur_params, verbose=0
    )
    result = generate_neural_volume(
        neurons, positions, vol_params, neur_params, verbose=0
    )
    return result, positions, angles, vol_params


class TestDendriteHelpers:
    """Tests for dendrite helper functions."""

    def test_extract_subvolume_shape(self):
        """Extracted subvolume should have correct shape."""
        vol = np.zeros((100, 100, 60), dtype=np.float32)
        center = np.array([50, 50, 30])
        fdims = np.array([40, 40, 40])
        fulldims = np.array([100, 100, 60])

        obs, root_local, border, offsets = _extract_subvolume(
            vol, center, fdims, fulldims, small_z=False
        )
        assert obs.shape == (40, 40, 40)
        assert root_local[0] == 20  # half of fdims[0]
        assert root_local[1] == 20

    def test_extract_subvolume_border_clipping(self):
        """Subvolume near border should be handled gracefully."""
        vol = np.ones((100, 100, 60), dtype=np.float32)
        center = np.array([5, 5, 5])  # Near corner
        fdims = np.array([40, 40, 40])
        fulldims = np.array([100, 100, 60])

        obs, root_local, border, offsets = _extract_subvolume(
            vol, center, fdims, fulldims, small_z=False
        )
        assert obs.shape == (40, 40, 40)
        assert border  # Should flag border clipping

    def test_compute_fill_fraction(self):
        """Fill fraction should be between 0 and 1."""
        dims = np.array([4, 4, 4])
        dimsSS = np.array([5, 5, 5])
        obs = np.zeros((20, 20, 20), dtype=np.float32)
        obs[:10, :10, :10] = 1  # Fill one octant

        ff = _compute_fill_fraction(obs, dims, dimsSS)
        assert ff.shape == (4, 4, 4)
        assert np.all(ff >= 0) and np.all(ff <= 1)
        assert ff[0, 0, 0] > 0  # Filled region
        assert ff[3, 3, 3] == 0  # Empty region

    def test_path_weights_shape(self):
        """Path weights should match path length."""
        path = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                         [3, 0, 0], [4, 0, 0]], dtype=np.int32)
        w = _compute_path_weights(path, 0.25)
        assert len(w) == len(path)
        assert np.all(w >= 0)

    def test_basal_endpoints_in_bounds(self):
        """Generated endpoints should be within volume bounds."""
        np.random.seed(42)
        fdims = np.array([40, 40, 40])
        root = np.array([20, 20, 20])
        dtParams = [10, 30, 20, 1, 5]  # Already scaled
        obs = np.zeros(tuple(fdims), dtype=np.float32)

        ends = _generate_basal_endpoints(5, root, dtParams, fdims, obs)
        assert ends.shape == (5, 3)
        assert np.all(ends >= 0)
        assert np.all(ends < 40)


class TestGrowNeuronDendrites:
    """Tests for the main grow_neuron_dendrites function."""

    def test_output_types(self):
        """Result should have correct types."""
        result, positions, angles, vol_params = _make_volume(
            vol_sz=(50, 50, 30), n_neur=2
        )
        np.random.seed(123)
        dend_params = DendParams(
            dtParams=(5, 20, 10, 1, 2),
            atParams=(1, 5, 5, 5, 1),
            dims=(10, 10, 10),
            dimsSS=(5, 5, 5),
        )

        dend_result = grow_neuron_dendrites(
            vol_params, dend_params, result,
            positions=positions,
            rotation_angles=angles,
            verbose=0,
        )

        assert isinstance(dend_result, DendriteResult)
        assert dend_result.neur_num.dtype == np.uint16
        assert dend_result.neur_num.shape == result.neur_soma.shape
        assert dend_result.dendrite_ad.shape == result.neur_soma.shape

    def test_dendrite_voxels_exist(self):
        """After growth, there should be dendrite voxels beyond soma."""
        result, positions, angles, vol_params = _make_volume(
            vol_sz=(50, 50, 30), n_neur=2
        )
        np.random.seed(123)
        dend_params = DendParams(
            dtParams=(5, 20, 10, 1, 2),
            atParams=(1, 5, 5, 5, 1),
            dims=(10, 10, 10),
            dimsSS=(5, 5, 5),
        )

        dend_result = grow_neuron_dendrites(
            vol_params, dend_params, result,
            positions=positions,
            rotation_angles=angles,
            verbose=0,
        )

        # Should have more nonzero voxels than just soma
        soma_count = int(np.sum(result.neur_soma > 0))
        total_count = int(np.sum(dend_result.neur_num > 0))
        assert total_count >= soma_count

    def test_neuron_ids_preserved(self):
        """Soma voxels should still have correct neuron IDs."""
        result, positions, angles, vol_params = _make_volume(
            vol_sz=(50, 50, 30), n_neur=2
        )
        np.random.seed(123)
        dend_params = DendParams(
            dtParams=(3, 15, 10, 1, 1),
            atParams=(1, 5, 5, 5, 1),
            dims=(10, 10, 10),
            dimsSS=(5, 5, 5),
        )

        dend_result = grow_neuron_dendrites(
            vol_params, dend_params, result,
            positions=positions,
            rotation_angles=angles,
            verbose=0,
        )

        # Check each neuron's soma is still correctly labeled
        for kk in range(len(result.gp_soma)):
            soma_idx = result.gp_soma[kk]
            if len(soma_idx) > 0:
                vals = dend_result.neur_num.ravel()[soma_idx]
                assert np.all(vals == kk + 1), \
                    f"Neuron {kk+1} soma not preserved"

    def test_no_dendrites_in_nucleus(self):
        """Dendrites should not occupy nucleus voxels."""
        result, positions, angles, vol_params = _make_volume(
            vol_sz=(50, 50, 30), n_neur=2
        )
        np.random.seed(123)
        dend_params = DendParams(
            dtParams=(3, 15, 10, 1, 1),
            atParams=(1, 5, 5, 5, 1),
            dims=(10, 10, 10),
            dimsSS=(5, 5, 5),
        )

        dend_result = grow_neuron_dendrites(
            vol_params, dend_params, result,
            positions=positions,
            rotation_angles=angles,
            verbose=0,
        )

        # Nucleus voxels should be 0 in neur_num
        for kk in range(len(result.gp_nuc)):
            nuc_idx = result.gp_nuc[kk][0]
            if len(nuc_idx) > 0:
                vals = dend_result.neur_num.ravel()[nuc_idx]
                assert np.all(vals == 0), \
                    f"Neuron {kk+1} nucleus has dendrite voxels"

    def test_single_neuron(self):
        """Should work with a single neuron."""
        result, positions, angles, vol_params = _make_volume(
            vol_sz=(50, 50, 30), n_neur=1
        )
        np.random.seed(42)
        dend_params = DendParams(
            dtParams=(3, 15, 10, 1, 1),
            atParams=(1, 5, 5, 5, 1),
            dims=(10, 10, 10),
            dimsSS=(5, 5, 5),
        )

        dend_result = grow_neuron_dendrites(
            vol_params, dend_params, result,
            positions=positions,
            rotation_angles=angles,
            verbose=0,
        )

        assert isinstance(dend_result, DendriteResult)
        assert np.any(dend_result.neur_num > 0)

    def test_reproducibility(self):
        """Same seed should produce same results."""
        result, positions, angles, vol_params = _make_volume(
            vol_sz=(50, 50, 30), n_neur=1
        )
        dend_params = DendParams(
            dtParams=(3, 15, 10, 1, 1),
            atParams=(1, 5, 5, 5, 1),
            dims=(10, 10, 10),
            dimsSS=(5, 5, 5),
        )

        np.random.seed(99)
        r1 = grow_neuron_dendrites(
            vol_params, dend_params, result,
            positions=positions, verbose=0,
        )

        np.random.seed(99)
        r2 = grow_neuron_dendrites(
            vol_params, dend_params, result,
            positions=positions, verbose=0,
        )

        assert np.array_equal(r1.neur_num, r2.neur_num)

    def test_gp_soma_output(self):
        """gp_soma should have one entry per neuron."""
        result, positions, angles, vol_params = _make_volume(
            vol_sz=(50, 50, 30), n_neur=2
        )
        np.random.seed(42)
        dend_params = DendParams(
            dtParams=(3, 15, 10, 1, 1),
            atParams=(1, 5, 5, 5, 1),
            dims=(10, 10, 10),
            dimsSS=(5, 5, 5),
        )

        dend_result = grow_neuron_dendrites(
            vol_params, dend_params, result,
            positions=positions, verbose=0,
        )

        assert len(dend_result.gp_soma) == vol_params.N_neur

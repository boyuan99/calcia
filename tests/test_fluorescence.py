"""Tests for Step 6: Cell fluorescence distribution."""

import numpy as np
import pytest

from calcia.algorithms.gaussian_process import sample_3d_gp
from calcia.config.params import VolumeParams, NeuronParams, DendParams
from calcia.volume.neurons import sample_dense_neurons
from calcia.volume.neural_volume import generate_neural_volume
from calcia.volume.dendrites import grow_neuron_dendrites, grow_apical_dendrites
from calcia.volume.fluorescence import (
    set_cell_fluorescence,
    FluorescenceResult,
    CellFluorescenceData,
)


# ======================================================================
# Tests for sample_3d_gp (FFT-based 3D GP sampler)
# ======================================================================


class TestSample3dGp:
    """Tests for the FFT-based 3D Gaussian Process sampler."""

    def test_output_shape(self):
        np.random.seed(42)
        result = sample_3d_gp((10, 12, 8), l_scale=np.array([[2.0]]), p_scale=0.5)
        assert result.shape == (10, 12, 8)

    def test_output_dtype(self):
        np.random.seed(42)
        result = sample_3d_gp((8, 8, 8), l_scale=np.array([[2.0]]), p_scale=0.5)
        assert result.dtype == np.float32

    def test_output_real(self):
        np.random.seed(42)
        result = sample_3d_gp((10, 10, 10), l_scale=np.array([[2.0]]), p_scale=0.5)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_reproducible(self):
        np.random.seed(42)
        r1 = sample_3d_gp((10, 10, 10), l_scale=np.array([[2.0]]), p_scale=0.5)
        np.random.seed(42)
        r2 = sample_3d_gp((10, 10, 10), l_scale=np.array([[2.0]]), p_scale=0.5)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds(self):
        np.random.seed(1)
        r1 = sample_3d_gp((10, 10, 10), l_scale=np.array([[2.0]]), p_scale=0.5)
        np.random.seed(2)
        r2 = sample_3d_gp((10, 10, 10), l_scale=np.array([[2.0]]), p_scale=0.5)
        assert not np.allclose(r1, r2)

    def test_mean_offset(self):
        np.random.seed(42)
        mu = 5.0
        result = sample_3d_gp(
            (20, 20, 20), l_scale=np.array([[2.0]]), p_scale=0.2, mu=mu
        )
        # Mean should be approximately mu (GP has zero mean, then we add mu)
        assert abs(np.mean(result) - mu) < 1.0

    def test_scalar_l_scale(self):
        np.random.seed(42)
        # Single scalar as 2D array with 1 column → should expand to (1, 3)
        result = sample_3d_gp((8, 8, 8), l_scale=np.array([[3.0]]), p_scale=0.5)
        assert result.shape == (8, 8, 8)

    def test_multi_scale(self):
        np.random.seed(42)
        l_scale = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        result = sample_3d_gp((10, 10, 10), l_scale=l_scale, p_scale=0.5)
        assert result.shape == (10, 10, 10)
        assert not np.any(np.isnan(result))

    def test_bin_mask(self):
        np.random.seed(42)
        mask = np.zeros((8, 8, 8), dtype=np.float32)
        mask[2:6, 2:6, 2:6] = 1.0
        result = sample_3d_gp(
            (8, 8, 8), l_scale=np.array([[2.0]]), p_scale=0.5, bin_mask=mask
        )
        # Outside the mask, values should be zero (mu=0)
        assert np.all(result[0, 0, :] == 0.0)
        # Inside the mask, values should be non-zero
        assert np.any(result[3, 3, :] != 0.0)


# ======================================================================
# Helper: run Steps 1-5 to produce inputs for Step 6
# ======================================================================


def _make_step5_result():
    """Run Steps 1-5 with small parameters for testing."""
    np.random.seed(42)
    vol_params = VolumeParams(vol_sz=(60, 60, 30), vres=2, N_neur=5)
    neur_params = NeuronParams(n_samps=150, nuc_fluorsc=0.5)
    dend_params = DendParams(
        dtParams=(8, 25, 15, 1, 3),
        atParams=(2, 5, 5, 5, 1),
        dims=(12, 12, 12),
        dimsSS=(5, 5, 5),
    )

    neurons, angles, positions = sample_dense_neurons(
        vol_params, neur_params, verbose=0
    )
    vol_result = generate_neural_volume(
        neurons, positions, vol_params, neur_params, verbose=0
    )

    np.random.seed(42)
    dend_result = grow_neuron_dendrites(
        vol_params, dend_params, vol_result,
        positions=positions, rotation_angles=angles, verbose=0,
    )

    np.random.seed(42)
    apical_result = grow_apical_dendrites(
        vol_params, dend_result.dend_params,
        dend_result, vol_result, verbose=0,
    )

    return vol_params, neur_params, dend_result.dend_params, \
        apical_result, vol_result, positions


# ======================================================================
# Tests for set_cell_fluorescence
# ======================================================================


class TestSetCellFluorescence:
    """Tests for the main cell fluorescence function."""

    @pytest.fixture(scope="class")
    def step6_data(self):
        """Run Steps 1-5, then Step 6."""
        (vol_params, neur_params, dend_params,
         apical_result, vol_result, positions) = _make_step5_result()

        np.random.seed(123)
        fluor_result = set_cell_fluorescence(
            vol_params, neur_params, dend_params,
            neur_num=apical_result.neur_num,
            neur_soma=vol_result.neur_soma,
            neur_num_ad=apical_result.neur_num_ad,
            positions=positions,
            neur_vol=vol_result.neur_vol,
            verbose=0,
        )
        return {
            "vol_params": vol_params,
            "neur_params": neur_params,
            "dend_params": dend_params,
            "apical_result": apical_result,
            "vol_result": vol_result,
            "positions": positions,
            "fluor_result": fluor_result,
        }

    def test_output_type(self, step6_data):
        result = step6_data["fluor_result"]
        assert isinstance(result, FluorescenceResult)
        assert isinstance(result.neur_vol, np.ndarray)
        assert result.neur_vol.dtype == np.float32

    def test_gp_vals_length(self, step6_data):
        result = step6_data["fluor_result"]
        vol_params = step6_data["vol_params"]
        expected = vol_params.N_neur + vol_params.N_den
        assert len(result.gp_vals) == expected

    def test_gp_vals_element_types(self, step6_data):
        result = step6_data["fluor_result"]
        for g in result.gp_vals:
            assert isinstance(g, CellFluorescenceData)
            assert g.indices.dtype == np.int32
            assert g.fluorescence.dtype == np.float32
            assert g.soma_mask.dtype == bool
            # All arrays same length
            assert len(g.indices) == len(g.fluorescence) == len(g.soma_mask)

    def test_neuron_has_fluorescence(self, step6_data):
        result = step6_data["fluor_result"]
        vol_params = step6_data["vol_params"]
        for kk in range(vol_params.N_neur):
            g = result.gp_vals[kk]
            if len(g.indices) > 0:
                assert np.all(g.fluorescence > 0)

    def test_soma_fluorescence_range(self, step6_data):
        """Soma fluorescence should be in [0.5, 1.5] after normalization."""
        result = step6_data["fluor_result"]
        vol_params = step6_data["vol_params"]
        for kk in range(vol_params.N_neur):
            g = result.gp_vals[kk]
            if np.any(g.soma_mask):
                soma_fluor = g.fluorescence[g.soma_mask]
                assert np.min(soma_fluor) >= 0.5 - 1e-5
                assert np.max(soma_fluor) <= 1.5 + 1e-5

    def test_dendrite_fluorescence_decay(self, step6_data):
        """Dendrite fluorescence should generally decrease with distance."""
        result = step6_data["fluor_result"]
        vol_params = step6_data["vol_params"]
        positions = step6_data["positions"]
        grid_shape = step6_data["apical_result"].neur_num.shape
        neur_num_ad_flat = step6_data["apical_result"].neur_num_ad.ravel()
        vres = vol_params.vres

        for kk in range(vol_params.N_neur):
            g = result.gp_vals[kk]
            # Exclude both soma and apical dendrite voxels (which have
            # constant fluorescence = 1.0 and would break decay analysis)
            ad_flags = neur_num_ad_flat[g.indices] == (kk + 1)
            dend_mask = ~g.soma_mask & ~ad_flags
            if np.sum(dend_mask) < 10:
                continue

            dend_idx = g.indices[dend_mask]
            dend_fl = g.fluorescence[dend_mask]
            rx, ry, rz = np.unravel_index(dend_idx, grid_shape)
            center = vres * positions[kk]
            dist = np.sqrt(
                (rx - center[0]) ** 2
                + (ry - center[1]) ** 2
                + (rz - center[2]) ** 2
            )
            # Split into near and far halves
            median_dist = np.median(dist)
            near = dend_fl[dist <= median_dist]
            far = dend_fl[dist > median_dist]
            if len(near) > 0 and len(far) > 0:
                assert np.mean(near) >= np.mean(far) - 0.01

    def test_apical_dendrite_uniform(self, step6_data):
        """Through-volume apical dendrites should have fluorescence = 1.0."""
        result = step6_data["fluor_result"]
        vol_params = step6_data["vol_params"]
        for kk in range(vol_params.N_neur, len(result.gp_vals)):
            g = result.gp_vals[kk]
            if len(g.indices) > 0:
                np.testing.assert_array_almost_equal(g.fluorescence, 1.0)
                assert not np.any(g.soma_mask)

    def test_soma_mask_correct(self, step6_data):
        """Soma mask should match neur_soma."""
        result = step6_data["fluor_result"]
        vol_params = step6_data["vol_params"]
        neur_soma = step6_data["vol_result"].neur_soma.ravel()
        for kk in range(vol_params.N_neur):
            g = result.gp_vals[kk]
            if len(g.indices) == 0:
                continue
            expected_soma = neur_soma[g.indices] == (kk + 1)
            np.testing.assert_array_equal(g.soma_mask, expected_soma)

    def test_neur_vol_updated(self, step6_data):
        """neur_vol should have fluorescence at component voxel locations."""
        result = step6_data["fluor_result"]
        vol_flat = result.neur_vol.ravel()
        for g in result.gp_vals:
            if len(g.indices) > 0:
                vals = vol_flat[g.indices]
                np.testing.assert_array_almost_equal(vals, g.fluorescence)

    def test_neur_vol_not_mutated(self, step6_data):
        """Original neur_vol should not be modified."""
        vol_result = step6_data["vol_result"]
        # The original neur_vol should still have nucleus fluorescence
        # and zeros elsewhere (not modified by Step 6)
        assert vol_result.neur_vol is not step6_data["fluor_result"].neur_vol

    def test_reproducibility(self):
        (vol_params, neur_params, dend_params,
         apical_result, vol_result, positions) = _make_step5_result()

        np.random.seed(99)
        r1 = set_cell_fluorescence(
            vol_params, neur_params, dend_params,
            apical_result.neur_num, vol_result.neur_soma,
            apical_result.neur_num_ad, positions, vol_result.neur_vol,
            verbose=0,
        )

        np.random.seed(99)
        r2 = set_cell_fluorescence(
            vol_params, neur_params, dend_params,
            apical_result.neur_num, vol_result.neur_soma,
            apical_result.neur_num_ad, positions, vol_result.neur_vol,
            verbose=0,
        )

        np.testing.assert_array_equal(r1.neur_vol, r2.neur_vol)
        for g1, g2 in zip(r1.gp_vals, r2.gp_vals):
            np.testing.assert_array_equal(g1.indices, g2.indices)
            np.testing.assert_array_equal(g1.fluorescence, g2.fluorescence)

    def test_indices_cover_all_voxels(self, step6_data):
        """All non-zero voxels in neur_num should appear in gp_vals."""
        result = step6_data["fluor_result"]
        vol_params = step6_data["vol_params"]
        neur_num = step6_data["apical_result"].neur_num
        numcomps = vol_params.N_neur + vol_params.N_den

        all_indices = np.concatenate([g.indices for g in result.gp_vals])
        all_indices_set = set(all_indices.tolist())

        expected_flat = np.flatnonzero(
            (neur_num.ravel() >= 1) & (neur_num.ravel() <= numcomps)
        )
        expected_set = set(expected_flat.tolist())

        assert all_indices_set == expected_set

    def test_zero_through_volume_dendrites(self):
        """Should work with N_den=0."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(40, 40, 20), vres=2, N_neur=3, N_den=0)
        neur_params = NeuronParams(n_samps=100, nuc_fluorsc=0.5)
        dend_params = DendParams(
            dtParams=(5, 15, 10, 1, 3),
            atParams=(1, 3, 3, 3, 1),
            dims=(10, 10, 10),
            dimsSS=(4, 4, 4),
        )

        neurons, angles, positions = sample_dense_neurons(
            vol_params, neur_params, verbose=0
        )
        vol_result = generate_neural_volume(
            neurons, positions, vol_params, neur_params, verbose=0
        )
        np.random.seed(42)
        dend_result = grow_neuron_dendrites(
            vol_params, dend_params, vol_result,
            positions=positions, rotation_angles=angles, verbose=0,
        )
        np.random.seed(42)
        apical_result = grow_apical_dendrites(
            vol_params, dend_result.dend_params,
            dend_result, vol_result, verbose=0,
        )

        np.random.seed(42)
        result = set_cell_fluorescence(
            vol_params, neur_params, dend_result.dend_params,
            apical_result.neur_num, vol_result.neur_soma,
            apical_result.neur_num_ad, positions, vol_result.neur_vol,
            verbose=0,
        )
        assert len(result.gp_vals) == vol_params.N_neur
        assert isinstance(result, FluorescenceResult)

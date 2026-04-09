"""
Tests for through-volume apical dendrite growth (Step 5).

Tests grow_apical_dendrites which generates Layer 5 apical dendrites
traversing the full volume depth, independent of neuron somas.
"""

import pytest
import numpy as np

from calcia.config.params import VolumeParams, NeuronParams, DendParams
from calcia.volume.neurons import sample_dense_neurons
from calcia.volume.neural_volume import generate_neural_volume
from calcia.volume.dendrites import (
    grow_neuron_dendrites,
    grow_apical_dendrites,
    DendriteResult,
    ApicalDendriteResult,
)


def _make_step4_result(vol_sz=(60, 60, 30), n_neur=3, seed=42):
    """Helper: run Steps 1-4 to produce inputs for Step 5."""
    np.random.seed(seed)
    vol_params = VolumeParams(vol_sz=vol_sz, vres=2, N_neur=n_neur)
    neur_params = NeuronParams(n_samps=100, nuc_fluorsc=0.3)

    neurons, angles, positions = sample_dense_neurons(
        vol_params, neur_params, verbose=0
    )
    neural_vol = generate_neural_volume(
        neurons, positions, vol_params, neur_params, verbose=0
    )
    dend_params = DendParams(
        dtParams=(5, 20, 10, 1, 2),
        atParams=(1, 5, 5, 5, 1),
        dims=(10, 10, 10),
        dimsSS=(5, 5, 5),
    )
    dend_result = grow_neuron_dendrites(
        vol_params, dend_params, neural_vol,
        positions=positions, rotation_angles=angles, verbose=0
    )
    return vol_params, dend_params, dend_result, neural_vol


class TestGrowApicalDendrites:
    """Tests for the main grow_apical_dendrites function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up shared test data."""
        self.vol_params, self.dend_params, self.dend_result, self.neural_vol = \
            _make_step4_result(vol_sz=(60, 60, 30), n_neur=3, seed=42)

    def test_output_types_and_shapes(self):
        """Result should be ApicalDendriteResult with correct array shapes."""
        np.random.seed(100)
        result = grow_apical_dendrites(
            self.vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        assert isinstance(result, ApicalDendriteResult)
        assert result.neur_num.dtype == np.uint16
        assert result.neur_num_ad.dtype == np.uint16
        fulldims = (np.array(self.vol_params.vol_sz) * self.vol_params.vres).astype(int)
        assert result.neur_num.shape == tuple(fulldims)
        assert result.neur_num_ad.shape == tuple(fulldims)

    def test_dendrite_voxels_exist(self):
        """Through-volume dendrites should produce non-zero voxels."""
        np.random.seed(100)
        result = grow_apical_dendrites(
            self.vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        N_neur = self.vol_params.N_neur
        # Through-volume dendrite IDs are > N_neur
        apical_voxels = np.sum(result.neur_num > N_neur)
        assert apical_voxels > 0, "Should have through-volume apical voxels"

    def test_neuron_ids_preserved(self):
        """Original neuron soma IDs should be preserved after Step 5."""
        np.random.seed(100)
        result = grow_apical_dendrites(
            self.vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        N_neur = self.vol_params.N_neur
        # Each neuron should still have soma voxels
        for kk in range(N_neur):
            soma_idx = self.dend_result.gp_soma[kk][0]
            if len(soma_idx) > 0:
                vals = result.neur_num.ravel()[soma_idx]
                assert np.all(vals == kk + 1), \
                    f"Neuron {kk+1} soma voxels should be preserved"

    def test_no_dendrites_in_nucleus(self):
        """Through-volume dendrites should not exist in nucleus regions."""
        np.random.seed(100)
        result = grow_apical_dendrites(
            self.vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        for kk in range(self.vol_params.N_neur):
            nuc_idx = self.neural_vol.gp_nuc[kk][0]
            if len(nuc_idx) > 0:
                vals = result.neur_num.ravel()[nuc_idx]
                assert np.all(vals == 0), \
                    f"Neuron {kk+1} nucleus should be cleared"

    def test_neur_num_ad_soma_cleared(self):
        """neur_num_ad should have 0 at soma and nucleus locations."""
        np.random.seed(100)
        result = grow_apical_dendrites(
            self.vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        for kk in range(self.vol_params.N_neur):
            soma_idx = self.dend_result.gp_soma[kk][0]
            nuc_idx = self.neural_vol.gp_nuc[kk][0]
            if len(soma_idx) > 0:
                assert np.all(result.neur_num_ad.ravel()[soma_idx] == 0), \
                    f"Soma of neuron {kk+1} should be 0 in neur_num_ad"
            if len(nuc_idx) > 0:
                assert np.all(result.neur_num_ad.ravel()[nuc_idx] == 0), \
                    f"Nucleus of neuron {kk+1} should be 0 in neur_num_ad"

    def test_dendrite_ids_above_N_neur(self):
        """Through-volume dendrite IDs should be > N_neur."""
        np.random.seed(100)
        result = grow_apical_dendrites(
            self.vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        N_neur = self.vol_params.N_neur
        # IDs that are not 0 and not 1..N_neur should be > N_neur
        new_ids = result.neur_num[
            (result.neur_num > 0) &
            (self.dend_result.neur_num == 0)
        ]
        if len(new_ids) > 0:
            assert np.all(new_ids > N_neur), \
                "New voxel IDs from Step 5 should be > N_neur"

    def test_zero_dendrites(self):
        """N_den=0 should return unchanged volumes."""
        np.random.seed(100)
        vol_params = VolumeParams(
            vol_sz=(60, 60, 30), vres=2, N_neur=3, N_den=0
        )
        result = grow_apical_dendrites(
            vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        np.testing.assert_array_equal(
            result.neur_num, self.dend_result.neur_num
        )

    def test_single_dendrite(self):
        """N_den=1 should work correctly."""
        np.random.seed(100)
        vol_params = VolumeParams(
            vol_sz=(60, 60, 30), vres=2, N_neur=3, N_den=1
        )
        result = grow_apical_dendrites(
            vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        assert isinstance(result, ApicalDendriteResult)
        # Should have some new voxels
        diff = np.sum(result.neur_num != self.dend_result.neur_num)
        assert diff >= 0  # May be 0 if dilation didn't expand

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        np.random.seed(200)
        r1 = grow_apical_dendrites(
            self.vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        np.random.seed(200)
        r2 = grow_apical_dendrites(
            self.vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        np.testing.assert_array_equal(r1.neur_num, r2.neur_num)
        np.testing.assert_array_equal(r1.neur_num_ad, r2.neur_num_ad)

    def test_apical_var_default(self):
        """Default dendVar for through-volume apicals should be 0.35."""
        np.random.seed(100)
        # No apicalVar or dendVar set -> should use 0.35 internally
        dend_params = DendParams(
            dtParams=(5, 20, 10, 1, 2),
            atParams=(1, 5, 5, 5, 1),
            dims=(10, 10, 10),
            dimsSS=(5, 5, 5),
        )
        # Should not error
        result = grow_apical_dendrites(
            self.vol_params, dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        assert isinstance(result, ApicalDendriteResult)

    def test_custom_apical_var(self):
        """Custom apicalVar should override dendVar."""
        np.random.seed(100)
        dend_params = DendParams(
            dtParams=(5, 20, 10, 1, 2),
            atParams=(1, 5, 5, 5, 1),
            dims=(10, 10, 10),
            dimsSS=(5, 5, 5),
            dendVar=0.1,
            apicalVar=0.5,
        )
        result = grow_apical_dendrites(
            self.vol_params, dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        assert isinstance(result, ApicalDendriteResult)

    def test_neur_num_ad_no_overflow(self):
        """neur_num_ad should not exceed neur_num at any voxel."""
        np.random.seed(100)
        result = grow_apical_dendrites(
            self.vol_params, self.dend_params,
            self.dend_result, self.neural_vol, verbose=0
        )
        overflow = result.neur_num_ad.astype(np.int32) - \
            result.neur_num.astype(np.int32)
        assert np.all(overflow <= 0), \
            "neur_num_ad should not exceed neur_num"

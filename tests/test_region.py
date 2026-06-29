"""
Tests for the region preset system (cortex vs striatum).

Covers:
* :func:`calcia.config.params.apply_region_defaults` value mapping.
* ``VolumeParams.region`` field and the striatum ``N_den=0`` branch.
* Backward compatibility: default region leaves params untouched.
* A tiny end-to-end striatum Phase-1 run: no apical dendrites and a
  vessel network with no pial surface sheet.
"""

import numpy as np
import pytest

from calcia.config.params import (
    DendParams,
    NeuronParams,
    VascParams,
    VolumeParams,
    apply_region_defaults,
)


class TestRegionField:
    def test_default_is_cortex(self):
        assert VolumeParams().region == "cortex"

    def test_cortex_derives_N_den_from_density(self):
        vp = VolumeParams(vol_sz=(100, 100, 50))
        assert vp.N_den > 0  # cortex still grows apical dendrites

    def test_striatum_forces_N_den_zero(self):
        vp = VolumeParams(vol_sz=(100, 100, 50), region="striatum")
        assert vp.N_den == 0

    def test_explicit_N_den_respected_even_for_striatum(self):
        vp = VolumeParams(region="striatum", N_den=7)
        assert vp.N_den == 7


class TestApplyRegionDefaults:
    def test_cortex_is_noop(self):
        vp = VolumeParams()
        npar, vpar, dpar = NeuronParams(), VascParams(), DendParams()
        before = (npar.neur_type, npar.avg_rad, vpar.depth_surf, dpar.atParams)
        apply_region_defaults(vp, npar, vpar, dpar)
        after = (npar.neur_type, npar.avg_rad, vpar.depth_surf, dpar.atParams)
        assert before == after

    def test_striatum_sets_msn_params(self):
        vp = VolumeParams(region="striatum")
        npar, vpar, dpar = NeuronParams(), VascParams(), DendParams()
        apply_region_defaults(vp, npar, vpar, dpar)
        # Neuron: MSN
        assert npar.neur_type == "spherical"
        assert npar.avg_rad == 8.0
        # Vasculature: no surface, more terminal
        assert vpar.depth_surf == 0.0
        assert vpar.distsc == 6.0
        # Dendrites: zero soma-attached apical, 5-6 isotropic basal
        assert dpar.atParams[0] == 0.0
        assert dpar.dtParams[0] == 5.0
        assert dpar.dtParams[1] == dpar.dtParams[2]  # isotropic field

    def test_unknown_region_raises(self):
        vp = VolumeParams()
        vp.region = "cerebellum"
        with pytest.raises(ValueError):
            apply_region_defaults(vp, NeuronParams(), VascParams(), DendParams())

    def test_old_pickle_without_region_treated_as_cortex(self):
        class _Legacy:  # mimics an unpickled VolumeParams lacking `region`
            pass
        npar, vpar, dpar = NeuronParams(), VascParams(), DendParams()
        # Should not raise and should be a no-op.
        apply_region_defaults(_Legacy(), npar, vpar, dpar)
        assert npar.neur_type == "pyramidal"


class TestStriatumEndToEnd:
    def test_tiny_striatum_volume(self):
        from calcia import simulate_neural_volume

        vol_params = VolumeParams(
            vol_sz=(60, 60, 40), vres=1, vol_depth=0, region="striatum",
        )
        out = simulate_neural_volume(vol_params=vol_params, seed=0, verbose=0)

        # No apical dendrites anywhere.
        assert out.neur_num_ad.max() == 0

        # Vasculature exists but has no pial surface sheet: the top z-slice
        # should not be saturated with vessels (cortex would put a dense
        # surface net there).
        assert out.neur_ves is not None
        ves = out.neur_ves
        assert ves.sum() > 0
        top_frac = ves[:, :, 0].mean()
        assert top_frac < 0.5

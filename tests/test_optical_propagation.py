"""
Tests for Phase 2 optical propagation simulation.

Covers:
  - PsfParams and TpmParams dataclasses
  - gaussian_psf_na
  - gaussian_beam_size
  - tpm_signal_scale
  - compute_collection_mask
  - compute_illumination_mask
  - simulate_optical_propagation (integration)
"""

import math

import numpy as np
import pytest

from calcia.config.params import PsfParams, TpmParams, VolumeParams
from calcia.optics import (
    OpticalPropagationResult,
    PsfTail,
    compute_collection_mask,
    compute_illumination_mask,
    gaussian_beam_size,
    gaussian_psf_na,
    simulate_optical_propagation,
)
from calcia.optics.signal import tpm_signal_scale


# ---------------------------------------------------------------------------
# PsfParams
# ---------------------------------------------------------------------------

class TestPsfParams:
    def test_defaults(self):
        p = PsfParams()
        assert p.na == pytest.approx(0.6)
        assert p.obj_na == pytest.approx(0.8)
        assert p.n == pytest.approx(1.35)
        assert p.n_diff == pytest.approx(0.02)
        assert p.lambda_um == pytest.approx(0.92)
        assert p.obj_fl == pytest.approx(4.5)
        assert p.ss == 2
        assert p.sampling == pytest.approx(50.0)
        assert p.psf_sz == (20.0, 20.0, 50.0)
        assert p.prop_sz == pytest.approx(10.0)
        assert p.blur == pytest.approx(3.0)
        assert p.tail_length == pytest.approx(50.0)
        assert p.psf_type == "gaussian"
        assert p.scaling == "two-photon"
        assert p.prop_crop is True
        assert p.fast_mask is True
        assert p.fm_sampling == pytest.approx(10.0)
        assert p.fm_fine_samp == 2
        assert p.fm_ss == 1

    def test_lambda_keyword_safe(self):
        # Must not raise — lambda_um is not a Python keyword
        p = PsfParams(lambda_um=1.04)
        assert p.lambda_um == pytest.approx(1.04)

    def test_hemo_abs_value(self):
        p = PsfParams()
        expected = 0.00674 * math.log(10)
        assert p.hemo_abs == pytest.approx(expected, rel=1e-6)

    def test_scatter_sz_length(self):
        p = PsfParams()
        assert len(p.scatter_sz) == 4
        assert len(p.scatter_wt) == 4

    def test_zernike_wt_length(self):
        p = PsfParams()
        assert len(p.zernike_wt) == 11

    def test_custom_na(self):
        p = PsfParams(na=0.45, obj_na=0.9)
        assert p.na == pytest.approx(0.45)
        assert p.obj_na == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TpmParams
# ---------------------------------------------------------------------------

class TestTpmParams:
    def test_defaults(self):
        p = TpmParams()
        assert p.nidx == pytest.approx(1.33)
        assert p.nac == pytest.approx(0.8)
        assert p.eta == pytest.approx(0.6)
        assert p.conc == pytest.approx(10.0)
        assert p.delta == pytest.approx(2.0)
        assert p.gp == pytest.approx(0.588)
        assert p.f == pytest.approx(80.0)
        assert p.tau == pytest.approx(150.0)
        assert p.pavg == pytest.approx(40.0)
        assert p.lambda_um == pytest.approx(0.92)

    def test_phi_computed(self):
        p = TpmParams()
        assert p.phi is not None
        # 0.8 * ((1 - sqrt(1-(0.8/1.33)^2))/2) * 0.4
        sa = (1.0 - math.sqrt(1.0 - (0.8 / 1.33) ** 2)) / 2.0
        expected = 0.8 * sa * 0.4
        assert p.phi == pytest.approx(expected, rel=1e-6)

    def test_phi_override(self):
        p = TpmParams(phi=0.123)
        assert p.phi == pytest.approx(0.123)

    def test_lambda_keyword_safe(self):
        p = TpmParams(lambda_um=1.04)
        assert p.lambda_um == pytest.approx(1.04)


# ---------------------------------------------------------------------------
# gaussian_psf_na
# ---------------------------------------------------------------------------

class TestGaussianPsfNa:
    def _make_psf(self, Nx=21, Ny=21, Nz=41, na=0.6):
        return gaussian_psf_na(
            na=na,
            lambda_um=0.92,
            sampling=(0.5, 0.5, 0.5),
            mat_size=(Nx, Ny, Nz),
            nidx=1.33,
        )

    def test_output_shape(self):
        psf, x, y, z = self._make_psf(21, 21, 41)
        assert psf.shape == (21, 21, 41)
        assert x.shape == (21,)
        assert y.shape == (21,)
        assert z.shape == (41,)

    def test_dtype_float32(self):
        psf, *_ = self._make_psf()
        assert psf.dtype == np.float32

    def test_peak_at_center(self):
        Nx, Ny, Nz = 21, 21, 41
        psf, *_ = self._make_psf(Nx, Ny, Nz)
        idx = np.unravel_index(np.argmax(psf), psf.shape)
        assert idx[0] == Nx // 2
        assert idx[1] == Ny // 2
        assert idx[2] == Nz // 2

    def test_values_nonnegative(self):
        psf, *_ = self._make_psf()
        assert float(psf.min()) >= 0.0

    def test_radial_symmetry_xy(self):
        psf, *_ = self._make_psf(21, 21, 41)
        cx, cy, cz = 10, 10, 20
        # PSF should be symmetric in XY around the centre
        for d in range(1, 5):
            np.testing.assert_allclose(
                psf[cx + d, cy, cz], psf[cx - d, cy, cz], rtol=1e-5
            )
            np.testing.assert_allclose(
                psf[cx, cy + d, cz], psf[cx, cy - d, cz], rtol=1e-5
            )

    def test_theta_tilt_shifts_peak(self):
        Nx, Ny, Nz = 21, 21, 51
        psf_notilt, *_ = gaussian_psf_na(
            0.6, 0.92, (0.5, 0.5, 0.5), (Nx, Ny, Nz), theta=0.0, nidx=1.33
        )
        psf_tilt, *_ = gaussian_psf_na(
            0.6, 0.92, (0.5, 0.5, 0.5), (Nx, Ny, Nz), theta=15.0, nidx=1.33
        )
        # With tilt, PSF at off-axis z should differ
        assert not np.allclose(psf_notilt, psf_tilt)

    def test_zero_na_raises(self):
        with pytest.raises(ValueError, match="na must be positive"):
            gaussian_psf_na(0.0, 0.92, (0.5, 0.5, 0.5), (21, 21, 41))

    def test_na_ge_nidx_raises(self):
        with pytest.raises(ValueError, match="must be less than nidx"):
            gaussian_psf_na(1.5, 0.92, (0.5, 0.5, 0.5), (21, 21, 41), nidx=1.33)

    def test_two_photon_squaring(self):
        # Two-photon PSF decays faster than linear intensity
        psf, x, y, z = self._make_psf(1, 1, 41)
        # At the centre voxel, intensity = 1 → psf = 1
        cz = 20
        assert psf[0, 0, cz] == pytest.approx(1.0, abs=1e-5)
        # Away from centre, psf < intensity (since intensity < 1, intensity^2 < intensity)
        assert psf[0, 0, cz + 5] < 1.0


# ---------------------------------------------------------------------------
# gaussian_beam_size
# ---------------------------------------------------------------------------

class TestGaussianBeamSize:
    def test_returns_3_element_array(self):
        p = PsfParams()
        result = gaussian_beam_size(p, dist=100.0)
        assert result.shape == (3,)

    def test_z_component_zero(self):
        p = PsfParams()
        result = gaussian_beam_size(p, dist=100.0)
        assert result[2] == 0.0

    def test_lateral_grows_with_dist(self):
        p = PsfParams()
        r1 = gaussian_beam_size(p, dist=50.0)
        r2 = gaussian_beam_size(p, dist=100.0)
        assert r2[0] >= r1[0]

    def test_apod_scaling(self):
        p = PsfParams()
        r1 = gaussian_beam_size(p, dist=100.0, apod=1.0)
        r2 = gaussian_beam_size(p, dist=100.0, apod=2.0)
        np.testing.assert_allclose(r2[:2], 2.0 * r1[:2], rtol=1e-6)

    def test_default_apod_is_2(self):
        p = PsfParams()
        r_default = gaussian_beam_size(p, dist=100.0)
        r_explicit = gaussian_beam_size(p, dist=100.0, apod=2.0)
        np.testing.assert_array_equal(r_default, r_explicit)

    def test_zero_dist_returns_zero_lateral(self):
        p = PsfParams()
        result = gaussian_beam_size(p, dist=0.0)
        # tan(...) * 0 * 1.5 → 0, ceil(0)*apod = 0
        assert result[0] == 0.0
        assert result[1] == 0.0


# ---------------------------------------------------------------------------
# tpm_signal_scale
# ---------------------------------------------------------------------------

class TestTpmSignalScale:
    def test_returns_positive_float(self):
        result = tpm_signal_scale(TpmParams())
        assert isinstance(result, float)
        assert result > 0.0

    def test_pavg_quadratic(self):
        p1 = TpmParams(pavg=40.0)
        p2 = TpmParams(pavg=80.0)
        r1 = tpm_signal_scale(p1)
        r2 = tpm_signal_scale(p2)
        assert r2 == pytest.approx(4.0 * r1, rel=1e-5)

    def test_conc_linear(self):
        p1 = TpmParams(conc=10.0)
        p2 = TpmParams(conc=20.0)
        r1 = tpm_signal_scale(p1)
        r2 = tpm_signal_scale(p2)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-5)

    def test_psf_params_override_changes_result(self):
        tpm = TpmParams()
        psf = PsfParams(n=1.40, obj_na=0.75, lambda_um=1.0)
        r_no_override = tpm_signal_scale(tpm)
        r_override = tpm_signal_scale(tpm, psf_params=psf)
        assert r_no_override != pytest.approx(r_override)

    def test_order_of_magnitude(self):
        # For typical settings, Ftavg should be in a physically plausible range
        result = tpm_signal_scale(TpmParams())
        assert result > 1e3


# ---------------------------------------------------------------------------
# compute_collection_mask
# ---------------------------------------------------------------------------

class TestComputeCollectionMask:
    def _vol(self):
        return VolumeParams(vol_sz=(40, 40, 20), vres=2, vol_depth=100, verbose=0)

    def test_no_vessels_returns_ones(self):
        mask = compute_collection_mask(self._vol(), PsfParams(), vessel_volume=None)
        np.testing.assert_array_equal(mask, np.ones_like(mask))

    def test_output_shape(self):
        v = self._vol()
        mask = compute_collection_mask(v, PsfParams(), vessel_volume=None)
        expected = (int(v.vol_sz[0] * v.vres), int(v.vol_sz[1] * v.vres))
        assert mask.shape == expected

    def test_dtype_float32(self):
        mask = compute_collection_mask(self._vol(), PsfParams(), vessel_volume=None)
        assert mask.dtype == np.float32

    def test_vessel_reduces_mask(self):
        v = self._vol()
        p = PsfParams()
        Nx = int(v.vol_sz[0] * v.vres)
        Ny = int(v.vol_sz[1] * v.vres)
        Nz = int((v.vol_depth + v.vol_sz[2]) * v.vres)
        # Vessel fills first quarter of Z (above focal plane)
        vessels = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        vessels[:, :, :Nz // 4] = 1.0
        mask = compute_collection_mask(v, p, vessel_volume=vessels)
        # With vessels, at least some values should be < 1
        assert float(mask.min()) < 1.0

    def test_values_in_zero_one(self):
        v = self._vol()
        Nx = int(v.vol_sz[0] * v.vres)
        Ny = int(v.vol_sz[1] * v.vres)
        Nz = int((v.vol_depth + v.vol_sz[2]) * v.vres)
        vessels = np.ones((Nx, Ny, Nz), dtype=np.float32)
        mask = compute_collection_mask(v, PsfParams(), vessel_volume=vessels)
        assert float(mask.min()) > 0.0
        assert float(mask.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# compute_illumination_mask
# ---------------------------------------------------------------------------

class TestComputeIlluminationMask:
    def _vol(self):
        return VolumeParams(vol_sz=(40, 40, 20), vres=2, vol_depth=100, verbose=0)

    def test_no_vessels_returns_ones(self):
        mask = compute_illumination_mask(self._vol(), PsfParams(), vessel_volume=None)
        np.testing.assert_array_equal(mask, np.ones_like(mask))

    def test_output_shape(self):
        v = self._vol()
        mask = compute_illumination_mask(v, PsfParams(), vessel_volume=None)
        expected = (int(v.vol_sz[0] * v.vres), int(v.vol_sz[1] * v.vres))
        assert mask.shape == expected

    def test_dtype_float32(self):
        mask = compute_illumination_mask(self._vol(), PsfParams(), vessel_volume=None)
        assert mask.dtype == np.float32

    def test_normalized_mean_one(self):
        v = self._vol()
        Nx = int(v.vol_sz[0] * v.vres)
        Ny = int(v.vol_sz[1] * v.vres)
        Nz = int((v.vol_depth + v.vol_sz[2]) * v.vres)
        vessels = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        vessels[:Nx // 2, :, :] = 1.0
        mask = compute_illumination_mask(v, PsfParams(), vessel_volume=vessels)
        assert float(mask.mean()) == pytest.approx(1.0, abs=1e-5)

    def test_vessel_reduces_mask_locally(self):
        v = self._vol()
        Nx = int(v.vol_sz[0] * v.vres)
        Ny = int(v.vol_sz[1] * v.vres)
        Nz = int((v.vol_depth + v.vol_sz[2]) * v.vres)
        # Block left half only
        vessels = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        vessels[:Nx // 2, :, :int(v.vol_depth * v.vres)] = 1.0
        mask = compute_illumination_mask(v, PsfParams(), vessel_volume=vessels)
        left_mean = float(mask[:Nx // 2, :].mean())
        right_mean = float(mask[Nx // 2:, :].mean())
        assert left_mean < right_mean


# ---------------------------------------------------------------------------
# simulate_optical_propagation  (integration)
# ---------------------------------------------------------------------------

class TestSimulateOpticalPropagation:
    def _vol_params(self):
        return VolumeParams(vol_sz=(40, 40, 20), vres=2, vol_depth=100, verbose=0)

    def _psf_params(self):
        return PsfParams(psf_sz=(10.0, 10.0, 20.0), psf_type="gaussian_analytical")

    def test_returns_result_type(self):
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert isinstance(result, OpticalPropagationResult)

    def test_psf_shape(self):
        v = self._vol_params()
        p = self._psf_params()
        result = simulate_optical_propagation(v, p)
        expected = tuple(int(round(s * v.vres)) for s in p.psf_sz)
        assert result.psf.shape == expected

    def test_mask_shape(self):
        v = self._vol_params()
        result = simulate_optical_propagation(v, self._psf_params())
        expected = (int(v.vol_sz[0] * v.vres), int(v.vol_sz[1] * v.vres))
        assert result.mask.shape == expected

    def test_col_mask_shape(self):
        v = self._vol_params()
        result = simulate_optical_propagation(v, self._psf_params())
        expected = (int(v.vol_sz[0] * v.vres), int(v.vol_sz[1] * v.vres))
        assert result.col_mask.shape == expected

    def test_psf_nonnegative(self):
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert float(result.psf.min()) >= 0.0

    def test_mask_mean_one(self):
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert float(result.mask.mean()) == pytest.approx(1.0, abs=1e-5)

    def test_col_mask_range(self):
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert float(result.col_mask.min()) > 0.0
        assert float(result.col_mask.max()) <= 1.0 + 1e-6

    def test_psf_top_bot_types(self):
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert isinstance(result.psf_top, PsfTail)
        assert isinstance(result.psf_bot, PsfTail)

    def test_no_vol_out(self):
        # Should work without providing vol_out (vessel-free case)
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert result is not None

    def test_params_stored(self):
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert "vol_params" in result.params
        assert "psf_params" in result.params

    def test_verbose_silent(self, capsys):
        simulate_optical_propagation(
            self._vol_params(), self._psf_params(), verbose=0
        )
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_default_psf_params(self):
        # Should use PsfParams() defaults without raising (use analytical for speed)
        result = simulate_optical_propagation(
            self._vol_params(),
            PsfParams(psf_type="gaussian_analytical"),
        )
        assert isinstance(result, OpticalPropagationResult)

    def test_psf_dtype_float32(self):
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert result.psf.dtype == np.float32

    def test_mask_dtype_float32(self):
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert result.mask.dtype == np.float32

    def test_col_mask_dtype_float32(self):
        result = simulate_optical_propagation(self._vol_params(), self._psf_params())
        assert result.col_mask.dtype == np.float32

    def test_reproducible(self):
        v = self._vol_params()
        p = self._psf_params()
        r1 = simulate_optical_propagation(v, p)
        r2 = simulate_optical_propagation(v, p)
        np.testing.assert_array_equal(r1.psf, r2.psf)
        np.testing.assert_array_equal(r1.mask, r2.mask)

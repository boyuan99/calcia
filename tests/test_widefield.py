"""
Tests for the widefield (single-photon) imaging path.

Covers:
* :func:`calcia.optics.gaussian_psf_na` scaling branch (I vs I^2).
* :func:`calcia.optics.widefield_signal_scale` basic behaviour.
* :func:`calcia.scanning.camera_noise` noise statistics.
* :func:`calcia.optics.simulate_optical_propagation` widefield dispatch.
* :func:`calcia.scanning.scan_widefield` end-to-end smoke test.
"""

import numpy as np
import pytest

from calcia.config.params import (
    CameraNoiseParams,
    PsfParams,
    ScanParams,
    SpikeParams,
    VolumeParams,
    WidefieldParams,
)
from calcia.optics.psf import gaussian_psf_na
from calcia.optics.signal import widefield_signal_scale
from calcia.scanning.noise import camera_noise


# ======================================================================
# Parameter dataclass tests
# ======================================================================

class TestWidefieldParams:
    def test_defaults(self):
        p = WidefieldParams()
        assert p.phi == pytest.approx(0.6)
        assert p.sigma_abs == pytest.approx(3.8e-16)
        assert p.lambda_ex_um == pytest.approx(0.488)
        # omega auto-computed from NA/n
        assert p.omega is not None
        assert p.omega > 0.0

    def test_omega_override(self):
        p = WidefieldParams(omega=0.1)
        assert p.omega == pytest.approx(0.1)


class TestCameraNoiseParams:
    def test_defaults(self):
        p = CameraNoiseParams()
        assert p.qe == pytest.approx(0.8)
        assert p.dark_rate == pytest.approx(0.3)
        assert p.read_noise == pytest.approx(1.6)
        assert p.bit_depth == 16


# ======================================================================
# PSF scaling branch
# ======================================================================

class TestGaussianPsfScaling:
    def test_widefield_is_linear_intensity(self):
        """widefield PSF == intensity; two-photon PSF == intensity^2."""
        sampling = (0.5, 0.5, 0.5)
        mat_size = (12, 12, 8)
        psf_tp, _, _, _ = gaussian_psf_na(
            na=0.6, lambda_um=0.92, sampling=sampling, mat_size=mat_size,
            nidx=1.33, scaling="two-photon",
        )
        psf_wf, _, _, _ = gaussian_psf_na(
            na=0.6, lambda_um=0.52, sampling=sampling, mat_size=mat_size,
            nidx=1.33, scaling="widefield",
        )
        # Centre peak == 1 for both (intensity^k at origin = 1 with our normalization)
        assert psf_tp[6, 6, 4] == pytest.approx(1.0, abs=1e-5)
        assert psf_wf[6, 6, 4] == pytest.approx(1.0, abs=1e-5)

    def test_one_photon_alias(self):
        sampling = (0.5, 0.5, 0.5)
        mat_size = (8, 8, 4)
        psf_a, _, _, _ = gaussian_psf_na(
            na=0.6, lambda_um=0.52, sampling=sampling, mat_size=mat_size,
            scaling="widefield",
        )
        psf_b, _, _, _ = gaussian_psf_na(
            na=0.6, lambda_um=0.52, sampling=sampling, mat_size=mat_size,
            scaling="one-photon",
        )
        np.testing.assert_array_equal(psf_a, psf_b)

    def test_two_photon_squares_intensity(self):
        """TPM PSF pointwise equals widefield PSF squared."""
        sampling = (0.5, 0.5, 0.5)
        mat_size = (10, 10, 6)
        psf_tp, _, _, _ = gaussian_psf_na(
            na=0.6, lambda_um=0.92, sampling=sampling, mat_size=mat_size,
            scaling="two-photon",
        )
        psf_wf, _, _, _ = gaussian_psf_na(
            na=0.6, lambda_um=0.92, sampling=sampling, mat_size=mat_size,
            scaling="widefield",
        )
        np.testing.assert_allclose(psf_tp, psf_wf ** 2, atol=1e-6)

    def test_invalid_scaling_raises(self):
        with pytest.raises(ValueError, match="Unknown scaling"):
            gaussian_psf_na(
                na=0.6, lambda_um=0.52, sampling=(0.5,) * 3,
                mat_size=(4, 4, 4), scaling="bogus",
            )


# ======================================================================
# Widefield signal scale
# ======================================================================

class TestWidefieldSignalScale:
    def test_positive(self):
        wf = WidefieldParams()
        val = widefield_signal_scale(wf)
        assert val > 0.0

    def test_linear_in_power(self):
        """Signal scales linearly with excitation power (P, not P^2)."""
        wf1 = WidefieldParams(pavg=1.0)
        wf2 = WidefieldParams(pavg=3.0)
        r1 = widefield_signal_scale(wf1)
        r2 = widefield_signal_scale(wf2)
        assert r2 / r1 == pytest.approx(3.0, rel=1e-6)

    def test_linear_in_conc(self):
        wf1 = WidefieldParams(conc=5.0)
        wf2 = WidefieldParams(conc=10.0)
        r1 = widefield_signal_scale(wf1)
        r2 = widefield_signal_scale(wf2)
        assert r2 / r1 == pytest.approx(2.0, rel=1e-6)


# ======================================================================
# Camera noise
# ======================================================================

class TestCameraNoise:
    def test_zero_signal_zero_noise(self):
        """No signal, no dark, no read -> output all zero."""
        rng = np.random.default_rng(0)
        cam = CameraNoiseParams(
            qe=1.0, dark_rate=0.0, t_exp=0.0,
            read_noise=0.0, pixel_gain_sigma=0.0, gain_e_per_adu=1.0,
        )
        img = np.zeros((16, 16), dtype=np.float32)
        out = camera_noise(img, cam, rng)
        assert out.shape == img.shape
        np.testing.assert_array_equal(out, 0.0)

    def test_read_noise_only(self):
        """With only read noise (on top of a baseline), output std ~ read_noise."""
        rng = np.random.default_rng(123)
        cam = CameraNoiseParams(
            qe=1.0, dark_rate=0.0, t_exp=0.0,
            read_noise=5.0, pixel_gain_sigma=0.0, gain_e_per_adu=1.0,
        )
        # Use a constant baseline well above zero so the read-noise Gaussian
        # does not get clipped. With qe=1 and Poisson(100), mean variance
        # from shot noise is 100; subtract that off to isolate read noise.
        img = np.full((128, 128), 100.0, dtype=np.float32)
        out = camera_noise(img, cam, rng)
        total_var = out.var()
        read_var = max(total_var - 100.0, 0.0)
        assert np.sqrt(read_var) == pytest.approx(5.0, rel=0.25)

    def test_shot_noise_variance(self):
        """Poisson variance of signal electrons ~= mean for large signals."""
        rng = np.random.default_rng(7)
        cam = CameraNoiseParams(
            qe=1.0, dark_rate=0.0, t_exp=0.0,
            read_noise=0.0, pixel_gain_sigma=0.0, gain_e_per_adu=1.0,
        )
        lam = 400.0
        img = np.full((128, 128), lam, dtype=np.float32)
        out = camera_noise(img, cam, rng)
        assert out.mean() == pytest.approx(lam, rel=0.05)
        assert out.var() == pytest.approx(lam, rel=0.2)

    def test_clip_to_bit_depth(self):
        rng = np.random.default_rng(1)
        cam = CameraNoiseParams(
            qe=1.0, dark_rate=0.0, t_exp=0.0,
            read_noise=0.0, pixel_gain_sigma=0.0, gain_e_per_adu=1.0,
            bit_depth=8,   # max = 255
        )
        img = np.full((8, 8), 1e5, dtype=np.float32)
        out = camera_noise(img, cam, rng)
        assert out.max() <= 255.0
        assert out.min() >= 0.0


# ======================================================================
# Phase 2 widefield propagation (no vessels, analytical)
# ======================================================================

class TestWidefieldOpticalPropagation:
    def test_dispatch_and_shape(self):
        from calcia.optics import simulate_optical_propagation

        vol_params = VolumeParams(
            vol_sz=(30, 30, 20), vres=1, min_dist=10.0, verbose=0,
        )
        psf_params = PsfParams(
            imaging_mode="widefield",
            psf_type="gaussian_analytical",
            psf_sz=(12, 12, 20),
            na=0.6, obj_na=0.8, n=1.35,
            lambda_em_um=0.52,
        )

        opt = simulate_optical_propagation(vol_params, psf_params)

        # PSF z extent = full volume z depth
        assert opt.psf.shape[2] == 20
        # Illumination mask is uniform
        assert opt.mask.shape == (30, 30)
        np.testing.assert_allclose(opt.mask, 1.0)
        # Collection mask exists and has correct shape
        assert opt.col_mask.shape == (30, 30)
        # PSF tails are zero-weighted placeholders
        assert opt.psf_top.weight == 0.0
        assert opt.psf_bot.weight == 0.0
        np.testing.assert_array_equal(opt.psf_top.weights, 0)

    def test_two_photon_path_unchanged(self):
        """Default (no imaging_mode) still routes to two-photon."""
        from calcia.optics import simulate_optical_propagation

        vol_params = VolumeParams(
            vol_sz=(20, 20, 10), vres=1, min_dist=8.0, verbose=0,
        )
        psf_params = PsfParams(
            psf_type="gaussian_analytical",
            psf_sz=(8, 8, 6),
        )
        opt = simulate_optical_propagation(vol_params, psf_params)

        # Analytical two-photon path PSF has psf_sz[2] depth, not full volume
        assert opt.psf.shape[2] == 6

    def test_psf_depth_attenuation_exponential(self):
        """Per-slice PSF sum must decay as exp(-2*z/L_scatter) with depth."""
        from calcia.optics import simulate_optical_propagation

        # Volume top at surface (vol_depth=0) so z index == absolute depth (um)
        vol_params = VolumeParams(
            vol_sz=(30, 30, 30), vres=1, min_dist=10.0, vol_depth=0,
            verbose=0,
        )
        L_scatter = 50.0
        psf_params = PsfParams(
            imaging_mode="widefield",
            psf_type="gaussian_analytical",
            psf_sz=(12, 12, 30),
            lambda_em_um=0.52,
            scatter_length_um_wf=L_scatter,
        )
        opt = simulate_optical_propagation(vol_params, psf_params)

        Nz = opt.psf.shape[2]
        slice_sums = opt.psf.sum(axis=(0, 1))
        z_um = np.arange(Nz, dtype=np.float64)  # vres=1 => z index == um

        # Expected attenuation: exp(-2 * z_um / L_scatter)
        expected = np.exp(-2.0 * z_um / L_scatter)
        np.testing.assert_allclose(slice_sums, expected, rtol=1e-4, atol=1e-5)

        # Fit log(sum) vs z, slope must match -2 / L_scatter
        slope, _ = np.polyfit(z_um, np.log(slice_sums + 1e-30), 1)
        assert slope == pytest.approx(-2.0 / L_scatter, rel=0.05)

    def test_psf_large_scatter_recovers_uniform(self):
        """With effectively infinite scatter length, all z-slices sum to 1."""
        from calcia.optics import simulate_optical_propagation

        vol_params = VolumeParams(
            vol_sz=(25, 25, 20), vres=1, min_dist=10.0, vol_depth=0,
            verbose=0,
        )
        psf_params = PsfParams(
            imaging_mode="widefield",
            psf_type="gaussian_analytical",
            psf_sz=(10, 10, 20),
            lambda_em_um=0.52,
            scatter_length_um_wf=1e9,
        )
        opt = simulate_optical_propagation(vol_params, psf_params)

        slice_sums = opt.psf.sum(axis=(0, 1))
        np.testing.assert_allclose(slice_sums, 1.0, atol=1e-3)

    def test_hemo_abs_wf_stronger_than_tpm(self):
        """Widefield collection mask uses hemo_abs_wf (~30x stronger absorb)."""
        from calcia.optics import simulate_optical_propagation
        from calcia.pipeline import NeuralVolumeOutput

        # Build a volume with a simple central vessel cylinder
        vol_params = VolumeParams(
            vol_sz=(30, 30, 20), vres=1, min_dist=10.0, vol_depth=50,
            verbose=0,
        )
        Nx = int(round(vol_params.vol_sz[0] * vol_params.vres))
        Ny = int(round(vol_params.vol_sz[1] * vol_params.vres))
        Nz = int(round(vol_params.vol_sz[2] * vol_params.vres))
        # Vasculature-grade Nx_v/Ny_v typically matches vol; top-down cylinder
        vessel = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
        cx, cy = Nx // 2, Ny // 2
        xv, yv = np.meshgrid(np.arange(Nx) - cx, np.arange(Ny) - cy,
                             indexing="ij")
        vessel_mask = (xv ** 2 + yv ** 2) <= 4 ** 2
        vessel[vessel_mask, :] = 1

        vol_out = NeuralVolumeOutput(
            neur_vol=np.zeros((Nx, Ny, Nz), dtype=np.float32),
            gp_nuc=[],
            gp_soma=[],
            gp_vals=[],
            neur_ves=vessel,
            bg_proc=[],
            locs=np.zeros((0, 3), dtype=np.float32),
            neur_num=np.zeros((Nx, Ny, Nz), dtype=np.uint16),
            neur_num_ad=np.zeros((Nx, Ny, Nz), dtype=np.uint16),
            gp_bgvals=[],
            params={"vol_params": vol_params},
        )

        # Two-photon reference
        psf_tpm = PsfParams(
            psf_type="gaussian_analytical",
            psf_sz=(10, 10, 10),
        )
        opt_tpm = simulate_optical_propagation(vol_params, psf_tpm, vol_out)

        # Widefield: same optics but imaging_mode on, scatter disabled so the
        # only difference driving col_mask is hemo_abs_wf.
        psf_wf = PsfParams(
            imaging_mode="widefield",
            psf_type="gaussian_analytical",
            psf_sz=(10, 10, 10),
            lambda_em_um=0.52,
            scatter_length_um_wf=1e9,
        )
        opt_wf = simulate_optical_propagation(vol_params, psf_wf, vol_out)

        # Widefield collection mask should be strictly darker on average
        # because hemo_abs_wf ~ 30x the tpm default.
        assert opt_wf.col_mask.mean() < opt_tpm.col_mask.mean() - 1e-3


# ======================================================================
# Phase 4 depth sensitivity
# ======================================================================

class TestWidefieldDepthResponse:
    def _make_inputs(self, rng, N_xy=40, Nz=20, vres=1, vol_depth=0,
                     nt=3, L_scatter=20.0):
        """Two neurons at opposite z-extremes; equal brightness and XY."""
        from calcia.optics.propagation import OpticalPropagationResult
        from calcia.optics.psf import PsfTail, gaussian_psf_na
        from calcia.pipeline import NeuralVolumeOutput
        from calcia.traces.traces import TimeTracesResult
        from calcia.volume.fluorescence import CellFluorescenceData

        # Two single-voxel "neurons": both at FOV center, one at z=0, one at z=Nz-1
        lin = lambda x, y, z: x * N_xy * Nz + y * Nz + z
        cx, cy = N_xy // 2, N_xy // 2
        gp_vals = [
            CellFluorescenceData(
                indices=np.array([lin(cx, cy, 0)], dtype=np.int32),
                fluorescence=np.array([1.0], dtype=np.float32),
                soma_mask=np.array([True]),
            ),
            CellFluorescenceData(
                indices=np.array([lin(cx, cy, Nz - 1)], dtype=np.int32),
                fluorescence=np.array([1.0], dtype=np.float32),
                soma_mask=np.array([True]),
            ),
        ]

        vol_out = NeuralVolumeOutput(
            neur_vol=np.zeros((N_xy, N_xy, Nz), dtype=np.float32),
            gp_nuc=[(np.array([], dtype=np.int32), 0.0)] * 2,
            gp_soma=[np.array([], dtype=np.int32)] * 2,
            gp_vals=gp_vals,
            neur_ves=None,
            bg_proc=[],
            locs=np.zeros((2, 3), dtype=np.float32),
            neur_num=np.zeros((N_xy, N_xy, Nz), dtype=np.uint16),
            neur_num_ad=np.zeros((N_xy, N_xy, Nz), dtype=np.uint16),
            gp_bgvals=[],
            params={},
        )

        # PSF with per-slice depth attenuation baked in
        psf_params = PsfParams(
            imaging_mode="widefield",
            psf_type="gaussian_analytical",
            psf_sz=(12, 12, 20),
            lambda_em_um=0.52,
            scatter_length_um_wf=L_scatter,
        )
        psf, _, _, _ = gaussian_psf_na(
            na=psf_params.obj_na,
            lambda_um=psf_params.lambda_em_um,
            sampling=(1.0 / vres, 1.0 / vres, 1.0 / vres),
            mat_size=(12, 12, Nz),
            nidx=psf_params.n,
            scaling="widefield",
        )
        slice_sums = psf.sum(axis=(0, 1), keepdims=True)
        slice_sums[slice_sums <= 0] = 1.0
        psf = psf / slice_sums
        z_idx = np.arange(Nz, dtype=np.float32)
        abs_depth = float(vol_depth) + z_idx / vres
        atten = np.exp(-2.0 * abs_depth / L_scatter).astype(np.float32)
        psf = (psf * atten[np.newaxis, np.newaxis, :]).astype(np.float32)

        mask = np.ones((N_xy, N_xy), dtype=np.float32)
        col_mask = np.ones((N_xy, N_xy), dtype=np.float32)
        zero_w = np.zeros((12, 12), dtype=np.float32)
        zero_m = np.zeros((N_xy, N_xy), dtype=np.float32)
        opt_out = OpticalPropagationResult(
            psf=psf, mask=mask,
            psf_top=PsfTail(weights=zero_w, mask=zero_m, weight=0.0),
            psf_bot=PsfTail(weights=zero_w.copy(), mask=zero_m.copy(),
                            weight=0.0),
            col_mask=col_mask,
            params={"psf_params": psf_params},
        )

        spike_params = SpikeParams(K=2, nt=nt, dt=1.0 / 30, verbose=0)
        # Constant "active" traces equal to 1.0 for both cells every frame
        soma = np.ones((2, nt), dtype=np.float32) + 0.5
        time_out = TimeTracesResult(
            soma=soma, dend=None, bg=None, spikes=None,
            mod_vals=np.ones(2, dtype=np.float32),
            params={"spike_params": spike_params, "cal_params": None},
        )
        return vol_out, opt_out, time_out, spike_params

    def test_shallow_neuron_brighter_than_deep(self):
        """Given equal activity, a shallow neuron (z=0) produces more signal
        than a deep one (z=Nz-1) — the difference must match exp(-2*depth/L)."""
        from calcia.scanning import scan_widefield
        from calcia.traces.traces import TimeTracesResult

        rng = np.random.default_rng(0)
        Nz = 20
        L = 20.0
        vol_out, opt_out, time_out, sp = self._make_inputs(
            rng, N_xy=40, Nz=Nz, vres=1, vol_depth=0, nt=3, L_scatter=L,
        )
        scan_params = ScanParams(
            scan_buff=4, motion=False, sfrac=1, verbose=0,
        )
        result = scan_widefield(
            vol_out, opt_out, time_out,
            scan_params=scan_params, spike_params=sp, seed=0,
        )

        # Each frame integrates two point sources; shallow at z=0 has
        # attenuation 1.0, deep at z=Nz-1 has attenuation exp(-2*(Nz-1)/L).
        # Because the two sources co-locate laterally, the frame total is
        # proportional to (1 + exp(...)). We instead compare runs with only
        # the shallow vs only the deep cell silenced to isolate the ratio.

        # Silence deep cell
        vol_shallow = vol_out
        time_shallow = TimeTracesResult(
            soma=np.vstack([time_out.soma[0:1], np.zeros_like(time_out.soma[0:1])]),
            dend=None, bg=None, spikes=None,
            mod_vals=time_out.mod_vals,
            params=time_out.params,
        )
        # Silence shallow cell
        time_deep = TimeTracesResult(
            soma=np.vstack([np.zeros_like(time_out.soma[0:1]), time_out.soma[1:2]]),
            dend=None, bg=None, spikes=None,
            mod_vals=time_out.mod_vals,
            params=time_out.params,
        )
        r_shallow = scan_widefield(
            vol_shallow, opt_out, time_shallow,
            scan_params=scan_params, spike_params=sp, seed=0,
        )
        r_deep = scan_widefield(
            vol_shallow, opt_out, time_deep,
            scan_params=scan_params, spike_params=sp, seed=0,
        )

        s_shallow = float(r_shallow.mov_raw.sum())
        s_deep = float(r_deep.mov_raw.sum())
        assert s_shallow > 0 and s_deep > 0
        # Expected ratio ~ exp(-2*(Nz-1)/L) = exp(-38/20) ~ 0.15
        expected_ratio = float(np.exp(-2.0 * (Nz - 1) / L))
        assert s_deep / s_shallow == pytest.approx(expected_ratio, rel=0.25)


# ======================================================================
# Phase 4 widefield scanning (smoke test)
# ======================================================================

class TestScanWidefield:
    @pytest.fixture
    def synthetic_inputs(self):
        """Minimal Phase 1/2/3 outputs for a widefield smoke test."""
        from calcia.optics.propagation import OpticalPropagationResult
        from calcia.optics.psf import PsfTail
        from calcia.pipeline import NeuralVolumeOutput
        from calcia.traces.traces import TimeTracesResult
        from calcia.volume.fluorescence import CellFluorescenceData

        rng = np.random.default_rng(42)
        N1, N2, N3 = 40, 40, 12
        K = 3
        Nt = 4

        neur_vol = rng.random((N1, N2, N3)).astype(np.float32) * 0.01
        gp_vals = []
        for _ in range(K):
            n_vox = 15
            indices = rng.integers(
                0, N1 * N2 * N3, size=n_vox
            ).astype(np.int32)
            fluor = rng.random(n_vox).astype(np.float32)
            soma_mask = np.zeros(n_vox, dtype=bool)
            soma_mask[:8] = True
            gp_vals.append(CellFluorescenceData(
                indices=indices, fluorescence=fluor, soma_mask=soma_mask,
            ))

        gp_nuc = [(np.array([], dtype=np.int32), 0.0)] * K

        vol_out = NeuralVolumeOutput(
            neur_vol=neur_vol,
            gp_nuc=gp_nuc,
            gp_soma=[np.array([], dtype=np.int32)] * K,
            gp_vals=gp_vals,
            neur_ves=None,
            bg_proc=[],
            locs=rng.random((K, 3)).astype(np.float32) * 30,
            neur_num=np.zeros((N1, N2, N3), dtype=np.uint16),
            neur_num_ad=np.zeros((N1, N2, N3), dtype=np.uint16),
            gp_bgvals=[],
            params={},
        )

        # Widefield PSF must span full z-depth, PSF XY <= volume XY
        psf_params = PsfParams(
            imaging_mode="widefield",
            psf_sz=(12, 12, 0),  # z replaced by full volume depth below
            lambda_em_um=0.52,
        )
        # Emulate what simulate_optical_propagation_widefield would build:
        from calcia.optics.psf import gaussian_psf_na as _gpsf
        psf, _, _, _ = _gpsf(
            na=psf_params.obj_na,
            lambda_um=psf_params.lambda_em_um,
            sampling=(1.0, 1.0, 1.0),
            mat_size=(12, 12, N3),
            nidx=psf_params.n,
            scaling="widefield",
        )
        slice_sums = psf.sum(axis=(0, 1), keepdims=True)
        slice_sums[slice_sums <= 0] = 1.0
        psf = (psf / slice_sums).astype(np.float32)

        mask = np.ones((N1, N2), dtype=np.float32)
        col_mask = np.ones((N1, N2), dtype=np.float32)
        zero_w = np.zeros((12, 12), dtype=np.float32)
        zero_m = np.zeros((N1, N2), dtype=np.float32)

        opt_out = OpticalPropagationResult(
            psf=psf,
            mask=mask,
            psf_top=PsfTail(weights=zero_w, mask=zero_m, weight=0.0),
            psf_bot=PsfTail(
                weights=zero_w.copy(), mask=zero_m.copy(), weight=0.0,
            ),
            col_mask=col_mask,
            params={"psf_params": psf_params},
        )

        soma = (rng.random((K, Nt)).astype(np.float32) + 1.0)
        dend = (rng.random((K, Nt)).astype(np.float32) + 1.0)
        spike_params = SpikeParams(K=K, nt=Nt, dt=1.0 / 30, verbose=0)
        time_out = TimeTracesResult(
            soma=soma, dend=dend, bg=None, spikes=None,
            mod_vals=np.ones(K, dtype=np.float32),
            params={"spike_params": spike_params, "cal_params": None},
        )
        return vol_out, opt_out, time_out, spike_params

    def test_output_shapes(self, synthetic_inputs):
        from calcia.scanning import scan_widefield

        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        scan_params = ScanParams(
            scan_buff=4, motion=False, sfrac=2, verbose=0,
        )
        wf_params = WidefieldParams(pavg=5.0)
        cam_params = CameraNoiseParams()

        result = scan_widefield(
            vol_out, opt_out, time_out,
            scan_params=scan_params,
            cam_params=cam_params,
            wf_params=wf_params,
            spike_params=spike_params,
            seed=7,
        )

        N1, N2 = 40, 40
        sfrac = 2
        buf = 4
        expected_h = N1 // sfrac - 2 * (buf // sfrac)
        expected_w = N2 // sfrac - 2 * (buf // sfrac)
        Nt = 4

        assert result.mov.shape == (expected_h, expected_w, Nt)
        assert result.mov_raw.shape == (expected_h, expected_w, Nt)
        assert result.mot_hist.shape == (3, Nt)
        assert result.mov.dtype == np.float32
        assert "wf_params" in result.params
        assert "cam_params" in result.params

    def test_deterministic_with_seed(self, synthetic_inputs):
        from calcia.scanning import scan_widefield

        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        scan_params = ScanParams(
            scan_buff=4, motion=False, sfrac=2, verbose=0,
        )
        r1 = scan_widefield(
            vol_out, opt_out, time_out,
            scan_params=scan_params, spike_params=spike_params, seed=11,
        )
        r2 = scan_widefield(
            vol_out, opt_out, time_out,
            scan_params=scan_params, spike_params=spike_params, seed=11,
        )
        np.testing.assert_array_equal(r1.mov, r2.mov)
        np.testing.assert_array_equal(r1.mov_raw, r2.mov_raw)

    def test_scan_volume_dispatches_to_widefield(self, synthetic_inputs):
        """scan_volume must route widefield opt_out to scan_widefield."""
        from calcia.scanning import scan_volume

        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        scan_params = ScanParams(
            scan_buff=4, motion=False, sfrac=2, verbose=0,
        )
        result = scan_volume(
            vol_out, opt_out, time_out,
            scan_params=scan_params,
            spike_params=spike_params,
            seed=3,
        )
        # Dispatcher stores widefield params, not tpm_params
        assert "wf_params" in result.params
        assert "tpm_params" not in result.params

    # --- In-focus / out-of-focus separation ---------------------------

    _sp = dict(scan_buff=4, motion=False, sfrac=2, verbose=0)

    def test_separate_focus_default_none(self, synthetic_inputs):
        from calcia.scanning import scan_widefield
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        r = scan_widefield(vol_out, opt_out, time_out,
                           scan_params=ScanParams(**self._sp),
                           spike_params=spike_params, seed=7)
        assert r.mov_infocus is None and r.mov_oof is None

    def test_separate_focus_invariant(self, synthetic_inputs):
        """Partial slab: mov_raw == infocus + oof, both non-trivial."""
        from calcia.scanning import scan_widefield
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        r = scan_widefield(vol_out, opt_out, time_out,
                           scan_params=ScanParams(**self._sp),
                           spike_params=spike_params, seed=7,
                           separate_focus=True, focus_slab_um=2.0)
        assert r.mov_infocus is not None and r.mov_oof is not None
        assert r.mov_infocus.shape == r.mov_raw.shape
        assert r.mov_oof.dtype == np.float32
        np.testing.assert_allclose(r.mov_raw, r.mov_infocus + r.mov_oof,
                                   rtol=1e-4, atol=1e-4)
        # A thin slab (z=6+/-1 of 12 planes) leaves real energy in both halves.
        assert np.any(r.mov_infocus != 0)
        assert np.any(r.mov_oof != 0)

    def test_separate_focus_full_slab_oof_zero(self, synthetic_inputs):
        """A slab covering the whole stack puts all light in focus."""
        from calcia.scanning import scan_widefield
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        r = scan_widefield(vol_out, opt_out, time_out,
                           scan_params=ScanParams(**self._sp),
                           spike_params=spike_params, seed=7,
                           separate_focus=True, focus_slab_um=1e6)
        np.testing.assert_array_equal(r.mov_oof, np.zeros_like(r.mov_oof))
        np.testing.assert_allclose(r.mov_infocus, r.mov_raw, rtol=1e-5,
                                   atol=1e-5)

    def test_separate_focus_default_path_identical(self, synthetic_inputs):
        """separate_focus must not perturb mov / mov_raw (same RNG draws)."""
        from calcia.scanning import scan_widefield
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        r_off = scan_widefield(vol_out, opt_out, time_out,
                               scan_params=ScanParams(**self._sp),
                               spike_params=spike_params, seed=7)
        r_on = scan_widefield(vol_out, opt_out, time_out,
                              scan_params=ScanParams(**self._sp),
                              spike_params=spike_params, seed=7,
                              separate_focus=True, focus_slab_um=2.0)
        np.testing.assert_array_equal(r_off.mov, r_on.mov)
        np.testing.assert_array_equal(r_off.mov_raw, r_on.mov_raw)

    def test_separate_focus_deterministic(self, synthetic_inputs):
        from calcia.scanning import scan_widefield
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        kw = dict(scan_params=ScanParams(**self._sp),
                  spike_params=spike_params, seed=7,
                  separate_focus=True, focus_slab_um=2.0)
        r1 = scan_widefield(vol_out, opt_out, time_out, **kw)
        r2 = scan_widefield(vol_out, opt_out, time_out, **kw)
        np.testing.assert_array_equal(r1.mov_infocus, r2.mov_infocus)
        np.testing.assert_array_equal(r1.mov_oof, r2.mov_oof)

    # ---- physio motion model (end-to-end) ----
    def test_physio_motion_default_unchanged(self, synthetic_inputs):
        """No motion_params (or randomwalk) => legacy integer-walk mot_hist."""
        from calcia.scanning import scan_widefield
        from calcia.config.params import MotionParams
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        sp = ScanParams(scan_buff=4, motion=True, sfrac=2, verbose=0)
        r_none = scan_widefield(vol_out, opt_out, time_out, scan_params=sp,
                                spike_params=spike_params, seed=3)
        r_rw = scan_widefield(vol_out, opt_out, time_out, scan_params=sp,
                              spike_params=spike_params,
                              motion_params=MotionParams(model="randomwalk"),
                              seed=3)
        np.testing.assert_array_equal(r_none.mot_hist, r_rw.mot_hist)
        # legacy shifts are integers
        assert np.allclose(r_none.mot_hist, np.round(r_none.mot_hist))

    def test_physio_motion_end_to_end(self, synthetic_inputs):
        from calcia.scanning import scan_widefield
        from calcia.config.params import MotionParams
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        sp = ScanParams(scan_buff=8, motion=True, sfrac=2, verbose=0)
        mp = MotionParams(model="physio", seed=5)
        r = scan_widefield(vol_out, opt_out, time_out, scan_params=sp,
                           spike_params=spike_params, motion_params=mp, seed=1)
        assert r.mov.shape[2] == time_out.soma.shape[1]
        # physio z-row is always zero; xy shifts are bounded by scan_buff
        assert np.all(np.abs(r.mot_hist[:2]) <= 8 + 1e-4)
        assert np.all(r.mot_hist[2] == 0)

    def test_physio_blur_hist_recorded(self, synthetic_inputs):
        """physio run exposes a (2, Nt) blur_hist; it equals the thresholded
        frame-to-frame shift difference (the streak smeared into each frame)."""
        from calcia.scanning import scan_widefield
        from calcia.config.params import MotionParams
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        sfrac = 2
        sp = ScanParams(scan_buff=8, motion=True, sfrac=sfrac, verbose=0)
        mp = MotionParams(model="physio", seed=5)
        r = scan_widefield(vol_out, opt_out, time_out, scan_params=sp,
                           spike_params=spike_params, motion_params=mp, seed=1)
        Nt = time_out.soma.shape[1]
        assert r.blur_hist is not None and r.blur_hist.shape == (2, Nt)
        assert np.all(r.blur_hist[:, 0] == 0)   # first frame never blurred
        # blur_hist == diff(mot_hist)*exposure_frac where |streak| >= min_len
        step = np.diff(r.mot_hist[:2], axis=1) * mp.exposure_frac
        mask = np.hypot(step[0], step[1]) >= mp.blur_min_px * sfrac
        exp = np.zeros_like(r.blur_hist)
        exp[:, 1:][:, mask] = step[:, mask]
        np.testing.assert_allclose(r.blur_hist, exp, atol=1e-5)
        # motion_params round-trips through the result params dict
        assert r.params["motion_params"] is mp

    def test_randomwalk_has_no_blur_hist(self, synthetic_inputs):
        """The legacy walk has no intra-frame blur, so blur_hist is None."""
        from calcia.scanning import scan_widefield
        from calcia.config.params import MotionParams
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        sp = ScanParams(scan_buff=4, motion=True, sfrac=2, verbose=0)
        r = scan_widefield(vol_out, opt_out, time_out, scan_params=sp,
                           spike_params=spike_params,
                           motion_params=MotionParams(model="randomwalk"), seed=3)
        assert r.blur_hist is None

    def test_physio_motion_disabled_when_motion_off(self, synthetic_inputs):
        """motion=False overrides the physio model (no shifts at all)."""
        from calcia.scanning import scan_widefield
        from calcia.config.params import MotionParams
        vol_out, opt_out, time_out, spike_params = synthetic_inputs
        sp = ScanParams(scan_buff=8, motion=False, sfrac=2, verbose=0)
        r = scan_widefield(vol_out, opt_out, time_out, scan_params=sp,
                           spike_params=spike_params,
                           motion_params=MotionParams(model="physio"), seed=1)
        assert np.all(r.mot_hist == 0)


# ======================================================================
# Physio motion model + intra-frame blur (unit tests)
# ======================================================================
class TestMotionModel:
    def test_trajectory_shape_and_bound(self):
        from calcia.config.params import MotionParams
        from calcia.scanning.motion import generate_motion_trajectory
        mp = MotionParams(model="physio")
        rng = np.random.default_rng(0)
        traj = generate_motion_trajectory(500, mp, vres=1.0, scan_buff=30, rng=rng)
        assert traj.shape == (500, 2)
        assert np.all(np.abs(traj) <= 30 + 1e-4)

    def test_trajectory_bound_clips_to_scan_buff(self):
        """bound_um larger than scan_buff => clipped at scan_buff voxels."""
        from calcia.config.params import MotionParams
        from calcia.scanning.motion import generate_motion_trajectory
        mp = MotionParams(model="physio", bound_um=100.0)
        rng = np.random.default_rng(1)
        traj = generate_motion_trajectory(2000, mp, vres=1.0, scan_buff=6, rng=rng)
        assert np.all(np.abs(traj) <= 6 + 1e-4)

    def test_trajectory_anisotropy_and_autocorr(self):
        """y-motion (sigma 4.8) exceeds x (2.0); position is autocorrelated."""
        from calcia.config.params import MotionParams
        from calcia.scanning.motion import generate_motion_trajectory
        mp = MotionParams(model="physio")
        rng = np.random.default_rng(2)
        t = generate_motion_trajectory(4000, mp, vres=1.0, scan_buff=40, rng=rng)
        assert t[:, 1].std() > 1.5 * t[:, 0].std()
        for i in (0, 1):
            x = t[:, i] - t[:, i].mean()
            ac1 = (x[:-1] * x[1:]).sum() / (x * x).sum()
            assert 0.6 < ac1 < 0.95   # correlated slow drift, not white

    def test_trajectory_vres_scales_voxels(self):
        from calcia.config.params import MotionParams
        from calcia.scanning.motion import generate_motion_trajectory
        mp = MotionParams(model="physio")
        a = generate_motion_trajectory(3000, mp, vres=1.0, scan_buff=100,
                                       rng=np.random.default_rng(4))
        b = generate_motion_trajectory(3000, mp, vres=2.0, scan_buff=200,
                                       rng=np.random.default_rng(4))
        # same um motion at vres=2 spans ~2x the voxels
        assert 1.7 < b[:, 1].std() / a[:, 1].std() < 2.3

    def test_streak_kernel_none_for_tiny(self):
        from calcia.scanning.motion import motion_streak_kernel
        assert motion_streak_kernel(0.3, 0.0, min_len=0.75) is None

    def test_streak_kernel_normalized_line(self):
        from calcia.scanning.motion import motion_streak_kernel
        k = motion_streak_kernel(10.0, 0.0)
        assert abs(k.sum() - 1.0) < 1e-5
        # horizontal streak: energy concentrated on the centre row
        cy = k.shape[0] // 2
        assert k[cy].sum() > 0.8

    def test_apply_blur_reduces_sharpness(self):
        from calcia.scanning.motion import apply_motion_blur
        rng = np.random.default_rng(0)
        img = np.zeros((40, 40), np.float32)
        img[20, 20] = 100.0      # a point source
        blurred = apply_motion_blur(img, 8.0, 0.0)
        assert blurred.max() < img.max()          # spread out
        assert abs(blurred.sum() - img.sum()) < 1e-2   # flux conserved

    def test_apply_blur_noop_for_tiny(self):
        from calcia.scanning.motion import apply_motion_blur
        img = np.random.default_rng(0).random((20, 20)).astype(np.float32)
        out = apply_motion_blur(img, 0.2, 0.0)
        np.testing.assert_array_equal(out, img)

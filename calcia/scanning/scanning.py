"""
Main scanning simulation pipeline.

Port of MATLAB: ``scan_volume.m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from scipy.signal import convolve2d

from ..config.params import NoiseParams, ScanParams, TpmParams
from ..optics.signal import tpm_signal_scale
from .convolution import blurred_back_comp, psf_fft, single_scan
from .motion import apply_row_shifts
from .noise import pixel_bleed, poisson_gauss_noise

if TYPE_CHECKING:
    from ..config.params import SpikeParams
    from ..optics.propagation import OpticalPropagationResult
    from ..pipeline import NeuralVolumeOutput
    from ..traces.traces import TimeTracesResult


@dataclass
class ScanResult:
    """Output of :func:`scan_volume`.

    Attributes
    ----------
    mov : np.ndarray
        ``(H, W, Nt)`` float32 noisy movie.
    mov_raw : np.ndarray
        ``(H, W, Nt)`` float32 clean movie (before noise model).
    mot_hist : np.ndarray
        ``(3, Nt)`` float32 motion history ``[x, y, z]``.
    params : dict
        Parameter objects used for the scan.
    mov_infocus : np.ndarray, optional
        ``(H, W, Nt)`` float32 clean in-focus image (before noise), produced
        only when ``scan_volume(..., separate_focus=True)``. Contains the
        in-focus contribution of all components (soma + dendrites + nucleus +
        axons), i.e. ``mov_raw`` minus the out-of-focus blur. ``None`` otherwise.
    mov_oof : np.ndarray, optional
        ``(H, W, Nt)`` float32 clean out-of-focus (defocus) background image
        (before noise), produced only when ``separate_focus=True``. By linearity
        ``mov_raw == mov_infocus + mov_oof`` (up to float). ``None`` otherwise.
    blur_hist : np.ndarray, optional
        ``(2, Nt)`` float32 per-frame intra-frame motion-blur streak ``[dx, dy]``
        in full-resolution **voxels** (same units as ``mot_hist``). This is the
        displacement the sample travelled during the exposure that was actually
        smeared into that frame; ``[0, 0]`` for frames with no applied blur (the
        first frame, streaks below ``MotionParams.blur_min_px``, or when blur is
        off). Only produced by the widefield ``physio`` motion model; ``None``
        for the legacy random walk / two-photon path.
    motion_gt : dict, optional
        Complete, lossless record of every motion component actually rendered
        into the movie. ``mot_hist`` and ``blur_hist`` are lossy summaries kept
        for backward compatibility; this is the full truth (sub-voxel rounding
        residual, un-thresholded and clamp-corrected blur streaks, per-row scan
        shear). See :func:`calcia.scanning.motion.describe_motion_gt` for the
        key list and units. Produced by both scanners.
    """
    mov: np.ndarray
    mov_raw: np.ndarray
    mot_hist: np.ndarray
    params: Dict
    mov_infocus: Optional[np.ndarray] = None
    mov_oof: Optional[np.ndarray] = None
    blur_hist: Optional[np.ndarray] = None
    motion_gt: Optional[Dict[str, np.ndarray]] = None


def scan_volume(
    vol_out: "NeuralVolumeOutput",
    opt_out: "OpticalPropagationResult",
    time_out: "TimeTracesResult",
    scan_params: Optional[ScanParams] = None,
    noise_params: Optional[NoiseParams] = None,
    tpm_params: Optional[TpmParams] = None,
    spike_params: Optional["SpikeParams"] = None,
    *,
    seed: Optional[int] = None,
    separate_focus: bool = False,
    focus_slab_um: Optional[float] = None,
) -> ScanResult:
    """Scan a 3-D neural volume and create a simulated two-photon movie.

    Port of MATLAB ``scan_volume.m``.

    Parameters
    ----------
    vol_out : NeuralVolumeOutput
        Phase 1 output (neural volume with component indexing).
    opt_out : OpticalPropagationResult
        Phase 2 output (PSF + masks).
    time_out : TimeTracesResult
        Phase 3 output (fluorescence time traces).
    scan_params : ScanParams, optional
        Scanning parameters (defaults used if ``None``).
    noise_params : NoiseParams, optional
        Noise model parameters (defaults used if ``None``).
    tpm_params : TpmParams, optional
        Two-photon microscope parameters (defaults used if ``None``).
    spike_params : SpikeParams, optional
        Spike parameters (needed for ``dt``).  If ``None``, uses the
        ``SpikeParams`` stored in ``time_out.params``.
    seed : int, optional
        Random seed for reproducibility.
    separate_focus : bool, default False
        If ``True``, additionally return noiseless ``mov_infocus`` and
        ``mov_oof`` (in-focus / out-of-focus decomposition) on the
        :class:`ScanResult`. The decomposition is exact by linearity of the
        convolution (``mov_raw == mov_infocus + mov_oof``) and adds no extra
        convolution — both images already exist as intermediates. When ``False``
        the code path is bit-identical to before and both fields are ``None``.
        On the widefield path the split is also supported but is NOT free:
        it costs one extra convolution per frame (see ``focus_slab_um``).
    focus_slab_um : float, optional
        Widefield only. Full thickness (in microns) of the in-focus slab,
        centered on the focal plane (volume z-midpoint). When ``None``, defaults
        to the widefield axial depth-of-field ``2*n*lambda_em/NA**2``. Ignored
        on the two-photon path (whose in-focus slab is fixed by the PSF z-extent).

    Returns
    -------
    ScanResult
        Simulated movie, clean movie, and motion history.
    """
    # ------------------------------------------------------------------
    # Widefield dispatch: if Phase 2 ran with imaging_mode='widefield',
    # delegate to scan_widefield and leave the two-photon code path below
    # untouched.
    # ------------------------------------------------------------------
    psf_params = opt_out.params.get("psf_params") if opt_out.params else None
    if getattr(psf_params, "imaging_mode", "two-photon") == "widefield":
        from .widefield import scan_widefield
        return scan_widefield(
            vol_out, opt_out, time_out,
            scan_params=scan_params,
            spike_params=spike_params,
            seed=seed,
            separate_focus=separate_focus,
            focus_slab_um=focus_slab_um,
        )

    # ------------------------------------------------------------------
    # Default parameters
    # ------------------------------------------------------------------
    if scan_params is None:
        scan_params = ScanParams()
    if noise_params is None:
        noise_params = NoiseParams()
    if tpm_params is None:
        tpm_params = TpmParams()
    if spike_params is None:
        spike_params = time_out.params["spike_params"]

    rng = np.random.default_rng(seed)

    scan_buff = scan_params.scan_buff
    mot_opt = scan_params.motion
    scan_avg = scan_params.scan_avg
    sfrac = scan_params.sfrac
    verbose = scan_params.verbose

    # ------------------------------------------------------------------
    # Compute signal scaling
    # ------------------------------------------------------------------
    sigscale = (
        tpm_signal_scale(tpm_params)
        * spike_params.dt
        * sfrac ** 2
        / (250 * 250)
    )

    # ------------------------------------------------------------------
    # Volume dimensions
    # ------------------------------------------------------------------
    neur_vol_3d = vol_out.neur_vol
    N1, N2, N3 = neur_vol_3d.shape
    PSF = opt_out.psf
    Np1, Np2, Np3 = PSF.shape

    if N1 < Np1 or N2 < Np2:
        raise ValueError("PSF extent is bigger than the volume!")
    if N3 < Np3:
        raise ValueError("PSF depth is larger than the volume depth!")

    # ------------------------------------------------------------------
    # Combined mask
    # ------------------------------------------------------------------
    t_mask = opt_out.mask * opt_out.col_mask
    t_thresh = 1e-5
    t_mask = np.maximum(t_mask, t_thresh)

    # ------------------------------------------------------------------
    # Activity traces
    # ------------------------------------------------------------------
    soma_trace = np.asarray(time_out.soma, dtype=np.float32)
    K, Nt = soma_trace.shape

    dend_trace = time_out.dend
    if dend_trace is None:
        dend_trace = soma_trace.copy()
    else:
        dend_trace = np.asarray(dend_trace, dtype=np.float32)

    bg_trace = time_out.bg
    if bg_trace is None:
        bg_trace = np.zeros((1, Nt), dtype=np.float32)
    else:
        bg_trace = np.asarray(bg_trace, dtype=np.float32)

    # Nuclear label mode
    nuc_label = scan_params.nuc_label >= 1
    if nuc_label:
        nuc_trace = soma_trace.copy()
        soma_trace = np.zeros_like(soma_trace)
        dend_trace = np.zeros_like(dend_trace)
        bg_trace = np.zeros_like(bg_trace)

    if verbose >= 1:
        print(f"  Initializing scanning parameters...")
        print(f"    - Volume size: [{N1}, {N2}, {N3}] voxels")
        print(f"    - Number of frames: {Nt}")
        print(f"    - Scan buffer: {scan_buff} pixels")
        print(f"    - Motion simulation: {mot_opt}")
        print(f"    - Subsampling factor: {sfrac}")

    # ------------------------------------------------------------------
    # Pre-split soma / dendrite component indexing with mask weighting
    # ------------------------------------------------------------------
    n_comp = len(vol_out.gp_vals)
    soma_idx_list = []
    soma_wt_list = []
    dend_idx_list = []
    dend_wt_list = []

    for i in range(n_comp):
        cfd = vol_out.gp_vals[i]
        s_mask = cfd.soma_mask

        s_idx = cfd.indices[s_mask]
        s_fl = cfd.fluorescence[s_mask].copy()
        d_idx = cfd.indices[~s_mask]
        d_fl = cfd.fluorescence[~s_mask].copy()

        # Apply lateral mask: project 3D C-order index to 2D
        if len(s_idx) > 0:
            i2d = _idx_to_2d(s_idx, N1, N2, N3)
            s_fl *= t_mask.ravel()[i2d]
        if len(d_idx) > 0:
            i2d = _idx_to_2d(d_idx, N1, N2, N3)
            d_fl *= t_mask.ravel()[i2d]

        soma_idx_list.append(s_idx)
        soma_wt_list.append(s_fl)
        dend_idx_list.append(d_idx)
        dend_wt_list.append(d_fl)

    # Pre-split axon / background
    axon_idx_list = []
    axon_wt_list = []
    if vol_out.bg_proc is not None:
        for bp in vol_out.bg_proc:
            a_idx = bp.indices
            a_fl = bp.fluorescence.copy()
            if len(a_idx) > 0:
                i2d = _idx_to_2d(a_idx, N1, N2, N3)
                a_fl *= t_mask.ravel()[i2d]
            axon_idx_list.append(a_idx)
            axon_wt_list.append(a_fl)

    # Nuclear volumes (for nuc_label mode)
    nuc_idx_list = []
    nuc_wt_list = []
    if nuc_label:
        for idx_arr, fl_val in vol_out.gp_nuc:
            if len(idx_arr) == 0:
                nuc_idx_list.append(idx_arr)
                nuc_wt_list.append(np.array([], dtype=np.float32))
                continue
            i2d = _idx_to_2d(idx_arr, N1, N2, N3)
            wt = t_mask.ravel()[i2d]
            nuc_idx_list.append(idx_arr)
            nuc_wt_list.append(wt.astype(np.float32))

    # ------------------------------------------------------------------
    # Activity baseline subtraction
    # ------------------------------------------------------------------
    cutoff = 1e-2

    soma_min = soma_trace.min(axis=1, keepdims=True)
    soma_act = soma_trace - soma_min
    soma_act[soma_act < cutoff] = 0
    soma_min = soma_min.ravel()

    dend_min = dend_trace.min(axis=1, keepdims=True)
    dend_act = dend_trace - dend_min
    dend_act[dend_act < cutoff] = 0
    dend_min = dend_min.ravel()

    bg_min = bg_trace.min(axis=1, keepdims=True)
    bg_act = bg_trace - bg_min
    bg_act[bg_act < cutoff] = 0
    bg_min = bg_min.ravel()

    if nuc_label:
        nuc_min = nuc_trace.min(axis=1, keepdims=True)
        nuc_act = nuc_trace - nuc_min
        nuc_act[nuc_act < cutoff] = 0
        nuc_min_v = nuc_min.ravel()

    # ------------------------------------------------------------------
    # Build baseline volume (f0vol)
    # ------------------------------------------------------------------
    if verbose >= 1:
        print("  Initializing the base volume...")

    f0vol = np.zeros((N1, N2, N3), dtype=np.float32)
    vol_flat = f0vol.ravel()

    # Soma baseline
    for ll in range(min(len(soma_idx_list), len(soma_min))):
        idx = soma_idx_list[ll]
        if len(idx) > 0 and soma_min[ll] != 0:
            vol_flat[idx] = soma_wt_list[ll] * soma_min[ll]

    # Nucleus baseline
    for ll in range(len(vol_out.gp_nuc)):
        nuc_idx, nuc_fl = vol_out.gp_nuc[ll]
        if len(nuc_idx) == 0 or nuc_fl == 0:
            continue
        if ll < len(soma_min):
            vol_flat[nuc_idx] = nuc_fl * soma_min[ll]

    # Dendrite baseline
    for ll in range(min(len(dend_idx_list), len(dend_min))):
        idx = dend_idx_list[ll]
        if len(idx) > 0 and dend_min[ll] != 0:
            vol_flat[idx] = dend_wt_list[ll] * dend_min[ll]

    # Background baseline (additive): axon/background voxels OVERLAP so they must
    # be scatter-ADDED. Batch all active components into one buffered np.bincount
    # instead of a per-component np.add.at loop (identical result, ~an order faster).
    _n_ax = min(len(axon_idx_list), len(bg_min))
    _bi = [axon_idx_list[ll] for ll in range(_n_ax)
           if len(axon_idx_list[ll]) > 0 and bg_min[ll] != 0]
    if _bi:
        _bw = [axon_wt_list[ll] * bg_min[ll] for ll in range(_n_ax)
               if len(axon_idx_list[ll]) > 0 and bg_min[ll] != 0]
        vol_flat += np.bincount(
            np.concatenate(_bi), weights=np.concatenate(_bw),
            minlength=vol_flat.size,
        ).astype(vol_flat.dtype, copy=False)

    # Nuclear label baseline
    if nuc_label:
        for ll in range(len(nuc_idx_list)):
            idx = nuc_idx_list[ll]
            if len(idx) > 0 and nuc_min_v[ll] != 0:
                vol_flat[idx] = nuc_wt_list[ll] * nuc_min_v[ll]

    # ------------------------------------------------------------------
    # Pre-compute PSF FFT
    # ------------------------------------------------------------------
    freq_psf = psf_fft((N1, N2, N3), PSF, scan_avg)

    # Out-of-focus PSF tails
    # Out-of-focus tails: check if weights have any nonzero content
    has_tails = (
        opt_out.psf_top is not None
        and np.any(opt_out.psf_top.weights != 0)
    )
    if has_tails:
        psfT = opt_out.psf_top
        psfB = opt_out.psf_bot
        # Normalise masks
        t_mask_top = psfT.mask / np.mean(psfT.mask)
        t_mask_bot = psfB.mask / np.mean(psfB.mask)
        # Use weights as convolution kernel (normalised to sum=1)
        wt_top = psfT.weights / np.sum(psfT.weights)
        wt_bot = psfB.weights / np.sum(psfB.weights)
        freq_psf_top = psf_fft((N1, N2, N3), wt_top[:, :, np.newaxis])
        freq_psf_bot = psf_fft((N1, N2, N3), wt_bot[:, :, np.newaxis])
        # Scalar tail energy (MATLAB psfT.weight / psfB.weight)
        tail_wt_top = psfT.weight
        tail_wt_bot = psfB.weight

    # ------------------------------------------------------------------
    # Motion parameters
    # ------------------------------------------------------------------
    z_base = N3 // 2 - Np3 // 2 + scan_params.zoffset
    z_loc = z_base
    x_loc = scan_buff + 1
    y_loc = scan_buff + 1

    if mot_opt:
        zmaxdiff = 2
        d_stps = np.array([-1, 1, 0, 0, 0, 0, 0])
        d_stpsZ = np.array([-1, 1] + [0] * 100)
        d_stps2 = np.arange(-3, 4)
        p_jump = 0.05
        maxshear = 1.0 / 200
    else:
        zmaxdiff = 0
        d_stps = np.array([0, 0, 0])
        d_stpsZ = np.array([0, 0, 0])
        d_stps2 = np.array([0, 0, 0])
        p_jump = 0.0
        maxshear = 0.0

    # ------------------------------------------------------------------
    # Output arrays
    # ------------------------------------------------------------------
    out_h = N1 // sfrac - 2 * (scan_buff // sfrac)
    out_w = N2 // sfrac - 2 * (scan_buff // sfrac)
    mov = np.zeros((out_h, out_w, Nt), dtype=np.float32)
    mov_raw = np.zeros((out_h, out_w, Nt), dtype=np.float32)
    mot_hist = np.zeros((3, Nt), dtype=np.float32)
    # Complete motion record (see motion.describe_motion_gt). mot_hist collapses
    # the raster to a single y_pos per frame; the per-ROW offsets below are the
    # intra-frame scan distortion it discards.
    _mgt = dict(
        shift_applied=np.zeros((3, Nt), dtype=np.float32),
        row_y_off=np.zeros((N1, Nt), dtype=np.float32),
        row_shear=np.zeros((N1, Nt), dtype=np.float32),
    )
    mov_infocus = (np.zeros((out_h, out_w, Nt), dtype=np.float32)
                   if separate_focus else None)
    mov_oof = (np.zeros((out_h, out_w, Nt), dtype=np.float32)
               if separate_focus else None)

    # ------------------------------------------------------------------
    # Per-frame scanning loop
    # ------------------------------------------------------------------
    if verbose >= 1:
        print("  Scanning volume...")

    n_soma = min(len(soma_idx_list), soma_act.shape[0])
    n_dend = min(len(dend_idx_list), dend_act.shape[0])
    n_bg = min(len(axon_idx_list), bg_act.shape[0])
    n_nuc = len(nuc_idx_list) if nuc_label else 0

    sfrac_int = int(sfrac) == sfrac

    def _post(img, x_pos, y_off):
        """Apply per-frame row shifts + downsampling (no RNG side effects).

        Identical operations to the inline ``clean_img`` post-processing, so
        reusing it on the in-focus / out-of-focus components keeps them aligned
        with ``mov_raw`` and leaves the default path numerically unchanged.
        """
        img = apply_row_shifts(img, scan_buff, x_pos, y_off)
        if sfrac_int:
            s = int(sfrac)
            img = convolve2d(img, np.ones((s, s), dtype=np.float32),
                             mode='same')
            img = img[::s, ::s]
        else:
            from scipy.ndimage import zoom
            img = sfrac ** 2 * zoom(img, 1.0 / sfrac, order=1)
        return img

    for kk in range(Nt):
        # --- Motion update ---
        if rng.random() > p_jump:
            x_loc = int(np.clip(x_loc + rng.choice(d_stps2),
                                1, 2 * scan_buff + 1))
            y_loc = int(np.clip(y_loc + rng.choice(d_stps2),
                                1, 2 * scan_buff + 1))

        x_pos = int(np.clip(x_loc + rng.choice(d_stps),
                            1, 2 * scan_buff + 1))
        y_pos = int(np.clip(y_loc + rng.choice(d_stps),
                            1, 2 * scan_buff + 1))
        z_loc = int(np.clip(z_loc + rng.choice(d_stpsZ),
                            z_base - zmaxdiff, z_base + zmaxdiff))
        z_loc = int(np.clip(z_loc + rng.choice(d_stps),
                            1, N3 - Np3 + 1))

        mot_hist[:, kk] = [x_pos, y_pos, z_loc]

        # Shearing vector
        start_flat = rng.integers(1, max(2, int(2 * N1 / 5)) + 1)
        ramp_len = max(1, int(round(rng.random() * 3 * N1 / 5)))
        shear_dir = (2 * (rng.random() - 0.5)) * maxshear * N1
        y_shr = np.zeros(N1)
        ramp = np.linspace(0, 1, ramp_len) * shear_dir
        end_ramp = start_flat + ramp_len
        if end_ramp <= N1:
            y_shr[start_flat:end_ramp] = ramp
            y_shr[end_ramp:] = ramp[-1] if len(ramp) > 0 else 0
        else:
            y_shr[start_flat:N1] = ramp[:N1 - start_flat]

        y_off = np.clip(
            y_pos + y_shr + rng.choice(d_stps, size=N1),
            1, 2 * scan_buff + 1,
        )
        y_off = np.round(y_off).astype(np.float64)

        _mgt["shift_applied"][:, kk] = [x_pos, y_pos, z_loc]
        _mgt["row_y_off"][:, kk] = y_off
        _mgt["row_shear"][:, kk] = y_shr

        # --- Build temporary volume ---
        tmp_vol = np.zeros((N1, N2, N3), dtype=np.float32)
        tmp_flat = tmp_vol.ravel()

        for ll in range(n_soma):
            a = soma_act[ll, kk]
            if a > 0 and len(soma_idx_list[ll]) > 0:
                tmp_flat[soma_idx_list[ll]] = soma_wt_list[ll] * a

        if nuc_label:
            for ll in range(n_nuc):
                a = nuc_act[ll, kk] if ll < nuc_act.shape[0] else 0
                if a > 0 and len(nuc_idx_list[ll]) > 0:
                    tmp_flat[nuc_idx_list[ll]] = nuc_wt_list[ll] * a

        for ll in range(n_dend):
            a = dend_act[ll, kk]
            if a > 0 and len(dend_idx_list[ll]) > 0:
                tmp_flat[dend_idx_list[ll]] = dend_wt_list[ll] * a

        # Axon/background voxels OVERLAP so they must be scatter-ADDED. np.add.at
        # is an unbuffered scatter and dominated Phase 4 (~62%): batch all active
        # components and do ONE buffered np.bincount instead (identical, faster).
        _ai = [axon_idx_list[ll] for ll in range(n_bg)
               if bg_act[ll, kk] > 0 and len(axon_idx_list[ll]) > 0]
        if _ai:
            _aw = [axon_wt_list[ll] * bg_act[ll, kk] for ll in range(n_bg)
                   if bg_act[ll, kk] > 0 and len(axon_idx_list[ll]) > 0]
            tmp_flat += np.bincount(
                np.concatenate(_ai),
                weights=np.concatenate(_aw),
                minlength=tmp_flat.size,
            ).astype(tmp_flat.dtype, copy=False)

        # --- PSF convolution ---
        z_start = max(0, z_loc - 1)  # MATLAB 1-based → Python 0-based
        z_end = min(N3, z_start + Np3)
        scan_vol = tmp_vol[:, :, z_start:z_end] + f0vol[:, :, z_start:z_end]

        clean_img = (sigscale / (2 * sfrac ** 2)) * single_scan(
            scan_vol, PSF.shape, freq_psf, scan_avg)

        # In-focus / out-of-focus split (opt-in, exact by linearity).
        if separate_focus:
            infocus_img = clean_img.copy()
            oof_img = np.zeros_like(clean_img)

        # --- Out-of-focus blur ---
        if has_tails:
            inv_mask = 1.0 / t_mask if t_mask is not None else None
            # Top contribution
            top_z = np.arange(0, z_start)
            if len(top_z) == 0:
                top_z = np.arange(N3)
            top_mask_2d = inv_mask * t_mask_top if inv_mask is not None else t_mask_top
            top_img = blurred_back_comp(
                tmp_vol, top_z, freq_psf_top[:, :, 0],
                tail_wt_top, top_mask_2d,
                z_scale=psfT.z_weights,
                extra_vols=[f0vol],
            )

            # Bottom contribution
            bot_z = np.arange(z_end, N3)
            if len(bot_z) == 0:
                bot_z = np.arange(N3)
            bot_mask_2d = inv_mask * t_mask_bot if inv_mask is not None else t_mask_bot
            bot_img = blurred_back_comp(
                tmp_vol, bot_z, freq_psf_bot[:, :, 0],
                tail_wt_bot, bot_mask_2d,
                z_scale=psfB.z_weights,
                extra_vols=[f0vol],
            )

            oof_contrib = (top_img + bot_img) * (sigscale / sfrac ** 2)
            clean_img += oof_contrib
            if separate_focus:
                oof_img = oof_contrib

        # --- Row shifts + downsample ---
        clean_img = _post(clean_img, x_pos, y_off)
        if separate_focus:
            infocus_img = _post(infocus_img, x_pos, y_off)
            oof_img = _post(oof_img, x_pos, y_off)

        # --- Noise model ---
        samp_img = poisson_gauss_noise(clean_img, noise_params, rng)
        samp_img = pixel_bleed(samp_img, noise_params.bleedp,
                               noise_params.bleedw, rng)

        # --- Store ---
        h, w = samp_img.shape
        mov[:h, :w, kk] = samp_img
        mov_raw[:h, :w, kk] = clean_img
        if separate_focus:
            mov_infocus[:h, :w, kk] = infocus_img
            mov_oof[:h, :w, kk] = oof_img

        if verbose >= 2 and (kk + 1) % max(1, Nt // 10) == 0:
            print(f"    Frame {kk + 1}/{Nt}")

    if verbose >= 1:
        print(f"  Scanning completed. Output movie size: "
              f"[{mov.shape[0]} x {mov.shape[1]} x {mov.shape[2]}]")

    return ScanResult(
        mov=mov,
        mov_raw=mov_raw,
        mot_hist=mot_hist,
        params={
            "scan_params": scan_params,
            "noise_params": noise_params,
            "tpm_params": tpm_params,
        },
        mov_infocus=mov_infocus,
        mov_oof=mov_oof,
        motion_gt=dict(
            _mgt,
            model=np.array("twophoton"),
            sfrac=np.float32(sfrac),
            scan_buff=np.int32(scan_buff),
            motion_enabled=np.bool_(bool(mot_opt)),
        ),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _idx_to_2d(
    linear_idx: np.ndarray,
    N1: int,
    N2: int,
    N3: int,
) -> np.ndarray:
    """Project C-order 3-D linear indices to 2-D lateral (row, col) linear index.

    For C-order shape (N1, N2, N3): linear = i*N2*N3 + j*N3 + k
    2-D lateral index (into (N1, N2) mask) = i*N2 + j
    """
    ij = linear_idx // N3
    return ij

"""
Widefield (single-photon, camera-based) scanning simulation.

Companion to :mod:`calcia.scanning.scanning`. Runs Phase 4 when the
upstream optical propagation used ``PsfParams.imaging_mode='widefield'``.
See ``docs/widefield_vs_twophoton.md`` for the physics motivation.

Key differences from :func:`calcia.scanning.scan_volume` (two-photon):

* Signal is proportional to excitation power (linear); uses
  :func:`calcia.optics.widefield_signal_scale`.
* Image formation is a 2-D convolution summed over **all** z-planes
  (no optical sectioning). The widefield PSF already spans the full
  volume z-range, so a single call to :func:`single_scan` implements
  the formula ``I(x,y) = sum_z  f[:,:,z]  *_2D  h[:,:,z - z_focus]``.
* Per-frame rigid XY shift instead of line-by-line raster motion.
* Camera noise (:func:`camera_noise`) replaces the PMT
  Poisson/lognormal/Gauss chain. No pixel bleed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from scipy.signal import convolve2d

from ..config.params import CameraNoiseParams, MotionParams, ScanParams, WidefieldParams
from ..optics.signal import widefield_signal_scale
from .convolution import psf_fft, single_scan
from .motion import (apply_motion_blur, generate_motion_trajectory,
                     resolve_streak)
from .noise import camera_noise
from .scanning import ScanResult, _idx_to_2d

if TYPE_CHECKING:
    from ..config.params import SpikeParams
    from ..optics.propagation import OpticalPropagationResult
    from ..pipeline import NeuralVolumeOutput
    from ..traces.traces import TimeTracesResult


def scan_widefield(
    vol_out: "NeuralVolumeOutput",
    opt_out: "OpticalPropagationResult",
    time_out: "TimeTracesResult",
    scan_params: Optional[ScanParams] = None,
    cam_params: Optional[CameraNoiseParams] = None,
    wf_params: Optional[WidefieldParams] = None,
    spike_params: Optional["SpikeParams"] = None,
    motion_params: Optional[MotionParams] = None,
    *,
    seed: Optional[int] = None,
    separate_focus: bool = False,
    focus_slab_um: Optional[float] = None,
) -> ScanResult:
    """Simulate widefield camera-based imaging of a neural volume.

    Parameters
    ----------
    vol_out : NeuralVolumeOutput
        Phase 1 output.
    opt_out : OpticalPropagationResult
        Phase 2 output from the widefield path. Its ``psf`` must span the
        full z-extent of the volume.
    time_out : TimeTracesResult
        Phase 3 output.
    scan_params : ScanParams, optional
        Reuses two-photon scanning parameters (buffer, sfrac, motion flag).
    cam_params : CameraNoiseParams, optional
        Camera noise configuration.
    wf_params : WidefieldParams, optional
        Widefield signal-scaling parameters.
    spike_params : SpikeParams, optional
        Needed for ``dt``. Falls back to ``time_out.params['spike_params']``.
    seed : int, optional
        Random seed for reproducibility.
    separate_focus : bool, default False
        If ``True``, additionally return noiseless ``mov_infocus`` and
        ``mov_oof`` on the :class:`ScanResult`. Unlike the two-photon path this
        is NOT free: it costs one extra ``single_scan`` per frame. The split is
        exact by linearity (``mov_raw == mov_infocus + mov_oof``); noise is still
        applied only to the combined image. When ``False`` the code path is
        unchanged and both fields are ``None``.
    focus_slab_um : float, optional
        Full thickness (microns) of the in-focus slab, centered on the focal
        plane (the volume z-midpoint, where the emission PSF is sharpest).
        Volume z-planes inside the slab form ``mov_infocus``; everything else
        (the defocused background) forms ``mov_oof``. When ``None``, defaults to
        the widefield axial depth-of-field ``2*n*lambda_em/NA**2``.

    Returns
    -------
    ScanResult
        Simulated movie, clean movie, and (XY) motion history (Z row is 0).
    """
    if scan_params is None:
        scan_params = ScanParams()
    if cam_params is None:
        cam_params = CameraNoiseParams()
    if wf_params is None:
        wf_params = WidefieldParams()
    if spike_params is None:
        spike_params = time_out.params["spike_params"]

    rng = np.random.default_rng(seed)

    scan_buff = scan_params.scan_buff
    mot_opt = scan_params.motion
    sfrac = scan_params.sfrac
    verbose = scan_params.verbose

    # ------------------------------------------------------------------
    # Signal scaling: photons per fluorescence unit per frame.
    # Widefield integrates over a full pixel (no point-scan normalization),
    # so unlike the two-photon path there is no /(250*250) divisor.
    # ------------------------------------------------------------------
    sigscale = widefield_signal_scale(wf_params) * spike_params.dt * sfrac ** 2

    neur_vol_3d = vol_out.neur_vol
    N1, N2, N3 = neur_vol_3d.shape
    PSF = opt_out.psf
    Np1, Np2, Np3 = PSF.shape

    if N1 < Np1 or N2 < Np2:
        raise ValueError("Widefield PSF XY extent is bigger than the volume!")
    if Np3 != N3:
        raise ValueError(
            f"Widefield PSF z-depth ({Np3}) must equal volume z-depth ({N3}). "
            f"Re-run simulate_optical_propagation with imaging_mode='widefield'."
        )

    # ------------------------------------------------------------------
    # In-focus slab (only needed when separate_focus=True). single_scan aligns
    # PSF z-slice k 1:1 with volume z-plane k, so the slab must be centred on
    # whichever plane the emission PSF is sharpest at. That is the array centre
    # ONLY on the legacy path; when PsfParams.wf_focal_depth_um is set,
    # simulate_optical_propagation slices a double-height PSF so the sharp
    # slice lands at k_focus = round(wf_focal_depth_um * vres) instead.
    # ------------------------------------------------------------------
    zmask_in = None
    if separate_focus:
        psf_params = opt_out.params.get("psf_params") if opt_out.params else None
        vol_params = opt_out.params.get("vol_params") if opt_out.params else None
        vres = getattr(vol_params, "vres", 1.0) or 1.0
        if focus_slab_um is None:
            na = getattr(psf_params, "obj_na", 0.35) or 0.35
            n = getattr(psf_params, "n", 1.33) or 1.33
            lam = getattr(psf_params, "lambda_em_um", 0.52) or 0.52
            focus_slab_um = 2.0 * n * lam / (na ** 2)
        focal_um = getattr(psf_params, "wf_focal_depth_um", None)
        if focal_um is None:
            z_focus = N3 // 2                       # legacy: sharp at mid-depth
        else:
            z_focus = int(round(min(max(float(focal_um), 0.0),
                                    (N3 - 1) / vres) * vres))
        slab_half = int(round(0.5 * float(focus_slab_um) * vres))
        z_idx = np.arange(N3)
        zmask_in = (np.abs(z_idx - z_focus) <= slab_half)
        if verbose >= 1:
            depth_note = ("mid-depth" if focal_um is None
                          else f"focal_depth={float(focal_um):.0f} um")
            print(f"    - In-focus slab: z={z_focus}+/-{slab_half} voxels "
                  f"({focus_slab_um:.1f} um, {int(zmask_in.sum())}/{N3} planes, "
                  f"{depth_note})")

    # ------------------------------------------------------------------
    # Combined lateral mask (uniform illumination * collection)
    # ------------------------------------------------------------------
    t_mask = opt_out.mask * opt_out.col_mask
    t_mask = np.maximum(t_mask, 1e-5)

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

    nuc_label = scan_params.nuc_label >= 1
    if nuc_label:
        nuc_trace = soma_trace.copy()
        soma_trace = np.zeros_like(soma_trace)
        dend_trace = np.zeros_like(dend_trace)
        bg_trace = np.zeros_like(bg_trace)

    if verbose >= 1:
        print("  Initializing widefield scanning parameters...")
        print(f"    - Volume size: [{N1}, {N2}, {N3}] voxels")
        print(f"    - Number of frames: {Nt}")
        print(f"    - Motion simulation: {mot_opt}")
        print(f"    - Subsampling factor: {sfrac}")

    # ------------------------------------------------------------------
    # Pre-split component indexing with mask weighting
    # ------------------------------------------------------------------
    n_comp = len(vol_out.gp_vals)
    soma_idx_list: list = []
    soma_wt_list: list = []
    dend_idx_list: list = []
    dend_wt_list: list = []

    for i in range(n_comp):
        cfd = vol_out.gp_vals[i]
        s_mask = cfd.soma_mask

        s_idx = cfd.indices[s_mask]
        s_fl = cfd.fluorescence[s_mask].copy()
        d_idx = cfd.indices[~s_mask]
        d_fl = cfd.fluorescence[~s_mask].copy()

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

    axon_idx_list: list = []
    axon_wt_list: list = []
    if vol_out.bg_proc is not None:
        for bp in vol_out.bg_proc:
            a_idx = bp.indices
            a_fl = bp.fluorescence.copy()
            if len(a_idx) > 0:
                i2d = _idx_to_2d(a_idx, N1, N2, N3)
                a_fl *= t_mask.ravel()[i2d]
            axon_idx_list.append(a_idx)
            axon_wt_list.append(a_fl)

    nuc_idx_list: list = []
    nuc_wt_list: list = []
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
    # Baseline (f0) volume
    # ------------------------------------------------------------------
    f0vol = np.zeros((N1, N2, N3), dtype=np.float32)
    vol_flat = f0vol.ravel()

    for ll in range(min(len(soma_idx_list), len(soma_min))):
        idx = soma_idx_list[ll]
        if len(idx) > 0 and soma_min[ll] != 0:
            vol_flat[idx] = soma_wt_list[ll] * soma_min[ll]

    for ll in range(len(vol_out.gp_nuc)):
        nuc_idx, nuc_fl = vol_out.gp_nuc[ll]
        if len(nuc_idx) == 0 or nuc_fl == 0:
            continue
        if ll < len(soma_min):
            vol_flat[nuc_idx] = nuc_fl * soma_min[ll]

    for ll in range(min(len(dend_idx_list), len(dend_min))):
        idx = dend_idx_list[ll]
        if len(idx) > 0 and dend_min[ll] != 0:
            vol_flat[idx] = dend_wt_list[ll] * dend_min[ll]

    # Batched scatter-add of the (overlapping) axon/background baseline into
    # f0vol via one np.bincount instead of a per-component np.add.at loop.
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

    if nuc_label:
        for ll in range(len(nuc_idx_list)):
            idx = nuc_idx_list[ll]
            if len(idx) > 0 and nuc_min_v[ll] != 0:
                vol_flat[idx] = nuc_wt_list[ll] * nuc_min_v[ll]

    # ------------------------------------------------------------------
    # PSF FFT pre-computation (full volume z extent, no z-sub-summing)
    # ------------------------------------------------------------------
    freq_psf = psf_fft((N1, N2, N3), PSF, z_sub=1)

    # ------------------------------------------------------------------
    # Motion setup: per-frame rigid XY shift
    #   * legacy 'randomwalk' (default): bounded +/-1 voxel integer walk.
    #   * 'physio': AR(1) drift+jitter + heavy-tailed jumps + intra-frame
    #     motion blur, fit to real NoRMCorre shifts (see MotionParams).
    # ------------------------------------------------------------------
    use_physio = (motion_params is not None
                  and motion_params.model == "physio" and mot_opt)
    physio_traj = None
    if use_physio:
        _vp = opt_out.params.get("vol_params") if opt_out.params else None
        vres = float(getattr(_vp, "vres", 1.0) or 1.0)
        mrng = (np.random.default_rng(motion_params.seed)
                if motion_params.seed is not None else rng)
        physio_traj = generate_motion_trajectory(
            Nt, motion_params, vres, scan_buff, mrng)
        if verbose >= 1:
            _pk = np.abs(physio_traj).max(0)
            print(f"    - Motion: physio model, |shift| max "
                  f"[{_pk[0]:.1f}, {_pk[1]:.1f}] vox, "
                  f"blur={'on' if motion_params.blur else 'off'}")

    if mot_opt:
        xy_step = np.array([-1, 0, 0, 0, 0, 0, 1])
    else:
        xy_step = np.array([0])

    x_shift = 0
    y_shift = 0

    # ------------------------------------------------------------------
    # Output arrays (same shape convention as scan_volume)
    # ------------------------------------------------------------------
    out_h = N1 // sfrac - 2 * (scan_buff // sfrac)
    out_w = N2 // sfrac - 2 * (scan_buff // sfrac)
    mov = np.zeros((out_h, out_w, Nt), dtype=np.float32)
    mov_raw = np.zeros((out_h, out_w, Nt), dtype=np.float32)
    mot_hist = np.zeros((3, Nt), dtype=np.float32)
    # Per-frame intra-frame blur streak [dx, dy] in full-res voxels (physio
    # only). Records the displacement actually smeared into each frame; 0 for
    # frames where no blur was applied. Ground truth for de-blur / registration.
    blur_hist = (np.zeros((2, Nt), dtype=np.float32) if use_physio else None)
    # Complete, unreduced motion record (see motion.describe_motion_gt). Unlike
    # mot_hist/blur_hist this keeps the parts the rendering pipeline discards:
    # the sub-voxel rounding residual and the threshold/clamp-corrected streak.
    _mgt = dict(
        shift_requested=np.zeros((2, Nt), dtype=np.float32),
        shift_applied=np.zeros((2, Nt), dtype=np.float32),
        blur_requested=np.zeros((2, Nt), dtype=np.float32),
        blur_applied=np.zeros((2, Nt), dtype=np.float32),
        blur_skipped=np.zeros(Nt, dtype=bool),
        blur_clipped=np.zeros(Nt, dtype=bool),
    )
    mov_infocus = (np.zeros((out_h, out_w, Nt), dtype=np.float32)
                   if separate_focus else None)
    mov_oof = (np.zeros((out_h, out_w, Nt), dtype=np.float32)
               if separate_focus else None)

    sfrac_int = int(sfrac) == sfrac

    def _post(img, x_shift, y_shift):
        """Rigid XY shift/crop + downsample (no RNG side effects).

        Identical to the inline ``clean_img`` post-processing so the in-focus /
        out-of-focus components stay aligned with ``mov_raw`` and the default
        path is numerically unchanged.
        """
        img = _rigid_shift_and_crop(img, scan_buff, x_shift, y_shift)
        if sfrac_int:
            s = int(sfrac)
            img = convolve2d(img, np.ones((s, s), dtype=np.float32),
                             mode='same')
            img = img[::s, ::s]
        else:
            from scipy.ndimage import zoom
            img = sfrac ** 2 * zoom(img, 1.0 / sfrac, order=1)
        return img

    if verbose >= 1:
        print("  Running widefield acquisition...")

    n_soma = min(len(soma_idx_list), soma_act.shape[0])
    n_dend = min(len(dend_idx_list), dend_act.shape[0])
    n_bg = min(len(axon_idx_list), bg_act.shape[0])
    n_nuc = len(nuc_idx_list) if nuc_label else 0

    for kk in range(Nt):
        # --- Per-frame rigid XY shift + intra-frame blur velocity ---
        blur_dx = blur_dy = 0.0
        if use_physio:
            xf, yf = float(physio_traj[kk, 0]), float(physio_traj[kk, 1])
            if kk > 0:
                blur_dx = (xf - float(physio_traj[kk - 1, 0]))
                blur_dy = (yf - float(physio_traj[kk - 1, 1]))
                # The sample travelled this far during the exposure whether or
                # not the blur stage is enabled -- record it unconditionally.
                _mgt["blur_requested"][:, kk] = [
                    blur_dx * motion_params.exposure_frac,
                    blur_dy * motion_params.exposure_frac]
            x_shift = int(round(xf))
            y_shift = int(round(yf))
            mot_hist[:, kk] = [xf, yf, 0]
            # mot_hist keeps the float request, but the pixels move by the
            # ROUNDED shift -- record both so the <=0.5 voxel residual survives.
            _mgt["shift_requested"][:, kk] = [xf, yf]
        else:
            # legacy bounded +/-1 voxel Brownian walk
            x_shift = int(np.clip(x_shift + rng.choice(xy_step),
                                  -scan_buff, scan_buff))
            y_shift = int(np.clip(y_shift + rng.choice(xy_step),
                                  -scan_buff, scan_buff))
            mot_hist[:, kk] = [x_shift, y_shift, 0]
            # Integer walk: request and application coincide exactly.
            _mgt["shift_requested"][:, kk] = [x_shift, y_shift]
        _mgt["shift_applied"][:, kk] = [x_shift, y_shift]

        # --- Build transient activity volume ---
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

        # Axon/background voxels OVERLAP (many processes share a voxel) so they
        # must be scatter-ADDED, not assigned. np.add.at is an unbuffered scatter
        # and dominated Phase 4 (~62%): batch all active components and do ONE
        # buffered np.bincount instead (identical result, ~an order faster).
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

        # --- Full-volume widefield convolution (sum over all z) ---
        scan_vol = tmp_vol + f0vol
        if separate_focus:
            # Partition the depth sum inside the transform: the per-z products
            # are already there, so the split costs one extra 2-D inverse FFT
            # rather than a second full scan. Exact by linearity.
            all_img, in_img = single_scan(scan_vol, PSF.shape, freq_psf, 1,
                                          z_groups=(slice(None), zmask_in))
            clean_img = sigscale * all_img
            infocus_img = sigscale * in_img
            oof_img = clean_img - infocus_img
        else:
            clean_img = sigscale * single_scan(scan_vol, PSF.shape, freq_psf, 1)

        # --- Intra-frame motion blur (physio model): the sample moves DURING
        # the exposure, so a fast frame is smeared along the motion direction.
        # Applied to the full-res clean image (before downsample); the streak
        # length is the intra-frame displacement in voxels. Convolution is
        # linear so the in-focus/out-of-focus split stays consistent. ---
        if use_physio and motion_params.blur and (blur_dx or blur_dy):
            bx = blur_dx * motion_params.exposure_frac
            by = blur_dy * motion_params.exposure_frac
            _mx = motion_params.blur_max_px * sfrac
            _mn = motion_params.blur_min_px * sfrac
            # Record the streak actually rendered into this frame (0 stays if
            # below _mn, where apply_motion_blur is a no-op) as ground truth.
            if np.hypot(bx, by) >= _mn:
                blur_hist[:, kk] = [bx, by]
            # blur_hist stores the UN-clamped request; motion_gt keeps both the
            # request and what motion_streak_kernel actually renders.
            _ax, _ay, _skip, _clip = resolve_streak(bx, by, max_len=_mx,
                                                    min_len=_mn)
            _mgt["blur_applied"][:, kk] = [_ax, _ay]
            _mgt["blur_skipped"][kk] = _skip
            _mgt["blur_clipped"][kk] = _clip
            clean_img = apply_motion_blur(clean_img, bx, by,
                                          max_len=_mx, min_len=_mn)
            if separate_focus:
                infocus_img = apply_motion_blur(infocus_img, bx, by,
                                                max_len=_mx, min_len=_mn)
                oof_img = clean_img - infocus_img

        # --- Rigid XY shift/crop + downsample ---
        clean_img = _post(clean_img, x_shift, y_shift)
        if separate_focus:
            infocus_img = _post(infocus_img, x_shift, y_shift)
            oof_img = _post(oof_img, x_shift, y_shift)

        # --- Camera noise ---
        samp_img = camera_noise(clean_img, cam_params, rng)

        h, w = samp_img.shape
        mov[:h, :w, kk] = samp_img
        mov_raw[:h, :w, kk] = clean_img
        if separate_focus:
            mov_infocus[:h, :w, kk] = infocus_img
            mov_oof[:h, :w, kk] = oof_img

        if verbose >= 2 and (kk + 1) % max(1, Nt // 10) == 0:
            print(f"    Frame {kk + 1}/{Nt}")

    if verbose >= 1:
        print(
            f"  Widefield scanning completed. Movie shape: "
            f"[{mov.shape[0]} x {mov.shape[1]} x {mov.shape[2]}]"
        )

    params: Dict = {
        "scan_params": scan_params,
        "cam_params": cam_params,
        "wf_params": wf_params,
        "motion_params": motion_params,
    }
    _mgt["shift_residual"] = (_mgt["shift_requested"]
                              - _mgt["shift_applied"]).astype(np.float32)
    _mgt.update(
        model=np.array("physio" if use_physio else "randomwalk"),
        sfrac=np.float32(sfrac),
        vres=np.float32(vres if use_physio else np.nan),
        scan_buff=np.int32(scan_buff),
        blur_enabled=np.bool_(bool(use_physio and motion_params.blur)),
        exposure_frac=np.float32(motion_params.exposure_frac
                                 if use_physio else np.nan),
    )
    return ScanResult(mov=mov, mov_raw=mov_raw, mot_hist=mot_hist, params=params,
                      mov_infocus=mov_infocus, mov_oof=mov_oof,
                      blur_hist=blur_hist, motion_gt=_mgt)


def _rigid_shift_and_crop(
    img: np.ndarray,
    buf: int,
    x_shift: int,
    y_shift: int,
) -> np.ndarray:
    """Apply an integer rigid shift and crop to the inner FOV.

    The input ``img`` has the full volume XY shape. The output is
    ``img[buf + x_shift : -buf + x_shift, buf + y_shift : -buf + y_shift]``
    (with edge clamping). Shift is the apparent displacement of the sample
    relative to the camera.
    """
    H, W = img.shape
    row_lo = buf + x_shift
    row_hi = H - buf + x_shift
    col_lo = buf + y_shift
    col_hi = W - buf + y_shift
    row_lo = max(0, min(H - 1, row_lo))
    row_hi = max(1, min(H, row_hi))
    col_lo = max(0, min(W - 1, col_lo))
    col_hi = max(1, min(W, col_hi))
    return img[row_lo:row_hi, col_lo:col_hi].astype(np.float32)

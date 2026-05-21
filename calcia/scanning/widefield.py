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

from ..config.params import CameraNoiseParams, ScanParams, WidefieldParams
from ..optics.signal import widefield_signal_scale
from .convolution import psf_fft, single_scan
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
    *,
    seed: Optional[int] = None,
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

    for ll in range(min(len(axon_idx_list), len(bg_min))):
        idx = axon_idx_list[ll]
        if len(idx) > 0 and bg_min[ll] != 0:
            np.add.at(vol_flat, idx, axon_wt_list[ll] * bg_min[ll])

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
    # ------------------------------------------------------------------
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

    sfrac_int = int(sfrac) == sfrac

    if verbose >= 1:
        print("  Running widefield acquisition...")

    n_soma = min(len(soma_idx_list), soma_act.shape[0])
    n_dend = min(len(dend_idx_list), dend_act.shape[0])
    n_bg = min(len(axon_idx_list), bg_act.shape[0])
    n_nuc = len(nuc_idx_list) if nuc_label else 0

    for kk in range(Nt):
        # --- Per-frame rigid XY shift (Brownian walk clipped to buffer) ---
        x_shift = int(np.clip(x_shift + rng.choice(xy_step),
                              -scan_buff, scan_buff))
        y_shift = int(np.clip(y_shift + rng.choice(xy_step),
                              -scan_buff, scan_buff))
        mot_hist[:, kk] = [x_shift, y_shift, 0]

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

        for ll in range(n_bg):
            a = bg_act[ll, kk]
            if a > 0 and len(axon_idx_list[ll]) > 0:
                np.add.at(tmp_flat, axon_idx_list[ll],
                          axon_wt_list[ll] * a)

        # --- Full-volume widefield convolution (sum over all z) ---
        scan_vol = tmp_vol + f0vol
        clean_img = sigscale * single_scan(scan_vol, PSF.shape, freq_psf, 1)

        # --- Rigid per-frame XY shift + crop to inner FOV ---
        clean_img = _rigid_shift_and_crop(clean_img, scan_buff,
                                          x_shift, y_shift)

        # --- Downsample by sfrac ---
        if sfrac_int:
            s = int(sfrac)
            clean_img = convolve2d(
                clean_img, np.ones((s, s), dtype=np.float32), mode='same',
            )
            clean_img = clean_img[::s, ::s]
        else:
            from scipy.ndimage import zoom
            clean_img = sfrac ** 2 * zoom(clean_img, 1.0 / sfrac, order=1)

        # --- Camera noise ---
        samp_img = camera_noise(clean_img, cam_params, rng)

        h, w = samp_img.shape
        mov[:h, :w, kk] = samp_img
        mov_raw[:h, :w, kk] = clean_img

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
    }
    return ScanResult(mov=mov, mov_raw=mov_raw, mot_hist=mot_hist, params=params)


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

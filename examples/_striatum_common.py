"""Shared helpers + calibrated defaults for the striatum widefield scripts.

Single source of truth for the 1P widefield striatum simulation parameters
and the rendering/IO helpers reused across:
  - demo_widefield_striatum_v1.py
  - striatum_demix_dataset.py
  - striatum_param_sweep.py

Calibration notes baked into the defaults:
  - bg_scale=0.1 : dense axon/neuropil is the dominant 1P wash-out source;
    0.1 takes brightest-frame spatial CV ~0.65 -> ~0.95 (cells visible).
  - smod="burst", burst_mean=0, rate=0.02 : decorrelated, stationary
    ~0.57 Hz activity suitable for demixing ground truth.

Heavy deps (calcia, tifffile, matplotlib) are imported lazily inside the
functions so `import _striatum_common` stays cheap (fast --help, etc).
"""
import pickle

import numpy as np


# ----------------------------------------------------------------------
# Calibrated parameter factories (return fresh instances; no shared state)
# ----------------------------------------------------------------------
def striatum_psf(obj_na=0.8, scatter_length_um_wf=None):
    from calcia.config.params import PsfParams
    kw = dict(imaging_mode="widefield", psf_type="gaussian_analytical",
              lambda_em_um=0.52, obj_na=obj_na, n=1.35,
              psf_sz=(12.0, 12.0, 20.0))
    if scatter_length_um_wf is not None:
        kw["scatter_length_um_wf"] = scatter_length_um_wf
    return PsfParams(**kw)


def striatum_spike(K, nt, dt, has_axons, *, rate=0.02, prot="GCaMP6f",
                   smod="burst", burst_mean=0, bg_scale=0.1, verbose=1):
    from calcia.config.params import SpikeParams
    return SpikeParams(K=K, nt=nt, dt=dt, N_bg=0, axonflag=has_axons,
                       rate=rate, prot=prot, smod_flag=smod,
                       burst_mean=burst_mean, bg_scale=bg_scale,
                       verbose=verbose)


def striatum_scan(verbose=1):
    from calcia.config.params import ScanParams
    return ScanParams(scan_buff=10, motion=True, sfrac=2, verbose=verbose)


def striatum_wf():
    from calcia.config.params import WidefieldParams
    return WidefieldParams(pavg=2.0, lambda_ex_um=0.488, qe_det=0.8)


def striatum_cam(dt):
    from calcia.config.params import CameraNoiseParams
    return CameraNoiseParams(qe=1.0, dark_rate=0.3, t_exp=dt,
                             read_noise=1.6, gain_e_per_adu=1.0)


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------
def load_phase1(cache_path):
    """Load a cached Phase-1 (vol_out, vol_params) tuple."""
    with open(cache_path, "rb") as f:
        vol_out, vol_params = pickle.load(f)
    return vol_out, vol_params


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def brightest_frame(mov):
    """Index of the frame with the highest mean intensity. mov: (H,W,T)."""
    return int(mov.reshape(-1, mov.shape[2]).mean(0).argmax())


def cv_bright(mov):
    """Spatial coefficient of variation of the brightest frame."""
    fr = mov[:, :, brightest_frame(mov)]
    return float(fr.std() / (fr.mean() + 1e-12))


def dF(mov, pct=10):
    """dF over a static per-pixel baseline (pct-th temporal percentile)."""
    f0 = np.percentile(mov, pct, axis=2, keepdims=True)
    return mov - f0


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def save_tif(arr, path, clip0=False):
    """Save (H,W,T) as a self-normalized 16-bit TIFF stack (0.5/99.5 pct)."""
    import tifffile
    a = arr.astype(np.float64)
    if clip0:
        a = np.clip(a, 0, None)
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = (np.clip(a, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(str(path), np.transpose(a, (2, 0, 1)), imagej=True)


# Backwards-compatible alias (demo used this name).
save_tiff_normalized = save_tif


def make_video(mov_noisy, mov_clean, path, dt, contrast=0.995, fps=30):
    """Side-by-side noisy|clean GIF (each self-normalized)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    nt = mov_noisy.shape[2]
    vmin_n = np.percentile(mov_noisy, (1 - contrast) * 100)
    vmax_n = np.percentile(mov_noisy, contrast * 100)
    vmin_c = np.percentile(mov_clean, (1 - contrast) * 100)
    vmax_c = np.percentile(mov_clean, contrast * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.90, bottom=0.02)
    im1 = ax1.imshow(mov_noisy[:, :, 0], cmap="gray",
                     vmin=vmin_n, vmax=vmax_n, aspect="equal")
    im2 = ax2.imshow(mov_clean[:, :, 0], cmap="gray",
                     vmin=vmin_c, vmax=vmax_c, aspect="equal")
    ax1.set_title("Noisy (widefield)", fontsize=10)
    ax2.set_title("Clean (widefield)", fontsize=10)
    ax1.axis("off"); ax2.axis("off")
    time_text = fig.suptitle("t = 0.000 s", fontsize=11)

    def update(frame):
        im1.set_data(mov_noisy[:, :, frame])
        im2.set_data(mov_clean[:, :, frame])
        time_text.set_text(f"t = {frame * dt:.3f} s")
        return im1, im2, time_text

    anim = FuncAnimation(fig, update, frames=nt, blit=True, interval=1)
    anim.save(str(path), writer="pillow", fps=fps)
    plt.close(fig)

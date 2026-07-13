"""Rendering + lightweight IO for scanned movies.

Save a scanned ``(H, W, T)`` movie as a self-normalised 16-bit TIFF stack, build
a side-by-side noisy|clean preview GIF, and load a cached Phase-1
``(vol_out, vol_params)`` tuple. Heavy deps (tifffile, matplotlib) are imported
lazily so ``import calcia.io`` stays cheap.
"""
import pickle

import numpy as np


def load_phase1(cache_path):
    """Load a cached Phase-1 (vol_out, vol_params) tuple."""
    with open(cache_path, "rb") as f:
        vol_out, vol_params = pickle.load(f)
    return vol_out, vol_params


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

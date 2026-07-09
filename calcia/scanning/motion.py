"""
Motion simulation and per-row image shifting for scanning.

Port of MATLAB: ``imgSubRowShift.m`` and inline motion code in
``scan_volume.m``.

Also provides the calcia-original realistic sample-motion model
(:func:`generate_motion_trajectory`) and intra-frame motion blur
(:func:`motion_streak_kernel`, :func:`apply_motion_blur`) used by the widefield
scanner when ``MotionParams.model == 'physio'``. See ``MotionParams`` for the
physics; the parameters were fit to real NoRMCorre rigid shifts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..config.params import MotionParams


def generate_motion_trajectory(
    nt: int,
    motion_params: "MotionParams",
    vres: float,
    scan_buff: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a per-frame rigid XY shift trajectory (physio model).

    Returns an ``(nt, 2)`` float array of ``(x, y)`` shifts in **voxels**
    (full-resolution grid units), matching the sign/units of the legacy
    integer random-walk trajectory so the scanner can consume either.

    The model is a per-axis AR(1) drift+jitter process plus rare heavy-tailed
    jumps, clipped to the crop margin::

        s_t = phi * s_{t-1} + sigma * eps_t   (+ jump with prob jump_prob)

    ``sigma`` and the jump/bound magnitudes are given in microns in
    ``MotionParams`` and converted to voxels here via ``vres``.
    """
    phi = float(motion_params.ar_phi)
    sig_um = np.asarray(motion_params.sigma_um, dtype=np.float64).ravel()
    if sig_um.size == 1:
        sig_um = np.array([sig_um[0], sig_um[0]])
    sigma_vox = sig_um[:2] * vres
    jump_sigma_vox = float(motion_params.jump_sigma_um) * vres

    # Clip bound: min(requested um bound, crop margin scan_buff voxels).
    if motion_params.bound_um is not None:
        bound_vox = min(float(motion_params.bound_um) * vres, float(scan_buff))
    else:
        bound_vox = float(scan_buff)

    traj = np.zeros((nt, 2), dtype=np.float64)
    s = np.zeros(2, dtype=np.float64)
    for t in range(nt):
        s = phi * s + sigma_vox * rng.standard_normal(2)
        if motion_params.jump_prob > 0 and rng.random() < motion_params.jump_prob:
            s = s + jump_sigma_vox * rng.standard_normal(2)
        s = np.clip(s, -bound_vox, bound_vox)
        traj[t] = s
    return traj.astype(np.float32)


def motion_streak_kernel(
    dx: float,
    dy: float,
    max_len: float = 40.0,
    min_len: float = 0.75,
) -> Optional[np.ndarray]:
    """Normalized line (streak) kernel for intra-frame motion blur.

    ``dx, dy`` is the intra-frame displacement (pixels) the sample travels
    during the exposure. Returns a small 2-D kernel (sum 1) that, when
    convolved with a sharp frame, smears it along that direction, or ``None``
    when the streak is below ``min_len`` (negligible).
    """
    L = float(np.hypot(dx, dy))
    if L < min_len:
        return None
    L = min(L, float(max_len))
    ux, uy = dx / np.hypot(dx, dy), dy / np.hypot(dx, dy)

    n = max(2, int(np.ceil(L)) + 1)
    ts = np.linspace(-0.5, 0.5, n) * L            # positions along the streak
    xs, ys = ts * ux, ts * uy                     # (col, row) offsets, px

    R = int(np.ceil(L / 2.0)) + 1
    size = 2 * R + 1
    ker = np.zeros((size, size), dtype=np.float64)
    # bilinear splat each sample point into the kernel centred at (R, R)
    for x, y in zip(xs, ys):
        cx, cy = R + x, R + y
        x0, y0 = int(np.floor(cx)), int(np.floor(cy))
        fx, fy = cx - x0, cy - y0
        for (yy, wy) in ((y0, 1 - fy), (y0 + 1, fy)):
            for (xx, wx) in ((x0, 1 - fx), (x0 + 1, fx)):
                if 0 <= yy < size and 0 <= xx < size:
                    ker[yy, xx] += wy * wx
    tot = ker.sum()
    if tot <= 0:
        return None
    return (ker / tot).astype(np.float32)


def apply_motion_blur(
    img: np.ndarray,
    dx: float,
    dy: float,
    max_len: float = 40.0,
    min_len: float = 0.75,
) -> np.ndarray:
    """Convolve ``img`` with the intra-frame motion streak for (dx, dy) px.

    No-op (returns ``img`` unchanged) when the streak is negligible.
    """
    ker = motion_streak_kernel(dx, dy, max_len=max_len, min_len=min_len)
    if ker is None:
        return img
    from scipy.ndimage import convolve
    return convolve(img.astype(np.float32), ker, mode="nearest").astype(np.float32)


def apply_row_shifts(
    img: np.ndarray,
    buf_sz: int,
    x_off: float,
    y_off: np.ndarray,
) -> np.ndarray:
    """Extract a sub-image with per-row sub-pixel shifts.

    Port of MATLAB ``imgSubRowShift.m``.

    Parameters
    ----------
    img : np.ndarray
        Input 2-D image of shape ``(H, W)``.
    buf_sz : int
        Buffer width (pixels cropped from each edge).
    x_off : float
        Row offset (same value for every row, or broadcast).
    y_off : np.ndarray
        1-D array of per-row column offsets (length = output height).

    Returns
    -------
    img_out : np.ndarray
        Shifted and cropped image, float32.
    """
    h, w = img.shape
    n_rows = len(y_off)

    # x_off and y_off are relative to buf_sz in MATLAB
    x_off_adj = x_off - buf_sz
    y_off_adj = y_off - buf_sz

    # Absolute row positions: x_off_adj + row_index (0-based)
    x_pos = x_off_adj + np.arange(n_rows, dtype=np.float64)

    # --- Row extraction with sub-pixel x-interpolation ---
    img_tmp = np.full((n_rows, w), np.nan, dtype=np.float32)
    for k in range(n_rows):
        xp = x_pos[k]
        frac = xp - np.floor(xp)
        r0 = int(np.floor(xp)) - 1  # MATLAB 1-based → Python 0-based
        r1 = int(np.ceil(xp)) - 1

        if abs(frac) < 1e-12:
            # Integer row
            r = int(round(xp)) - 1
            if 0 <= r < h:
                img_tmp[k, :] = img[r, :]
        else:
            row0 = img[r0, :] if 0 <= r0 < h else np.full(w, np.nan, dtype=np.float32)
            row1 = img[r1, :] if 0 <= r1 < h else np.full(w, np.nan, dtype=np.float32)
            img_tmp[k, :] = row0 * (1 - frac) + row1 * frac

    # --- Column shifting with sub-pixel y-interpolation ---
    offset = int(np.ceil(np.max(np.abs(y_off_adj)))) if len(y_off_adj) > 0 else 0
    # Pad columns with NaN on both sides
    padded = np.full((n_rows, w + 2 * offset), np.nan, dtype=np.float32)
    padded[:, offset:offset + w] = img_tmp

    img_out = np.zeros((n_rows, w), dtype=np.float32)
    for k in range(n_rows):
        yo = y_off_adj[k]
        frac = yo - np.floor(yo)
        base = int(np.floor(yo)) + offset  # index into padded
        row1 = padded[k, base:base + w]
        row2 = padded[k, base + 1:base + 1 + w] if (base + w) < padded.shape[1] else np.full(w, np.nan, dtype=np.float32)
        if abs(frac) < 1e-12:
            img_out[k, :] = row1
        else:
            img_out[k, :] = row1 * (1 - frac) + row2 * frac

    # Crop buffer from all edges
    img_out = img_out[buf_sz:n_rows - buf_sz, buf_sz:w - buf_sz]
    return img_out.astype(np.float32)

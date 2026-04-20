"""
Motion simulation and per-row image shifting for scanning.

Port of MATLAB: ``imgSubRowShift.m`` and inline motion code in
``scan_volume.m``.
"""

from __future__ import annotations

import numpy as np


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

"""Unified loaders for segmentation-algorithm outputs.

Every algorithm is normalised to an :class:`AlgoResult`:

  * ``centroids`` — ``(n, 2)`` weighted footprint centroids as ``(col, row)`` in
    the movie's ``(H, W)`` pixel frame, *before* the GT->frame calibration
    (offset / axis swap) which :mod:`matching` resolves.
  * ``masks`` — list of length ``n``; each is a 1-D array of flat pixel indices
    ``row * W + col`` (``None`` if the algorithm did not persist footprints).
  * ``traces`` — ``(n, T)`` temporal traces (``None`` if not persisted).

Native-1p algorithms with usable footprints+traces: DeepWonder, CNMF-E,
MIN1PIPE.  SUNS2 / DeepCaImX persisted counts only -> ``count_only``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class AlgoResult:
    name: str
    centroids: np.ndarray            # (n, 2) col,row
    masks: Optional[List[np.ndarray]]  # per-neuron flat pixel idx, or None
    traces: Optional[np.ndarray]     # (n, T) or None
    H: int
    W: int
    count_only: bool = False
    meta: dict = None

    @property
    def n(self) -> int:
        if self.centroids is not None and len(self.centroids):
            return len(self.centroids)
        return int((self.meta or {}).get("n", 0))


def _centroids_from_sparse(A_rows: sp.csr_matrix, W: int, thresh_frac: float = 0.0):
    """A_rows: csr (n, H*W). Returns (centroids col,row), masks (flat idx)."""
    n = A_rows.shape[0]
    cen = np.zeros((n, 2))
    masks = []
    for i in range(n):
        s, e = A_rows.indptr[i], A_rows.indptr[i + 1]
        idx = A_rows.indices[s:e]
        w = A_rows.data[s:e].astype(float)
        if len(idx) == 0:
            masks.append(idx)
            continue
        if thresh_frac > 0:
            keep = w >= thresh_frac * w.max()
            idx, w = idx[keep], w[keep]
        masks.append(idx)
        cen[i, 0] = ((idx % W) * w).sum() / w.sum()
        cen[i, 1] = ((idx // W) * w).sum() / w.sum()
    return cen, masks


def _load_footprints_npz(path, H, W):
    """DeepWonder / CNMF-E footprint npz: scipy-sparse or manual csr fields."""
    try:
        A = sp.load_npz(path).tocsr()
    except Exception:
        f = np.load(path)
        A = sp.csr_matrix((f["data"], f["indices"], f["indptr"]),
                          shape=(len(f["indptr"]) - 1, H * W))
    if A.shape[0] == H * W:
        A = A.T.tocsr()
    return A


def load_deepwonder(seg_dir: str, H=820, W=820, name=None) -> AlgoResult:
    import scipy.io as sio
    A = _load_footprints_npz(os.path.join(seg_dir, "footprints_A.npz"), H, W)
    cen, masks = _centroids_from_sparse(A, W)
    C = sio.loadmat(os.path.join(seg_dir, "infer_results.mat"))["C"]
    return AlgoResult(name or "DeepWonder/" + os.path.basename(seg_dir),
                      cen, masks, np.asarray(C, float), H, W)


def load_cnmfe(cfg_dir: str, H=820, W=820, name=None, thresh_frac=0.2) -> AlgoResult:
    A = _load_footprints_npz(os.path.join(cfg_dir, "A.npz"), H, W)
    cen, masks = _centroids_from_sparse(A, W, thresh_frac=thresh_frac)
    C = np.load(os.path.join(cfg_dir, "C.npy"))
    return AlgoResult(name or "CNMFE/" + os.path.basename(cfg_dir),
                      cen, masks, np.asarray(C, float), H, W)


def load_min1pipe(cfg_dir: str, H=820, W=820, name=None, thresh_frac=0.2) -> AlgoResult:
    """MIN1PIPE summary.mat (v7.3). Footprints at pixh*pixw (downsampled);
    upsampled to (H,W) by filling each source pixel as a 2x2 block."""
    import h5py
    with h5py.File(os.path.join(cfg_dir, "summary.mat"), "r") as f:
        roifn = f["roifn"][()]
        sigfn = f["sigfn"][()]
        pixh = int(np.array(f["pixh"]).ravel()[0])
        pixw = int(np.array(f["pixw"]).ravel()[0])
    P = pixh * pixw
    if roifn.shape[0] == P:
        roifn = roifn.T
    if sigfn.shape[0] != roifn.shape[0]:
        sigfn = sigfn.T
    up_r, up_c = H / pixh, W / pixw
    n = roifn.shape[0]
    cen = np.zeros((n, 2))
    masks = []
    for i in range(n):
        fpi = roifn[i].reshape(pixh, pixw, order="F")
        m = fpi >= thresh_frac * fpi.max() if fpi.max() > 0 else fpi > 0
        r, c = np.where(m)
        if len(r) == 0:
            masks.append(np.array([], int))
            continue
        # fill each downsampled source pixel as a 2x2 block in the full frame
        r0 = np.floor(r * up_r); c0 = np.floor(c * up_c)
        rr = np.concatenate([r0, r0 + 1, r0, r0 + 1]).astype(int)
        cc = np.concatenate([c0, c0, c0 + 1, c0 + 1]).astype(int)
        rr = np.clip(rr, 0, H - 1)
        cc = np.clip(cc, 0, W - 1)
        masks.append(rr * W + cc)
        cen[i, 0] = cc.mean()
        cen[i, 1] = rr.mean()
    return AlgoResult(name or "MIN1PIPE/" + os.path.basename(cfg_dir),
                      cen, masks, np.asarray(sigfn, float), H, W)


def load_count_only(name: str, n: int, **meta) -> AlgoResult:
    meta = dict(meta); meta["n"] = int(n)
    return AlgoResult(name, np.empty((0, 2)), None, None, 820, 820,
                      count_only=True, meta=meta)

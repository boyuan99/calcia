"""Calibrate the GT->movie-frame mapping and match components to neurons.

The scan maps a neuron at volume ``(x,y)`` to pixel ``(x*vres-buff)/sfrac`` but
(a) which axis is the movie row vs column and (b) the residual motion-correction
offset differ per result, so we *calibrate* both by maximising the number of
component centroids that land near a GT soma, then do a greedy one-to-one match
(each component owns its nearest still-free neuron within ``radius_px``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .gt import GroundTruth
from .loaders import AlgoResult


@dataclass
class Calibration:
    swap: bool
    dc: float
    dr: float
    n_matched_at_calib: int

    def gt_px(self, gt: GroundTruth) -> np.ndarray:
        """GT soma centres placed into this result's (col, row) pixel frame."""
        s = gt.vres / gt.sfrac
        a = gt.locs_um[:, 0] * s
        b = gt.locs_um[:, 1] * s
        col = (b if self.swap else a) + self.dc
        row = (a if self.swap else b) + self.dr
        return np.column_stack([col, row])


@dataclass
class Matching:
    calib: Calibration
    gt_px: np.ndarray             # (N,2) col,row
    owner: np.ndarray             # (n_comp,) GT index each comp owns, -1 if none
    gt_comp: np.ndarray           # (N,) comp index owning each GT, -1 if none
    detected: np.ndarray          # (N,) bool = gt_comp>=0
    match_dist: np.ndarray        # (N,) px distance to owning comp, nan if none


def calibrate(gt: GroundTruth, res: AlgoResult, radius: float = 3.0,
              coarse=range(-40, 41, 2)) -> Calibration:
    cen = res.centroids
    s = gt.vres / gt.sfrac
    a = gt.locs_um[:, 0] * s
    b = gt.locs_um[:, 1] * s
    best = None
    for swap in (False, True):
        gcol = b if swap else a
        grow = a if swap else b
        tree = cKDTree(np.column_stack([gcol, grow]))
        for dc in coarse:
            for dr in coarse:
                d, _ = tree.query(np.column_stack([cen[:, 0] - dc, cen[:, 1] - dr]),
                                  distance_upper_bound=radius)
                nm = int(np.isfinite(d).sum())
                if best is None or nm > best[0]:
                    best = (nm, swap, dc, dr)
    _, swap, dc, dr = best
    gcol = b if swap else a
    grow = a if swap else b
    tree = cKDTree(np.column_stack([gcol, grow]))
    for ddc in np.arange(dc - 2, dc + 2.01, 0.5):
        for ddr in np.arange(dr - 2, dr + 2.01, 0.5):
            d, _ = tree.query(np.column_stack([cen[:, 0] - ddc, cen[:, 1] - ddr]),
                              distance_upper_bound=radius)
            nm = int(np.isfinite(d).sum())
            if nm > best[0]:
                best = (nm, swap, ddc, ddr)
    return Calibration(bool(best[1]), float(best[2]), float(best[3]), int(best[0]))


def match(gt: GroundTruth, res: AlgoResult, calib: Calibration,
          radius: float = 4.0) -> Matching:
    gt_px = calib.gt_px(gt)
    cen = res.centroids
    tree = cKDTree(gt_px)
    dd, gg = tree.query(cen, distance_upper_bound=radius)
    order = np.argsort(np.where(np.isfinite(dd), dd, np.inf))
    N = gt.n
    n = len(cen)
    gt_comp = np.full(N, -1, int)
    owner = np.full(n, -1, int)
    match_dist = np.full(N, np.nan)
    for i in order:
        if not np.isfinite(dd[i]):
            break
        g = gg[i]
        if gt_comp[g] == -1:
            gt_comp[g] = i
            owner[i] = g
            match_dist[g] = dd[i]
    return Matching(calib, gt_px, owner, gt_comp, gt_comp >= 0, match_dist)

"""Which ground-truth neurons are prone to being merged / confused.

Two neurons get conflated by a segmentation algorithm when their somata sit
within roughly one spatial footprint of each other *and both* carry enough
signal to matter.  This module builds a **confusability graph** over the
bright, infected, in-FOV neurons:

  * an edge (i, j) exists when the lateral soma distance < ``radius_um`` and
    both cells are above a brightness percentile;
  * edges are optionally weighted by trace correlation (temporally similar
    neighbours are the hardest to demix and the most likely to leak signal);
  * connected components are "confusable groups" — sets of cells an algorithm
    is at risk of merging into a single ROI.

Per neuron it reports a merge-risk ``score`` and a ``contamination`` estimate
(how much brighter-neighbour signal would leak into its trace if merged),
which downstream-impact analysis uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .gt import GroundTruth
from .detectability import Detectability


@dataclass
class ConfusabilityConfig:
    radius_um: float = 10.0            # merge if soma centres closer than this
    brightness_percentile: float = 50.0   # only cells this bright (infected in-FOV pool) can confuse
    use_trace_similarity: bool = True


@dataclass
class Confusability:
    radius_um: float
    bright: np.ndarray            # (N,) bool, cells eligible to confuse
    n_neighbors: np.ndarray       # (N,) # of bright neighbours within radius
    score: np.ndarray             # (N,) [0,1] merge-risk (isolated=0)
    contamination: np.ndarray     # (N,) brightness-weighted neighbour leakage potential
    group_id: np.ndarray          # (N,) connected-component id among bright cells (-1 = isolated/dim)
    group_sizes: np.ndarray       # sizes indexed by group_id
    pairs: np.ndarray             # (M,2) confusable pair indices (into N)
    pair_dist_um: np.ndarray      # (M,)
    pair_trace_corr: np.ndarray   # (M,)
    cfg: ConfusabilityConfig

    def summary(self) -> str:
        nb = self.bright.sum()
        in_grp = (self.group_id >= 0) & (self.n_neighbors > 0)
        big = self.group_sizes[self.group_sizes >= 2]
        return (f"Confusability (radius={self.radius_um:g} um, {int(nb)} bright cells):\n"
                f"  cells with >=1 bright neighbour: {int((self.n_neighbors>0).sum())} "
                f"({100*(self.n_neighbors>0).sum()/max(nb,1):.1f}% of bright)\n"
                f"  confusable groups (>=2 cells): {int(len(big))} "
                f"covering {int(big.sum())} cells; largest = {int(big.max()) if len(big) else 0}\n"
                f"  confusable pairs: {len(self.pairs)}")


def _pair_corr(traces, i, j):
    a = traces[i] - traces[i].mean()
    b = traces[j] - traces[j].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def analyze(gt: GroundTruth, det: Detectability,
            cfg: ConfusabilityConfig | None = None) -> Confusability:
    cfg = cfg or ConfusabilityConfig()
    N = gt.n
    pool = det.infected & det.in_fov
    thr = np.percentile(det.optical_brightness[pool], cfg.brightness_percentile) if pool.sum() else np.inf
    bright = pool & (det.optical_brightness >= thr)
    idx = np.where(bright)[0]

    n_neigh = np.zeros(N, int)
    contam = np.zeros(N, float)
    group_id = np.full(N, -1, int)
    pairs = np.empty((0, 2), int)
    pdist = np.empty(0)
    pcorr = np.empty(0)
    group_sizes = np.zeros(0, int)

    if len(idx) >= 2:
        xy = gt.locs_um[idx, :2]
        tree = cKDTree(xy)
        pair_list = tree.query_pairs(cfg.radius_um, output_type="ndarray")  # local indices into idx
        ob = det.optical_brightness
        # graph adjacency among bright cells
        rows, cols = [], []
        P = []
        for a_loc, b_loc in pair_list:
            gi, gj = idx[a_loc], idx[b_loc]
            n_neigh[gi] += 1
            n_neigh[gj] += 1
            # contamination: fraction of the (self+neighbour) light that is the neighbour's
            contam[gi] += ob[gj] / (ob[gi] + ob[gj] + 1e-12)
            contam[gj] += ob[gi] / (ob[gi] + ob[gj] + 1e-12)
            rows += [a_loc, b_loc]
            cols += [b_loc, a_loc]
            d = float(np.hypot(*(xy[a_loc] - xy[b_loc])))
            c = _pair_corr(gt.traces, gi, gj) if cfg.use_trace_similarity else 0.0
            P.append((gi, gj, d, c))
        if P:
            pairs = np.array([[p[0], p[1]] for p in P], int)
            pdist = np.array([p[2] for p in P])
            pcorr = np.array([p[3] for p in P])
        # connected components
        m = len(idx)
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m)) if rows else csr_matrix((m, m))
        ncomp, labels = connected_components(A, directed=False)
        group_id[idx] = labels
        group_sizes = np.bincount(labels, minlength=ncomp)

    # per-neuron merge-risk score: neighbours count normalised, dim/isolated = 0
    score = np.zeros(N)
    if n_neigh.max() > 0:
        score = np.clip(n_neigh / np.percentile(n_neigh[n_neigh > 0], 95), 0, 1)
    return Confusability(radius_um=cfg.radius_um, bright=bright, n_neighbors=n_neigh,
                         score=score, contamination=contam, group_id=group_id,
                         group_sizes=group_sizes, pairs=pairs, pair_dist_um=pdist,
                         pair_trace_corr=pcorr, cfg=cfg)

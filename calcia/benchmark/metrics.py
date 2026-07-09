"""GT-anchored detection / trace-fidelity / separation metrics.

Detection is judged **two ways**, kept explicit because they answer different
questions:

  * *spatial* — a neuron is detected if a component centroid matched it
    (position only);
  * *temporally-gated* — additionally the recovered trace must correlate with
    the neuron's true trace above a threshold (position AND signal).

Recall is reported against the *detectable* pool (see
:mod:`calcia.benchmark.detectability`) so uninfected / invisible cells do not
unfairly deflate it, and is broken down by detectability category.

Separation quantifies under-segmentation: a mask that covers >= 2 bright
neurons has *merged* them; its single trace can match at most one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .gt import GroundTruth
from .detectability import Detectability, CATEGORIES
from .confusability import Confusability
from .loaders import AlgoResult
from .matching import Matching


def _pearson_rows(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / (np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1)) + 1e-12)


@dataclass
class SegMetrics:
    name: str
    n_detected: int
    precision: float
    # recall on the detectable pool, spatial and temporally gated
    recall_spatial: float
    recall_corr05: float
    recall_corr07: float
    n_detectable: int
    # trace fidelity on matched detectable cells
    median_corr: float
    frac_corr_gt07: float
    trace_valid_frac: float          # recall_corr07 / recall_spatial
    # separation / under-segmentation
    merge_rate: float                # frac masks covering >=2 bright cells
    frac_bright_not_separated: float
    median_mask_area_px: float
    # per-category spatial recall
    recall_by_category: Dict[str, float] = field(default_factory=dict)
    gt_corr: np.ndarray = None       # (N,) per-GT trace corr with owner (nan if unmatched)

    def summary(self) -> str:
        rc = "  ".join(f"{k}={v:.2f}" for k, v in self.recall_by_category.items()
                       if k in ("hard", "detectable", "easy"))
        return (f"[{self.name}] n={self.n_detected} prec={self.precision:.2f} | "
                f"recall(detectable): spatial={self.recall_spatial:.2f} "
                f"@0.5={self.recall_corr05:.2f} @0.7={self.recall_corr07:.2f} "
                f"(trace-valid {self.trace_valid_frac:.0%}) | "
                f"corr={self.median_corr:.2f} | merge={self.merge_rate:.2f} "
                f"area={self.median_mask_area_px:.0f}px\n    recall by cat: {rc}")


def compute(gt: GroundTruth, res: AlgoResult, det: Detectability,
            conf: Confusability, mt: Matching,
            radius: float = 4.0) -> SegMetrics:
    N = gt.n
    n = len(res.centroids)
    gt_px = mt.gt_px

    # precision: fraction of components landing within radius of an infected GT
    from scipy.spatial import cKDTree
    inf_idx = np.where(det.infected)[0]
    tree = cKDTree(gt_px[inf_idx]) if len(inf_idx) else None
    if tree is not None and n:
        d, _ = tree.query(res.centroids, distance_upper_bound=radius)
        precision = float(np.isfinite(d).mean())
    else:
        precision = float("nan")

    # per-GT trace corr with owning component
    gt_corr = np.full(N, np.nan)
    if res.traces is not None:
        matched = np.where(mt.gt_comp >= 0)[0]
        if len(matched):
            comps = mt.gt_comp[matched]
            gt_corr[matched] = _pearson_rows(res.traces[comps], gt.traces[matched])

    dpool = det.detectable
    ndet = int(dpool.sum())

    def rec(mask):
        return float((mt.detected & dpool & mask).sum() / max(ndet, 1))
    recall_spatial = float(mt.detected[dpool].mean()) if ndet else float("nan")
    recall_c05 = rec(gt_corr > 0.5)
    recall_c07 = rec(gt_corr > 0.7)

    mc = gt_corr[mt.detected & dpool & ~np.isnan(gt_corr)]
    median_corr = float(np.median(mc)) if len(mc) else float("nan")
    frac07 = float((mc > 0.7).mean()) if len(mc) else float("nan")
    tv = float(recall_c07 / recall_spatial) if recall_spatial > 0 else float("nan")

    # per-category spatial recall
    rbc = {}
    for c in CATEGORIES:
        m = det.category == c
        rbc[c] = float(mt.detected[m].mean()) if m.sum() else float("nan")

    # separation via mask coverage of bright cells
    merge_rate = not_sep = med_area = float("nan")
    if res.masks is not None and n:
        H, W = res.H, res.W
        lab = np.zeros(H * W, np.int32)
        for i, idx in enumerate(res.masks):
            if len(idx):
                lab[idx] = i + 1
        rr = np.clip(np.round(gt_px[:, 1]).astype(int), 0, H - 1)
        cc = np.clip(np.round(gt_px[:, 0]).astype(int), 0, W - 1)
        gpc = lab[rr * W + cc]
        bright = conf.bright
        cov = np.zeros(n + 1, int)
        for g in np.where(bright)[0]:
            if gpc[g] > 0:
                cov[gpc[g]] += 1
        merge_rate = float((cov[1:] >= 2).sum() / n)
        cb = np.where(bright & (gpc > 0))[0]
        share = sum(1 for g in cb if cov[gpc[g]] >= 2)
        not_sep = float(share / max(len(cb), 1))
        med_area = float(np.median([len(m) for m in res.masks if len(m)]))

    return SegMetrics(
        name=res.name, n_detected=n, precision=precision,
        recall_spatial=recall_spatial, recall_corr05=recall_c05, recall_corr07=recall_c07,
        n_detectable=ndet, median_corr=median_corr, frac_corr_gt07=frac07,
        trace_valid_frac=tv, merge_rate=merge_rate,
        frac_bright_not_separated=not_sep, median_mask_area_px=med_area,
        recall_by_category=rbc, gt_corr=gt_corr,
    )

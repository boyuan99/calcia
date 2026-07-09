"""What unclean (contaminated / merged) traces cost downstream analysis.

Two downstream uses are simulated on the matched detectable cells, comparing
the algorithm's recovered traces against the neurons' true traces:

1. **Functional connectivity** — the pairwise trace-correlation matrix is the
   substrate for most population analyses.  Signal leakage between neighbours
   inflates correlations and invents functional "edges" that are not real.  We
   compare the recovered correlation matrix to the ground-truth one and, for
   physically confusable pairs, measure how much their correlation is inflated.

2. **Event / transient detection** — spikes inferred from a contaminated trace
   inherit a neighbour's transients (false positives) or lose their own to the
   mixture.  We detect events on GT vs recovered traces and score them.

Also reports per-cell amplitude bias (contamination lifts baseline / rescales
dF/F).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .gt import GroundTruth
from .detectability import Detectability
from .confusability import Confusability
from .loaders import AlgoResult
from .matching import Matching


def _zscore(x):
    med = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=-1, keepdims=True) * 1.4826 + 1e-9
    return (x - med) / mad


def detect_events(traces, z_thr=3.0):
    """Rising-edge peaks above z_thr (robust z). Returns list of event-frame arrays."""
    Z = _zscore(np.atleast_2d(traces).astype(float))
    out = []
    for z in Z:
        above = z > z_thr
        # local peak among supra-threshold frames
        peak = above.copy()
        peak[1:-1] &= (z[1:-1] >= z[:-2]) & (z[1:-1] >= z[2:])
        out.append(np.where(peak)[0])
    return out


def _event_f1(gt_ev, al_ev, tol=2):
    if len(gt_ev) == 0 and len(al_ev) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_ev) == 0 or len(al_ev) == 0:
        return 0.0, 0.0, 0.0
    used = np.zeros(len(al_ev), bool)
    tp = 0
    for g in gt_ev:
        cand = np.where((~used) & (np.abs(al_ev - g) <= tol))[0]
        if len(cand):
            used[cand[np.argmin(np.abs(al_ev[cand] - g))]] = True
            tp += 1
    prec = tp / len(al_ev)
    rec = tp / len(gt_ev)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _corr_matrix(T):
    T = T - T.mean(1, keepdims=True)
    n = np.sqrt((T ** 2).sum(1)) + 1e-12
    return (T @ T.T) / np.outer(n, n)


@dataclass
class DownstreamImpact:
    n_cells: int                      # matched detectable cells used
    # connectivity
    corr_matrix_rmse: float           # off-diagonal RMSE(recovered - GT)
    corr_offdiag_r: float             # corr between recovered & GT off-diagonals
    spurious_edge_rate: float         # frac pairs GT<0.2 but recovered>0.5
    confusable_pair_inflation: float  # median (recovered-GT) corr on confusable pairs
    n_confusable_pairs_used: int
    # events
    event_f1_median: float
    event_precision_median: float
    event_recall_median: float
    # signal quality (scale-invariant: SNR = peak / robust-noise)
    snr_ratio_median: float           # recovered SNR / GT SNR
    snr_recovered_median: float
    snr_gt_median: float
    extras: Dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Downstream impact on {self.n_cells} matched detectable cells:\n"
                f"  connectivity: off-diag RMSE={self.corr_matrix_rmse:.3f}, "
                f"recovered-vs-GT off-diag r={self.corr_offdiag_r:.2f}, "
                f"spurious edges={self.spurious_edge_rate:.1%}\n"
                f"  confusable pairs (n={self.n_confusable_pairs_used}): "
                f"median corr inflation={self.confusable_pair_inflation:+.2f}\n"
                f"  event detection: F1={self.event_f1_median:.2f} "
                f"(P={self.event_precision_median:.2f} R={self.event_recall_median:.2f})\n"
                f"  SNR recovered={self.snr_recovered_median:.1f} vs GT={self.snr_gt_median:.1f} "
                f"(ratio={self.snr_ratio_median:.2f})")


def assess(gt: GroundTruth, res: AlgoResult, det: Detectability,
           conf: Confusability, mt: Matching, z_thr: float = 3.0,
           event_tol: int = 2) -> DownstreamImpact:
    if res.traces is None:
        raise ValueError("algorithm has no traces; downstream impact needs them")
    # matched detectable cells
    cells = np.where(mt.detected & det.detectable)[0]
    comps = mt.gt_comp[cells]
    G = gt.traces[cells].astype(float)
    A = res.traces[comps].astype(float)

    # scale-invariant signal quality: SNR = peak / robust noise std
    def snr(x):
        med = np.median(x, axis=1, keepdims=True)
        noise = np.median(np.abs(x - med), axis=1) * 1.4826 + 1e-9
        return (np.percentile(x, 99, axis=1) - med[:, 0]) / noise
    snr_g = snr(G); snr_a = snr(A)
    snr_ratio = snr_a / (snr_g + 1e-9)

    # events
    gev = detect_events(G, z_thr)
    aev = detect_events(A, z_thr)
    f1s, ps, rs = [], [], []
    for ge, ae in zip(gev, aev):
        p, r, f = _event_f1(ge, ae, event_tol)
        ps.append(p); rs.append(r); f1s.append(f)

    # connectivity matrices
    Rg = _corr_matrix(G)
    Ra = _corr_matrix(A)
    iu = np.triu_indices(len(cells), k=1)
    og, oa = Rg[iu], Ra[iu]
    rmse = float(np.sqrt(np.mean((oa - og) ** 2)))
    offr = float(np.corrcoef(og, oa)[0, 1]) if len(og) > 2 else float("nan")
    spurious = float(np.mean((og < 0.2) & (oa > 0.5))) if len(og) else float("nan")

    # confusable-pair inflation: pairs (from conf) whose BOTH cells are in `cells`
    pos = {int(g): k for k, g in enumerate(cells)}
    infl = []
    for a, b in conf.pairs:
        if a in pos and b in pos:
            ia, ib = pos[a], pos[b]
            infl.append(Ra[ia, ib] - Rg[ia, ib])
    infl = np.array(infl)

    return DownstreamImpact(
        n_cells=len(cells),
        corr_matrix_rmse=rmse, corr_offdiag_r=offr, spurious_edge_rate=spurious,
        confusable_pair_inflation=float(np.median(infl)) if len(infl) else float("nan"),
        n_confusable_pairs_used=int(len(infl)),
        event_f1_median=float(np.median(f1s)) if f1s else float("nan"),
        event_precision_median=float(np.median(ps)) if ps else float("nan"),
        event_recall_median=float(np.median(rs)) if rs else float("nan"),
        snr_ratio_median=float(np.median(snr_ratio)) if len(snr_ratio) else float("nan"),
        snr_recovered_median=float(np.median(snr_a)) if len(snr_a) else float("nan"),
        snr_gt_median=float(np.median(snr_g)) if len(snr_g) else float("nan"),
        extras=dict(event_f1=np.array(f1s), snr_ratio=snr_ratio,
                    offdiag_gt=og, offdiag_alg=oa, confusable_inflation=infl),
    )

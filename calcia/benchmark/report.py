"""Orchestration: characterise a run, evaluate algorithm results, emit report.

Public entry points:

  * :func:`characterize_run` -> ``(gt, det, conf)`` for a simulation run dir.
  * :func:`evaluate_result`  -> a flat metrics dict for one loaded result.
  * :func:`discover`         -> auto-load a benchmark ``results/`` tree.
  * :func:`run_benchmark`    -> full pipeline: GT characterisation + every
    result + figures + ``BENCHMARK_REPORT.md`` + ``summary.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import List, Optional

import numpy as np

from .gt import GroundTruth
from .detectability import (characterize, Detectability, DetectabilityConfig,
                            CATEGORIES)
from .confusability import analyze, Confusability
from .loaders import (AlgoResult, load_deepwonder, load_cnmfe, load_min1pipe,
                      load_count_only)
from . import matching, metrics, downstream


def characterize_run(run_dir: str, cfg: DetectabilityConfig | None = None):
    """``(gt, det, conf)`` for a run. ``cfg`` picks the detectability criterion;
    the default one needs a cached footprint render in the run directory."""
    gt = GroundTruth.from_run(run_dir)
    det = characterize(gt, cfg)
    conf = analyze(gt, det)
    return gt, det, conf


def evaluate_result(gt: GroundTruth, det: Detectability, conf: Confusability,
                    res: AlgoResult, want_downstream: bool = True) -> dict:
    if res.count_only:
        return {"name": res.name, "n_detected": int(res.n), "count_only": True}
    cal = matching.calibrate(gt, res)
    mt = matching.match(gt, res, cal)
    m = metrics.compute(gt, res, det, conf, mt)
    rec = {k: v for k, v in asdict(m).items() if k != "gt_corr"}
    rec["calib"] = {"swap": cal.swap, "dc": cal.dc, "dr": cal.dr,
                    "n_matched": cal.n_matched_at_calib}
    if want_downstream and res.traces is not None:
        di = downstream.assess(gt, res, det, conf, mt)
        rec["downstream"] = {k: v for k, v in asdict(di).items() if k != "extras"}
    return rec


def discover(results_dir: str) -> List[AlgoResult]:
    """Load every result found under a benchmark ``results/`` tree."""
    out = []
    dw = os.path.join(results_dir, "DeepWonder")
    if os.path.isdir(dw):
        for d in sorted(os.listdir(dw)):
            if d.startswith("SEG_") and os.path.isfile(os.path.join(dw, d, "footprints_A.npz")):
                out.append(load_deepwonder(os.path.join(dw, d)))
    cn = os.path.join(results_dir, "CNMFE")
    if os.path.isdir(cn):
        for d in sorted(os.listdir(cn)):
            if os.path.isfile(os.path.join(cn, d, "A.npz")):
                out.append(load_cnmfe(os.path.join(cn, d)))
    mp = os.path.join(results_dir, "MIN1PIPE")
    if os.path.isdir(mp):
        for d in sorted(os.listdir(mp)):
            if os.path.isfile(os.path.join(mp, d, "summary.mat")):
                out.append(load_min1pipe(os.path.join(mp, d)))
    # count-only (masks not persisted)
    for algo in ("SUNS2", "DeepCaImX"):
        p = os.path.join(results_dir, algo, "movie_result.npz")
        if os.path.isfile(p):
            d = np.load(p, allow_pickle=True)
            if "raw" not in d.files or "n" in d.files:  # no masks -> count only
                out.append(load_count_only(algo, int(d["n"])))
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _colors():
    return {"DeepWonder": "#2a7fff", "CNMFE": "#ff7f0e", "MIN1PIPE": "#2ca02c",
            "SUNS2": "#9467bd", "DeepCaImX": "#8c564b"}


def fig_detectability(gt: GroundTruth, det: Detectability, conf: Confusability,
                      path: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    base = gt.base_px()
    catcol = {"uninfected": "#333", "out_of_fov": "#777", "invisible": "#d62728",
              "hard": "#ff7f0e", "detectable": "#2ca02c", "easy": "#17becf"}
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    inf = det.in_fov
    for c in CATEGORIES:
        m = (det.category == c) & inf
        if m.sum():
            ax[0].scatter(base[m, 0], base[m, 1], s=5, c=catcol[c], label=f"{c} ({m.sum()})",
                          alpha=.5, edgecolors="none")
    ax[0].set_title("Detectability category (in-FOV)"); ax[0].invert_yaxis()
    ax[0].legend(fontsize=7, markerscale=2); ax[0].set_aspect("equal")
    # depth vs brightness colored by category
    pool = det.infected & det.in_fov
    sc = ax[1].scatter(det.depth_um[pool], det.optical_brightness[pool], s=4,
                       c=det.score[pool], cmap="viridis", alpha=.5)
    ax[1].set_xlabel("depth z (um)"); ax[1].set_ylabel("optical brightness"); ax[1].set_yscale("log")
    ax[1].set_title("Optical brightness vs depth (color=detectability score)")
    plt.colorbar(sc, ax=ax[1], fraction=.046)
    # confusable group sizes
    gs = conf.group_sizes[conf.group_sizes >= 2]
    ax[2].hist(gs, bins=np.arange(1.5, max(6, gs.max() if len(gs) else 3) + 1.5),
               rwidth=.85, color="#c44")
    ax[2].set_xlabel("cells per confusable group"); ax[2].set_ylabel("# groups")
    ax[2].set_title(f"Merge-prone groups (r={conf.radius_um:g}um): "
                    f"{int(len(gs))} groups, {int(gs.sum()) if len(gs) else 0} cells")
    plt.suptitle("Ground-truth detectability & confusability", fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def fig_comparison(records: List[dict], path: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    col = _colors()
    R = [r for r in records if not r.get("count_only")]
    if not R:
        return
    labels = [r["name"].split("/")[-1] for r in R]
    algos = [r["name"].split("/")[0] for r in R]
    cols = [col.get(a, "#888") for a in algos]
    x = np.arange(len(R))
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    # recall by category (hard/detectable/easy) stacked as grouped
    cats = ["hard", "detectable", "easy"]
    w = 0.26
    for j, c in enumerate(cats):
        vals = [r["recall_by_category"].get(c, np.nan) for r in R]
        ax[0].bar(x + (j - 1) * w, vals, w, label=c, alpha=.85)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("spatial recall"); ax[0].set_title("Recall by detectability category")
    ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3)
    # tradeoff scatter: strict recall vs corr, ring~merge, size~n
    for r in R:
        a = r["name"].split("/")[0]
        ax[1].scatter(r["recall_corr07"], r["median_corr"], s=40 + r["n_detected"] / 6,
                      c=col.get(a, "#888"), alpha=.6, edgecolors="k",
                      linewidths=.5 + 3 * (r.get("merge_rate") or 0))
        ax[1].annotate(r["name"].split("/")[-1], (r["recall_corr07"], r["median_corr"]),
                       fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("strict recall (spatial AND corr>0.7)"); ax[1].set_ylabel("median trace corr")
    ax[1].set_title("Recall vs fidelity (size~#det, ring~merge)"); ax[1].grid(alpha=.3)
    ax[1].legend(handles=[Patch(color=c, label=a) for a, c in col.items() if a in algos], fontsize=8)
    # downstream: connectivity off-diag r + confusable inflation
    haved = [r for r in R if "downstream" in r]
    if haved:
        xd = np.arange(len(haved))
        offr = [r["downstream"]["corr_offdiag_r"] for r in haved]
        infl = [r["downstream"]["confusable_pair_inflation"] for r in haved]
        ax[2].bar(xd - .2, offr, .4, label="connectivity fidelity (off-diag r)", color="#3a7")
        ax[2].bar(xd + .2, infl, .4, label="confusable-pair corr inflation", color="#c44")
        ax[2].set_xticks(xd); ax[2].set_xticklabels([r["name"].split("/")[-1] for r in haved],
                                                    rotation=60, ha="right", fontsize=7)
        ax[2].axhline(0, color="k", lw=.6); ax[2].legend(fontsize=8)
        ax[2].set_title("Downstream: connectivity fidelity vs spurious coupling")
    plt.suptitle("Cross-algorithm segmentation benchmark (GT-anchored)", fontsize=13)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def _fmt_table(records):
    hdr = ("| algo/config | #det | prec | R@spatial | R@0.7 | trace-valid | corr | "
           "merge | area px | conn.r | infl |")
    sep = "|" + "|".join(["---"] * 11) + "|"
    rows = [hdr, sep]
    for r in records:
        if r.get("count_only"):
            rows.append(f"| {r['name']} | {r['n_detected']} | (count only) |||||||||")
            continue
        d = r.get("downstream", {})
        rows.append(
            f"| {r['name']} | {r['n_detected']} | {r['precision']:.2f} | "
            f"{r['recall_spatial']:.2f} | {r['recall_corr07']:.2f} | "
            f"{r['trace_valid_frac']:.0%} | {r['median_corr']:.2f} | "
            f"{r['merge_rate']:.2f} | {r['median_mask_area_px']:.0f} | "
            f"{d.get('corr_offdiag_r', float('nan')):.2f} | "
            f"{d.get('confusable_pair_inflation', float('nan')):+.2f} |")
    return "\n".join(rows)


def run_benchmark(gt_run_dir: str, results, out_dir: str,
                  make_figures: bool = True) -> dict:
    """Full pipeline. ``results`` is a list of AlgoResult (e.g. from :func:`discover`)."""
    os.makedirs(out_dir, exist_ok=True)
    gt, det, conf = characterize_run(gt_run_dir)
    records = [evaluate_result(gt, det, conf, r) for r in results]

    if make_figures:
        fig_detectability(gt, det, conf, os.path.join(out_dir, "fig_detectability.png"))
        fig_comparison(records, os.path.join(out_dir, "fig_comparison.png"))

    summary = {
        "gt_run": gt_run_dir,
        "detectability_counts": det.counts,
        "n_detectable_pool": int(det.detectable.sum()),
        "confusability": {"radius_um": conf.radius_um,
                          "n_confusable_pairs": int(len(conf.pairs)),
                          "n_groups_ge2": int((conf.group_sizes >= 2).sum())},
        "results": records,
    }
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=float)

    md = ["# Segmentation benchmark vs ground truth", "",
          f"GT run: `{os.path.basename(gt_run_dir)}`", "",
          "## Ground-truth detectability", "```", det.summary(), conf.summary(), "```", "",
          "## Algorithm comparison", "",
          "Recall is on the **detectable pool** "
          f"({int(det.detectable.sum())} infected, bright, in-FOV cells). "
          "`trace-valid` = strict/spatial recall; `conn.r` = connectivity fidelity "
          "(recovered vs GT off-diagonal correlation); `infl` = spurious correlation "
          "inflation on confusable pairs.", "",
          _fmt_table(records), "",
          "See `fig_detectability.png` and `fig_comparison.png`."]
    open(os.path.join(out_dir, "BENCHMARK_REPORT.md"), "w", encoding="utf-8").write("\n".join(md))
    return summary

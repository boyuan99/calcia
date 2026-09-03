"""Command-line entry point for the segmentation benchmark.

    python -m calcia.benchmark --gt <run_dir> --results <results_dir> --out <out_dir>

``--gt`` is a calcia simulation run directory (has traces.npz / params.pkl /
optics.npz).  ``--results`` is a benchmark ``results/`` tree with
DeepWonder/CNMFE/MIN1PIPE (+ count-only SUNS2/DeepCaImX) sub-folders.  If
``--results`` is omitted, only the ground-truth characterisation is produced.
"""

from __future__ import annotations

import argparse
import os


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calcia.benchmark",
                                 description="Evaluate spatio-temporal segmentation vs calcia GT")
    ap.add_argument("--gt", required=True, help="simulation run directory (ground truth)")
    ap.add_argument("--results", help="benchmark results/ tree to evaluate")
    ap.add_argument("--out", help="output directory (default: <results>/analysis or <gt>/seg_eval)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--radius-um", type=float, default=10.0, help="confusability merge radius (um)")
    ap.add_argument("--criterion", default=None,
                    choices=["identifiability", "percentile", "absolute_snr", "functional"],
                    help="detectability standard (default: identifiability, which "
                         "needs a cached footprints_all.npz in the run directory)")
    args = ap.parse_args(argv)

    from . import report
    from .confusability import ConfusabilityConfig, analyze
    from .detectability import DetectabilityConfig

    cfg = DetectabilityConfig(criterion=args.criterion) if args.criterion else None

    if not args.results:
        gt, det, conf = report.characterize_run(args.gt, cfg)
        print(det.summary()); print(conf.summary())
        return

    out = args.out or os.path.join(args.results, "analysis")
    results = report.discover(args.results)
    print(f"discovered {len(results)} results:", ", ".join(r.name for r in results))
    # allow overriding the confusability radius via a re-characterise
    gt, det, _ = report.characterize_run(args.gt, cfg)
    conf = analyze(gt, det, ConfusabilityConfig(radius_um=args.radius_um))
    records = [report.evaluate_result(gt, det, conf, r) for r in results]
    os.makedirs(out, exist_ok=True)
    if not args.no_figures:
        report.fig_detectability(gt, det, conf, os.path.join(out, "fig_detectability.png"))
        report.fig_comparison(records, os.path.join(out, "fig_comparison.png"))
    import json
    summary = {"gt_run": args.gt, "detectability_counts": det.counts,
               "n_detectable_pool": int(det.detectable.sum()), "results": records}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    open(os.path.join(out, "BENCHMARK_REPORT.md"), "w", encoding="utf-8").write(
        report._fmt_table(records))
    for r in records:
        if r.get("count_only"):
            print(f"[{r['name']}] n={r['n_detected']} (count only)")
        else:
            print(f"[{r['name']}] n={r['n_detected']} R@spatial={r['recall_spatial']:.2f} "
                  f"R@0.7={r['recall_corr07']:.2f} corr={r['median_corr']:.2f} "
                  f"merge={r['merge_rate']:.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()

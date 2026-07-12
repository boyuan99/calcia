"""Demo: evaluate spatio-temporal segmentation algorithms against calcia GT.

Runs the full :mod:`calcia.benchmark` pipeline on a simulation run and a tree of
algorithm results:

  1. characterise the ground truth — which neurons are detectable (by depth /
     brightness / AAV expression / illumination) and which are merge-prone;
  2. evaluate each algorithm — detection (spatial + trace-gated), fidelity,
     under-segmentation;
  3. assess downstream damage — functional-connectivity distortion and spurious
     coupling from contaminated traces;
  4. write figures + ``BENCHMARK_REPORT.md`` + ``summary.json``.

Usage (paths default to the striatum benchmark bundle):

    conda run -n calcia python examples/demo_seg_benchmark.py \
        --gt   examples/output/striatum_v1_1700um_physio-motion_20260706_114842 \
        --results data/algo_benchmark_striatum_v1_1700um_physio-motion/results
"""

from __future__ import annotations

import argparse
import os

from calcia.benchmark import report

DEFAULT_GT = "examples/output/striatum_v1_1700um_physio-motion_20260706_114842"
DEFAULT_RESULTS = "data/algo_benchmark_striatum_v1_1700um_physio-motion/results"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "seg_eval")

    gt, det, conf = report.characterize_run(args.gt)
    print("=" * 70)
    print(det.summary())
    print(conf.summary())
    print("=" * 70)

    results = report.discover(args.results)
    print(f"Evaluating {len(results)} algorithm results...\n")
    summary = report.run_benchmark(args.gt, results, out)

    for r in summary["results"]:
        if r.get("count_only"):
            print(f"  {r['name']:26s} n={r['n_detected']:5d}  (count only)")
            continue
        d = r.get("downstream", {})
        print(f"  {r['name']:26s} n={r['n_detected']:5d}  "
              f"R@spatial={r['recall_spatial']:.2f} R@0.7={r['recall_corr07']:.2f}  "
              f"corr={r['median_corr']:.2f}  merge={r['merge_rate']:.2f}  "
              f"conn.r={d.get('corr_offdiag_r', float('nan')):.2f}  "
              f"infl={d.get('confusable_pair_inflation', float('nan')):+.2f}")
    print(f"\nWrote report + figures to: {out}")


if __name__ == "__main__":
    import _instrument; _instrument.start()  # run log + pyinstrument (mandated)
    main()

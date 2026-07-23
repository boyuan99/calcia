"""Backfill / rebuild the viz_cache bundle for EXISTING run directories, without
re-scanning. viz_cache is built by ``calcia.viz.prep.prep_run`` purely from a
run's already-saved artifacts (metadata.json, cell_footprints.pkl, traces.npz),
so any finished run can get its interactive-viz bundle after the fact — e.g. runs
that were scanned with ``--no-viz`` during fast parameter tuning.

Usage:
  python examples/rebuild_viz.py RUN_DIR [RUN_DIR ...]   # specific runs
  python examples/rebuild_viz.py --all                   # every run under output/ missing viz_cache
  python examples/rebuild_viz.py --all --force           # rebuild even if it already exists
"""
import argparse
import glob
import os
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def has_bundle(run_dir):
    """A run can be viz-built only if its saved footprint bundle is present."""
    return os.path.isfile(os.path.join(run_dir, "cell_footprints.pkl"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", help="run directories to (re)build")
    ap.add_argument("--all", action="store_true",
                    help="all run dirs under examples/output/ that have a saved bundle")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if viz_cache already exists")
    args = ap.parse_args()

    candidates = list(args.run_dirs)
    if args.all:
        candidates += [d for d in glob.glob(os.path.join(OUT, "*"))
                       if os.path.isdir(d)]

    seen, targets = set(), []
    for d in candidates:
        d = os.path.abspath(d)
        if d in seen:
            continue
        seen.add(d)
        name = os.path.basename(d)
        if not os.path.isdir(d):
            print(f"skip (not a dir): {d}")
            continue
        if not has_bundle(d):
            print(f"skip (no cell_footprints.pkl, nothing to build from): {name}")
            continue
        if os.path.isdir(os.path.join(d, "viz_cache")) and not args.force:
            print(f"skip (viz_cache exists; --force to rebuild): {name}")
            continue
        targets.append(d)

    if not targets:
        print("nothing to do.")
        return

    from calcia.viz.prep import prep_run
    print(f"building viz_cache for {len(targets)} run(s)...")
    ok = bad = 0
    for d in targets:
        print(f"\n=== {os.path.basename(d)} ===")
        try:
            prep_run(d, verbose=True)
            ok += 1
        except Exception:
            bad += 1
            traceback.print_exc()
    print(f"\ndone: {ok} built, {bad} failed.")


if __name__ == "__main__":
    main()

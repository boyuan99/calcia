"""Side-by-side comparison of Python vs MATLAB overlap stats.

Usage as a CLI::

    python -m calcia.diagnostics.compare_with_matlab \\
        path/to/matlab_overlap_stats.mat           # compare against current Python run
    python -m calcia.diagnostics.compare_with_matlab \\
        path/to/matlab_overlap_stats.mat \\
        --python-mat path/to/python_overlap_stats.mat

Or programmatically::

    from calcia.diagnostics.overlap import summarize
    from calcia.diagnostics.compare_with_matlab import compare
    py_report = summarize(vol_out)
    compare(py_report, matlab_mat_path='matlab_overlap_stats.mat')

The MATLAB file must have been saved by ``measure_overlap.m`` (variable
``stats``). The goal is to determine whether any Python/MATLAB divergence
in the overlap numbers points to a port regression; order-of-magnitude
agreement counts as "shared model behavior".
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

import numpy as np

from .overlap import OverlapReport


def load_matlab_stats(path: str) -> Dict[str, Any]:
    """Load a ``stats`` struct written by ``measure_overlap.m``."""
    from scipy.io import loadmat
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "stats" not in raw:
        raise KeyError(f"{path} does not contain variable 'stats'")
    s = raw["stats"]

    def _to_dict(x):
        if hasattr(x, "_fieldnames"):
            return {n: getattr(x, n) for n in x._fieldnames}
        return x

    return {
        "total_voxels": int(s.total_voxels),
        "component_counts": {
            k: int(v) for k, v in _to_dict(s.component_counts).items()
        },
        "owner_hist": np.asarray(s.owner_hist, dtype=np.int64).ravel(),
        "pair_matrix": np.asarray(s.pair_matrix, dtype=np.int64),
        "pair_keys": [str(k) for k in s.pair_keys],
        "vessel_overlap": {
            k: (int(v[0]), float(v[1]))
            for k, v in _to_dict(s.vessel_overlap).items()
        } if hasattr(s, "vessel_overlap") and hasattr(s.vessel_overlap, "_fieldnames")
        else {},
    }


def report_to_dict(report: OverlapReport) -> Dict[str, Any]:
    """Convert an :class:`OverlapReport` to the same dict shape."""
    return {
        "total_voxels": int(report.total_voxels),
        "component_counts": dict(report.component_counts),
        "owner_hist": np.asarray(report.owner_hist, dtype=np.int64).ravel(),
        "pair_matrix": np.asarray(report.pair_matrix, dtype=np.int64),
        "pair_keys": list(report.pair_keys),
        "vessel_overlap": dict(report.vessel_overlap),
    }


def _fmt(n: int) -> str:
    return f"{int(n):,}"


def compare(
    py_report: Optional[OverlapReport] = None,
    *,
    py_mat_path: Optional[str] = None,
    matlab_mat_path: Optional[str] = None,
    out=sys.stdout,
) -> None:
    """Print Python and MATLAB stats side by side.

    Provide exactly one of ``py_report`` or ``py_mat_path`` (the latter
    is a file produced by dumping :func:`report_to_dict` to .mat/.npz).
    Pass ``matlab_mat_path`` to load MATLAB stats.
    """
    if py_report is not None:
        py = report_to_dict(py_report)
    elif py_mat_path is not None:
        py = load_matlab_stats(py_mat_path)  # same schema
    else:
        raise ValueError("Provide py_report or py_mat_path")

    if matlab_mat_path is None:
        raise ValueError("matlab_mat_path is required")
    ml = load_matlab_stats(matlab_mat_path)

    bar = "=" * 76
    print(bar, file=out)
    print("Python vs MATLAB overlap stats", file=out)
    print(bar, file=out)
    print(
        f"{'total_voxels':<22s} {'Python':>14s} {'MATLAB':>14s} {'rel diff':>10s}",
        file=out,
    )
    py_tv, ml_tv = py["total_voxels"], ml["total_voxels"]
    rel = _rel_diff(py_tv, ml_tv)
    print(f"{'':<22s} {_fmt(py_tv):>14s} {_fmt(ml_tv):>14s} {rel:>9.2f}%", file=out)

    print(f"\n{'Per-component voxels':<22s}", file=out)
    header = f"{'':<22s} {'Python':>14s} {'MATLAB':>14s} {'rel diff':>10s}"
    print(header, file=out)
    keys = list(py["component_counts"].keys())
    for k in keys:
        p = py["component_counts"].get(k, 0)
        m = ml["component_counts"].get(k, 0)
        rel = _rel_diff(p, m)
        print(f"  {k:<20s} {_fmt(p):>14s} {_fmt(m):>14s} {rel:>9.2f}%", file=out)

    print(f"\n{'Owner histogram (0..5 neural owners per voxel)':<22s}", file=out)
    print(header, file=out)
    py_h = py["owner_hist"]
    ml_h = ml["owner_hist"]
    n_bins = max(len(py_h), len(ml_h))
    for i in range(n_bins):
        p = int(py_h[i]) if i < len(py_h) else 0
        m = int(ml_h[i]) if i < len(ml_h) else 0
        rel = _rel_diff(p, m)
        label = f"  {i} owner(s)"
        print(f"  {label:<20s} {_fmt(p):>14s} {_fmt(m):>14s} {rel:>9.2f}%", file=out)

    if py["vessel_overlap"] or ml["vessel_overlap"]:
        print("\nVessel overlap fraction per component:", file=out)
        print(
            f"{'':<22s} {'Python %':>14s} {'MATLAB %':>14s} {'abs diff':>10s}",
            file=out,
        )
        all_keys = set(py["vessel_overlap"]) | set(ml["vessel_overlap"])
        for k in sorted(all_keys):
            pf = 100 * py["vessel_overlap"].get(k, (0, 0.0))[1]
            mf = 100 * ml["vessel_overlap"].get(k, (0, 0.0))[1]
            diff = pf - mf
            print(
                f"  {k:<20s} {pf:>13.3f}% {mf:>13.3f}% {diff:>9.3f}%",
                file=out,
            )

    print(bar, file=out)


def _rel_diff(py: int, ml: int) -> float:
    """Relative difference in %, guarded against division by zero."""
    if ml == 0 and py == 0:
        return 0.0
    if ml == 0:
        return float("inf")
    return 100.0 * (py - ml) / ml


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Python vs MATLAB overlap stats.",
    )
    parser.add_argument(
        "matlab_mat",
        help="Path to matlab_overlap_stats.mat (from measure_overlap.m).",
    )
    parser.add_argument(
        "--python-mat",
        default=None,
        help=(
            "Optional path to a .mat dump of Python stats (same schema). "
            "If omitted, the script runs a fresh Python pipeline per --demo-*."
        ),
    )
    parser.add_argument(
        "--demo-vol-sz",
        default="100,100,50",
        help="Comma-separated vol_sz for the demo Python run (default 100,100,50).",
    )
    parser.add_argument("--demo-seed", type=int, default=0)
    parser.add_argument("--demo-n-neur", type=int, default=20)
    args = parser.parse_args()

    if args.python_mat is not None:
        compare(
            py_mat_path=args.python_mat,
            matlab_mat_path=args.matlab_mat,
        )
        return 0

    # Demo path: run a small Python pipeline in-process
    from calcia.config.params import VolumeParams
    from calcia.pipeline import simulate_neural_volume
    from .overlap import summarize

    vol_sz = tuple(int(s) for s in args.demo_vol_sz.split(","))
    print(f"Running demo Python pipeline vol_sz={vol_sz}, seed={args.demo_seed}...")
    vol_out = simulate_neural_volume(
        VolumeParams(
            vol_sz=vol_sz, vres=1, N_neur=args.demo_n_neur,
            N_den=10, N_bg=10, verbose=0,
        ),
        seed=args.demo_seed, verbose=0,
    )
    report = summarize(vol_out)
    compare(py_report=report, matlab_mat_path=args.matlab_mat)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

"""Headless end-to-end smoke test (no Qt, no display).

Exercises the whole non-UI core -- load -> geometry -> backend-agnostic scene
-> off-screen render -> screenshot -- and prints linkage sanity numbers.  This
is what proves the pipeline works before any interactive window is opened, and
it doubles as the trame-backend feasibility check (same off-screen Plotter).

    python -m calcia.viz.render_check <run_dir> [--out shot.png]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyvista as pv

from . import model, runs
from .geometry import GeometryCache
from .linkage import NeuronTable
from .scene3d import Scene3D


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="run directory; omit to use the most recent run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--neuron", type=int, default=0)
    ap.add_argument("--vessel-ds", type=int, default=1)
    ap.add_argument("--dendrites", choices=["points", "tube"], default="points")
    args = ap.parse_args(argv)

    run_dir = runs.resolve(args.run_dir)
    if not run_dir:
        raise SystemExit("no run_dir given and no runs found under "
                         f"{runs.default_root()}")

    pv.OFF_SCREEN = True
    try:
        run = model.load(run_dir, load_vessels=True)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"could not load run {run_dir!r}: "
                         f"{type(exc).__name__}: {exc}")
    table = NeuronTable(run)
    geom = GeometryCache(run)

    # ---- linkage sanity --------------------------------------------------
    i = args.neuron
    vox_all = run.neuron_voxels(i, "all")
    vox_soma = run.neuron_voxels(i, "soma")
    vox_dend = run.neuron_voxels(i, "dend")
    fp = table.footprint(i)
    r, c = table.soma_movie_xy(i)
    dff = run.dff()
    print("── linkage sanity ─────────────────────────────────────────────")
    print(f"neuron #{i}: voxels all={len(vox_all)} soma={len(vox_soma)} "
          f"dend={len(vox_dend)}")
    print(f"  footprint nonzero px = {(fp > 0).sum()}, soma@movie=({r:.1f},{c:.1f})")
    print(f"  trace dF/F: min={dff[i].min():.2f} max={dff[i].max():.2f}")
    # round-trip pick: soma grid loc -> pick_from_3d should return i
    back = table.pick_from_3d(run.soma_grid_locs()[i])
    print(f"  pick_from_3d(soma_loc[{i}]) -> {back}  {'OK' if back == i else 'MISMATCH'}")

    # ---- geometry --------------------------------------------------------
    ves = geom.vessels(downsample=args.vessel_ds)
    somapts = geom.soma_points()
    dend = geom.dendrite(i, mode=args.dendrites)
    soma_surf = geom.soma_surface(i)
    print("── geometry ───────────────────────────────────────────────────")
    print(f"  vessels: {ves.n_points} pts / {ves.n_faces} faces")
    print(f"  soma cloud: {somapts.n_points} pts")
    print(f"  neuron soma surf: {soma_surf.n_points} pts, "
          f"dend({args.dendrites}): {dend.n_points} pts")

    # ---- off-screen scene render ----------------------------------------
    frame = args.frame if args.frame is not None else run.nt - 1
    p = pv.Plotter(off_screen=True, window_size=(1200, 900))
    scene = Scene3D(p, run, geom, table, dendrite_mode=args.dendrites)
    scene.build(show_vessels=True, vessel_downsample=args.vessel_ds)
    scene.set_frame(frame, render=False)
    scene.select(i, render=False)
    p.camera_position = "iso"
    out = args.out or os.path.join(run.run_dir, "viz_cache", "render_check.png")
    p.screenshot(out)
    print("── render ─────────────────────────────────────────────────────")
    print(f"  frame={frame} selected=#{i}")
    print(f"  screenshot -> {out}  ({os.path.getsize(out)} bytes)")
    print("OK")


if __name__ == "__main__":
    main()

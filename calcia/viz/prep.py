"""Pre-compute the visualization bundle for a run (headless, memory-safe).

The interactive viewer only ever needs to read small cached assets if they
already exist:

    viz_cache/vessels_*.vtp      blood-vessel surface mesh   (~10 MB)
    viz_cache/soma_mesh_dec*.vtu merged all-soma surfaces     (bounded, decimated)
    viz_cache/soma_outlines.npz  2D soma outline polylines    (few MB)

Building these is the one-time expensive step (marching cubes over every
neuron, contour extraction).  Doing it *here* -- once, at delivery time, in a
single guarded headless process -- means the viewer never has to touch the
multi-GB ``cell_footprints.pkl`` / phase-1 pickle to build them, and toggles
are instant.

Two ways to run:

  * standalone, over an existing run:
        python -m calcia.viz.prep examples/output/<run_dir>

  * from a demo that just finished (data still in RAM): call
    :func:`prep_run` with ``neur_ves=vol_out.neur_ves`` so it never re-reads the
    7.5 GB phase-1 pickle.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from . import model, runs
from .geometry import GeometryCache
from .linkage import NeuronTable


def prep_run(run_dir, *, neur_ves=None, decimate: float = 0.8,
             outline_k: int = 16, verbose: bool = True):
    """Build (and cache) the viz bundle for ``run_dir``.

    Parameters
    ----------
    neur_ves : optional in-memory vessel voxels (from a demo's ``vol_out``); if
        given, the multi-GB phase-1 pickle is never read.  If omitted,
        ``model.load`` is cache-first (skips the pickle when a vessel ``.vtp``
        already exists) and otherwise loads it under the RAM guard.
    decimate : per-soma face-reduction fraction for the all-soma mesh.
    outline_k : number of colour buckets for the 2D outlines.

    Returns the list of asset filenames written / present.
    """
    run = model.load(run_dir, load_vessels=(neur_ves is None), verbose=verbose)
    if neur_ves is not None:
        run.neur_ves = np.asarray(neur_ves)

    geom = GeometryCache(run)
    table = NeuronTable(run)
    made = []

    ves = geom.vessels()                       # loads .vtp cache or builds it
    if ves.n_points:
        made.append(os.path.basename(geom.vessels_cache_path()))
        if verbose:
            print(f"[prep] vessels: {ves.n_points} pts")

    mesh = geom.all_soma_surfaces(decimate=decimate, verbose=verbose)
    if mesh.n_points:
        made.append(f"soma_mesh_dec{decimate}.vtu")
        if verbose:
            print(f"[prep] soma mesh: {mesh.n_points} pts (decimate={decimate})")

    table.soma_contour_buckets(outline_k)      # computes + writes the .npz
    if os.path.exists(table._outlines_cache_path()):
        made.append("soma_outlines.npz")
        if verbose:
            print(f"[prep] soma outlines: {outline_k} colour buckets")

    if verbose:
        print(f"[prep] bundle ready in {geom.cache_dir}")
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Pre-compute the visualization bundle for a run.")
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="run directory; omit to use the most recent run")
    ap.add_argument("--decimate", type=float, default=0.8,
                    help="per-soma face reduction for the all-soma mesh")
    ap.add_argument("--outline-k", type=int, default=16,
                    help="number of colour buckets for the 2D outlines")
    args = ap.parse_args(argv)

    run_dir = runs.resolve(args.run_dir)
    if not run_dir:
        raise SystemExit("no run_dir given and no runs found under "
                         f"{runs.default_root()}")
    made = prep_run(run_dir, decimate=args.decimate, outline_k=args.outline_k)
    print("wrote:", ", ".join(made) if made else "(nothing)")


if __name__ == "__main__":
    main()

"""Save a full simulation run as a self-describing artifact bundle + a one-page
human-readable report.

This is the standard persistence paradigm for a finished calcia run: given the
core pipeline objects (volume, optics, traces, scan) plus a metadata dict and the
scanned movie, write the complete, crash-safe artifact set into a run directory.
The metadata SCHEMA is the caller's (passed in as a dict and serialized as-is) —
this module only orchestrates the writing, in an order chosen so a crash or full
disk still leaves a usable movie + metadata + ground-truth traces on disk.
"""
import json
import os
import pickle
import traceback

import numpy as np

from .render import make_video, save_tif


def save_full_bundle(run_dir, *, noisy, clean, vol_out, vol_params, opt_out,
                     time_out, scan_out, params_dict, metadata, dt,
                     make_gif=True, make_viz=True, verbose=True):
    """Write the full reproducible artifact set for one finished run.

    Writes: movies.npz (clean+noisy THW + mot_hist/blur_hist), movie_noisy.tif,
    movie_clean.tif, optics.npz, cell_footprints.pkl, params.pkl, traces.npz
    (soma/dend/bg/spikes + soma_neurons/locs ground truth), metadata.json,
    report.md, movie.gif, and the viz_cache bundle. Channel-specific ground truth
    (e.g. tdtomato_expression.npz) is saved by the caller separately.
    """
    os.makedirs(run_dir, exist_ok=True)
    saved, failed = [], []

    def _step(name, fn, critical=False):
        """Run one save step in isolation. A failure is logged LOUDLY but does
        NOT abort the remaining steps."""
        try:
            fn()
            saved.append(name)
        except Exception as e:
            failed.append(name)
            print(f"  [save] {'CRITICAL ' if critical else ''}FAILED {name}: {e}")
            traceback.print_exc()

    # Order matters for crash-safety: write the CHEAP + IRREPLACEABLE artifacts
    # first (the scanned movie, metadata, ground-truth traces), then the HEAVY /
    # OPTIONAL tail (multi-GB footprints, viz meshes, gif). If the tail dies or
    # fills the disk, the run dir still holds a usable movie + metadata + traces.

    def _save_movies():  # the irreplaceable scan output -> first
        movies = dict(
            mov_clean=np.transpose(clean, (2, 0, 1)).astype(np.float32),
            mov_noisy=np.transpose(noisy, (2, 0, 1)).astype(np.float32),
            axes=np.array("THW"))
        if getattr(scan_out, "mot_hist", None) is not None:
            movies["mot_hist"] = scan_out.mot_hist
        if getattr(scan_out, "blur_hist", None) is not None:
            movies["blur_hist"] = scan_out.blur_hist
        np.savez_compressed(os.path.join(run_dir, "movies.npz"), **movies)
    _step("movies.npz", _save_movies, critical=True)

    _step("movie_noisy.tif",
          lambda: save_tif(noisy, os.path.join(run_dir, "movie_noisy.tif")))
    _step("movie_clean.tif",
          lambda: save_tif(clean, os.path.join(run_dir, "movie_clean.tif")))

    def _save_metadata():
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    _step("metadata.json", _save_metadata)

    def _save_params():
        with open(os.path.join(run_dir, "params.pkl"), "wb") as f:
            pickle.dump(params_dict, f)
    _step("params.pkl", _save_params)

    def _save_traces():  # ground truth -> before the heavy tail
        gp = vol_out.gp_vals
        n_soma = sum(1 for g in gp if getattr(g, "soma_mask", None) is not None
                     and np.any(np.asarray(g.soma_mask)))
        locs = np.asarray(vol_out.locs)
        soma = np.asarray(time_out.soma, dtype=np.float32)
        traces = dict(n_soma=np.int64(n_soma), trace_axes=np.array("KT"),
                      locs_axes=np.array("Kxyz"), soma=soma, locs=locs,
                      soma_neurons=soma[:n_soma], soma_locs=locs[:n_soma])
        if time_out.spikes is not None:
            traces["spikes"] = time_out.spikes
            if time_out.spikes.shape[0] == soma.shape[0]:
                traces["spikes_neurons"] = time_out.spikes[:n_soma]
        if time_out.dend is not None:
            traces["dend"] = np.asarray(time_out.dend, dtype=np.float32)
            traces["dend_neurons"] = traces["dend"][:n_soma]
        if time_out.bg is not None:
            traces["bg"] = np.asarray(time_out.bg, dtype=np.float32)
        np.savez_compressed(os.path.join(run_dir, "traces.npz"), **traces)
    _step("traces.npz", _save_traces)

    def _save_optics():
        optics = dict(psf=opt_out.psf)
        if getattr(opt_out, "mask", None) is not None:
            optics["mask"] = opt_out.mask
        if getattr(opt_out, "col_mask", None) is not None:
            optics["col_mask"] = opt_out.col_mask
        np.savez_compressed(os.path.join(run_dir, "optics.npz"), **optics)
    _step("optics.npz", _save_optics)

    # --- heavy / optional tail ---
    def _save_footprints():  # multi-GB -> after everything critical is on disk
        with open(os.path.join(run_dir, "cell_footprints.pkl"), "wb") as f:
            pickle.dump(dict(gp_vals=vol_out.gp_vals, bg_proc=vol_out.bg_proc,
                             locs=vol_out.locs,
                             neur_vol_shape=vol_out.neur_vol.shape), f)
    _step("cell_footprints.pkl", _save_footprints)

    _step("report.md", lambda: write_summary_report(
        run_dir, neur_ves=getattr(vol_out, "neur_ves", None), verbose=False))

    if make_viz:
        def _save_viz():
            from calcia.viz.prep import prep_run
            prep_run(run_dir, neur_ves=getattr(vol_out, "neur_ves", None),
                     verbose=verbose)
        _step("viz_cache", _save_viz)

    if make_gif:
        _step("movie.gif", lambda: make_video(
            noisy, clean, os.path.join(run_dir, "movie.gif"), dt=dt, fps=30))

    if verbose:
        msg = f"  saved full bundle -> {os.path.basename(os.path.normpath(run_dir))}"
        if failed:
            msg += f"   [FAILED: {', '.join(failed)}]"
        print(msg)


def write_summary_report(run_dir, *, neur_ves=None, title=None, write=True,
                         verbose=True):
    """Human-readable one-page summary of a finished run.

    Reads the run's own saved artifacts (``metadata.json``, ``traces.npz``,
    ``cell_footprints.pkl``) so it works both inline (called after a run is saved)
    and standalone on any existing run dir. Reports the FOV / pixel geometry and
    the component inventory (neurons, background processes, spikes, and — when
    ``neur_ves`` is supplied — blood-vessel voxels).

    Parameters
    ----------
    run_dir : str
        A finished run directory.
    neur_ves : np.ndarray, optional
        In-memory vessel voxel volume (``vol_out.neur_ves``) so the report can
        state vessel voxel count + volume fraction. Omit (standalone use) and the
        vessel line falls back to noting the viz mesh.
    title : str, optional
        Report heading. Defaults to ``"<REGION> WIDEFIELD SIMULATION — SUMMARY
        REPORT"`` derived from ``metadata['region']``.

    Returns
    -------
    str : the report text (also written to ``<run_dir>/report.md`` if ``write``).
    """
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    cfg = meta.get("config", {})

    if title is None:
        region = meta.get("region", "")
        title = (f"{region.upper()} WIDEFIELD SIMULATION — SUMMARY REPORT"
                 if region else "WIDEFIELD SIMULATION — SUMMARY REPORT")

    # --- geometry ---
    vol_sz = meta["vol_sz"]                       # [x, y, z] um
    vres = meta["vres"]                           # vox / um
    sfrac = cfg.get("sfrac", 2)
    nt = meta["nt"]
    dt = meta["dt"]
    fov_x, fov_y, depth = vol_sz
    grid = (fov_x * vres, fov_y * vres, depth * vres)
    n_grid = grid[0] * grid[1] * grid[2]
    mov_shape = meta["movie_shape"]               # [H, W, T]
    H, W = mov_shape[0], mov_shape[1]
    um_per_px = sfrac / vres

    # --- components (prefer traces.npz for exact per-row counts) ---
    n_soma = int(meta.get("n_soma", meta.get("N_neur", 0)))
    n_comp = int(meta.get("N_soma_traces", n_soma))
    n_bg = n_comp - n_soma
    neuron_spikes = None
    tp = os.path.join(run_dir, "traces.npz")
    if os.path.exists(tp):
        z = np.load(tp)
        if "spikes_neurons" in z:
            neuron_spikes = int(np.asarray(z["spikes_neurons"]).sum())
    total_spikes = int(meta.get("total_spikes", 0))

    # --- vessels ---
    if neur_ves is not None:
        ves = np.asarray(neur_ves)
        n_ves = int((ves > 0).sum())
        ves_line = (f"  Blood vessels:         {n_ves:>12,} voxels "
                    f"({100.0 * n_ves / n_grid:.1f}% of volume)")
    elif os.path.exists(os.path.join(run_dir, "viz_cache")):
        ves_line = ("  Blood vessels:         vascular network present "
                    "(see viz_cache/vessels_*.vtp; voxel count needs the "
                    "Phase-1 volume)")
    else:
        ves_line = "  Blood vessels:         (not recorded)"

    dur = nt * dt
    spike_hz = (neuron_spikes / n_soma / dur) if (neuron_spikes and n_soma) else None

    L = []
    L.append(title)
    L.append(f"run:       {os.path.basename(os.path.normpath(run_dir))}")
    L.append(f"region:    {meta.get('region','?')}   indicator: {meta.get('prot','?')}"
             f"   motion: {cfg.get('motion_model','randomwalk')}")
    L.append("")
    L.append("FIELD OF VIEW")
    L.append(f"  Lateral FOV:           {fov_x} x {fov_y} um  "
             f"({fov_x/1000:.2f} x {fov_y/1000:.2f} mm)")
    L.append(f"  Imaged depth:          {depth} um")
    L.append(f"  Voxel resolution:      {vres} vox/um  ->  grid "
             f"{grid[0]} x {grid[1]} x {grid[2]} = {n_grid/1e6:.1f} M voxels")
    L.append("")
    L.append("OUTPUT MOVIE")
    L.append(f"  Frame size:            {H} x {W} px  ({H*W:,} px/frame)")
    L.append(f"  Pixel size:            {um_per_px:g} um/px  "
             f"(sfrac={sfrac} downsample / vres={vres})")
    L.append(f"  Frames:                {nt}  @ {1/dt:.0f} Hz  ->  {dur:.1f} s")
    L.append(f"  Total pixels:          {H*W*nt/1e6:.1f} M  ({H} x {W} x {nt})")
    L.append("")
    L.append("COMPONENTS")
    L.append(f"  Fluorescing neurons:   {n_soma:>12,}  (labelled somata)")
    L.append(f"  Background processes:  {n_bg:>12,}  (bg dendrites + axons)")
    L.append(f"  Total trace components:{n_comp:>12,}")
    L.append(ves_line)
    if neuron_spikes is not None:
        L.append(f"  Neuron spikes ({dur:.0f}s):    {neuron_spikes:>12,}"
                 + (f"  (~{spike_hz:.2f} Hz/neuron mean)" if spike_hz else ""))
    L.append(f"  Total spikes (all rows):{total_spikes:>11,}")
    text = "\n".join(L)

    if write:
        with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write("```\n" + text + "\n```\n")
    if verbose:
        print("\n" + text + "\n")
    return text

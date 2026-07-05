"""Striatum widefield — per-parameter TIF troubleshooting sweep.

Reuses the cached medium striatum Phase-1 volume and emits ONE TIF (+ a
brightest-frame PNG) per parameter value, plus a per-axis montage PNG, so the
background "wash-out" can be compared by eye across:

  NA, scatter_length_um_wf, out-of-focus weight, focus_slab_um, imaging depth.

See docs/background_brightness_troubleshooting.md and the plan file.

Run:
    # smoke (validate code path, few min):
    conda run -n calcia python examples/striatum_param_sweep.py --nt 6 --max-per-axis 1
    # full sweep (~2.5-3h, run in background):
    conda run -n calcia python examples/striatum_param_sweep.py
    # subset:
    conda run -n calcia python examples/striatum_param_sweep.py --axes NA depth
"""
import argparse
import copy
import datetime as _dt
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _striatum_common as C
from dataclasses import replace

PHASE1 = os.path.join(os.path.dirname(__file__), "output", "_shared",
                      "phase1_f1d312ce32.pkl")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
DT = 1.0 / 30
SEED = 42

# Axis value lists
NA_VALUES = [0.4, 0.6, 0.8, 1.0]
SCATTER_VALUES = [20.0, 45.0, 70.0]
OOF_WEIGHTS = [1.0, 0.3, 0.1, 0.0]
SLAB_VALUES = [2.0, 10.0, 30.0]
DEPTH_VALUES = [30, 60, 100]  # um
ALL_AXES = ["NA", "scatter", "oof", "slab", "depth"]


# ----------------------------------------------------------------------
# Rendering helpers (save_tif / brightest_frame / cv_bright / dF come from
# _striatum_common as C.*; save_png / save_montage are sweep-specific).
# ----------------------------------------------------------------------
def save_png(frame, path, title=""):
    hi = np.percentile(frame, 99.5)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=90)
    ax.imshow(frame, cmap="gray", vmin=0, vmax=hi if hi > 0 else 1)
    ax.set_title(title, fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def save_montage(items, path, title):
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2), dpi=90)
    if n == 1:
        axes = [axes]
    for ax, (label, frame) in zip(axes, items):
        hi = np.percentile(frame, 99.5)
        ax.imshow(frame, cmap="gray", vmin=0, vmax=hi if hi > 0 else 1)
        ax.set_title(label, fontsize=10); ax.axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


# ----------------------------------------------------------------------
# Depth masking: keep only voxels with z = idx % N3 < z_cut. C-order
# (scanning.py:597). Reuses Phase 1 (no rebuild); grid shape unchanged.
# ----------------------------------------------------------------------
def mask_depth(vol_out, depth_um, vres, N3):
    z_cut = int(depth_um * vres)
    v = copy.copy(vol_out)
    new_gp = []
    for cfd in vol_out.gp_vals:
        keep = (cfd.indices % N3) < z_cut
        c = copy.copy(cfd)
        c.indices = cfd.indices[keep]
        c.fluorescence = cfd.fluorescence[keep]
        c.soma_mask = cfd.soma_mask[keep]
        new_gp.append(c)
    v.gp_vals = new_gp
    if vol_out.bg_proc:
        new_bg = []
        for bp in vol_out.bg_proc:
            keep = (bp.indices % N3) < z_cut
            b = copy.copy(bp)
            b.indices = bp.indices[keep]
            b.fluorescence = bp.fluorescence[keep]
            new_bg.append(b)
        v.bg_proc = new_bg
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=90)
    ap.add_argument("--max-per-axis", type=int, default=None,
                    help="cap values per axis (smoke test)")
    ap.add_argument("--axes", nargs="+", default=ALL_AXES, choices=ALL_AXES)
    args = ap.parse_args()
    nt = args.nt

    def cap(lst):
        return lst if args.max_per_axis is None else lst[:args.max_per_axis]

    from calcia import simulate_optical_propagation, generate_time_traces
    from calcia.scanning import scan_widefield

    # De-washed sparse mode; the sweep overrides obj_na / scatter per PSF below.
    cfg = C.StriatumConfig.dewashed(nt=nt, fps=1.0 / DT, seed=SEED)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, f"param_sweep_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"== param sweep ==\n  out: {run_dir}\n  nt={nt}  axes={args.axes}")

    # ---- Phase 1 (load) ----
    t0 = time.time()
    vol_out, vol_params = C.load_phase1(PHASE1)
    N3 = vol_out.neur_vol.shape[2]
    vres = vol_params.vres
    print(f"  phase1 loaded {time.time()-t0:.0f}s  grid={vol_out.neur_vol.shape}")

    # ---- Phase 3 (once; depth-masking changes voxels within components,
    #      not component count, so traces stay valid). Calibrated defaults
    #      (bg_scale=0.1, burst) come from _striatum_common. ----
    K = len(vol_out.gp_vals)
    sp = cfg.build_spike(K, len(vol_out.bg_proc) > 0, verbose=0)
    time_out = generate_time_traces(
        spike_params=sp, cal_params=cfg.build_cal(),
        n_locs=vol_out.locs, verbose=0)

    scan_params = cfg.build_scan(verbose=0)
    cam_params = cfg.build_cam()
    wf_params = cfg.build_wf()

    def make_psf(na=0.8, scatter=70.0):
        return replace(cfg, obj_na=na, scatter_length_um_wf=scatter).build_psf()

    def do_scan(vol, opt, sep=False, slab=None):
        return scan_widefield(
            vol_out=vol, opt_out=opt, time_out=time_out,
            scan_params=scan_params, cam_params=cam_params,
            wf_params=wf_params, spike_params=sp, seed=SEED,
            separate_focus=sep, focus_slab_um=slab)

    metrics = {}
    meta = dict(timestamp=ts, nt=nt, axes=args.axes, timings={})

    def emit(axis, label, mov):
        """Save tif + dF tif + brightest-frame png; record metrics."""
        C.save_tif(mov, os.path.join(run_dir, f"{axis}_{label}.tif"))
        C.save_tif(C.dF(mov), os.path.join(run_dir, f"{axis}_{label}_dF.tif"),
                   clip0=True)
        fr = mov[:, :, C.brightest_frame(mov)]
        save_png(fr, os.path.join(run_dir, f"{axis}_{label}.png"),
                 title=f"{axis}={label}")
        metrics[f"{axis}_{label}"] = dict(mean=float(mov.mean()),
                                          cv_bright=C.cv_bright(mov))
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return label, fr

    # ============ Axis: NA ============
    if "NA" in args.axes:
        items = []
        for na in cap(NA_VALUES):
            t0 = time.time()
            opt = simulate_optical_propagation(vol_params=vol_params,
                                               psf_params=make_psf(na=na),
                                               vol_out=vol_out, verbose=0)
            so = do_scan(vol_out, opt)
            items.append(emit("NA", f"{na}", so.mov_raw.astype(np.float32)))
            meta["timings"][f"NA_{na}"] = time.time() - t0
            print(f"  NA={na}  {meta['timings'][f'NA_{na}']:.0f}s")
        save_montage(items, os.path.join(run_dir, "montage_NA.png"),
                     "NA sweep (brightest frame)")

    # ============ Axis: scatter ============
    if "scatter" in args.axes:
        items = []
        for sc in cap(SCATTER_VALUES):
            t0 = time.time()
            opt = simulate_optical_propagation(vol_params=vol_params,
                                               psf_params=make_psf(scatter=sc),
                                               vol_out=vol_out, verbose=0)
            so = do_scan(vol_out, opt)
            items.append(emit("scatter", f"{sc:.0f}", so.mov_raw.astype(np.float32)))
            meta["timings"][f"scatter_{sc}"] = time.time() - t0
            print(f"  scatter={sc}  {meta['timings'][f'scatter_{sc}']:.0f}s")
        save_montage(items, os.path.join(run_dir, "montage_scatter.png"),
                     "scatter_length sweep")

    # separate_focus scans are reused by both oof and slab axes; cache them.
    sep_cache = {}

    def get_split(slab):
        if slab not in sep_cache:
            opt = simulate_optical_propagation(vol_params=vol_params,
                                               psf_params=make_psf(),
                                               vol_out=vol_out, verbose=0)
            so = do_scan(vol_out, opt, sep=True, slab=slab)
            sep_cache[slab] = (so.mov_infocus.astype(np.float32),
                               so.mov_oof.astype(np.float32))
        return sep_cache[slab]

    # ============ Axis: oof-weight (offline blend from slab=10 split) ============
    if "oof" in args.axes:
        t0 = time.time()
        infocus, oof = get_split(10.0)
        meta["timings"]["oof_split_scan"] = time.time() - t0
        items = []
        for w in cap(OOF_WEIGHTS):
            items.append(emit("oof", f"w{w}", infocus + w * oof))
        save_montage(items, os.path.join(run_dir, "montage_oof.png"),
                     "out-of-focus weight (infocus + w*oof)")

    # ============ Axis: focus_slab (show in-focus at each DOF) ============
    if "slab" in args.axes:
        items = []
        for slab in cap(SLAB_VALUES):
            t0 = time.time()
            infocus, _ = get_split(slab)
            meta["timings"][f"slab_{slab}"] = time.time() - t0
            items.append(emit("slab", f"{slab:.0f}", infocus))
            print(f"  slab={slab}")
        save_montage(items, os.path.join(run_dir, "montage_slab.png"),
                     "focus_slab in-focus signal")

    # ============ Axis: depth (mask deep voxels, reuse Phase 1) ============
    if "depth" in args.axes:
        items = []
        opt = simulate_optical_propagation(vol_params=vol_params,
                                           psf_params=make_psf(),
                                           vol_out=vol_out, verbose=0)
        for d in cap(DEPTH_VALUES):
            t0 = time.time()
            vol_d = mask_depth(vol_out, d, vres, N3) if d * vres < N3 else vol_out
            so = do_scan(vol_d, opt)
            items.append(emit("depth", f"{d}um", so.mov_raw.astype(np.float32)))
            meta["timings"][f"depth_{d}"] = time.time() - t0
            print(f"  depth={d}um  {meta['timings'][f'depth_{d}']:.0f}s")
        save_montage(items, os.path.join(run_dir, "montage_depth.png"),
                     "imaging depth sweep")

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDONE -> {run_dir}")


if __name__ == "__main__":
    main()

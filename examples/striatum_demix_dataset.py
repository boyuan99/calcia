"""Generate a long (30 s) striatum widefield demixing dataset.

Reuses the cached medium Phase-1 volume, runs ONE scan_widefield with
separate_focus=True at nt=900, producing the demixing input/ground-truth pair:

  - movie_input  (mov_raw)      : full washed widefield movie = what a demixer sees
  - movie_infocus (mov_infocus) : in-focus focal-layer cells = visual ground truth
  - ground_truth.npz            : spikes, soma traces, locs (+ both movies, full precision)

focus_slab_um=10 gives a ~10 um in-focus layer (a visible, moderately clean
ground truth; use a thinner slab for stricter in-focus). Calibrated defaults
(bg_scale=0.1, burst spikes) live in _striatum_common.

Run:  conda run -n calcia python examples/striatum_demix_dataset.py
"""
import datetime as _dt
import json
import os
import time

import numpy as np

import _striatum_common as C

PHASE1 = os.path.join(os.path.dirname(__file__), "output", "_shared",
                      "phase1_f1d312ce32.pkl")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
NT = 900
DT = 1.0 / 30
SEED = 42
FOCUS_SLAB_UM = 10.0


def main():
    from calcia import simulate_optical_propagation, generate_time_traces
    from calcia.scanning import scan_widefield

    # De-washed sparse mode (individual cells visible; no cosmetics).
    cfg = C.StriatumConfig.dewashed(nt=NT, fps=1.0 / DT, seed=SEED)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, f"striatum_demix_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"== demix dataset ==\n  out: {run_dir}\n  nt={NT} ({NT*DT:.0f}s)  "
          f"focus_slab={FOCUS_SLAB_UM}um")

    t0 = time.time()
    vol_out, vol_params = C.load_phase1(PHASE1)
    print(f"  phase1 loaded {time.time()-t0:.0f}s  grid={vol_out.neur_vol.shape}")

    opt_out = simulate_optical_propagation(
        vol_params=vol_params, psf_params=cfg.build_psf(),
        vol_out=vol_out, verbose=0)

    K = len(vol_out.gp_vals)
    sp = cfg.build_spike(K, len(vol_out.bg_proc) > 0)
    time_out = generate_time_traces(
        spike_params=sp, cal_params=cfg.build_cal(),
        n_locs=vol_out.locs, verbose=1)
    n_spk = int(time_out.spikes.sum()) if time_out.spikes is not None else 0
    print(f"  phase3 done  soma {time_out.soma.shape}  spikes={n_spk}")

    print(f"  scanning (separate_focus=True, nt={NT}) ... ~2.5h")
    t0 = time.time()
    so = scan_widefield(
        vol_out=vol_out, opt_out=opt_out, time_out=time_out,
        scan_params=cfg.build_scan(), cam_params=cfg.build_cam(),
        wf_params=cfg.build_wf(), spike_params=sp, seed=SEED,
        separate_focus=True, focus_slab_um=FOCUS_SLAB_UM)
    scan_s = time.time() - t0
    print(f"  scan done {scan_s:.0f}s")

    mov_input = so.mov_raw.astype(np.float32)      # full washed = demixer input
    infocus = so.mov_infocus.astype(np.float32)    # ground-truth focal cells
    oof = so.mov_oof.astype(np.float32)

    # full-precision ground-truth bundle
    np.savez_compressed(
        os.path.join(run_dir, "ground_truth.npz"),
        mov_input=mov_input, mov_infocus=infocus, mov_oof=oof,
        soma=time_out.soma.astype(np.float32),
        spikes=(time_out.spikes if time_out.spikes is not None
                else np.zeros((0, NT), dtype=np.uint8)),
        locs=vol_out.locs)
    print("  saved ground_truth.npz")

    # viewable TIFs
    C.save_tif(mov_input, os.path.join(run_dir, "movie_input.tif"))
    C.save_tif(infocus, os.path.join(run_dir, "movie_infocus.tif"))
    C.save_tif(C.dF(mov_input), os.path.join(run_dir, "movie_input_dF.tif"),
               clip0=True)
    print("  saved TIFs")

    meta = dict(timestamp=ts, nt=NT, dt=DT, focus_slab_um=FOCUS_SLAB_UM,
                total_spikes=n_spk, scan_seconds=scan_s,
                input_mean=float(mov_input.mean()), input_cv=C.cv_bright(mov_input),
                infocus_mean=float(infocus.mean()), infocus_cv=C.cv_bright(infocus),
                oof_frac=float(oof.mean() / (mov_input.mean() + 1e-9)))
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  input CV={meta['input_cv']:.3f}  infocus CV={meta['infocus_cv']:.3f}  "
          f"oof_frac={meta['oof_frac']:.3f}")
    print(f"\nDONE -> {run_dir}")


if __name__ == "__main__":
    main()

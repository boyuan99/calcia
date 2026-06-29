"""Striatum widefield simulation.

1P widefield simulation sized for striatum imaging via a cranial window
over the exposed striatum surface.

Default config (the long run):
  - vol_sz = (500, 500, 300) um   vol_depth = 0   vres = 2
  - nt = 300 frames @ 30 Hz (10 s)
  - prot = GCaMP6f (placeholder; GCaMP8f is not in the protein table yet)

Smoke test (--smoke):
  - vol_sz = (80, 80, 50)   nt = 30   (~1-2 min)
  Use this FIRST to verify the whole save pipeline before committing to
  the multi-hour run.

This iteration uses the EXISTING widefield pipeline as-is — no striatum-
specific modifications to neuron type / vasculature topology yet.

Checkpointing: every phase writes its output to the run folder as soon
as it finishes, so a crash in a later (e.g. Phase 4 FFT) does not waste
earlier compute. Critical ground-truth data is written BEFORE the
optional GIF preview.

Output layout:
    output/
      _shared/
        phase1_<signature>.pkl        shared phase 1 cache (reused across runs)
      striatum_v1_<YYYYMMDD_HHMMSS>/
        metadata.json
        cell_footprints.pkl           per-cell space templates (after Phase 1)
        optics.npz                    psf + masks            (after Phase 2)
        traces.npz                    soma/dend/bg/spikes    (after Phase 3)
        movies.npz                    mov_clean/mov_noisy    (after Phase 4)
        params.pkl                    dataclass instances    (after Phase 4)
        movie_noisy.tif / movie_clean.tif
        movie.gif                     preview (saved LAST; non-critical)
        profile_phaseN.html           pyinstrument profiles

Run:
    conda run -n calcia python examples/demo_widefield_striatum_v1.py --smoke
    conda run -n calcia python examples/demo_widefield_striatum_v1.py
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import pickle
import time

import numpy as np
from pyinstrument import Profiler

import _striatum_common as C


OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
SHARED_DIR = os.path.join(OUTPUT_ROOT, "_shared")


def parse_args():
    p = argparse.ArgumentParser(description="Striatum widefield simulation")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny fast run (80x80x50, 30 frames) to verify the "
                        "pipeline + save logic before the long run")
    p.add_argument("--medium", action="store_true",
                   help="Medium volume (250x250x100, 300 frames) — large "
                        "enough to assess background wash-out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nt", type=int, default=None,
                   help="Number of frames (overrides preset)")
    p.add_argument("--rate", type=float, default=0.02,
                   help="Baseline spike intensity. In burst mode the empirical "
                        "rate is ~rate*96 Hz (0.02 -> ~2 Hz). Default 0.02")
    p.add_argument("--smod", type=str, default="burst",
                   choices=["burst", "hawkes"],
                   help="Spike model. 'burst' = independent per-neuron Poisson "
                        "(decorrelated, stationary, good for demixing). "
                        "'hawkes' = coupled (produces synchronized avalanches). "
                        "Default 'burst'")
    p.add_argument("--burst-mean", type=int, default=0, dest="burst_mean",
                   help="Mean intra-burst spike count (burst mode). 0 = single "
                        "spikes / pure Poisson. Default 0")
    p.add_argument("--bg-scale", type=float, default=0.1, dest="bg_scale",
                   help="Neuropil/axon trace amplitude scale. The dense axon "
                        "background is the dominant 1P widefield wash-out source; "
                        "0.1 takes brightest-frame spatial CV from ~0.65 to ~0.95 "
                        "(cells become visible). 1.0 = raw NAOMi amplitude "
                        "(washed). Default 0.1")
    p.add_argument("--vres", type=int, default=2,
                   help="Voxels per um (default 2; use 1 to cut memory ~8x)")
    p.add_argument("--prot", type=str, default="GCaMP6f")
    return p.parse_args()


def phase1_signature(vol_sz, vol_depth, vres, seed, region):
    h = hashlib.sha1()
    h.update(repr((tuple(vol_sz), vol_depth, vres, seed, region)).encode())
    return h.hexdigest()[:10]


def save_profile(profiler, run_dir, name):
    html_path = os.path.join(run_dir, f"profile_{name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(profiler.output_html())
    print(profiler.output_text(unicode=True, color=False, show_all=False))
    print(f"  Profile saved: {html_path}")


# Rendering helpers (save_tiff_normalized, make_video) come from
# _striatum_common as C.*; save_profile is demo-specific (pyinstrument).


def main():
    args = parse_args()

    # -------- resolve config --------
    if args.smoke:
        vol_sz = (80, 80, 50)
        nt = args.nt if args.nt is not None else 30
        tag = "striatum_smoke"
    elif args.medium:
        vol_sz = (250, 250, 100)
        nt = args.nt if args.nt is not None else 300
        tag = "striatum_medium"
    else:
        vol_sz = (500, 500, 300)
        nt = args.nt if args.nt is not None else 300
        tag = "striatum_v1"
    vol_depth = 0
    vres = args.vres
    seed = args.seed
    rate = args.rate
    smod = args.smod
    burst_mean = args.burst_mean
    bg_scale = args.bg_scale
    prot = args.prot
    dt = 1.0 / 30

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, f"{tag}_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(SHARED_DIR, exist_ok=True)

    n_vox = (vol_sz[0] * vres) * (vol_sz[1] * vres) * (vol_sz[2] * vres)
    print("=" * 60)
    print(f"Striatum widefield  ({'SMOKE TEST' if args.smoke else 'full run'})")
    print(f"  Run dir:   {run_dir}")
    print(f"  Volume:    {vol_sz} um   vol_depth={vol_depth}   vres={vres}")
    print(f"  Voxels:    {n_vox:,}  ({n_vox * 4 / 1e9:.2f} GB / float32 array)")
    print(f"  Frames:    {nt} ({nt * dt:.1f} s at {1/dt:.0f} Hz)")
    print(f"  Seed:      {seed}   prot={prot}")
    print(f"  Spikes:    smod={smod}  rate={rate}  burst_mean={burst_mean}"
          + (f"  (~{rate*96:.1f} Hz expected)" if smod == "burst" else ""))
    print(f"  Neuropil:  bg_scale={bg_scale}"
          + ("  (washed)" if bg_scale >= 1.0 else "  (de-washed)"))
    print("=" * 60)

    from calcia import (
        simulate_neural_volume,
        simulate_optical_propagation,
        generate_time_traces,
    )
    from calcia.config.params import CalciumParams, VolumeParams
    from calcia.scanning import scan_widefield

    timings = {}
    profiler = Profiler()
    vol_params = VolumeParams(
        vol_sz=vol_sz, vres=vres, vol_depth=vol_depth, region="striatum",
    )

    # ==================================================================
    # Phase 1: Neural volume (shared cache keyed on geometry params)
    # ==================================================================
    sig = phase1_signature(vol_sz, vol_depth, vres, seed, "striatum")
    phase1_cache = os.path.join(SHARED_DIR, f"phase1_{sig}.pkl")
    print(f"\n[PHASE 1] Neural volume   (sig={sig})")
    print(f"  shared cache: {phase1_cache}")
    t0 = time.time()
    if os.path.exists(phase1_cache):
        print("  -> cache hit, loading")
        with open(phase1_cache, "rb") as f:
            vol_out, vol_params = pickle.load(f)
    else:
        print("  -> cache miss, generating (slow)")
        profiler.start()
        vol_out = simulate_neural_volume(
            vol_params=vol_params, seed=seed, verbose=1,
        )
        profiler.stop()
        save_profile(profiler, run_dir, "phase1")
        profiler.reset()
        vol_params = vol_out.params["vol_params"]
        with open(phase1_cache, "wb") as f:
            pickle.dump((vol_out, vol_params), f)
        print(f"  -> saved shared cache "
              f"({os.path.getsize(phase1_cache)/1e9:.1f} GB)")
    timings["phase1"] = time.time() - t0
    print(f"  done in {timings['phase1']:.1f}s   "
          f"N_neur={vol_params.N_neur}  grid={vol_out.neur_vol.shape}")

    # CHECKPOINT: per-cell footprints (from Phase 1) — save immediately
    with open(os.path.join(run_dir, "cell_footprints.pkl"), "wb") as f:
        pickle.dump(dict(
            gp_vals=vol_out.gp_vals,
            bg_proc=vol_out.bg_proc,
            locs=vol_out.locs,
            neur_vol_shape=vol_out.neur_vol.shape,
        ), f)
    print("  checkpoint: cell_footprints.pkl")

    # ==================================================================
    # Phase 2: Optical propagation (widefield)
    # ==================================================================
    print("\n[PHASE 2] Optical propagation (widefield)")
    t0 = time.time()
    psf_params = C.striatum_psf()
    profiler.start()
    opt_out = simulate_optical_propagation(
        vol_params=vol_params, psf_params=psf_params,
        vol_out=vol_out, verbose=1,
    )
    profiler.stop()
    save_profile(profiler, run_dir, "phase2")
    profiler.reset()
    timings["phase2"] = time.time() - t0
    print(f"  done in {timings['phase2']:.1f}s   PSF {opt_out.psf.shape}")

    # CHECKPOINT: optics — save immediately
    optics = dict(psf=opt_out.psf)
    if getattr(opt_out, "mask", None) is not None:
        optics["mask"] = opt_out.mask
    if getattr(opt_out, "col_mask", None) is not None:
        optics["col_mask"] = opt_out.col_mask
    np.savez_compressed(os.path.join(run_dir, "optics.npz"), **optics)
    print("  checkpoint: optics.npz")

    # ==================================================================
    # Phase 3: Time traces
    # ==================================================================
    print("\n[PHASE 3] Time traces")
    t0 = time.time()
    K = len(vol_out.gp_vals)
    has_axons = len(vol_out.bg_proc) > 0
    spike_params = C.striatum_spike(
        K, nt, dt, has_axons, rate=rate, prot=prot, smod=smod,
        burst_mean=burst_mean, bg_scale=bg_scale, verbose=1)
    cal_params = CalciumParams(prot_type=prot.lower())
    profiler.start()
    time_out = generate_time_traces(
        spike_params=spike_params, cal_params=cal_params,
        n_locs=vol_out.locs, verbose=1,
    )
    profiler.stop()
    save_profile(profiler, run_dir, "phase3")
    profiler.reset()
    timings["phase3"] = time.time() - t0
    n_spk = int(time_out.spikes.sum()) if time_out.spikes is not None else 0
    print(f"  done in {timings['phase3']:.1f}s   soma {time_out.soma.shape}   "
          f"total spikes={n_spk}")

    # CHECKPOINT: traces + spikes (ground truth) — save immediately, this is
    # the critical demixing ground truth and protects against a Phase 4 crash.
    traces = dict(
        soma=time_out.soma.astype(np.float32),
        spikes=(time_out.spikes if time_out.spikes is not None
                else np.zeros((0, nt), dtype=np.uint8)),
        locs=vol_out.locs,
    )
    if time_out.dend is not None:
        traces["dend"] = time_out.dend.astype(np.float32)
    if time_out.bg is not None:
        traces["bg"] = time_out.bg.astype(np.float32)
    np.savez_compressed(os.path.join(run_dir, "traces.npz"), **traces)
    print("  checkpoint: traces.npz")

    # ==================================================================
    # Phase 4: Widefield camera scan
    # ==================================================================
    print("\n[PHASE 4] Widefield camera scanning")
    t0 = time.time()
    scan_params = C.striatum_scan(verbose=1)
    wf_params = C.striatum_wf()
    cam_params = C.striatum_cam(dt)
    profiler.start()
    scan_out = scan_widefield(
        vol_out=vol_out, opt_out=opt_out, time_out=time_out,
        scan_params=scan_params, cam_params=cam_params,
        wf_params=wf_params, spike_params=spike_params, seed=seed,
    )
    profiler.stop()
    save_profile(profiler, run_dir, "phase4")
    profiler.reset()
    timings["phase4"] = time.time() - t0
    print(f"  done in {timings['phase4']:.1f}s   movie {scan_out.mov.shape}")
    print(f"  noisy [{scan_out.mov.min():.1f}, {scan_out.mov.max():.1f}]   "
          f"clean [{scan_out.mov_raw.min():.3g}, {scan_out.mov_raw.max():.3g}]")

    # ==================================================================
    # Save outputs — CRITICAL DATA FIRST, GIF LAST
    # ==================================================================
    print("\n[SAVE] writing run outputs (critical first, GIF last)")
    t0 = time.time()

    # 1. movies.npz (full-precision) — critical
    movies = dict(
        mov_clean=scan_out.mov_raw.astype(np.float32),
        mov_noisy=scan_out.mov.astype(np.float32),
    )
    if getattr(scan_out, "mot_hist", None) is not None:
        movies["mot_hist"] = scan_out.mot_hist
    np.savez_compressed(os.path.join(run_dir, "movies.npz"), **movies)
    print("  saved movies.npz")

    # 2. params.pkl — reproducibility
    with open(os.path.join(run_dir, "params.pkl"), "wb") as f:
        pickle.dump(dict(
            vol_params=vol_params, psf_params=psf_params,
            spike_params=spike_params, cal_params=cal_params,
            scan_params=scan_params, wf_params=wf_params,
            cam_params=cam_params,
        ), f)
    print("  saved params.pkl")

    # 3. metadata.json
    meta = dict(
        tag=tag, smoke=args.smoke,
        timestamp=_dt.datetime.now().isoformat(),
        seed=seed, region="striatum",
        vol_sz=list(vol_sz), vol_depth=vol_depth, vres=vres,
        nt=nt, dt=dt, prot=prot, rate=rate,
        smod=smod, burst_mean=burst_mean, bg_scale=bg_scale,
        N_neur=int(vol_params.N_neur),
        N_soma_traces=int(time_out.soma.shape[0]),
        total_spikes=n_spk,
        movie_shape=list(scan_out.mov.shape),
        timings_seconds={k: float(v) for k, v in timings.items()},
        timings_total_seconds=float(sum(timings.values())),
        phase1_cache=phase1_cache,
        profiles=sorted(f for f in os.listdir(run_dir)
                        if f.startswith("profile_") and f.endswith(".html")),
    )
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  saved metadata.json")

    # 4. Display TIFFs (derived; can be regenerated from movies.npz)
    C.save_tiff_normalized(scan_out.mov,
                           os.path.join(run_dir, "movie_noisy.tif"))
    C.save_tiff_normalized(scan_out.mov_raw,
                           os.path.join(run_dir, "movie_clean.tif"))
    print("  saved display TIFFs")

    # 5. GIF preview LAST — non-critical, can hang/fail without data loss
    try:
        C.make_video(scan_out.mov, scan_out.mov_raw,
                     os.path.join(run_dir, "movie.gif"), dt=dt, fps=30)
        print("  saved movie.gif")
    except Exception as e:
        print(f"  WARNING: GIF generation failed (data already saved): {e}")

    timings["save"] = time.time() - t0

    print("\n" + "=" * 60)
    print("DONE")
    print("  " + "  ".join(f"{k}={v:.0f}s" for k, v in timings.items()))
    print(f"  total {sum(timings.values()):.1f}s")
    print(f"  output: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

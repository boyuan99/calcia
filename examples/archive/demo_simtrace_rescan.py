"""Re-scan an EXISTING striatum volume with calcium traces sourced from
*sim-trace* (ensemble-recruitment) instead of calcia's built-in burst/Hawkes.

WHY
The default GCaMP demo generates spikes with calcia's own independent
burst-Poisson (or spatial Hawkes) — every soma flickers on its own. Real
population activity is CORRELATED: each event recruits a sub-population (an
ensemble). ``sim-trace`` (github.com/boyuan99/sim-trace) models exactly this —
``simtrace.ensemble.simulate_recruitment`` fires spatially-scattered ensembles
with a tunable overlap regime. This script feeds those spikes through calcia's
biophysical calcium ODE (via ``generate_time_traces_recruitment``) and scans the
SAME volume with the design-pure BEST optics (two-scale PSF, composite OFF, flat
illum), so the ONLY change vs the co-registered GCaMP run is the activity
STRUCTURE. Ground-truth ensemble membership is saved for downstream scoring.

Core + demos are untouched; the only library addition is the new bridge function
``calcia.traces.simtrace_bridge.generate_time_traces_recruitment``.

Run:
    conda run -n calcia python examples/demo_simtrace_rescan.py --smoke
    conda run -n calcia python examples/demo_simtrace_rescan.py \
        --match-run deepthinves_s7_500um_flat_stub --regime partial
"""
import argparse
import datetime as _dt
import json
import os
import time

import numpy as np

import sys  # archived: add examples/ (parent dir) so sibling imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _striatum_common as C
from demo_static_tdtomato_matched import load_matched_volume

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_MATCH = "deepthinves_s7_500um_flat_stub"
PSF_SUPPORT_UM = 100.0


def parse_args():
    p = argparse.ArgumentParser(description="sim-trace recruitment re-scan of a volume")
    p.add_argument("--match-run", default=DEFAULT_MATCH, dest="match_run",
                   help="Stub / run dir whose Phase-1 volume to re-scan (co-registered).")
    p.add_argument("--regime", default="partial", choices=["high", "partial", "low"],
                   help="Ensemble overlap regime between conditions.")
    p.add_argument("--n-conditions", type=int, default=6, dest="n_conditions")
    p.add_argument("--ensemble-frac", type=float, default=0.06, dest="ensemble_frac",
                   help="Fraction of components recruited per condition.")
    p.add_argument("--iti-s", type=float, default=1.0, dest="iti_s")
    p.add_argument("--evoked-rate", type=float, default=8.0, dest="evoked_rate")
    p.add_argument("--halo-um", type=float, default=28.0, dest="halo_um")
    p.add_argument("--halo-weight", type=float, default=0.8, dest="halo_weight")
    p.add_argument("--bg-scale", type=float, default=2.0, dest="bg_scale")
    p.add_argument("--pavg", type=float, default=2.0)
    p.add_argument("--nt", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    return p.parse_args()


def run(vol_out, vol_params, nt, seed, args, focal_um):
    from calcia import simulate_optical_propagation
    from calcia.scanning import scan_widefield
    from calcia.config.params import PsfParams
    from calcia.traces.simtrace_bridge import generate_time_traces_recruitment

    cfg = C.StriatumConfig(vol_um=vol_params.vol_sz[0], depth_um=vol_params.vol_sz[2],
                           vres=vol_params.vres, nt=nt, seed=seed,
                           motion_model="physio", prot="GCaMP6f",
                           bg_scale=args.bg_scale, pavg=args.pavg)
    depth_um = vol_params.vol_sz[2]
    focal = focal_um if focal_um is not None else min(cfg.focal_depth_um, depth_um / 2)
    supp = min(PSF_SUPPORT_UM, 0.5 * vol_params.vol_sz[0])

    # --- Phase 2: BEST optics (two-scale PSF, flat) ---
    psf_params = PsfParams(imaging_mode="widefield", psf_type="gaussian_analytical",
                           lambda_em_um=cfg.lambda_em_um, obj_na=cfg.obj_na,
                           n=cfg.n_index, psf_sz=(supp, supp, cfg.psf_sz[2]),
                           wf_focal_depth_um=focal)
    opt_out = simulate_optical_propagation(vol_params=vol_params, psf_params=psf_params,
                                           vol_out=vol_out, verbose=0)
    if args.halo_weight and args.halo_weight > 0:
        opt_out.psf = C.broaden_psf_two_scale(opt_out.psf, args.halo_um,
                                              args.halo_weight, vol_params.vres)
        print(f"  two-scale PSF: core + halo {args.halo_um}um w={args.halo_weight}")

    # --- Phase 3: sim-trace recruitment spikes -> calcia calcium ODE ---
    K = len(vol_out.gp_vals)
    has_axons = len(vol_out.bg_proc) > 0
    spike_params = cfg.build_spike(K, has_axons, verbose=0)
    cal_params = cfg.build_cal()
    print(f"  sim-trace recruitment: K={K} regime={args.regime} "
          f"n_cond={args.n_conditions} ens_frac={args.ensemble_frac}")
    time_out, session = generate_time_traces_recruitment(
        spike_params=spike_params, cal_params=cal_params, n_locs=vol_out.locs,
        regime=args.regime, seed=seed, n_conditions=args.n_conditions,
        ensemble_frac=args.ensemble_frac, iti_s=args.iti_s,
        evoked_rate=args.evoked_rate, verbose=0)
    ever_recruited = int(session.recruited.any(axis=0).sum()) if session.recruited.size else 0
    n_events = int(session.recruited.sum())
    print(f"  ensembles: {session.membership.shape[0]} conditions x "
          f"{int(session.membership.sum(1).mean())} cells/ensemble; "
          f"{len(session.trial_onsets)} trials; recruited {ever_recruited}/{K} cells "
          f"({n_events} cell-events)")

    # --- Phase 4: scan (composite OFF, flat stub -> no vignette) ---
    scan_params = cfg.build_scan(verbose=0)
    motion_params = cfg.build_motion()
    scan_out = scan_widefield(vol_out=vol_out, opt_out=opt_out, time_out=time_out,
                              scan_params=scan_params, cam_params=cfg.build_cam(),
                              wf_params=cfg.build_wf(), motion_params=motion_params,
                              spike_params=spike_params, seed=seed)
    noisy, clean = scan_out.mov, scan_out.mov_raw
    params_dict = dict(vol_params=vol_params, psf_params=psf_params,
                       spike_params=spike_params, cal_params=cal_params,
                       scan_params=scan_params, wf_params=cfg.build_wf(),
                       cam_params=cfg.build_cam(), motion_params=motion_params)
    return noisy, clean, time_out, scan_out, opt_out, params_dict, session


def dff_stats(mov, bias):
    sig = np.clip(mov - bias, 0, None)
    f0 = np.percentile(sig, 10, axis=2, keepdims=True)
    dff = (sig - f0) / (f0 + 1e-6)
    return float(np.percentile(dff, 99)), float(np.median(mov.mean(2)))


def main():
    args = parse_args()
    import _instrument; _instrument.start("scan_simtrace")
    t_wall = time.time()

    if args.smoke:
        from demo_static_indicator import load_or_build_phase1
        print("=== sim-trace recruitment re-scan (SMOKE) ===")
        seed = args.seed or 42
        vol_out, vol_params = load_or_build_phase1((80, 80, 50), 0, 2, seed)
        nt = args.nt or 24
        match_name, focal_um = "smoke", None
    else:
        print(f"=== sim-trace recruitment re-scan (matched to {args.match_run}) ===")
        vol_out, vol_params, meta = load_matched_volume(args.match_run)
        seed = args.seed if args.seed is not None else int(meta["seed"])
        nt = args.nt if args.nt is not None else int(meta["nt"])
        match_name = os.path.basename(os.path.normpath(args.match_run))
        focal_um = meta.get("focal_depth_um")

    C.fill_nuclei(vol_out)
    print(f"  frames={nt} seed={seed} regime={args.regime}")

    t0 = time.time()
    noisy, clean, time_out, scan_out, opt_out, params_dict, session = run(
        vol_out, vol_params, nt, seed, args, focal_um)
    bias = C.StriatumConfig().build_cam().bias
    d99, med = dff_stats(noisy, bias)
    print(f"  done ({time.time()-t0:.1f}s)  movie {noisy.shape}  "
          f"dff_p99={d99:.3f} (real ~0.20)  median={med:.0f}")

    if args.no_save:
        return

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fov = vol_params.vol_sz[0]
    tag = ("simtrace_rescan_smoke" if args.smoke
           else f"gcamp_simtrace_{args.regime}_{fov}um")
    run_dir = os.path.join(OUTPUT_ROOT, f"{tag}_{ts}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)

    n_soma = sum(1 for g in vol_out.gp_vals
                 if getattr(g, "soma_mask", None) is not None
                 and np.any(np.asarray(g.soma_mask)))
    total_spikes = int(time_out.spikes.sum()) if time_out.spikes is not None else 0
    meta = dict(kind="gcamp_simtrace_recruitment", matched_run=match_name,
                region="striatum", prot="GCaMP6f", trace_source="simtrace-recruitment",
                regime=args.regime, n_conditions=args.n_conditions,
                ensemble_frac=args.ensemble_frac, iti_s=args.iti_s,
                evoked_rate=args.evoked_rate, n_trials=int(len(session.trial_onsets)),
                optics_method="two-scale-psf", halo_um=args.halo_um,
                halo_weight=args.halo_weight, composite=False,
                focal_depth_um=focal_um, motion_model="physio",
                seed=int(seed), nt=int(nt), dt=1/20,
                vol_sz=list(vol_params.vol_sz), vres=int(vol_params.vres),
                N_neur=int(getattr(vol_params, "N_neur", 0)),
                n_soma=int(n_soma), N_soma_traces=int(time_out.soma.shape[0]),
                total_spikes=total_spikes, movie_shape=list(noisy.shape),
                dff_p99=d99, median=med,
                config=dict(sfrac=params_dict["scan_params"].sfrac, motion_model="physio"),
                timestamp=_dt.datetime.now().isoformat())

    C.save_full_bundle(run_dir, noisy=noisy, clean=clean, vol_out=vol_out,
                       vol_params=vol_params, opt_out=opt_out, time_out=time_out,
                       scan_out=scan_out, params_dict=params_dict, metadata=meta,
                       dt=1/20, make_viz=not args.no_viz)
    # sim-trace ground truth: ensemble membership + per-trial recruitment
    np.savez_compressed(
        os.path.join(run_dir, "simtrace_groundtruth.npz"),
        membership=session.membership, recruited=session.recruited,
        trial_onsets=session.trial_onsets, trial_conditions=session.trial_conditions,
        regime=str(args.regime),
        note=np.array("membership: n_conditions x K ensembles; rows align with "
                      "gp_vals / soma prefix of the co-registered GCaMP run"))
    print(f"\nTotal wall time: {time.time()-t_wall:.1f}s")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()

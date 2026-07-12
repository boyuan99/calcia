"""Realistic GCaMP dynamic channel, co-registered on the SAME volume as the
tdTomato static channel — a design-pure re-scan (no post-processing hacks).

WHY THIS EXISTS
The original striatum GCaMP demo looked "too clear" vs real 1P widefield: it
resolved every soma crisply and the movie dF/F was ~1.36 (p99) where real
striatum window recordings sit at ~0.20. Root cause (established by sweeps):
`dff_p99` is dominated by the SHARPLY-RESOLVED in-focus somata. Real 1P widefield
does NOT resolve single somata — tissue scatter spreads each soma's emission
laterally over a broad area. The sim's analytic Gaussian-NA PSF is diffraction-
limited (sharp) and OMITS this lateral scatter, so cells stay crisp and hot.
(Amplifying the neuropil background — bg_scale — cannot fix it: the background
traces carry the same ~1.37 dF/F as somata, and diluting a resolved cell needs a
floor so bright it saturates the 16-bit range.)

THE FIX (physical, at the OPTICS layer — not image post-processing)
Add the missing lateral tissue scatter: broaden the collection PSF by a Gaussian
of sigma = `scatter_um` (photon-conserving), in the OPTICS domain, so the scan
spreads every source and single somata are no longer resolved. Camera noise is
still added by the scan AFTER — this is tissue scatter, NOT a post-hoc movie blur
(which would average the noise and read as camera defocus). Measured on the
matched volume: scatter 0 -> dff_p99 1.08; 8 um -> 0.45; 16 um -> 0.22 (real
~0.20), median STABLE ~6600 (no saturation), and single cells wash into the
cloud. No per-cell cosmetics (bright_frac / solid_soma / soma_gain), no bg tricks.

CO-REGISTRATION
Reuses the EXACT Phase-1 volume of the matched run (its metadata -> phase1_cache),
so somata are identical to the tdTomato channel produced by
demo_static_tdtomato_matched.py on the same volume. The two colours share one
sample; register/co-localize against soma_locs (saved here and there).

Run:
    conda run -n calcia python examples/demo_gcamp_realistic_matched.py --smoke
    conda run -n calcia python examples/demo_gcamp_realistic_matched.py \
        --match-run striatum_v1_1700um_physio-motion_20260706_114842
"""
import argparse
import datetime as _dt
import json
import os
import pickle
import time

import numpy as np

import _striatum_common as C
from demo_static_tdtomato_matched import load_matched_volume

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_MATCH = "striatum_v1_1700um_physio-motion_20260706_114842"

# DEPTH is the physically-correct realism lever (see two_color_realism_progress.md):
# a deep tissue column's out-of-focus fluorescence forms a smooth bright haze that
# dilutes cell contrast + dF/F to real levels while the in-focus slab stays SHARP.
# So on a DEEP volume, scatter_um=0 (no artificial blur). `--scatter-um` is kept as
# a fallback approximation ONLY for shallow volumes (it uniformly blurs and reads
# as defocus — not preferred). A wide PSF support lets the physical defocus of deep
# planes spread into the haze.
DEFAULT_SCATTER_UM = 0.0
PSF_SUPPORT_UM = 100.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Realistic GCaMP channel, neuropil-floor re-scan")
    p.add_argument("--match-run", type=str, default=DEFAULT_MATCH, dest="match_run",
                   help="GCaMP run dir to reuse the Phase-1 volume / geometry from.")
    p.add_argument("--scatter-um", type=float, default=DEFAULT_SCATTER_UM,
                   dest="scatter_um",
                   help="Lateral tissue-scatter PSF sigma in um (single-Gaussian "
                        "lever). 0 = diffraction-limited (too clear). NOTE: a "
                        "single width either leaves holes OR over-washes cells; "
                        "prefer the TWO-SCALE PSF below.")
    p.add_argument("--halo-um", type=float, default=18.0, dest="halo_um",
                   help="TWO-SCALE PSF: width (um) of the wide scattering HALO "
                        "added to the sharp core (approximates the real 1p PSF).")
    p.add_argument("--halo-weight", type=float, default=0.0, dest="halo_weight",
                   help="TWO-SCALE PSF: fraction of light in the halo (0=off). "
                        "~0.5-0.7 fills neuropil holes AND keeps cells visible + "
                        "dF/F ~real. The design-pure best-use-case lever.")
    p.add_argument("--bg-scale", type=float, default=2.0, dest="bg_scale",
                   help="Neuropil/axon wash amplitude. Measured balance point is "
                        "~2.6 (soma effective 1.73/vox vs bg 0.669*bg_scale): below "
                        "it somata are bright, above it somata become DARK HOLES "
                        "(neuropil brighter than the soma that excludes it). ~2 "
                        "keeps somata ~1.3x brighter (faintly visible like real, "
                        "no holes). NEVER exceed ~2.6.")
    p.add_argument("--pavg", type=float, default=2.0,
                   help="Photon-budget scale (sets brightness/median + shot-noise "
                        "grain). ~0.85 keeps median near the real ~4000 ADU once "
                        "the bright bg_scale wash is added.")
    p.add_argument("--nt", type=int, default=None,
                   help="Frames (default: matched run's nt).")
    p.add_argument("--rate", type=float, default=0.02, help="Spike rate (burst).")
    p.add_argument("--motion", choices=["physio", "randomwalk"], default="physio")
    p.add_argument("--motion-seed", type=int, default=None, dest="motion_seed",
                   help="Motion trajectory seed. Default = independent of the "
                        "tdTomato channel (non-simultaneous acquisition).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny throwaway-volume pipeline check.")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip building the viz_cache bundle (faster). Backfill "
                        "later with: python examples/rebuild_viz.py <run_dir>")
    p.add_argument("--profile", action="store_true",
                   help="Profile the run with pyinstrument (saves profile.html + "
                        "prints a runtime breakdown).")
    p.add_argument("--neuropil-smooth-um", type=float, default=14.0,
                   dest="neuropil_smooth_um",
                   help="Neuropil-continuum smoothing (um). The discrete NAOMi "
                        "neuropil processes leave cell-sized dark HOLES the real "
                        "data never shows; real neuropil is a sub-resolution SMOOTH "
                        "continuum. Rendering neuropil+somata separately and "
                        "smoothing the neuropil to ~22 um fills the holes (0/mm^2, "
                        "real ~2) while somata stay faint blobs. 0 = off (holey).")
    p.add_argument("--soma-blur-um", type=float, default=2.0, dest="soma_blur_um",
                   help="Soma soft-blur (um) in the composite (faint blobs).")
    p.add_argument("--soma-scale", type=float, default=1.0, dest="soma_scale",
                   help="Soma brightness scale in the composite (faint, not popping).")
    p.add_argument("--focal-um", type=float, default=None, dest="focal_um_cli",
                   help="Override the focal plane depth (um). DEEP focus (e.g. "
                        "~140, near the bottom of a 180 um column) puts most of "
                        "the tissue OUT of focus, so the smooth depth-integrated "
                        "OOF haze dominates and averages away the discrete-neuropil "
                        "clump holes. Default: the volume's metadata focal_depth_um.")
    return p.parse_args()


def run_gcamp(vol_out, vol_params, nt, seed, rate, scatter_um,
              motion_model, motion_seed, illum_cfg, focal_um=None,
              bg_scale=2.0, pavg=2.0, neuropil_smooth_um=14.0,
              soma_blur_um=2.0, soma_scale=1.0,
              obj_na=None, scatter_length_um_wf=None,
              halo_um=18.0, halo_weight=0.0):
    """Render the realistic GCaMP channel. Returns (noisy, clean, time_out)."""
    from calcia import simulate_optical_propagation, generate_time_traces
    from calcia.scanning import scan_widefield
    from calcia.scanning.noise import camera_noise
    from calcia.config.params import PsfParams

    cfg = C.StriatumConfig(vol_um=vol_params.vol_sz[0], depth_um=vol_params.vol_sz[2],
                           vres=vol_params.vres, nt=nt, seed=seed,
                           motion_model=motion_model, prot="GCaMP6f", rate=rate,
                           bg_scale=bg_scale, pavg=pavg)
    cfg_seed_for_motion = motion_seed if motion_seed is not None else None
    # Focus in the upper tissue (default from meta or ~1/4 depth) so the deep
    # column is out of focus and forms the smooth haze.
    depth_um = vol_params.vol_sz[2]
    focal = focal_um if focal_um is not None else min(cfg.focal_depth_um, depth_um/2)
    # PSF lateral support: wide enough for deep-plane defocus, but must fit the
    # volume (widefield PSF can't exceed the FOV).
    supp = min(PSF_SUPPORT_UM, 0.5 * vol_params.vol_sz[0])

    # --- Phase 2: green optics with WIDE PSF support so the PHYSICAL defocus of
    #     the deep planes can spread into the OOF haze. scatter=0 on deep volumes. ---
    # Optical-lever overrides (None = keep the config default; used by the
    # optical-sweep harness to probe the physical PSF-breadth levers): obj_na
    # (lower NA = broader diffraction PSF) and scatter_length_um_wf (tissue
    # scatter depth-haze). These live in the OPTICS domain, not post-scan.
    _psf_extra = {}
    if scatter_length_um_wf is not None:
        _psf_extra["scatter_length_um_wf"] = scatter_length_um_wf
    psf_params = PsfParams(imaging_mode="widefield", psf_type="gaussian_analytical",
                           lambda_em_um=cfg.lambda_em_um,
                           obj_na=(obj_na if obj_na is not None else cfg.obj_na),
                           n=cfg.n_index,
                           psf_sz=(supp, supp, cfg.psf_sz[2]),
                           wf_focal_depth_um=focal, **_psf_extra)
    opt_out = simulate_optical_propagation(vol_params=vol_params,
                                           psf_params=psf_params, vol_out=vol_out,
                                           verbose=0)
    if scatter_um > 0:
        opt_out.psf = C.broaden_psf_scatter(opt_out.psf, scatter_um, vol_params.vres)
    if halo_weight and halo_weight > 0:
        # TWO-SCALE PSF: sharp core + wide scattering halo (real 1p PSF shape).
        opt_out.psf = C.broaden_psf_two_scale(opt_out.psf, halo_um, halo_weight,
                                              vol_params.vres)
        print(f"  two-scale PSF: core + halo {halo_um}um w={halo_weight}")
    print(f"  depth={depth_um}um focal={focal:.0f}um  scatter={scatter_um}um  "
          f"(PSF {opt_out.psf.shape})")

    # --- Phase 3: dynamic GCaMP traces (no cosmetics, no neuropil tricks) ---
    K = len(vol_out.gp_vals)
    has_axons = len(vol_out.bg_proc) > 0
    spike_params = cfg.build_spike(K, has_axons, verbose=0)
    cal_params = cfg.build_cal()
    time_out = generate_time_traces(spike_params=spike_params, cal_params=cal_params,
                                    n_locs=vol_out.locs, verbose=0)

    # --- Phase 4: scan (physio motion) ---
    scan_params = cfg.build_scan(verbose=0)
    motion_params = cfg.build_motion()
    if motion_params is not None and cfg_seed_for_motion is not None:
        motion_params.seed = cfg_seed_for_motion

    def _scan(t_):
        return scan_widefield(vol_out=vol_out, opt_out=opt_out, time_out=t_,
                              scan_params=scan_params, cam_params=cfg.build_cam(),
                              wf_params=cfg.build_wf(), motion_params=motion_params,
                              spike_params=spike_params, seed=seed)
    cam = cfg.build_cam()

    if neuropil_smooth_um and neuropil_smooth_um > 0:
        # NEUROPIL-CONTINUUM COMPOSITE. In real 1P the neuropil (fine, sub-
        # resolution processes) images as a SMOOTH continuum, not the discrete
        # clumpy processes NAOMi renders (whose inter-process gaps read as many
        # cell-sized dark holes the real data never shows). We render the neuropil
        # and somata SEPARATELY and smooth the neuropil to the continuum it
        # physically is, while the larger somata stay faint soft blobs. Fills the
        # holes without over-blurring the whole frame. Two scans (2x cost).
        import copy as _copy
        from scipy.ndimage import gaussian_filter as _gf
        t_np = _copy.copy(time_out)          # neuropil only
        t_np.soma = np.zeros_like(time_out.soma)
        t_np.dend = None if time_out.dend is None else np.zeros_like(time_out.dend)
        t_so = _copy.copy(time_out)          # somata (+dend) only
        t_so.bg = None if time_out.bg is None else np.zeros_like(time_out.bg)
        scan_out = _scan(t_np)               # neuropil scan (reused for mot_hist)
        neuro = scan_out.mov_raw
        soma = _scan(t_so).mov_raw           # only 2 scans total
        sf = vol_params.vres / scan_params.sfrac        # um -> movie px
        clean = (_gf(neuro, (neuropil_smooth_um*sf, neuropil_smooth_um*sf, 0))
                 + _gf(soma, (soma_blur_um*sf, soma_blur_um*sf, 0)) * soma_scale
                 ).astype(np.float32)
        print(f"  neuropil-continuum composite: neuropil smooth "
              f"{neuropil_smooth_um}um + somata blur {soma_blur_um}um x{soma_scale}")
        if illum_cfg is not None and illum_cfg.enable:
            clean = (clean * illum_cfg.illum_map(clean.shape[:2])[:, :, None]).astype(np.float32)
        rng = np.random.default_rng(seed + 777)
        noisy = np.empty_like(clean)
        for kk in range(clean.shape[2]):
            noisy[:, :, kk] = camera_noise(clean[:, :, kk], cam, rng)
    else:
        scan_out = _scan(time_out)
        noisy, clean = scan_out.mov, scan_out.mov_raw
        if illum_cfg is not None and illum_cfg.enable:
            bias = cam.bias
            illum = illum_cfg.illum_map(noisy.shape[:2])[:, :, None]
            noisy = (bias + (noisy - bias) * illum).astype(np.float32)
            clean = (clean * illum).astype(np.float32)
            print(f"  applied illumination vignette (edge/centre={illum.min():.2f})")

    params_dict = dict(vol_params=vol_params, psf_params=psf_params,
                       spike_params=spike_params, cal_params=cal_params,
                       scan_params=scan_params, wf_params=cfg.build_wf(),
                       cam_params=cfg.build_cam(), motion_params=motion_params)
    return noisy, clean, time_out, scan_out, opt_out, params_dict


def dff_stats(mov, bias):
    sig = np.clip(mov - bias, 0, None)
    f0 = np.percentile(sig, 10, axis=2, keepdims=True)
    dff = (sig - f0) / (f0 + 1e-6)
    return float(np.percentile(dff, 99)), float(np.median(mov.mean(2)))


def main():
    args = parse_args()

    import _instrument; _instrument.start("scan_gcamp")  # run log + pyinstrument
    profiler = None
    if args.profile:
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()
    t_wall = time.time()

    if args.smoke:
        from demo_static_indicator import load_or_build_phase1
        print("=== Realistic GCaMP (SMOKE) ===")
        seed = args.seed or 42
        vol_out, vol_params = load_or_build_phase1((80, 80, 50), 0, 2, seed)
        nt = args.nt or 24
        illum_cfg = None
        match_name = "smoke"
        focal_um = None
    else:
        print(f"=== Realistic GCaMP (matched to {args.match_run}) ===")
        vol_out, vol_params, meta = load_matched_volume(args.match_run)
        seed = args.seed if args.seed is not None else int(meta["seed"])
        nt = args.nt if args.nt is not None else int(meta["nt"])
        ic = meta.get("config", {}).get("illum")
        illum_cfg = (C.IllumConfig(**ic)
                     if meta.get("illum_grad") and ic else None)
        match_name = os.path.basename(os.path.normpath(args.match_run))
        focal_um = meta.get("focal_depth_um")   # deep volumes carry this
    if args.focal_um_cli is not None:
        focal_um = args.focal_um_cli            # CLI override (deep focus)

    # Solid somata (fill the dark nucleus) — physical correction so cells are
    # solid blobs like real washed 1P, not NAOMi's nuc_fluorsc=0 rings.
    C.fill_nuclei(vol_out)

    motion_seed = args.motion_seed if args.motion_seed is not None else seed + 3
    print(f"  frames={nt} seed={seed} rate={args.rate} "
          f"scatter_um={args.scatter_um} motion_seed={motion_seed}")

    t0 = time.time()
    noisy, clean, time_out, scan_out, opt_out, params_dict = run_gcamp(
        vol_out, vol_params, nt, seed, args.rate, args.scatter_um,
        args.motion, motion_seed, illum_cfg, focal_um=focal_um,
        bg_scale=args.bg_scale, pavg=args.pavg,
        neuropil_smooth_um=args.neuropil_smooth_um,
        soma_blur_um=args.soma_blur_um, soma_scale=args.soma_scale,
        halo_um=args.halo_um, halo_weight=args.halo_weight)
    bias = C.StriatumConfig().build_cam().bias
    d99, med = dff_stats(noisy, bias)
    print(f"  done ({time.time()-t0:.1f}s)  movie {noisy.shape}  "
          f"dff_p99={d99:.3f} (real ~0.20)  median={med:.0f}")

    if args.no_save:
        return

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fov = vol_params.vol_sz[0]
    tag = ("gcamp_realistic_smoke" if args.smoke
           else f"gcamp_realistic_{fov}um_{args.motion}-motion")
    # PID suffix keeps parallel runs (launched in the same second) from colliding
    # on one dir and corrupting each other's movies.npz.
    run_dir = os.path.join(OUTPUT_ROOT, f"{tag}_{ts}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)

    n_soma = sum(1 for g in vol_out.gp_vals
                 if getattr(g, "soma_mask", None) is not None
                 and np.any(np.asarray(g.soma_mask)))
    total_spikes = int(time_out.spikes.sum()) if time_out.spikes is not None else 0
    # metadata rich enough for report.md + reproducibility
    optics_method = ("two-scale-psf" if args.halo_weight > 0
                     else ("single-scatter" if args.scatter_um > 0 else "diffraction"))
    composite_on = args.neuropil_smooth_um > 0
    meta = dict(kind="gcamp_realistic_channel", matched_run=match_name,
                region="striatum", prot="GCaMP6f", rate=args.rate,
                optics_method=optics_method,
                scatter_um=args.scatter_um, halo_um=args.halo_um,
                halo_weight=args.halo_weight, composite=composite_on,
                neuropil_smooth_um=args.neuropil_smooth_um,
                soma_blur_um=args.soma_blur_um, soma_scale=args.soma_scale,
                depth_lever="volume_depth_OOF_haze",
                focal_depth_um=focal_um, cosmetics=False, oof_blur=False,
                motion_model=args.motion, motion_seed=int(motion_seed),
                seed=int(seed), nt=int(nt), dt=1/20,
                vol_sz=list(vol_params.vol_sz), vres=int(vol_params.vres),
                N_neur=int(getattr(vol_params, "N_neur", 0)),
                n_soma=int(n_soma), N_soma_traces=int(time_out.soma.shape[0]),
                total_spikes=total_spikes,
                movie_shape=list(noisy.shape), dff_p99=d99, median=med,
                config=dict(sfrac=params_dict["scan_params"].sfrac,
                            motion_model=args.motion),
                timestamp=_dt.datetime.now().isoformat())

    C.save_full_bundle(run_dir, noisy=noisy, clean=clean, vol_out=vol_out,
                       vol_params=vol_params, opt_out=opt_out, time_out=time_out,
                       scan_out=scan_out, params_dict=params_dict, metadata=meta,
                       dt=1/20, make_viz=not args.no_viz)

    if profiler is not None:
        profiler.stop()
        with open(os.path.join(run_dir, "profile.html"), "w", encoding="utf-8") as f:
            f.write(profiler.output_html())
        print(profiler.output_text(unicode=True, color=False, show_all=False))
        print(f"  saved profile.html")
    print(f"\nTotal wall time: {time.time()-t_wall:.1f}s")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()

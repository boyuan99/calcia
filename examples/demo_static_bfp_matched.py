"""Sparse (2%) static BFP structural channel, CO-REGISTERED to a GCaMP run.

The GCaMP demo (``demo_widefield_striatum_v1.py``) images a DYNAMIC calcium
indicator. This script produces a matching BLUE structural channel for the SAME
sample: constitutive BFP in a SPARSE (default 2%) genetically-defined subset of
the same neurons. BFP is NOT a calcium sensor — its brightness is fixed by
expression level — so the movie is a single structural image plus camera noise
and (physio) sample motion (``SpikeParams.dyn_type='static'``).

BFP is NUCLEAR-ENRICHED: the signal is bright punctate nuclei over a dim diffuse
cytoplasm/neuropil background, exactly the regime the ``bfp`` preset in
``demo_static_indicator.py`` was tuned for (temporal_cv + p999 + histogram tail
vs real striatum). This is a sibling of ``demo_static_tdtomato_matched.py`` — same
co-registration machinery, a different (nuclear, sparse) label.

WHAT MAKES IT "MATCHED" (a two-colour registration test bench):
  * It reuses the EXACT Phase-1 volume of the GCaMP run (read from that run's
    ``metadata.json -> phase1_cache``), so cell ``i`` here IS cell ``i`` in the
    GCaMP run's ``traces.npz``. The volumes are pixel-registered by construction;
    the only cross-channel misalignment is the optics (blue vs green defocus) plus
    an INDEPENDENT per-channel motion trajectory (models non-simultaneous
    acquisition). Pass ``--motion-seed <gcamp_seed+3>`` to motion-lock the two.

SPARSE 2% INFECTION (the whole point of this variant):
  * Only ``--label-frac`` (default 0.02 = 2%) of neurons express BFP. This models
    a sparse AAV: a small genetically-defined subset. The expressing-cell IDs are
    saved as ground truth so downstream code can score co-localization against the
    known correspondence.
  * The neuropil/axon background is down-sampled to the SAME fraction by default
    (``--bg-frac``, default = label_frac): the neuropil of a 2%-labelled sparse
    population is ~2% as dense, so the field is sparse bright nuclei scattered on a
    near-black background — NOT the thick diffuse wash of a dense (tdt/hSyn) label.

Expression is rendered by zeroing the footprint fluorescence AND the nuclear
fluorescence of un-infected cells (values-only edit of a freshly loaded volume;
geometry ground truth in ``locs`` is untouched). Background processes are
independently down-sampled to ``bg_frac``.

Run:
    # quick end-to-end sanity check on a tiny volume (no big cache needed):
    conda run -n calcia python examples/demo_static_bfp_matched.py --smoke

    # the real matched channel for a 1700 um GCaMP run (long: Phase 4 ~20 min):
    conda run -n calcia python examples/demo_static_bfp_matched.py \
        --match-run striatum_v1_1700um_physio-motion_20260706_114842
"""
import argparse
import datetime as _dt
import json
import os
import time

import numpy as np

import _striatum_common as C
# Reuse the validated bfp static preset + stats and the co-registration helpers
# from the sibling demos (importing only defines constants/functions).
from demo_static_indicator import (STATIC_PRESETS, summary_stats,
                                    print_comparison, load_or_build_phase1)
from demo_static_tdtomato_matched import (load_matched_volume, select_expressing,
                                          OUTPUT_ROOT, DEFAULT_MATCH)


def parse_args():
    p = argparse.ArgumentParser(
        description="Sparse (2%) static BFP channel co-registered to a GCaMP run")
    p.add_argument("--match-run", type=str, default=DEFAULT_MATCH,
                   dest="match_run",
                   help="GCaMP run directory (under examples/output/) to match. "
                        "Its metadata.json supplies the Phase-1 cache, FOV, vres, "
                        "seed, frame count and illumination gradient so the BFP "
                        "channel is geometrically identical.")
    p.add_argument("--label-frac", type=float, default=0.02, dest="label_frac",
                   help="Fraction of neurons INFECTED (nuclear BFP expression). "
                        "Default 0.02 = sparse 2%% AAV. Seed-reproducible subset.")
    p.add_argument("--bg-frac", type=float, default=None, dest="bg_frac",
                   help="Fraction of background (neuropil/axon) processes kept. "
                        "Default = label_frac: the neuropil of a 2%%-labelled "
                        "SPARSE population is ~2%% as dense, so the field is sparse "
                        "bright nuclei on a near-black background. Set to 1.0 to "
                        "keep a full diffuse wash (dense-label regime).")
    p.add_argument("--nuc-frac", type=float, default=1.0, dest="nuc_frac",
                   help="Of the INFECTED cells, fraction that show a bright nucleus "
                        "(default 1.0: every infected cell fluoresces its nucleus; "
                        "per-cell brightness still varies via the heavy-tailed "
                        "expression model). Lower it for partial nuclear labelling.")
    p.add_argument("--nt", type=int, default=None,
                   help="Frames (default: same as the matched GCaMP run).")
    p.add_argument("--image-crop-um", type=float, default=0.0, dest="image_crop_um",
                   help="Crop this many um from each lateral side of the OUTPUT "
                        "movie (drop background edge pile-up / match a cropped GCaMP "
                        "run so both colour channels image the SAME central FOV).")
    p.add_argument("--motion", choices=["physio", "randomwalk"], default="physio",
                   help="Sample-motion model (default physio, matching the run).")
    p.add_argument("--motion-seed", type=int, default=None, dest="motion_seed",
                   help="Seed for the BFP channel's motion trajectory. Default = an "
                        "INDEPENDENT trajectory (distinct from the matched GCaMP "
                        "run's motion), modelling non-simultaneous acquisition. Pass "
                        "the GCaMP run's motion seed (its run seed + 3) to share one "
                        "trajectory (simultaneous dual-colour).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny 80x80x50 / 20-frame run on a throwaway volume to "
                        "verify the pipeline WITHOUT the multi-GB matched cache.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override seed (default: the matched run's seed).")
    p.add_argument("--out-dir", type=str, default=None, dest="out_dir",
                   help="Root directory for the output run folder (default: "
                        "examples/output). Point it at a sibling experiment's "
                        "outputs/ to keep the matched BFP channel next to the run "
                        "it registers against.")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-illum", action="store_true",
                   help="Disable the illumination vignette (matched run's gradient "
                        "is applied by default; real BFP has a moderate vignette).")
    p.add_argument("--illum-floor", type=float, default=None, dest="illum_floor",
                   help="Override the vignette floor (edge/centre). Higher = more "
                        "uniform.")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip building the viz_cache bundle (faster).")
    p.add_argument("--profile", action="store_true",
                   help="Profile the run with pyinstrument (saves profile.html).")
    return p.parse_args()


def run_bfp(vol_out, vol_params, nt, seed, expr_ids, motion_model, illum_cfg,
            motion_seed, bg_frac, nuc_frac, focal_um=None,
            halo_um=0.0, halo_weight=0.0):
    """Render the sparse static BFP channel. Returns (noisy, clean) H x W x T.

    halo_weight>0 adds a MODEST two-scale scattering halo to the PSF (1P scatter
    realism). Keep it small for a nuclear label: a big halo (tdt's 0.8/28um) would
    smear the sparse nuclei into a wash and destroy their identifiability.
    """
    from calcia import simulate_optical_propagation, generate_time_traces
    from calcia.scanning import scan_widefield
    from calcia.scanning.noise import camera_noise
    from calcia.config.params import (PsfParams, WidefieldParams, SpikeParams,
                                      ScanParams, CameraNoiseParams, MotionParams)
    from scipy.ndimage import gaussian_filter

    P = STATIC_PRESETS["bfp"]
    expr_set = set(int(i) for i in expr_ids)

    # --- expression: silence NON-infected cells (values-only). BFP is nuclear-
    #     enriched, so an un-infected cell must emit NOTHING from either its
    #     footprint (soma + dendrites) OR its nucleus. The nucleus loop in
    #     scan_widefield paints gp_nuc unconditionally, so zeroing the footprint
    #     alone is not enough — gp_nuc must be zeroed too. ---
    nuc_rng = np.random.default_rng(seed + 99)
    n_off = n_nuc = 0
    for i in range(len(vol_out.gp_vals)):
        infected = i in expr_set
        if not infected:
            vol_out.gp_vals[i].fluorescence[:] = 0.0
            n_off += 1
    # Nuclear brightness: infected cells get nuc_fl (a nuc_frac subset if <1),
    # everyone else 0. Per-cell brightness still varies through the heavy-tailed
    # static expression model (min_mod=gamma) at scan time.
    for i in range(len(vol_out.gp_nuc)):
        idx, _ = vol_out.gp_nuc[i]
        on = (i in expr_set) and (nuc_rng.random() < nuc_frac)
        vol_out.gp_nuc[i] = (np.asarray(idx), float(P["nuc_fl"]) if on else 0.0)
        n_nuc += int(on)
    print(f"  BFP+ (infected): {len(expr_set)} cells   bright nuclei: {n_nuc}   "
          f"silenced: {n_off}")

    # --- background axons/neuropil down-sampled to the sparse labelling fraction.
    #     A 2%-labelled sparse population has ~2% as dense a neuropil, so the field
    #     is sparse bright nuclei on a near-black background (not a diffuse wash). ---
    if vol_out.bg_proc and bg_frac < 1.0:
        bg_rng = np.random.default_rng(seed + 54321)
        keep = bg_rng.random(len(vol_out.bg_proc)) < bg_frac
        for j, bp in enumerate(vol_out.bg_proc):
            if not keep[j]:
                bp.fluorescence[:] = 0.0
        print(f"  background processes kept: {int(keep.sum())}/{len(keep)} "
              f"(bg_frac={bg_frac:.3f})")
    else:
        print(f"  background processes: ALL {len(vol_out.bg_proc)} kept "
              f"(bg_frac={bg_frac})")

    # --- Phase 2: widefield optics (BLUE emission). BFP is punctate / high-
    #     contrast, so a compact PSF (sharp nuclei) — NOT the wide wash PSF the
    #     dense tdt channel uses. ---
    depth_um = vol_params.vol_sz[2]
    focal = focal_um if focal_um is not None else min(30.0, depth_um / 2)
    # A halo needs PSF support wide enough to contain it; keep compact otherwise.
    supp = (min(60.0, 0.5 * vol_params.vol_sz[0]) if halo_weight and halo_weight > 0
            else 12.0)
    psf_params = PsfParams(
        imaging_mode="widefield", psf_type="gaussian_analytical",
        lambda_em_um=P["lambda_em_um"], obj_na=0.8, n=1.35,
        psf_sz=(supp, supp, 20.0), wf_focal_depth_um=focal,
        scatter_length_um_wf=P["scatter_length_um_wf"])
    opt_out = simulate_optical_propagation(
        vol_params=vol_params, psf_params=psf_params, vol_out=vol_out, verbose=0)
    if halo_weight and halo_weight > 0:
        # MODEST two-scale scatter halo (sharp nuclear core + narrow scatter skirt).
        opt_out.psf = C.broaden_psf_two_scale(opt_out.psf, halo_um, halo_weight,
                                              vol_params.vres)
        print(f"  two-scale PSF: core + modest halo {halo_um}um w={halo_weight}")
    print(f"  depth={depth_um}um focal={focal:.0f}um  (PSF {opt_out.psf.shape})")

    # --- Phase 3: STATIC traces (constant per cell). The heavy-tailed expression
    #     spread (min_mod=gamma) makes the nuclei a bright, heavy-tailed population
    #     rather than uniform dots. ---
    K = len(vol_out.gp_vals)
    has_axons = P["axonflag"] and len(vol_out.bg_proc) > 0
    sp_kw = {} if P["gamma"] is None else dict(min_mod=P["gamma"])
    spike_params = SpikeParams(
        K=K, nt=nt, dt=1 / 20, N_bg=0, dyn_type="static", prot="bfp",
        dendflag=P["dendflag"], axonflag=has_axons, bg_scale=P["bg_scale"],
        verbose=0, **sp_kw)
    time_out = generate_time_traces(spike_params=spike_params,
                                    n_locs=vol_out.locs, verbose=0)

    # --- Phase 4: widefield camera scan (physio motion) ---
    wf_params = WidefieldParams(pavg=P["pavg"], lambda_ex_um=P["lambda_ex_um"],
                                sigma_abs=P["sigma_abs"], phi=P["phi"],
                                qe_det=P["qe_det"])
    _buff = 30 if motion_model == "physio" else 10
    scan_params = ScanParams(scan_buff=_buff, motion=True, sfrac=2, verbose=0)
    motion_params = (MotionParams(model="physio", seed=motion_seed)
                     if motion_model == "physio" else None)
    cam = CameraNoiseParams(qe=1.0, dark_rate=P["dark_rate"], t_exp=1 / 20,
                            read_noise=P["read_noise"], gain_e_per_adu=P["gain"],
                            bias=P["bias"])
    scan_out = scan_widefield(vol_out=vol_out, opt_out=opt_out, time_out=time_out,
                              scan_params=scan_params, cam_params=cam,
                              motion_params=motion_params, wf_params=wf_params,
                              spike_params=spike_params, seed=seed)

    # --- Out-of-focus haze: blur the CLEAN photon image, THEN re-add camera noise
    #     per pixel (optical blur precedes the sensor; blurring the noisy movie
    #     would average the noise away and kill the temporal CV). ---
    if P["oof_blur_um"] > 0:
        blur_px = P["oof_blur_um"] * vol_params.vres / scan_params.sfrac
        clean = gaussian_filter(scan_out.mov_raw, sigma=(blur_px, blur_px, 0))
        rng = np.random.default_rng(seed + 777)
        noisy = np.empty_like(clean)
        for kk in range(clean.shape[2]):
            noisy[:, :, kk] = camera_noise(clean[:, :, kk], cam, rng)
        noisy = noisy.astype(np.float32); clean = clean.astype(np.float32)
    else:
        noisy, clean = scan_out.mov, scan_out.mov_raw

    # --- Non-uniform illumination gradient, IDENTICAL to the matched GCaMP run so
    #     both colour channels share the same vignette (registration-relevant). ---
    if illum_cfg is not None and illum_cfg.enable:
        bias = cam.bias
        illum = illum_cfg.illum_map(noisy.shape[:2])[:, :, None]
        noisy = (bias + (noisy - bias) * illum).astype(np.float32)
        clean = (clean * illum).astype(np.float32)
        print(f"  applied matched illumination gradient "
              f"(edge/centre={illum.min():.2f})")

    params_dict = dict(vol_params=vol_params, psf_params=psf_params,
                       spike_params=spike_params, scan_params=scan_params,
                       wf_params=wf_params, cam_params=cam,
                       motion_params=motion_params)
    return noisy, clean, opt_out, time_out, scan_out, params_dict


def main():
    args = parse_args()
    import _instrument; _instrument.start("scan_bfp")  # run log + pyinstrument
    profiler = None
    if args.profile:
        from pyinstrument import Profiler
        profiler = Profiler(); profiler.start()
    _twall = time.time()

    if args.smoke:
        print("=" * 60)
        print("Sparse static BFP  (SMOKE — throwaway 80x80x50 volume)")
        print("=" * 60)
        vol_sz = (80, 80, 50); vres = 2; seed = args.seed or 42
        nt = args.nt or 20
        vol_out, vol_params = load_or_build_phase1(vol_sz, 0, vres, seed)
        illum_cfg = None
        match_name = "smoke"
        focal_um = None
    else:
        print("=" * 60)
        print(f"Sparse static BFP  (matched to {args.match_run})")
        print("=" * 60)
        vol_out, vol_params, match_meta = load_matched_volume(args.match_run)
        seed = args.seed if args.seed is not None else int(match_meta["seed"])
        nt = args.nt if args.nt is not None else int(match_meta["nt"])
        vres = int(match_meta["vres"])
        ic = match_meta.get("config", {}).get("illum")
        if args.no_illum:
            illum_cfg = None
        elif ic is not None and args.illum_floor is not None:
            ic = dict(ic); ic["floor"] = args.illum_floor
            illum_cfg = C.IllumConfig(**ic)
        else:
            illum_cfg = (C.IllumConfig(**ic)
                         if match_meta.get("illum_grad") and ic else None)
        match_name = os.path.basename(os.path.normpath(args.match_run))
        focal_um = match_meta.get("focal_depth_um")

    # Sparse neuropil follows the infection fraction unless overridden. Clamp both
    # to [0, 1]; 2% by default.
    bg_frac = args.bg_frac if args.bg_frac is not None else args.label_frac

    # NOTE: no fill_nuclei — BFP is nuclear-ENRICHED, the bright nucleus IS the
    # signal. fill_nuclei (cytoplasmic tdt) would merge nuclei into somata and zero
    # gp_nuc, destroying that signal.

    motion_seed = (args.motion_seed if args.motion_seed is not None
                   else seed + 1234)
    shared = (motion_seed == seed + 3)
    print(f"  volume {vol_params.vol_sz} um  vres={vres}  frames={nt}  "
          f"seed={seed}  label_frac={args.label_frac}  bg_frac={bg_frac:.3f}  "
          f"motion={args.motion}")
    print(f"  motion_seed={motion_seed}  "
          f"({'SHARED with GCaMP' if shared else 'INDEPENDENT of GCaMP'})")

    # --- infection subset (co-registration ground truth) ---
    expr_ids, soma_ids = select_expressing(vol_out, args.label_frac, seed)
    soma_locs = np.asarray(vol_out.locs)[soma_ids]

    t0 = time.time()
    noisy, clean, opt_out, time_out, scan_out, params_dict = run_bfp(
        vol_out, vol_params, nt, seed, expr_ids, args.motion, illum_cfg,
        motion_seed, bg_frac, args.nuc_frac, focal_um=focal_um)

    if args.image_crop_um and args.image_crop_um > 0:
        cpx = int(round(args.image_crop_um * vres / params_dict["scan_params"].sfrac))
        if cpx > 0:
            noisy = noisy[cpx:-cpx, cpx:-cpx, :]
            clean = clean[cpx:-cpx, cpx:-cpx, :]
            for _a in ("mov", "mov_raw", "mov_infocus", "mov_oof"):
                _m = getattr(scan_out, _a, None)
                if _m is not None:
                    setattr(scan_out, _a, _m[cpx:-cpx, cpx:-cpx, :])
            print(f"  imaged central FOV: cropped {cpx}px/side "
                  f"({args.image_crop_um:.0f}um) -> {noisy.shape[:2]}")
    print(f"  done ({time.time()-t0:.1f}s)  movie {noisy.shape}  "
          f"noisy[{noisy.min():.0f}, {noisy.max():.0f}]")
    print_comparison("bfp", noisy)

    if args.no_save:
        return

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fov = vol_params.vol_sz[0]
    frac_tag = f"{args.label_frac*100:.0f}pct"
    tag = ("striatum_bfp_static_smoke" if args.smoke
           else f"striatum_bfp_static_{frac_tag}_{fov}um_{args.motion}-motion")
    out_root = args.out_dir if args.out_dir else OUTPUT_ROOT
    os.makedirs(out_root, exist_ok=True)
    run_dir = os.path.join(out_root, f"{tag}_{ts}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)

    meta = dict(
        kind="static_bfp_channel", matched_run=match_name,
        matched_gcamp_metadata=(None if args.smoke else args.match_run),
        region="striatum", prot="bfp", dyn_type="static", nuclear=True,
        label_frac=float(args.label_frac), bg_frac=float(bg_frac),
        nuc_frac=float(args.nuc_frac),
        optics_method="diffraction",
        focal_depth_um=focal_um,
        n_infected=int(len(expr_ids)), n_soma_total=int(len(soma_ids)),
        n_soma=int(len(soma_ids)), N_soma_traces=int(time_out.soma.shape[0]),
        total_spikes=0,
        motion_model=args.motion, seed=int(seed), motion_seed=int(motion_seed),
        motion_shared_with_gcamp=bool(shared), nt=int(nt), dt=1 / 20,
        image_crop_um=float(args.image_crop_um),
        vres=int(vres), vol_sz=list(vol_params.vol_sz),
        N_neur=int(getattr(vol_params, "N_neur", 0)),
        lambda_em_um=STATIC_PRESETS["bfp"]["lambda_em_um"],
        movie_shape=list(noisy.shape),
        config=dict(sfrac=params_dict["scan_params"].sfrac,
                    motion_model=args.motion),
        timestamp=_dt.datetime.now().isoformat(), stats=summary_stats(noisy))

    C.save_full_bundle(run_dir, noisy=noisy, clean=clean, vol_out=vol_out,
                       vol_params=vol_params, opt_out=opt_out, time_out=time_out,
                       scan_out=scan_out, params_dict=params_dict, metadata=meta,
                       dt=1 / 20, make_viz=not args.no_viz)

    # --- channel-specific co-registration ground truth (which cells are infected) ---
    expr_mask = np.isin(soma_ids, expr_ids)
    np.savez_compressed(
        os.path.join(run_dir, "bfp_expression.npz"),
        expr_ids=expr_ids, soma_ids=soma_ids, soma_locs=soma_locs,
        expr_mask=expr_mask, label_frac=np.float64(args.label_frac),
        note=np.array("expr_ids index the matched GCaMP run's soma_neurons/"
                      "soma_locs (same Phase-1 volume)"))

    if profiler is not None:
        profiler.stop()
        with open(os.path.join(run_dir, "profile.html"), "w", encoding="utf-8") as f:
            f.write(profiler.output_html())
        print(profiler.output_text(unicode=True, color=False, show_all=False))
    print(f"\nTotal wall time: {time.time()-_twall:.1f}s")
    print(f"Output: {run_dir}")
    print(f"  bfp_expression.npz  ({len(expr_ids)}/{len(soma_ids)} cells "
          f"infected; expr_ids align with the GCaMP run's soma rows)")


if __name__ == "__main__":
    main()

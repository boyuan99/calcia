"""Static tdTomato structural channel, CO-REGISTERED to a GCaMP dynamic run.

The GCaMP demo (``demo_widefield_striatum_v1.py``) images a DYNAMIC calcium
indicator: fluorescence tracks spikes, the movie flickers with activity. This
script produces the matching RED structural channel for the SAME sample:
constitutive tdTomato in a random subset of the same neurons. tdTomato is NOT a
calcium sensor — its brightness is fixed by expression level — so the movie is a
single structural image plus camera noise and (physio) sample motion. That is
exactly the ``SpikeParams.dyn_type='static'`` regime already validated by
``demo_static_indicator.py`` (tdt preset: 5/5 summary stats vs real striatum).

WHAT MAKES IT "MATCHED" (the whole point — a two-colour registration test bench):
  * It reuses the EXACT Phase-1 volume of the GCaMP run (read from that run's
    ``metadata.json -> phase1_cache``). Neuron positions are therefore identical,
    so cell ``i`` here IS cell ``i`` in the GCaMP run's ``traces.npz``
    (``soma_neurons[i]`` / ``soma_locs[i]``). The underlying volumes are pixel-
    registered by construction, so the ONLY cross-channel misalignment is the
    optics (red vs green defocus) plus an INDEPENDENT per-channel motion
    trajectory (this red channel uses a motion seed distinct from the GCaMP run
    by default, modelling non-simultaneous acquisition) — exactly the residual a
    real two-colour registration pipeline must recover. Pass ``--motion-seed``
    equal to the GCaMP run's motion seed to instead motion-lock the two channels
    (simultaneous dual-colour).
  * Only a fraction (default 1/2) of neurons express tdTomato. This mirrors
    sparse structural labelling: the green channel reports activity in all cells,
    the red channel marks a genetically-defined subset. The expressing-cell IDs
    are saved as ground truth so downstream code can score co-localization /
    registration error against the known correspondence.

This does NOT modify any core calcia code or the GCaMP demo. It is a standalone
companion script built on the existing static-indicator machinery.

Non-expression is rendered by zeroing the footprint fluorescence of unlabelled
cells (values-only edit of a freshly loaded volume; geometry ground truth in
``locs`` is untouched). Background axons/neuropil are independently down-sampled
to the same labelling fraction (the neuropil of a 50%-labelled population is
~50% as dense).

Run:
    # quick end-to-end sanity check on a tiny volume (no big cache needed):
    conda run -n calcia python examples/demo_static_tdtomato_matched.py --smoke

    # the real matched channel for the 1700 um GCaMP run (long: Phase 4 ~20 min):
    conda run -n calcia python examples/demo_static_tdtomato_matched.py \
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
# Reuse the validated tdt static preset + stats from the sibling demo. Importing
# the module only defines constants/functions (its work is under __main__).
from demo_static_indicator import (STATIC_PRESETS, summary_stats,
                                    print_comparison, load_or_build_phase1)


OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_MATCH = "striatum_v1_1700um_physio-motion_20260706_114842"


def parse_args():
    p = argparse.ArgumentParser(
        description="Static tdTomato channel co-registered to a GCaMP run")
    p.add_argument("--match-run", type=str, default=DEFAULT_MATCH,
                   dest="match_run",
                   help="GCaMP run directory (under examples/output/) to match. "
                        "Its metadata.json supplies the Phase-1 cache, FOV, vres, "
                        "seed, frame count and illumination gradient so the "
                        "tdTomato channel is geometrically identical.")
    p.add_argument("--label-frac", type=float, default=0.5, dest="label_frac",
                   help="Fraction of neurons whose SOMA expresses tdTomato "
                        "(default 0.5). Seed-reproducible random subset.")
    p.add_argument("--bg-scale", type=float, default=None, dest="bg_scale",
                   help="Neuropil/axon wash amplitude (overrides the tdt preset "
                        "2.5). Higher = more washed (cells diluted into neuropil).")
    p.add_argument("--neuropil-smooth-um", type=float, default=14.0,
                   dest="neuropil_smooth_um",
                   help="Neuropil-continuum smoothing (um): render neuropil+somata "
                        "separately and smooth the neuropil to the sub-resolution "
                        "continuum, filling the discrete-process cell-holes real "
                        "data never shows. 0 = off.")
    p.add_argument("--soma-blur-um", type=float, default=4.0, dest="soma_blur_um")
    p.add_argument("--soma-scale", type=float, default=1.0, dest="soma_scale")
    p.add_argument("--bg-frac", type=float, default=1.0, dest="bg_frac",
                   help="Fraction of background (neuropil/axon) processes kept "
                        "(default 1.0 = FULL diffuse wash). The out-of-focus "
                        "neuropil haze is the DOMINANT signal in real 1P tdt "
                        "widefield and is what smooths the field into a low-"
                        "contrast cloud (real spatial CV ~0.11); halving it lets "
                        "the sparse somata show through as a too-sharp granular "
                        "field (CV ~0.29). Keep it at 1.0 unless you specifically "
                        "want de-washed single cells. Decoupled from --label-frac "
                        "because the neuropil integrates a dense overlapping "
                        "population over depth and stays a thick wash even when "
                        "only ~half the somata are labelled.")
    p.add_argument("--scatter-um", type=float, default=0.0, dest="scatter_um",
                   help="Lateral scatter PSF sigma in um. DEFAULT 0: on a DEEP "
                        "volume the out-of-focus haze from the tissue column washes "
                        "cells physically (depth is the real lever). Use >0 only as "
                        "a fallback approximation on shallow volumes (uniform blur, "
                        "reads as defocus).")
    p.add_argument("--halo-um", type=float, default=18.0, dest="halo_um",
                   help="TWO-SCALE PSF: width (um) of the wide scattering HALO "
                        "added to the sharp core (approximates the real 1p PSF).")
    p.add_argument("--halo-weight", type=float, default=0.0, dest="halo_weight",
                   help="TWO-SCALE PSF: fraction of light in the halo (0=off). "
                        "~0.5-0.7 fills neuropil holes AND keeps cells + texture. "
                        "The design-pure best-use-case lever.")
    p.add_argument("--nt", type=int, default=None,
                   help="Frames (default: same as the matched GCaMP run). Lower "
                        "it for a faster preview on the same volume.")
    p.add_argument("--image-crop-um", type=float, default=0.0, dest="image_crop_um",
                   help="Crop this many um from each lateral side of the OUTPUT "
                        "movie. Use it to (a) drop the background edge pile-up frame "
                        "and (b) match a GCaMP run generated with --gen-margin-um / "
                        "--image-crop-um so the two colour channels image the SAME "
                        "central FOV. Pass the same value as the matched GCaMP run.")
    p.add_argument("--motion", choices=["physio", "randomwalk"], default="physio",
                   help="Sample-motion model (default physio, matching the run).")
    p.add_argument("--motion-seed", type=int, default=None, dest="motion_seed",
                   help="Seed for the tdTomato channel's motion trajectory. "
                        "Default = an INDEPENDENT trajectory (distinct from the "
                        "matched GCaMP run's motion), modelling non-simultaneous "
                        "acquisition so a registration pipeline must recover the "
                        "cross-channel misalignment. Pass the GCaMP run's motion "
                        "seed (its run seed + 3) to instead share one trajectory "
                        "(simultaneous dual-colour).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny 80x80x50 / 20-frame run on a throwaway volume to "
                        "verify the pipeline WITHOUT the multi-GB matched cache.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override seed (default: the matched run's seed).")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-illum", action="store_true",
                   help="Disable the illumination vignette. Real tdt is nearly "
                        "UNIFORM (floor/median ~0.80); the GCaMP vignette darkens "
                        "the edges and weakens the background floor.")
    p.add_argument("--illum-floor", type=float, default=None, dest="illum_floor",
                   help="Override the vignette floor (edge/centre). Higher = "
                        "weaker vignette (more uniform). Real tdt ~ uniform.")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip building the viz_cache bundle (faster). Backfill "
                        "later with: python examples/archive/rebuild_viz.py <run_dir>")
    p.add_argument("--profile", action="store_true",
                   help="Profile the run with pyinstrument (saves profile.html).")
    return p.parse_args()


def load_matched_volume(match_run):
    """Load the EXACT Phase-1 volume the GCaMP run used, from its metadata.

    Returns (vol_out, vol_params, meta_dict). Guarantees identical neuron
    positions -> per-cell co-registration with that run's traces.npz.
    """
    run_dir = match_run
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(OUTPUT_ROOT, match_run)
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"matched run not found: {meta_path}")
    meta = json.load(open(meta_path))
    cache = meta["phase1_cache"]
    if not os.path.exists(cache):
        raise SystemExit(
            f"matched run's Phase-1 cache is missing:\n  {cache}\n"
            f"Re-run the GCaMP demo (or point --match-run at a run whose "
            f"cache still exists).")
    print(f"  loading matched Phase-1 cache ({os.path.getsize(cache)/1e9:.1f} GB)")
    with open(cache, "rb") as f:
        vol_out, vol_params = pickle.load(f)
    return vol_out, vol_params, meta


def select_expressing(vol_out, label_frac, seed):
    """Pick the random subset of somata that express tdTomato.

    Returns (expr_ids sorted ndarray, soma_ids ndarray). Cell ordering matches
    the GCaMP run: the somata are the contiguous prefix of gp_vals, so these IDs
    index directly into that run's soma_neurons / soma_locs.
    """
    soma_ids = np.array([i for i, cfd in enumerate(vol_out.gp_vals)
                         if np.any(np.asarray(cfd.soma_mask))], dtype=np.int64)
    rng = np.random.default_rng(seed + 12345)
    n_expr = int(round(label_frac * len(soma_ids)))
    expr = (np.sort(rng.choice(soma_ids, n_expr, replace=False))
            if n_expr else np.array([], dtype=np.int64))
    return expr, soma_ids


def run_tdt(vol_out, vol_params, nt, seed, expr_ids, label_frac,
            motion_model, illum_cfg, motion_seed, bg_frac, scatter_um,
            focal_um=None, bg_scale=None, neuropil_smooth_um=14.0,
            soma_blur_um=4.0, soma_scale=1.0, halo_um=18.0, halo_weight=0.0):
    """Render the static tdTomato channel. Returns (noisy, clean) H x W x T."""
    from calcia import simulate_optical_propagation, generate_time_traces
    from calcia.scanning import scan_widefield
    from calcia.scanning.noise import camera_noise
    from calcia.config.params import (PsfParams, WidefieldParams, SpikeParams,
                                      ScanParams, CameraNoiseParams, MotionParams)
    from scipy.ndimage import gaussian_filter

    P = STATIC_PRESETS["tdt"]
    expr_set = set(int(i) for i in expr_ids)

    # --- expression: zero the footprints of NON-expressing cells (values-only;
    #     soma + dendrite voxels of an unlabelled cell emit nothing). ---
    n_off = 0
    for i in range(len(vol_out.gp_vals)):
        if i not in expr_set:
            vol_out.gp_vals[i].fluorescence[:] = 0.0
            n_off += 1
    print(f"  tdTomato+: {len(expr_set)} cells   silenced: {n_off}")

    # --- background axons/neuropil. This diffuse out-of-focus haze DOMINATES
    #     real 1P tdt widefield and smooths the field into a low-contrast cloud
    #     (real spatial CV ~0.11). Keep it FULL (bg_frac=1.0) by default: even
    #     when only ~half the somata are labelled the neuropil integrates a dense
    #     overlapping population over depth and stays a thick wash. Down-sampling
    #     it (bg_frac<1) lets the sparse somata show through as a too-sharp
    #     granular field. ---
    if vol_out.bg_proc and bg_frac < 1.0:
        bg_rng = np.random.default_rng(seed + 54321)
        keep = bg_rng.random(len(vol_out.bg_proc)) < bg_frac
        for j, bp in enumerate(vol_out.bg_proc):
            if not keep[j]:
                bp.fluorescence[:] = 0.0
        print(f"  background processes kept: {int(keep.sum())}/{len(keep)} "
              f"(bg_frac={bg_frac})")
    else:
        print(f"  background processes: ALL {len(vol_out.bg_proc)} kept "
              f"(full wash, bg_frac={bg_frac})")

    # --- Phase 2: widefield optics (RED emission). On a DEEP volume the physical
    #     out-of-focus haze from the tissue column washes the cells (depth is the
    #     lever); focus in the upper tissue so the deep column defocuses. WIDE PSF
    #     support lets that defocus spread. scatter_um>0 is a shallow-volume fallback. ---
    depth_um = vol_params.vol_sz[2]
    focal = focal_um if focal_um is not None else min(30.0, depth_um / 2)
    supp = min(100.0, 0.5 * vol_params.vol_sz[0])
    psf_params = PsfParams(
        imaging_mode="widefield", psf_type="gaussian_analytical",
        lambda_em_um=P["lambda_em_um"], obj_na=0.8, n=1.35,
        psf_sz=(supp, supp, 20.0), wf_focal_depth_um=focal,
        scatter_length_um_wf=P["scatter_length_um_wf"])
    opt_out = simulate_optical_propagation(
        vol_params=vol_params, psf_params=psf_params, vol_out=vol_out, verbose=0)
    if scatter_um > 0:
        opt_out.psf = C.broaden_psf_scatter(opt_out.psf, scatter_um, vol_params.vres)
    if halo_weight and halo_weight > 0:
        # TWO-SCALE PSF: sharp core + wide scattering halo (real 1p PSF shape).
        opt_out.psf = C.broaden_psf_two_scale(opt_out.psf, halo_um, halo_weight,
                                              vol_params.vres)
        print(f"  two-scale PSF: core + halo {halo_um}um w={halo_weight}")
    print(f"  depth={depth_um}um focal={focal:.0f}um  scatter={scatter_um}um  "
          f"(PSF {opt_out.psf.shape})")

    # --- Phase 3: STATIC traces (constant per cell, no spikes / calcium) ---
    K = len(vol_out.gp_vals)
    has_axons = P["axonflag"] and len(vol_out.bg_proc) > 0
    _bgs = P["bg_scale"] if bg_scale is None else bg_scale
    spike_params = SpikeParams(
        K=K, nt=nt, dt=1 / 20, N_bg=0, dyn_type="static", prot="tdt",
        dendflag=P["dendflag"], axonflag=has_axons, bg_scale=_bgs,
        verbose=0)
    time_out = generate_time_traces(spike_params=spike_params,
                                    n_locs=vol_out.locs, verbose=0)

    # --- Phase 4: widefield camera scan (physio motion) ---
    wf_params = WidefieldParams(pavg=P["pavg"], lambda_ex_um=P["lambda_ex_um"],
                                sigma_abs=P["sigma_abs"], phi=P["phi"],
                                qe_det=P["qe_det"])
    _buff = 30 if motion_model == "physio" else 10
    scan_params = ScanParams(scan_buff=_buff, motion=True, sfrac=2, verbose=0)
    # Independent motion by default (motion_seed distinct from the GCaMP run's
    # seed+3), so the two channels are NOT motion-locked.
    motion_params = (MotionParams(model="physio", seed=motion_seed)
                     if motion_model == "physio" else None)
    cam = CameraNoiseParams(qe=1.0, dark_rate=P["dark_rate"], t_exp=1 / 20,
                            read_noise=P["read_noise"], gain_e_per_adu=P["gain"],
                            bias=P["bias"])
    def _scan(t_):
        return scan_widefield(vol_out=vol_out, opt_out=opt_out, time_out=t_,
                              scan_params=scan_params, cam_params=cam,
                              motion_params=motion_params, wf_params=wf_params,
                              spike_params=spike_params, seed=seed)

    if neuropil_smooth_um and neuropil_smooth_um > 0:
        # NEUROPIL-CONTINUUM COMPOSITE (see the GCaMP demo): render neuropil +
        # somata separately and smooth the neuropil to the sub-resolution SMOOTH
        # continuum it physically is, filling the discrete-process cell-holes the
        # real data never shows, while somata stay faint soft blobs.
        import copy as _copy
        from scipy.ndimage import gaussian_filter as _gf
        t_np = _copy.copy(time_out); t_np.soma = np.zeros_like(time_out.soma)
        t_np.dend = None if time_out.dend is None else np.zeros_like(time_out.dend)
        t_so = _copy.copy(time_out)
        t_so.bg = None if time_out.bg is None else np.zeros_like(time_out.bg)
        scan_out = _scan(t_np); neuro = scan_out.mov_raw   # reuse for mot_hist
        soma = _scan(t_so).mov_raw                          # 2 scans total
        sf = vol_params.vres / scan_params.sfrac
        clean = (_gf(neuro, (neuropil_smooth_um*sf, neuropil_smooth_um*sf, 0))
                 + _gf(soma, (soma_blur_um*sf, soma_blur_um*sf, 0)) * soma_scale
                 ).astype(np.float32)
        print(f"  neuropil-continuum composite: neuropil {neuropil_smooth_um}um "
              f"+ somata {soma_blur_um}um x{soma_scale}")
        rng = np.random.default_rng(seed + 777)
        noisy = np.empty_like(clean)
        for kk in range(clean.shape[2]):
            noisy[:, :, kk] = camera_noise(clean[:, :, kk], cam, rng)
    else:
        scan_out = _scan(time_out)
        noisy, clean = scan_out.mov, scan_out.mov_raw

    # --- Non-uniform illumination gradient, IDENTICAL to the matched GCaMP run
    #     so both colour channels share the same vignette (registration-relevant).
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
    import _instrument; _instrument.start("scan_tdt")  # run log + pyinstrument
    profiler = None
    if args.profile:
        from pyinstrument import Profiler
        profiler = Profiler(); profiler.start()
    _twall = time.time()

    if args.smoke:
        print("=" * 60)
        print("Static tdTomato  (SMOKE — throwaway 80x80x50 volume)")
        print("=" * 60)
        vol_sz = (80, 80, 50); vres = 2; seed = args.seed or 42
        nt = args.nt or 20
        vol_out, vol_params = load_or_build_phase1(vol_sz, 0, vres, seed)
        illum_cfg = None
        match_meta = None
        match_name = "smoke"
        focal_um = None
    else:
        print("=" * 60)
        print(f"Static tdTomato  (matched to {args.match_run})")
        print("=" * 60)
        vol_out, vol_params, match_meta = load_matched_volume(args.match_run)
        seed = args.seed if args.seed is not None else int(match_meta["seed"])
        nt = args.nt if args.nt is not None else int(match_meta["nt"])
        vres = int(match_meta["vres"])
        # Match the run's illumination gradient exactly (or None if it had none).
        # Real tdt (data/real/tdt-bfp) is nearly UNIFORM (floor/median ~0.80), so
        # the strong GCaMP vignette (floor 0.05) wrongly darkens the edges and
        # drops the background floor; --no-illum / --illum-floor make it uniform.
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
        focal_um = match_meta.get("focal_depth_um")   # deep volumes carry this

    # Solid somata (fill the dark nucleus) BEFORE expression masking, so expressing
    # cells are solid blobs like real washed 1P and non-expressing cells (zeroed in
    # run_tdt) stay fully dark. Physical correction, not a cosmetic.
    C.fill_nuclei(vol_out)

    # Independent motion trajectory by default: distinct from the matched GCaMP
    # run's motion seed (its run seed + 3), so the red channel is NOT motion-
    # locked to green (models non-simultaneous acquisition). Pass --motion-seed
    # <gcamp_seed+3> to share one trajectory.
    motion_seed = (args.motion_seed if args.motion_seed is not None
                   else seed + 1234)
    shared = (motion_seed == seed + 3)
    print(f"  volume {vol_params.vol_sz} um  vres={vres}  frames={nt}  "
          f"seed={seed}  label_frac={args.label_frac}  motion={args.motion}")
    print(f"  motion_seed={motion_seed}  "
          f"({'SHARED with GCaMP' if shared else 'INDEPENDENT of GCaMP'})")

    # --- expression subset (co-registration ground truth) ---
    expr_ids, soma_ids = select_expressing(vol_out, args.label_frac, seed)
    soma_locs = np.asarray(vol_out.locs)[soma_ids]

    t0 = time.time()
    noisy, clean, opt_out, time_out, scan_out, params_dict = run_tdt(
        vol_out, vol_params, nt, seed, expr_ids, args.label_frac, args.motion,
        illum_cfg, motion_seed, args.bg_frac, args.scatter_um, focal_um=focal_um,
        bg_scale=args.bg_scale, neuropil_smooth_um=args.neuropil_smooth_um,
        soma_blur_um=args.soma_blur_um, soma_scale=args.soma_scale,
        halo_um=args.halo_um, halo_weight=args.halo_weight)
    # Image only the central FOV: drop the background neuropil edge pile-up frame
    # and match a GCaMP run cropped the same way (same central region -> the two
    # colour channels stay co-registered). Crop in movie px: um * vres / sfrac.
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
    print_comparison("tdt", noisy)

    if args.no_save:
        return

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fov = vol_params.vol_sz[0]
    tag = ("striatum_tdt_static_smoke" if args.smoke
           else f"striatum_tdt_static_{fov}um_{args.motion}-motion")
    # PID suffix keeps parallel runs (launched in the same second) from colliding
    # on one dir and corrupting each other's movies.npz.
    run_dir = os.path.join(OUTPUT_ROOT, f"{tag}_{ts}_{os.getpid()}")
    os.makedirs(run_dir, exist_ok=True)

    meta = dict(
        kind="static_tdtomato_channel", matched_run=match_name,
        matched_gcamp_metadata=(None if args.smoke else args.match_run),
        region="striatum", prot="tdt", dyn_type="static",
        label_frac=float(args.label_frac), bg_frac=float(args.bg_frac),
        optics_method=("two-scale-psf" if args.halo_weight > 0
                       else ("single-scatter" if args.scatter_um > 0 else "diffraction")),
        scatter_um=float(args.scatter_um), halo_um=float(args.halo_um),
        halo_weight=float(args.halo_weight),
        composite=bool(args.neuropil_smooth_um > 0),
        neuropil_smooth_um=float(args.neuropil_smooth_um),
        soma_blur_um=float(args.soma_blur_um), soma_scale=float(args.soma_scale),
        depth_lever="volume_depth_OOF_haze",
        focal_depth_um=focal_um,
        n_expressing=int(len(expr_ids)), n_soma_total=int(len(soma_ids)),
        n_soma=int(len(soma_ids)), N_soma_traces=int(time_out.soma.shape[0]),
        total_spikes=0,
        motion_model=args.motion, seed=int(seed), motion_seed=int(motion_seed),
        motion_shared_with_gcamp=bool(shared), nt=int(nt), dt=1/20,
        image_crop_um=float(args.image_crop_um),
        vres=int(vres), vol_sz=list(vol_params.vol_sz),
        N_neur=int(getattr(vol_params, "N_neur", 0)),
        lambda_em_um=STATIC_PRESETS["tdt"]["lambda_em_um"],
        movie_shape=list(noisy.shape),
        config=dict(sfrac=params_dict["scan_params"].sfrac,
                    motion_model=args.motion),
        timestamp=_dt.datetime.now().isoformat(), stats=summary_stats(noisy))

    # full reproducible bundle (movies/optics/footprints/params/traces/report/viz)
    C.save_full_bundle(run_dir, noisy=noisy, clean=clean, vol_out=vol_out,
                       vol_params=vol_params, opt_out=opt_out, time_out=time_out,
                       scan_out=scan_out, params_dict=params_dict, metadata=meta,
                       dt=1/20, make_viz=not args.no_viz)

    # --- channel-specific co-registration ground truth (which cells express) ---
    expr_mask = np.isin(soma_ids, expr_ids)
    np.savez_compressed(
        os.path.join(run_dir, "tdtomato_expression.npz"),
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
    print(f"  tdtomato_expression.npz  ({len(expr_ids)}/{len(soma_ids)} cells "
          f"express; expr_ids align with the GCaMP run's soma rows)")


if __name__ == "__main__":
    main()

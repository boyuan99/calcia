"""Striatum widefield simulation.

1P widefield simulation sized for striatum imaging via a cranial window
over the exposed striatum surface.

Default config (the long run) — imaging-window prep, NOT a GRIN miniscope:
  - vol_sz = (1500, 1500, 150) um   vol_depth = 0   vres = 1
    (large ~1.5 mm window FOV; thin ~150 um scatter-limited 1P depth)
  - N_neur = 1000 cleanly-resolvable cells via SPARSE labelling
    (~3% of anatomical density, ~440 cells/mm^2 projected). The anatomical
    neur_density default would instead give ~33,750 overlapping/washed cells.
  - nt = 200 frames @ 20 Hz (10 s) — matches the real striatum window
    recordings (200-frame, ~1.7 mm FOV, GCaMP tiffs)
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
    p.add_argument("--vol-um", type=int, default=None, dest="vol_um",
                   help="Lateral FOV in um for the full run (square). Overrides "
                        "the (1000) preset; depth stays 60 um. Use 1700 to cover "
                        "the full ~1.7 mm real-sample FOV.")
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
    p.add_argument("--bg-scale", type=float, default=1.0, dest="bg_scale",
                   help="Neuropil/axon trace amplitude scale. Default 1.0 (raw "
                        "NAOMi wash) for the DENSE/washed striatum mode: bright "
                        "neuropil fills the field into a smooth cloud like the "
                        "real samples. Use 0.1 for the old sparse de-washed mode "
                        "where individual cells are visible. The dense axon "
                        "background is the dominant 1P widefield wash-out source; "
                        "0.1 takes brightest-frame spatial CV from ~0.65 to ~0.95 "
                        "(cells become visible). 1.0 = raw NAOMi amplitude "
                        "(washed). Default 0.1")
    p.add_argument("--vres", type=int, default=None,
                   help="Voxels per um (overrides preset; smoke/medium default "
                        "2, full-run defaults to 1 because the 1.5 mm window FOV "
                        "would otherwise need ~10 GB/array). Use 1 to cut memory "
                        "~8x")
    p.add_argument("--n-neur", type=int, default=None, dest="n_neur",
                   help="Number of fluorescing (labelled) neurons. Overrides the "
                        "preset. The full run targets ~1000 cleanly-resolvable "
                        "cells (sparse Cre+virus labelling, ~3%% of anatomical "
                        "density); leave None on smoke/medium to use the full "
                        "anatomical density (neur_density-driven).")
    p.add_argument("--neur-density", type=float, default=None, dest="neur_density",
                   help="Neuron density in neurons/mm^3 (density-driven path; "
                        "only used when --n-neur is not set). Default preset is "
                        "the full anatomical 1e5. Pass e.g. 70000 for a "
                        "70%%-density volume; the run dir is tagged lowdens<pct>.")
    p.add_argument("--soma-gain", type=float, default=None, dest="soma_gain",
                   help="Multiply soma-voxel fluorescence by this factor after "
                        "Phase 1 (values-only rescale of gp_vals via soma_mask; "
                        "no shape re-run). Real GCaMP concentrates in the soma, "
                        "so somata should outshine neuropil; the raw NAOMi values "
                        "leave them ~equal, washed out by out-of-focus dendrite "
                        "haze in widefield. Full-run default 3.0 lifts mean-image "
                        "CV from ~0.44 toward the real ~0.73 (cells become "
                        "visible). 1.0 = no boost.")
    p.add_argument("--no-illum", action="store_true",
                   help="Disable the non-uniform (Gaussian) widefield "
                        "illumination profile. The full run applies it by "
                        "default: real 1P widefield is dominated by a bright-"
                        "centre/dark-edge illumination gradient (LED beam + "
                        "window vignetting) that NAOMi's uniform model lacks — "
                        "it is the single biggest visual difference from the "
                        "real striatum samples.")
    p.add_argument("--motion-model", type=str, default="randomwalk",
                   dest="motion_model", choices=["randomwalk", "physio"],
                   help="Sample-motion model for Phase 4. 'randomwalk' (legacy "
                        "default) = bounded +/-1 voxel integer walk. 'physio' = "
                        "realistic AR(1) drift+jitter + heavy-tailed jumps + "
                        "intra-frame motion blur, fit to real NoRMCorre striatum "
                        "shifts (see data/real/398_09192025_gcamp_mc_shifts.mat). "
                        "physio auto-bumps scan_buff to 30 (real range ~+/-26 um).")
    p.add_argument("--prot", type=str, default="GCaMP6f")
    p.add_argument("--dendrite-strategy", type=str, default="morphology",
                   dest="dendrite_strategy",
                   choices=["morphology", "field", "space_colonization"],
                   help="Phase-1 dendrite generation strategy: 'morphology' "
                        "(Dijkstra, default), 'field' (fast statistical density "
                        "cloud -- makes dense full runs tractable), or "
                        "'space_colonization'.")
    p.add_argument("--depth-um", type=int, default=None, dest="depth_um",
                   help="Volume/imaging depth in um (overrides preset). Deep "
                        "(e.g. 180) gives the washed out-of-focus haze of real 1P.")
    p.add_argument("--psf-support-um", type=float, default=None,
                   dest="psf_support_um",
                   help="Lateral PSF array support (um). Preset default is 12; the "
                        "two-scale halo is CLIPPED to this footprint, so a wide halo "
                        "needs a wide support. KEEPER uses 100 (-> 30um FWHM blobs).")
    p.add_argument("--halo-um", type=float, default=0.0, dest="halo_um",
                   help="Two-scale PSF halo radius (um). >0 broadens the emission "
                        "PSF into a soft wide halo (KEEPER realism); 0 = off. "
                        "Needs --psf-support-um wide enough (>~3x halo) or it clips.")
    p.add_argument("--halo-weight", type=float, default=0.8, dest="halo_weight",
                   help="Two-scale PSF halo weight (fraction of energy in the halo).")
    p.add_argument("--bright-frac", type=float, default=None, dest="bright_frac",
                   help="Fraction of somata boosted so they pop out (cosmetic). "
                        "0 = none -> cells blend into the wash (KEEPER-like). "
                        "Overrides preset (full preset default is 0.2).")
    p.add_argument("--focal-depth", type=float, default=None, dest="focal_depth",
                   help="Focal-plane depth (um into the volume); overrides preset.")
    p.add_argument("--hemo-abs-mult", type=float, default=1.0, dest="hemo_abs_mult",
                   help="Scale the widefield hemoglobin absorption (col mask). >1 "
                        "deepens vessel shadows toward black (KEEPER-like) without "
                        "regenerating the volume. 1.0 = preset default.")
    p.add_argument("--gen-margin-um", type=float, default=0.0, dest="gen_margin_um",
                   help="Generate the volume this many um LARGER on each lateral "
                        "side, then image only the central target FOV. Pushes the "
                        "background neuropil edge pile-up outside the imaged region "
                        "(clean borders while keeping a bright washed background). "
                        "Requires regenerating Phase-1 at the larger size.")
    p.add_argument("--image-crop-um", type=float, default=0.0, dest="image_crop_um",
                   help="Crop this many um from each lateral side of the OUTPUT "
                        "movie (no change to generation, so any cached volume is "
                        "reused). Drops the background edge pile-up frame for free "
                        "at the cost of a smaller imaged FOV. ~110 clears the frame "
                        "with a 100um-support PSF. Composes with --gen-margin-um.")
    p.add_argument("--neuropil-fill", type=float, default=None, dest="neuropil_fill",
                   help="AxonParams.maxfill: fraction of the background volume the "
                        "axon neuropil fills (preset default 0.5). RAISE it to "
                        "generate a DENSER neuropil that fills the discrete-process "
                        "gaps (the dark 'holes') at the source, instead of blurring "
                        "over them with the PSF. Changes Phase-1 -> new cache.")
    p.add_argument("--process-thickness", type=float, default=None,
                   dest="process_thickness",
                   help="DendParams.thicknessScale (preset default 0.5): dilation "
                        "width of dendrite/neuropil processes. Higher = fatter "
                        "processes that close the inter-process gaps. Changes "
                        "Phase-1 -> new cache.")
    p.add_argument("--simtrace-design", type=str, default=None,
                   dest="simtrace_design",
                   choices=["hawkes_smallworld", "hawkes_scale_free",
                            "hmm_gated_hawkes", "shared_drive_osc",
                            "hmm_gated_drive"],
                   help="Drive Phase 3 spikes from a sim-trace coupling design "
                        "(via calcia.traces.simtrace_bridge) instead of the "
                        "built-in burst/hawkes. Reuses the cached Phase 1 "
                        "volume unchanged. The hawkes_* / *_hawkes designs build "
                        "a dense K x K matrix (only feasible for K up to a few "
                        "thousand); on a full-size dense volume (K ~ 50k) use a "
                        "scalable design: 'shared_drive_osc' (B: shared rhythm) "
                        "or 'hmm_gated_drive' (E over B: brain-state-gated "
                        "rhythm), neither of which builds a K x K matrix.")
    p.add_argument("--simtrace-rate", type=float, default=0.08,
                   dest="simtrace_rate",
                   help="Baseline rate scale for the sim-trace design "
                        "(Gamma scale for per-neuron mu; ~0.08 -> a few Hz). "
                        "Only used with --simtrace-design.")
    p.add_argument("--no-viz-prep", action="store_true",
                   help="skip pre-building the visualization bundle "
                        "(vessel/soma meshes + 2D outlines) at the end")
    return p.parse_args()


def phase1_signature(vol_sz, vol_depth, vres, seed, region, n_neur,
                     neur_density=None, strategy="morphology",
                     neuropil_fill=None, process_thickness=None):
    h = hashlib.sha1()
    h.update(repr((tuple(vol_sz), vol_depth, vres, seed, region,
                   n_neur, neur_density, strategy,
                   neuropil_fill, process_thickness)).encode())
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
    # Each preset carries its own default vres and N_neur. The full run models
    # an imaging-window prep (NOT a GRIN miniscope): a large ~1.5 mm FOV but
    # only a thin ~150 um scatter-limited depth. ~1000 cleanly-resolvable cells
    # are placed by SPARSE labelling (N_neur=1000 ~ 3% of anatomical density,
    # areal ~440 cells/mm^2), not by the full anatomical neur_density default
    # (which would give ~33,750 overlapping/washed neurons here). vres defaults
    # to 1 for the full run because vres=2 would need ~10 GB/array at this FOV.
    # Each preset is a StriatumConfig (single source of truth). smoke/medium are
    # "clean" physical runs (no illum gradient / soma boosting / blur cosmetics);
    # the full run is the DENSE/WASHED main line (anatomical density + bg_scale
    # ~1.0 + cosmetics) that matches the real striatum samples. 1P widefield only
    # images ~50-80 um deep, so the volume is a shallow 60 um.
    if args.smoke:
        tag = "striatum_smoke"
        cfg = C.StriatumConfig(
            vol_um=80, depth_um=50, vres=2,
            nt=(args.nt if args.nt is not None else 30),
            solid_soma=False, bright_frac=0.0, oof_blur_um=0.0,
            illum=C.IllumConfig(enable=False))
    elif args.medium:
        tag = "striatum_medium"
        cfg = C.StriatumConfig(
            vol_um=250, depth_um=100, vres=2,
            nt=(args.nt if args.nt is not None else 300),
            solid_soma=False, bright_frac=0.0, oof_blur_um=0.0,
            illum=C.IllumConfig(enable=False))
    else:
        tag = "striatum_v1"
        cfg = C.StriatumConfig(
            vol_um=(args.vol_um if args.vol_um is not None else 1000),
            depth_um=60, vres=1,
            nt=(args.nt if args.nt is not None else 200))

    # CLI overrides applied to every preset
    cfg.seed = args.seed
    cfg.rate = args.rate
    cfg.smod = args.smod
    cfg.burst_mean = args.burst_mean
    cfg.bg_scale = args.bg_scale
    cfg.motion_model = args.motion_model
    cfg.prot = args.prot
    if args.depth_um is not None:
        cfg.depth_um = args.depth_um
    if args.bright_frac is not None:
        cfg.bright_frac = args.bright_frac
    if args.focal_depth is not None:
        cfg.focal_depth_um = args.focal_depth
    if args.psf_support_um is not None:
        cfg.psf_sz = (args.psf_support_um, args.psf_support_um, cfg.psf_sz[2])
    # Generate a larger volume than we image, then crop the movie to the central
    # target FOV (see --gen-margin-um). Keeps the imaged field free of the
    # background neuropil edge pile-up without dimming the wash. cfg.vol_um now
    # holds the GENERATED size (so vol_sz / phase1 signature / density scale with
    # it); target_vol_um is the FOV we actually keep.
    target_vol_um = cfg.vol_um
    if args.gen_margin_um > 0:
        cfg.vol_um = int(round(target_vol_um + 2 * args.gen_margin_um))
    if args.vres is not None:
        cfg.vres = args.vres
    if args.n_neur is not None:
        cfg.n_neur = args.n_neur
    if args.neur_density is not None:
        cfg.neur_density = args.neur_density
    if args.soma_gain is not None:
        cfg.soma_gain = args.soma_gain
    if args.no_illum:
        cfg.illum.enable = False

    # Make the run self-identifying when driven by a sim-trace design, so its
    # output dir is not confused with a native-trace run of the same volume.
    if args.simtrace_design is not None:
        tag = f"{tag}_simtrace_{args.simtrace_design}"

    # Self-identify density-driven runs whose density differs from the full
    # anatomical 1e5/mm^3, so a low-density volume's output dir is never
    # confused with the full-density main line.
    if cfg.n_neur is None and cfg.neur_density != 1e5:
        tag = f"{tag}_lowdens{round(cfg.neur_density / 1e5 * 100)}pct"

    # Derived locals (the pipeline body below reads these; cfg stays the single
    # source they come from).
    vol_sz = cfg.vol_sz
    vol_depth = cfg.vol_depth
    vres = cfg.vres
    n_neur = cfg.n_neur
    nt = cfg.nt
    soma_gain = cfg.soma_gain
    solid_soma = cfg.solid_soma
    bright_frac = cfg.bright_frac
    bright_gain = cfg.bright_gain
    oof_blur_um = cfg.oof_blur_um
    illum_grad = cfg.illum.enable
    seed = cfg.seed
    rate = cfg.rate
    smod = cfg.smod
    burst_mean = cfg.burst_mean
    bg_scale = cfg.bg_scale
    prot = cfg.prot
    dt = cfg.dt

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, f"{tag}_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(SHARED_DIR, exist_ok=True)
    # Real-time run log: tee stdout/stderr (line-buffered) so full-run progress
    # streams live to disk and survives a kill (verbose=1 below feeds it).
    C.tee_stdout(f"{tag}_{ts}", output_dir=run_dir)

    n_vox = (vol_sz[0] * vres) * (vol_sz[1] * vres) * (vol_sz[2] * vres)
    print("=" * 60)
    print(f"Striatum widefield  ({'SMOKE TEST' if args.smoke else 'full run'})")
    print(f"  Run dir:   {run_dir}")
    print(f"  Volume:    {vol_sz} um   vol_depth={vol_depth}   vres={vres}")
    print(f"  Voxels:    {n_vox:,}  ({n_vox * 4 / 1e9:.2f} GB / float32 array)")
    print(f"  Neurons:   N_neur={n_neur if n_neur is not None else 'density-driven'}"
          + (f"  (~{n_neur/(np.prod(vol_sz)/1e9):.0f}/mm^3, "
             f"{n_neur/(vol_sz[0]*vol_sz[1]/1e6):.0f}/mm^2 projected)"
             if n_neur is not None else ""))
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
    from calcia.scanning import scan_widefield

    timings = {}
    profiler = Profiler()
    vol_params = cfg.build_vol_params()

    # ==================================================================
    # Phase 1: Neural volume (shared cache keyed on geometry params)
    # ==================================================================
    sig = phase1_signature(vol_sz, vol_depth, vres, seed, "striatum", n_neur,
                           None if n_neur is not None else cfg.neur_density,
                           strategy=args.dendrite_strategy,
                           neuropil_fill=args.neuropil_fill,
                           process_thickness=args.process_thickness)
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
        # Denser neuropil (fills the discrete-process 'holes' at the source rather
        # than blurring over them). maxfill raises axon fill fraction; thicknessScale
        # fattens processes. Region defaults overwrite dtParams/atParams but leave
        # thicknessScale/maxfill, so these overrides stick.
        _axon_params = _dend_params = None
        if args.neuropil_fill is not None:
            from calcia.config.params import AxonParams
            _axon_params = AxonParams(maxfill=args.neuropil_fill)
            print(f"  neuropil density: AxonParams.maxfill={args.neuropil_fill} "
                  f"(default 0.5)")
        if args.process_thickness is not None:
            from calcia.config.params import DendParams
            _dend_params = DendParams(thicknessScale=args.process_thickness)
            print(f"  process thickness: DendParams.thicknessScale="
                  f"{args.process_thickness} (default 0.5)")
        profiler.start()
        vol_out = simulate_neural_volume(
            vol_params=vol_params, seed=seed, verbose=1,
            dendrite_strategy=args.dendrite_strategy,
            axon_params=_axon_params, dend_params=_dend_params,
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

    # SOMA BRIGHTNESS re-calibration (post-Phase-1, NOT baked into the shared
    # cache). Real GCaMP concentrates in the soma cytoplasm, so somata outshine
    # the neuropil; raw NAOMi values make them ~equal, so the sparse somata get
    # washed out by out-of-focus dendrite haze in widefield (soma/bg contrast
    # ~1.0, mean-image CV ~0.44 vs real ~0.73). Boosting soma-voxel fluorescence
    # (values-only rescale via soma_mask — no shape re-run) lifts cells above
    # the haze so they are visible.
    if soma_gain != 1.0:
        n_boost = 0
        for cfd in vol_out.gp_vals:
            sm = np.asarray(cfd.soma_mask)
            if sm.any():
                cfd.fluorescence[sm] *= soma_gain
                n_boost += 1
        print(f"  soma_gain={soma_gain}: boosted {n_boost} soma footprints")

    # SOLID SOMA: make cells SOLID bright blobs, not rings. NAOMi gives the
    # nucleus zero fluorescence (nuc_fluorsc=0), leaving a dark centre (the
    # cytoplasmic-GCaMP "ring"), but the real washed 1P samples show cells as
    # solid light blobs. Merge each nucleus's voxels into its soma footprint
    # with the soma's own (median) fluorescence + soma_mask, so the nucleus is
    # as bright AND activity-modulated as the cytoplasm. Values-only edit of the
    # cached volume (no shape re-run).
    if solid_soma:
        n_solid = 0
        for i in range(min(len(vol_out.gp_vals), len(vol_out.gp_nuc))):
            nuc_idx = np.asarray(vol_out.gp_nuc[i][0])
            cfd = vol_out.gp_vals[i]
            sm = np.asarray(cfd.soma_mask)
            if len(nuc_idx) == 0 or not sm.any():
                continue
            fill = float(np.median(cfd.fluorescence[sm]))
            cfd.indices = np.concatenate([cfd.indices, nuc_idx])
            cfd.fluorescence = np.concatenate(
                [cfd.fluorescence,
                 np.full(len(nuc_idx), fill, cfd.fluorescence.dtype)])
            cfd.soma_mask = np.concatenate([sm, np.ones(len(nuc_idx), bool)])
            n_solid += 1
        vol_out.gp_nuc = [(np.array([], dtype=np.int64), 0.0)
                          for _ in vol_out.gp_nuc]
        print(f"  solid_soma: filled {n_solid} nuclei -> solid cells (no rings)")

    # BRIGHTNESS HETEROGENEITY: the real 1P data is a STRONG washed background
    # with a SUBSET of neurons that stay clearly resolvable on top (heterogeneous
    # GCaMP expression / activity). A UNIFORM wash (post-blur) wrongly smears
    # those clear cells away. Instead boost a random fraction of somata so they
    # stand out sharply while the rest blend into the dense neuropil haze — no
    # uniform blur needed. Values-only edit; subset is seed-reproducible.
    if bright_frac > 0.0:
        soma_ids = [i for i, cfd in enumerate(vol_out.gp_vals)
                    if np.any(np.asarray(cfd.soma_mask))]
        rng = np.random.default_rng(seed + 1)
        n_bright = int(round(bright_frac * len(soma_ids)))
        bright_ids = set(rng.choice(soma_ids, n_bright, replace=False)
                         .tolist()) if n_bright else set()
        for i in bright_ids:
            cfd = vol_out.gp_vals[i]
            sm = np.asarray(cfd.soma_mask)
            cfd.fluorescence[sm] *= bright_gain
        print(f"  bright_frac={bright_frac}: {n_bright}/{len(soma_ids)} somata "
              f"x{bright_gain} (clear cells on washed background)")

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
    psf_params = cfg.build_psf()
    if args.hemo_abs_mult != 1.0:
        from dataclasses import replace as _replace
        psf_params = _replace(
            psf_params, hemo_abs_wf=psf_params.hemo_abs_wf * args.hemo_abs_mult)
        print(f"  hemoglobin absorption x{args.hemo_abs_mult} "
              f"-> hemo_abs_wf={psf_params.hemo_abs_wf:.4f} (deeper vessels)")
    profiler.start()
    opt_out = simulate_optical_propagation(
        vol_params=vol_params, psf_params=psf_params,
        vol_out=vol_out, verbose=1,
    )
    profiler.stop()
    save_profile(profiler, run_dir, "phase2")
    profiler.reset()
    # Two-scale PSF lateral-scatter broadening (KEEPER realism practice): add a
    # soft wide halo to the diffraction-limited emission PSF so resolved cells
    # wash into a smooth cloud (real 1P look) instead of staying sharp points.
    if args.halo_um and args.halo_um > 0:
        opt_out.psf = C.broaden_psf_two_scale(
            opt_out.psf, args.halo_um, args.halo_weight, vres)
        print(f"  two-scale PSF: halo={args.halo_um}um weight={args.halo_weight}"
              f"  -> PSF {opt_out.psf.shape}")
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
    spike_params = cfg.build_spike(K, has_axons)
    cal_params = cfg.build_cal()
    profiler.start()
    if args.simtrace_design is not None:
        # Phase 3 spikes come from a sim-trace coupling design; the cached
        # Phase 1 volume is reused unchanged. use_locs=False avoids building a
        # K x K x D spatial-distance matrix (K ~ thousands here).
        from calcia.traces.simtrace_bridge import (
            generate_time_traces_simtrace, SCALABLE_DESIGNS,
            hawkes_smallworld, hawkes_scale_free, hmm_gated_hawkes,
            shared_drive_osc, hmm_gated_drive,
        )
        _factories = {
            "hawkes_smallworld": lambda: hawkes_smallworld(
                rate=args.simtrace_rate, use_locs=False),
            "hawkes_scale_free": lambda: hawkes_scale_free(
                rate=args.simtrace_rate),
            "hmm_gated_hawkes": lambda: hmm_gated_hawkes(
                rate=args.simtrace_rate, use_locs=False),
            "shared_drive_osc": lambda: shared_drive_osc(),
            "hmm_gated_drive": lambda: hmm_gated_drive(),
        }
        # A dense K x K coupling matrix is infeasible past a few thousand
        # components; block it rather than OOM on a full-size volume.
        if args.simtrace_design not in SCALABLE_DESIGNS and K > 5000:
            raise SystemExit(
                f"Design '{args.simtrace_design}' builds a dense {K}x{K} "
                f"coupling matrix (~{K*K*8/1e9:.0f} GB) — infeasible for "
                f"K={K}. Use a scalable design on this volume: "
                f"{sorted(SCALABLE_DESIGNS)}.")
        print(f"  Phase 3 driven by sim-trace design: {args.simtrace_design} "
              f"(rate={args.simtrace_rate})")
        time_out = generate_time_traces_simtrace(
            spike_params=spike_params, cal_params=cal_params,
            model_factory=_factories[args.simtrace_design](),
            n_locs=vol_out.locs, seed=seed, verbose=1,
        )
    else:
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
    #
    # SELF-DESCRIBING LAYOUT: the per-component trace arrays hold BOTH the real
    # neuron somata AND the appended background/neuropil processes, so their row
    # count (e.g. 2321) is NOT the neuron count. The pipeline appends bg
    # dendrites AFTER the somata, so the somata are the contiguous prefix
    # [:n_soma]. We store n_soma explicitly and pre-split the ground-truth
    # neuron rows so downstream code never has to read the log or reverse-
    # engineer footprint sizes. n_soma is counted from soma_mask (robust to
    # placing fewer cells than N_neur requested). locs[:n_soma] are the soma
    # centres (verified C-order aligned with the somatic rows).
    gp = vol_out.gp_vals
    n_soma = sum(1 for g in gp
                 if getattr(g, "soma_mask", None) is not None
                 and np.any(np.asarray(g.soma_mask)))
    locs = np.asarray(vol_out.locs)
    soma = time_out.soma.astype(np.float32)
    spikes = (time_out.spikes if time_out.spikes is not None
              else np.zeros((0, nt), dtype=np.uint8))
    traces = dict(
        n_soma=np.int64(n_soma),          # split point: rows [:n_soma] = neurons
        trace_axes=np.array("KT"),        # trace arrays are (K=component, T=frame)
        locs_axes=np.array("Kxyz"),       # locs are (K, 3) voxel coords x,y,z
        soma=soma,                        # full per-component (neurons + bg)
        spikes=spikes,
        locs=locs,
        soma_neurons=soma[:n_soma],       # ground-truth: the real cell somata
        soma_locs=locs[:n_soma],          # their centres (x,y,z voxels)
    )
    if spikes.shape[0] == soma.shape[0]:
        traces["spikes_neurons"] = spikes[:n_soma]
    if time_out.dend is not None:
        dend = time_out.dend.astype(np.float32)
        traces["dend"] = dend
        traces["dend_neurons"] = dend[:n_soma]
    if time_out.bg is not None:
        traces["bg"] = time_out.bg.astype(np.float32)
    np.savez_compressed(os.path.join(run_dir, "traces.npz"), **traces)
    print(f"  checkpoint: traces.npz  ({n_soma} real neurons + "
          f"{soma.shape[0] - n_soma} background components; "
          f"use soma_neurons / n_soma)")

    # ==================================================================
    # Phase 4: Widefield camera scan
    # ==================================================================
    print("\n[PHASE 4] Widefield camera scanning")
    t0 = time.time()
    scan_params = cfg.build_scan()
    wf_params = cfg.build_wf()
    cam_params = cfg.build_cam()
    motion_params = cfg.build_motion()
    if motion_params is not None:
        print(f"  motion model: physio (scan_buff={scan_params.scan_buff}, "
              f"seed={motion_params.seed})")
    profiler.start()
    scan_out = scan_widefield(
        vol_out=vol_out, opt_out=opt_out, time_out=time_out,
        scan_params=scan_params, cam_params=cam_params,
        wf_params=wf_params, spike_params=spike_params,
        motion_params=motion_params, seed=seed,
    )
    profiler.stop()
    save_profile(profiler, run_dir, "phase4")
    profiler.reset()
    timings["phase4"] = time.time() - t0
    print(f"  done in {timings['phase4']:.1f}s   movie {scan_out.mov.shape}")

    # Image only the central FOV, leaving the background neuropil edge pile-up
    # (which grows with bg_scale and is spread ~PSF-half-width inward) outside the
    # imaged field. Two contributions, both cropped from the output movie:
    #   * gen_margin_um: extra volume generated beyond the target FOV, and
    #   * image_crop_um: extra crop of an as-generated volume (no regen).
    # Crop is symmetric in movie pixels: um * vres / sfrac.
    _crop_um = args.gen_margin_um + args.image_crop_um
    if _crop_um > 0:
        cpx = int(round(_crop_um * vres / scan_params.sfrac))
        if cpx > 0:
            for _attr in ("mov", "mov_raw", "mov_infocus", "mov_oof"):
                _m = getattr(scan_out, _attr, None)
                if _m is not None:
                    setattr(scan_out, _attr, _m[cpx:-cpx, cpx:-cpx, :])
            print(f"  imaged central FOV: cropped {cpx}px/side ({_crop_um:.0f}um) "
                  f"-> {scan_out.mov.shape[:2]}  (generated {cfg.vol_um}um, "
                  f"imaged ~{target_vol_um - 2 * args.image_crop_um:.0f}um)")

    # Out-of-focus haze (smoothing): real 1P widefield integrates a large
    # defocused PSF over the whole depth, blurring the field into a SMOOTH
    # washed cloud. NAOMi's widefield PSF keeps too much fine structure, so the
    # raw sim field is grainy/speckled (dark-void area ~35% vs real ~1.7%). A
    # spatial Gaussian on the signal (above the bias pedestal) approximates the
    # extra out-of-focus blur and matches the real smoothness. ~20 um sigma.
    bias = cam_params.bias
    if illum_grad and oof_blur_um > 0:
        from scipy.ndimage import gaussian_filter
        blur_px = oof_blur_um * vres / scan_params.sfrac   # um -> movie px
        for mv in (scan_out.mov, scan_out.mov_raw):
            sig = mv - (bias if mv is scan_out.mov else 0.0)
            sig = gaussian_filter(sig, sigma=(blur_px, blur_px, 0))
            mv[...] = sig + (bias if mv is scan_out.mov else 0.0)
        print(f"  applied out-of-focus blur ({oof_blur_um:.0f} um = {blur_px:.1f} px)")

    # Non-uniform widefield illumination (bright centre -> dark edges) — the
    # dominant visual feature of real 1P widefield that NAOMi's uniform model
    # lacks. Excitation non-uniformity scales the photon signal, not the camera
    # bias pedestal, so multiply the signal ABOVE the bias floor. Applied to
    # both noisy and clean movies (mov is H x W x T).
    if illum_grad:
        illum = cfg.illum_map(scan_out.mov.shape[:2])[:, :, None]
        scan_out.mov = (bias + (scan_out.mov - bias) * illum).astype(np.float32)
        scan_out.mov_raw = (scan_out.mov_raw * illum).astype(np.float32)
        print(f"  applied Gaussian illumination gradient "
              f"(edge/centre = {illum.min():.2f})")
    print(f"  noisy [{scan_out.mov.min():.1f}, {scan_out.mov.max():.1f}]   "
          f"clean [{scan_out.mov_raw.min():.3g}, {scan_out.mov_raw.max():.3g}]")

    # ==================================================================
    # Save outputs — CRITICAL DATA FIRST, GIF LAST
    # ==================================================================
    print("\n[SAVE] writing run outputs (critical first, GIF last)")
    t0 = time.time()

    # 1. movies.npz (full-precision) — critical
    # The internal pipeline uses (H, W, T) but the saved movie is transposed to
    # (T, H, W) = (frames, height, width) so it matches the real striatum tiffs
    # and the ImageJ/tifffile convention. `axes` labels it so the meaning of
    # each dimension is unambiguous (frame axis first, not last).
    movies = dict(
        mov_clean=np.transpose(scan_out.mov_raw, (2, 0, 1)).astype(np.float32),
        mov_noisy=np.transpose(scan_out.mov, (2, 0, 1)).astype(np.float32),
        axes=np.array("THW"),          # T=frames, H=height(Y), W=width(X)
    )
    if getattr(scan_out, "mot_hist", None) is not None:
        movies["mot_hist"] = scan_out.mot_hist   # (3, T) applied XY(Z) shift
    if getattr(scan_out, "blur_hist", None) is not None:
        # (2, T) per-frame intra-frame motion-blur streak [dx, dy] in voxels
        # (physio motion only). Ground truth for de-blur / registration.
        movies["blur_hist"] = scan_out.blur_hist
    np.savez_compressed(os.path.join(run_dir, "movies.npz"), **movies)
    print(f"  saved movies.npz   mov_noisy {movies['mov_noisy'].shape} (T,H,W)")

    # 2. params.pkl — reproducibility
    with open(os.path.join(run_dir, "params.pkl"), "wb") as f:
        pickle.dump(dict(
            vol_params=vol_params, psf_params=psf_params,
            spike_params=spike_params, cal_params=cal_params,
            scan_params=scan_params, wf_params=wf_params,
            cam_params=cam_params, motion_params=motion_params,
        ), f)
    print("  saved params.pkl")

    # 3. metadata.json
    meta = dict(
        tag=tag, smoke=args.smoke,
        config=cfg.as_dict(),          # complete reproducible parameter record
        timestamp=_dt.datetime.now().isoformat(),
        seed=seed, region="striatum",
        vol_sz=list(vol_sz), vol_depth=vol_depth, vres=vres,
        nt=nt, dt=dt, prot=prot, rate=rate,
        smod=smod, burst_mean=burst_mean, bg_scale=bg_scale,
        # Records the actual Phase-3 spike source: when set, the built-in
        # smod/rate above are bypassed by the sim-trace design.
        simtrace_design=args.simtrace_design,
        simtrace_rate=(args.simtrace_rate
                       if args.simtrace_design is not None else None),
        soma_gain=float(soma_gain),
        illum_grad=bool(illum_grad),
        N_neur=int(vol_params.N_neur),
        n_soma=int(n_soma),                       # real neurons placed (ground truth)
        N_soma_traces=int(time_out.soma.shape[0]),  # incl. appended bg components
        total_spikes=n_spk,
        movie_shape=list(scan_out.mov.shape),
        gen_margin_um=float(args.gen_margin_um),
        image_crop_um=float(args.image_crop_um),
        imaged_fov_um=int(target_vol_um - 2 * args.image_crop_um),  # central FOV kept
        timings_seconds={k: float(v) for k, v in timings.items()},
        timings_total_seconds=float(sum(timings.values())),
        phase1_cache=phase1_cache,
        profiles=sorted(f for f in os.listdir(run_dir)
                        if f.startswith("profile_") and f.endswith(".html")),
    )
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  saved metadata.json")

    # 3a. Human-readable summary report (FOV / pixel geometry + component
    # inventory). Reads the just-saved metadata + traces; neur_ves gives the
    # vessel voxel count without reloading the Phase-1 volume.
    try:
        C.write_summary_report(run_dir, neur_ves=vol_out.neur_ves)
        print("  saved report.md")
    except Exception as e:
        print(f"  WARNING: report generation failed (data already saved): {e}")

    # 3b. Visualization bundle (on by default). Built here from the in-memory
    # vol_out so it never re-reads the multi-GB phase-1 pickle. Non-critical:
    # the run data is already saved; a failure here must not lose it.
    if not args.no_viz_prep:
        try:
            from calcia.viz.prep import prep_run
            made = prep_run(run_dir, neur_ves=vol_out.neur_ves, verbose=True)
            print(f"  saved viz bundle: {', '.join(made)}")
        except Exception as e:
            print(f"  WARNING: viz bundle prep failed (data already saved): {e}")

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
    import _instrument; _instrument.start()  # run log + pyinstrument (mandated)
    main()

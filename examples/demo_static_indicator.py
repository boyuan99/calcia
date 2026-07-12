"""Static (non-calcium) structural-label 1P widefield simulation: tdTomato / BFP.

Most calcia demos image a DYNAMIC calcium indicator (GCaMP): fluorescence tracks
spikes through the calcium ODE, so the movie flickers with activity. Constitutive
STRUCTURAL labels — tdTomato, BFP, a GFP cell-fill — are not calcium sensitive:
their brightness is fixed by expression level, so the movie is a single structural
image plus camera noise and sample motion. This script simulates that regime.

The machinery is one new trace mode, ``SpikeParams.dyn_type='static'`` (constant
per-cell fluorescence, no spikes / no calcium ODE), plus a per-channel choice of
where the label sits:

  tdTomato  cytoplasmic fill  -> soma + dendrites + dense neuropil (a smooth,
                                 washed cloud; low spatial contrast)
  BFP       nuclear-enriched  -> bright punctate nuclei on a dim diffuse
                                 background (high contrast, heavy bright tail)

Both are STATIC: temporal variation in the output is pure camera shot/read noise
plus the per-frame motion jitter, exactly like a real static-label recording.

The per-channel presets below were tuned against real striatum window recordings
(``C:/Users/boyuan/Downloads/tdt-bfp``, 1152x1152 @ 20 Hz). Pass ``--compare`` to
print the sim-vs-real summary statistics (needs that folder + h5py).

Physics notes on the two knobs that decouple "brightness" from "temporal noise":
  * A real static-label recording is DIM and read at HIGH analog gain, so a pixel
    holds only ~100 detected photons even though it digitizes to a few-thousand
    ADU. Temporal CV = 1/sqrt(photons) ~ 0.1 comes from that low photon count;
    the ADU level comes from the gain. So we drive photon count with ``pavg`` and
    ADU level with ``gain_e_per_adu`` (<1 = high gain) independently.
  * Out-of-focus haze is an OPTICAL blur (before the sensor), so it must smooth
    the clean photon image and THEN have camera noise added on top. Blurring the
    already-noisy movie would spatially average the noise away and collapse the
    temporal CV — so this script re-derives the noisy movie after the blur.

Run:
    conda run -n calcia python examples/demo_static_indicator.py --channel tdt --smoke
    conda run -n calcia python examples/demo_static_indicator.py --channel bfp --compare
    conda run -n calcia python examples/demo_static_indicator.py --channel both
"""
import argparse
import datetime as _dt
import glob
import os
import pickle
import time

import numpy as np

import _striatum_common as C


OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
SHARED_DIR = os.path.join(OUTPUT_ROOT, "_shared")
REAL_DIR = "C:/Users/boyuan/Downloads/tdt-bfp"


# ======================================================================
# Per-channel presets (tuned against the real tdt-bfp striatum recordings)
# ======================================================================
# Each preset carries: optics (emission wavelength, tissue scatter length),
# widefield excitation, the label localization (cytoplasmic vs nuclear), the
# brightness/photon split (pavg + gain), the diffuse-background scale, the
# out-of-focus blur, and the camera bias pedestal. The values are intensive
# (CV / ratios / per-pixel ADU) so they hold across FOV size.
STATIC_PRESETS = {
    "tdt": dict(
        # Cytoplasmic tdTomato: green-LED excited red emission, dense fill.
        lambda_em_um=0.581, lambda_ex_um=0.555, scatter_length_um_wf=150.0,
        sigma_abs=4.5e-16, phi=0.69, qe_det=0.8,
        nuclear=False, dendflag=True, axonflag=True,
        bg_scale=2.5, soma_gain=1.0, nuc_fl=0.0, gamma=None,
        pavg=0.09, gain=0.04, oof_blur_um=12.0,
        bias=470.0, dark_rate=0.3, read_noise=1.6,
    ),
    "bfp": dict(
        # Nuclear-enriched BFP: violet-LED excited blue emission. Bright
        # punctate nuclei (nuc_fl, heavy-tailed expression) over a dim
        # cytoplasm/neuropil background (bg_scale, soma_gain).
        lambda_em_um=0.457, lambda_ex_um=0.405, scatter_length_um_wf=55.0,
        sigma_abs=2.5e-16, phi=0.55, qe_det=0.75,
        nuclear=True, dendflag=True, axonflag=True,
        bg_scale=1.0, soma_gain=1.0, nuc_fl=5.0, nuc_frac=0.5,
        gamma=(0.5, 1.6),
        pavg=0.5, gain=0.045, oof_blur_um=2.0,
        bias=250.0, dark_rate=0.3, read_noise=1.6,
    ),
}
# Match quality vs the real recordings (medium 1000 um FOV):
#   tdt  5/5 summary stats in range (excellent) + the washed-cloud texture and
#        intensity histogram both match.
#   bfp  temporal_cv and p999 in range, median ~within 10%, and the intensity
#        HISTOGRAM overlaps the real one across the full bright-nuclei tail. Two
#        gaps remain, both rooted in Phase 1 (not the static trace / scan model):
#          - spatial_cv (~0.55 vs 0.43): the MSN phase-1 gives ~6000 small
#            (~5.5 um) nuclei, so even at nuc_frac=0.5 the labelled nuclei are
#            denser and smaller than the real sparse, larger BFP+ nuclei. Denser
#            small dots read as higher contrast.
#          - floor_frac (~0.5 vs 0.13-0.30): the real mean image includes the
#            dark window vignette and DARK BLOOD VESSELS punching the field; the
#            striatum preset deliberately thins vessels (so they fade under the
#            tdTomato wash) and this demo adds no vignette, so the sim floor
#            cannot drop as low.
#        Both need a phase-1 change (fewer/larger nuclei + visible vessels for a
#        nuclear channel), tracked separately; the static-indicator machinery
#        itself is validated by the tdt match.

# Real-recording summary-stat targets (min, max) across the 4 real files.
REAL_TARGETS = {
    "tdt": dict(spatial_cv=(0.08, 0.11), temporal_cv=(0.08, 0.12),
                median=(2260, 4000), floor_frac=(0.70, 0.82),
                p999_over_med=(1.30, 1.70)),
    "bfp": dict(spatial_cv=(0.35, 0.43), temporal_cv=(0.11, 0.14),
                median=(1500, 1700), floor_frac=(0.13, 0.30),
                p999_over_med=(5.00, 6.50)),
}


def parse_args():
    p = argparse.ArgumentParser(description="Static-label 1P widefield (tdT/BFP)")
    p.add_argument("--channel", choices=["tdt", "bfp", "both"], default="tdt")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny 80x80x50 / 20-frame run to verify the pipeline")
    p.add_argument("--vol-um", type=int, default=1000, dest="vol_um")
    p.add_argument("--depth-um", type=int, default=60, dest="depth_um")
    p.add_argument("--vres", type=int, default=1)
    p.add_argument("--nt", type=int, default=None, help="frames (default 60)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare", action="store_true",
                   help="Print sim-vs-real summary stats (needs the real folder)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing TIFF/npz/gif (stats only)")
    p.add_argument("--motion", choices=["physio", "randomwalk"], default="physio",
                   help="Sample-motion model. 'physio' (default) = realistic "
                        "AR(1) drift+jitter + heavy-tailed jumps + intra-frame "
                        "blur, fit to real NoRMCorre shifts. 'randomwalk' = "
                        "legacy small bounded integer walk.")
    return p.parse_args()


def phase1_signature(vol_sz, vol_depth, vres, seed, region, n_neur):
    import hashlib
    h = hashlib.sha1()
    h.update(repr((tuple(vol_sz), vol_depth, vres, seed, region, n_neur)).encode())
    return h.hexdigest()[:10]


def load_or_build_phase1(vol_sz, vol_depth, vres, seed):
    """Load the shared striatum phase-1 cache, building it if absent."""
    from calcia import simulate_neural_volume
    from calcia.config.params import VolumeParams
    sig = phase1_signature(vol_sz, vol_depth, vres, seed, "striatum", None)
    cache = os.path.join(SHARED_DIR, f"phase1_{sig}.pkl")
    os.makedirs(SHARED_DIR, exist_ok=True)
    if os.path.exists(cache):
        print(f"  phase1 cache hit: {os.path.basename(cache)}")
        with open(cache, "rb") as f:
            return pickle.load(f)
    print(f"  phase1 cache miss -> generating ({sig})")
    vp = VolumeParams(vol_sz=vol_sz, vres=vres, vol_depth=vol_depth,
                      region="striatum", N_neur=None)
    vol_out = simulate_neural_volume(vol_params=vp, seed=seed, verbose=1)
    vp = vol_out.params["vol_params"]
    with open(cache, "wb") as f:
        pickle.dump((vol_out, vp), f)
    return vol_out, vp


# ======================================================================
# One static channel end-to-end
# ======================================================================
def run_channel(channel, vol_out, vol_params, nt, seed, motion_model="physio"):
    """Return (mov_noisy, mov_clean) H x W x T for one static channel."""
    from calcia import simulate_optical_propagation, generate_time_traces
    from calcia.scanning import scan_widefield
    from calcia.scanning.noise import camera_noise
    from calcia.config.params import (PsfParams, WidefieldParams, SpikeParams,
                                      ScanParams, CameraNoiseParams, MotionParams)
    from scipy.ndimage import gaussian_filter

    P = STATIC_PRESETS[channel]

    # --- values-only edits of the (freshly loaded) volume ---
    if P["soma_gain"] != 1.0:
        for cfd in vol_out.gp_vals:
            sm = np.asarray(cfd.soma_mask)
            if sm.any():
                cfd.fluorescence[sm] *= P["soma_gain"]
    if P["nuc_fl"] > 0.0:
        # Bright nuclei rendered as nuc_fl * mod_vals[cell] on top of the dim
        # cytoplasm/neuropil background (nuc_label stays 0). Only nuc_frac of
        # cells get a bright nucleus (sparse nuclear labelling); the rest keep
        # their background but no bright nucleus, so the field is not saturated
        # with dots the way full labelling makes it.
        nuc_frac = P.get("nuc_frac", 1.0)
        rng = np.random.default_rng(seed + 99)
        keep = rng.random(len(vol_out.gp_nuc)) < nuc_frac
        vol_out.gp_nuc = [
            (np.asarray(idx), float(P["nuc_fl"]) if keep[i] else 0.0)
            for i, (idx, _) in enumerate(vol_out.gp_nuc)]

    # --- Phase 2: widefield optics ---
    psf_params = PsfParams(
        imaging_mode="widefield", psf_type="gaussian_analytical",
        lambda_em_um=P["lambda_em_um"], obj_na=0.8, n=1.35,
        psf_sz=(12.0, 12.0, 20.0), wf_focal_depth_um=30.0,
        scatter_length_um_wf=P["scatter_length_um_wf"])
    opt_out = simulate_optical_propagation(
        vol_params=vol_params, psf_params=psf_params, vol_out=vol_out, verbose=0)

    # --- Phase 3: STATIC traces (constant per cell, no spikes / calcium) ---
    K = len(vol_out.gp_vals)
    has_axons = P["axonflag"] and len(vol_out.bg_proc) > 0
    sp_kw = {} if P["gamma"] is None else dict(min_mod=P["gamma"])
    spike_params = SpikeParams(
        K=K, nt=nt, dt=1 / 20, N_bg=0, dyn_type="static", prot=channel,
        dendflag=P["dendflag"], axonflag=has_axons, bg_scale=P["bg_scale"],
        verbose=0, **sp_kw)
    time_out = generate_time_traces(spike_params=spike_params,
                                    n_locs=vol_out.locs, verbose=0)

    # --- Phase 4: widefield camera scan ---
    wf_params = WidefieldParams(pavg=P["pavg"], lambda_ex_um=P["lambda_ex_um"],
                                sigma_abs=P["sigma_abs"], phi=P["phi"],
                                qe_det=P["qe_det"])
    # 'physio' motion needs a larger crop margin: real striatum shifts reach
    # ~+/-26 um and the scan_buff bounds them. randomwalk keeps the small buffer.
    _buff = 30 if motion_model == "physio" else 10
    scan_params = ScanParams(scan_buff=_buff, motion=True, sfrac=2, verbose=0)
    motion_params = (MotionParams(model="physio", seed=seed + 3)
                     if motion_model == "physio" else None)
    cam = CameraNoiseParams(qe=1.0, dark_rate=P["dark_rate"], t_exp=1 / 20,
                            read_noise=P["read_noise"], gain_e_per_adu=P["gain"],
                            bias=P["bias"])
    scan_out = scan_widefield(vol_out=vol_out, opt_out=opt_out, time_out=time_out,
                              scan_params=scan_params, cam_params=cam,
                              motion_params=motion_params,
                              wf_params=wf_params, spike_params=spike_params,
                              seed=seed)

    # --- Out-of-focus haze: blur the CLEAN photon image, THEN re-add camera
    # noise per pixel (optical blur precedes the sensor; blurring the noisy
    # movie would average the noise away and kill the temporal CV). ---
    if P["oof_blur_um"] > 0:
        blur_px = P["oof_blur_um"] * vol_params.vres / scan_params.sfrac
        clean = gaussian_filter(scan_out.mov_raw, sigma=(blur_px, blur_px, 0))
        rng = np.random.default_rng(seed + 777)
        noisy = np.empty_like(clean)
        for kk in range(clean.shape[2]):
            noisy[:, :, kk] = camera_noise(clean[:, :, kk], cam, rng)
        return noisy.astype(np.float32), clean.astype(np.float32)

    return scan_out.mov, scan_out.mov_raw


# ======================================================================
# Stats + comparison
# ======================================================================
def summary_stats(mov):
    """mov: (H,W,T). Summary matching the real-data analysis."""
    mean_img = mov.mean(2)
    std_t = mov.std(2)
    bright = mean_img > np.percentile(mean_img, 50)
    cv_t = float(np.median(std_t[bright] / (mean_img[bright] + 1e-6)))
    p = np.percentile(mean_img, [1, 50, 90, 99, 99.9])
    return dict(median=float(p[1]), mean=float(mean_img.mean()),
                max=float(mean_img.max()),
                spatial_cv=float(mean_img.std() / mean_img.mean()),
                temporal_cv=cv_t, floor_frac=float(p[0] / (p[1] + 1e-9)),
                p999_over_med=float(p[4] / (p[1] + 1e-9)),
                pctiles=p.astype(int).tolist())


def print_comparison(channel, mov):
    st = summary_stats(mov)
    tgt = REAL_TARGETS[channel]
    print(f"\n  {channel.upper()} static widefield vs real striatum recordings")
    print(f"    median={st['median']:.0f}  mean={st['mean']:.0f}  "
          f"max={st['max']:.0f}  pctiles[1,50,90,99,99.9]={st['pctiles']}")
    for key in ("spatial_cv", "temporal_cv", "median", "floor_frac",
                "p999_over_med"):
        lo, hi = tgt[key]
        ok = "OK" if lo <= st[key] <= hi else "  "
        print(f"    [{ok}] {key:16s}= {st[key]:9.3f}   real [{lo}, {hi}]")


def real_mean_image(channel):
    import h5py
    files = sorted(glob.glob(os.path.join(REAL_DIR, f"*_{channel}_mc.h5")))
    if not files:
        return None
    with h5py.File(files[0], "r") as h:
        img = h["images"]
        idx = np.linspace(0, img.shape[0] - 1, min(img.shape[0], 40)).astype(int)
        return np.asarray(img[idx], dtype=np.float32).mean(0)


def save_comparison_png(channel, mov, path):
    real = real_mean_image(channel)
    if real is None:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sim = mov.mean(2)
    c = real.shape[0] // 2
    hw = min(c, sim.shape[0])
    rcrop = real[c - hw:c + hw, c - hw:c + hw]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, (im, ttl) in zip(ax, [
        (rcrop, f"REAL {channel}\nCV={real.std()/real.mean():.3f}"),
        (sim, f"SIM {channel}\nCV={sim.std()/sim.mean():.3f}"),
        (None, "intensity hist (norm to median)")]):
        if im is not None:
            lo, hi = np.percentile(im, [1, 99])
            a.imshow(im, cmap="gray", vmin=lo, vmax=hi)
            a.set_title(ttl, fontsize=10)
            a.axis("off")
        else:
            a.hist(real.ravel() / np.median(real), bins=80, log=True,
                   alpha=0.5, density=True, label="real")
            a.hist(sim.ravel() / np.median(sim), bins=80, log=True,
                   alpha=0.5, density=True, label="sim")
            a.set_xlim(0, 6); a.legend(fontsize=9); a.set_title(ttl, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return True


# ======================================================================
def main():
    args = parse_args()
    channels = ["tdt", "bfp"] if args.channel == "both" else [args.channel]

    if args.smoke:
        vol_sz = (80, 80, 50); vres = 2; nt = args.nt or 20
    else:
        vol_sz = (args.vol_um, args.vol_um, args.depth_um)
        vres = args.vres; nt = args.nt or 60
    vol_depth = 0

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, f"static_{args.channel}_{ts}")
    if not args.no_save:
        os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("Static-label 1P widefield  " + ("(SMOKE)" if args.smoke else ""))
    print(f"  Volume: {vol_sz} um  vres={vres}  frames={nt}  seed={args.seed}")
    print(f"  Channels: {channels}")
    print("=" * 60)

    print("\n[PHASE 1] striatum neural volume (shared cache)")
    t0 = time.time()
    # Build/cache once (returns are discarded so the multi-GB volume is not held
    # across the loop); each channel reloads a fresh copy since run_channel
    # applies values-only edits to the footprints.
    vol_out, vol_params = load_or_build_phase1(vol_sz, vol_depth, vres, args.seed)
    print(f"  N_neur={vol_params.N_neur}  grid={vol_out.neur_vol.shape}  "
          f"({time.time()-t0:.1f}s)")
    del vol_out, vol_params

    for ch in channels:
        print(f"\n[CHANNEL {ch}]  ({'nuclear' if STATIC_PRESETS[ch]['nuclear'] else 'cytoplasmic'} static label)")
        t0 = time.time()
        # Reload a fresh volume copy per channel (run_channel edits values).
        vo, vp = load_or_build_phase1(vol_sz, vol_depth, vres, args.seed)
        mov, mov_clean = run_channel(ch, vo, vp, nt, args.seed,
                                     motion_model=args.motion)
        print(f"  done ({time.time()-t0:.1f}s)  movie {mov.shape}  "
              f"noisy[{mov.min():.0f}, {mov.max():.0f}]")

        # Summary vs the real-recording targets is always informative.
        print_comparison(ch, mov)

        if not args.no_save:
            np.savez_compressed(
                os.path.join(run_dir, f"movie_{ch}.npz"),
                mov_noisy=np.transpose(mov, (2, 0, 1)).astype(np.float32),
                mov_clean=np.transpose(mov_clean, (2, 0, 1)).astype(np.float32),
                axes=np.array("THW"))
            C.save_tif(mov, os.path.join(run_dir, f"movie_{ch}_noisy.tif"))
            try:
                if save_comparison_png(
                        ch, mov, os.path.join(run_dir, f"compare_{ch}.png")):
                    print(f"  saved compare_{ch}.png")
            except Exception as e:
                print(f"  (comparison png skipped: {e})")
            print(f"  saved movie_{ch}.npz + movie_{ch}_noisy.tif")

    if not args.no_save:
        print(f"\nOutput: {run_dir}")


if __name__ == "__main__":
    import _instrument; _instrument.start()  # run log + pyinstrument (mandated)
    main()

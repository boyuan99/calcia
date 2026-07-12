"""Demo: Run the full NAOMi simulation pipeline (Phase 1-4).

Generates a simulated two-photon calcium imaging video end-to-end:

  Phase 1: simulate_neural_volume   - tissue geometry + fluorescence
  Phase 2: simulate_optical_propagation - PSF + illumination masks
  Phase 3: generate_time_traces     - spike trains + calcium dynamics
  Phase 4: scan_volume              - raster scan -> noisy movie

Outputs:
  - TIFF stack  (noisy + clean)
  - GIF video   (noisy + clean, side-by-side)
  - pyinstrument profiling report (HTML)

Usage:
    conda run -n calcia python examples/demo_full_pipeline.py
    conda run -n calcia python examples/demo_full_pipeline.py --small   # fast test
    conda run -n calcia python examples/demo_full_pipeline.py --large   # full-size
"""

import argparse
import os

import numpy as np
from pyinstrument import Profiler


def parse_args():
    parser = argparse.ArgumentParser(
        description="NAOMi full pipeline demo (calcia)")
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--small", action="store_true",
        help="Small volume for quick testing (~1 min)")
    size.add_argument(
        "--large", action="store_true",
        help="Full MATLAB-default size (250x250x100 um, ~10+ min)")
    size.add_argument(
        "--match-matlab", action="store_true",
        help="Match MATLAB demo_full_pipeline.m parameters exactly")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)")
    parser.add_argument(
        "--nt", type=int, default=None,
        help="Number of time frames (overrides preset)")
    parser.add_argument(
        "--bg-scale", type=float, default=1.0,
        help="Neuropil/background brightness scale (1.0=NAOMi default; "
             "lower dims the diffuse background, e.g. 0.25)")
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: examples/output)")
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip GIF video generation")
    parser.add_argument(
        "--no-tiff", action="store_true",
        help="Skip TIFF stack saving")
    parser.add_argument(
        "--no-profile", action="store_true",
        help="Skip pyinstrument profiling")
    parser.add_argument(
        "--cache", action="store_true",
        help="Cache Phase 1-2 results to disk; reuse on subsequent runs")
    return parser.parse_args()


def make_video(mov_noisy, mov_clean, path, dt, contrast=0.995, fps=30):
    """Create a side-by-side GIF video (noisy | clean).

    Parameters
    ----------
    mov_noisy, mov_clean : ndarray, (H, W, Nt)
    path : str
        Output GIF path.
    dt : float
        Frame interval in seconds.
    contrast : float
        Percentile for clipping (0-1).
    fps : int
        Playback frame rate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    nt = mov_noisy.shape[2]

    # Compute display range from full movie
    vmin_n = np.percentile(mov_noisy, (1 - contrast) * 100)
    vmax_n = np.percentile(mov_noisy, contrast * 100)
    vmin_c = np.percentile(mov_clean, (1 - contrast) * 100)
    vmax_c = np.percentile(mov_clean, contrast * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98,
                        top=0.90, bottom=0.02)

    im1 = ax1.imshow(mov_noisy[:, :, 0], cmap="gray",
                     vmin=vmin_n, vmax=vmax_n, aspect="equal")
    im2 = ax2.imshow(mov_clean[:, :, 0], cmap="gray",
                     vmin=vmin_c, vmax=vmax_c, aspect="equal")
    ax1.set_title("Noisy", fontsize=10)
    ax2.set_title("Clean", fontsize=10)
    ax1.axis("off")
    ax2.axis("off")
    time_text = fig.suptitle("t = 0.000 s", fontsize=11)

    def update(frame):
        im1.set_data(mov_noisy[:, :, frame])
        im2.set_data(mov_clean[:, :, frame])
        time_text.set_text(f"t = {frame * dt:.3f} s")
        return im1, im2, time_text

    anim = FuncAnimation(fig, update, frames=nt, blit=True, interval=1)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Video saved: {path}")


def save_tiff(mov, path):
    """Save a (H, W, Nt) movie as a 16-bit TIFF stack."""
    import tifffile

    # Normalize to 16-bit range
    arr = mov.astype(np.float64)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr * 65535, 0, 65535).astype(np.uint16)

    # tifffile expects (Nt, H, W)
    arr = np.transpose(arr, (2, 0, 1))
    tifffile.imwrite(path, arr, imagej=True)
    print(f"  TIFF saved: {path}")


def main():
    args = parse_args()

    # ---- Import calcia (after arg parsing for fast --help) ----
    from calcia import (
        simulate_neural_volume,
        simulate_optical_propagation,
        generate_time_traces,
        scan_volume,
    )
    from calcia.config.params import (
        VolumeParams, PsfParams, SpikeParams, CalciumParams,
        ScanParams, NoiseParams, TpmParams,
    )

    # ---- Choose parameter preset ----
    if args.match_matlab:
        # Exactly match MATLAB demo_full_pipeline.m parameters
        vol_sz = (250, 250, 100)
        vol_depth = 100
        nt = args.nt or 300
        preset = "match-matlab"
    elif args.small:
        vol_sz = (80, 80, 50)
        vol_depth = 50
        nt = args.nt or 300
        preset = "small"
    elif args.large:
        vol_sz = (250, 250, 100)
        vol_depth = 100
        nt = args.nt or 1500
        preset = "large"
    else:
        # Medium default: reasonable quality, manageable runtime
        vol_sz = (100, 100, 50)
        vol_depth = 50
        nt = args.nt or 500
        preset = "medium"

    seed = args.seed
    dt = 1 / 30  # 30 Hz frame rate

    outdir = args.outdir or os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    tag = f"{preset}_seed{seed}"

    print("=" * 60)
    print("NAOMi Full Pipeline Demo (calcia)")
    print(f"  Preset:  {preset}")
    print(f"  Volume:  {vol_sz} um, depth={vol_depth} um")
    print(f"  Frames:  {nt} ({nt * dt:.1f} s at {1/dt:.0f} Hz)")
    print(f"  Seed:    {seed}")
    print("=" * 60)

    vol_params = VolumeParams(
        vol_sz=vol_sz,
        vres=2,
        vol_depth=vol_depth,
    )

    # ---- Profiler setup ----
    profiler = None if args.no_profile else Profiler()

    # ---- Phase 1-2 caching ----
    import pickle
    cache_path = os.path.join(outdir, f"phase12_cache_{tag}.pkl")
    psf_params = PsfParams()

    if args.cache and os.path.exists(cache_path):
        print(f"\nLoading Phase 1-2 from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            vol_out, vol_params, opt_out = pickle.load(f)
        print(f"  Neurons: {vol_params.N_neur}, "
              f"Volume grid: {vol_out.neur_vol.shape}")
        print(f"  PSF shape: {opt_out.psf.shape}")
    else:
        # ==================================================================
        # Phase 1: Neural Volume
        # ==================================================================
        print("\n" + "-" * 60)
        print("PHASE 1: Neural Volume Simulation")
        print("-" * 60)

        if profiler:
            profiler.start()

        vol_out = simulate_neural_volume(
            vol_params=vol_params,
            seed=seed,
            verbose=1,
        )
        vol_params = vol_out.params["vol_params"]

        if profiler:
            profiler.stop()
            print(profiler.output_text(unicode=True, color=False, show_all=False))
            _save_profile(profiler, outdir, f"profile_phase1_{tag}")
            profiler.reset()

        print(f"  Neurons: {vol_params.N_neur}, "
              f"Volume grid: {vol_out.neur_vol.shape}")

        # ==================================================================
        # Phase 2: Optical Propagation
        # ==================================================================
        print("\n" + "-" * 60)
        print("PHASE 2: Optical Propagation (PSF)")
        print("-" * 60)

        if profiler:
            profiler.start()

        opt_out = simulate_optical_propagation(
            vol_params=vol_params,
            psf_params=psf_params,
            vol_out=vol_out,
            verbose=1,
        )

        if profiler:
            profiler.stop()
            print(profiler.output_text(unicode=True, color=False, show_all=False))
            _save_profile(profiler, outdir, f"profile_phase2_{tag}")
            profiler.reset()

        print(f"  PSF shape: {opt_out.psf.shape}")

        if args.cache:
            print(f"  Caching Phase 1-2 to: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump((vol_out, vol_params, opt_out), f)

    # ==================================================================
    # Phase 3: Time Traces
    # ==================================================================
    print("\n" + "-" * 60)
    print("PHASE 3: Time Traces (Spikes + Calcium)")
    print("-" * 60)

    # Number of active components = all gp_vals (neurons + apical + bg dendrites)
    # Matches MATLAB: spike_opts.K = size(vol_out.gp_vals, 1)
    K = len(vol_out.gp_vals)
    has_axons = len(vol_out.bg_proc) > 0

    spike_params = SpikeParams(
        K=K,
        nt=nt,
        dt=dt,
        # N_bg and axonflag are mutually exclusive:
        # axonflag=True  -> bg traces from sorted axon processes (Phase 1)
        # N_bg > 0       -> bg traces from generic GP components
        N_bg=0,
        axonflag=has_axons,
        rate=0.25,
        prot="GCaMP6f",
        bg_scale=args.bg_scale,
    )
    cal_params = CalciumParams(prot_type="gcamp6f")

    if profiler:
        profiler.start()

    time_out = generate_time_traces(
        spike_params=spike_params,
        cal_params=cal_params,
        n_locs=vol_out.locs,
        verbose=1,
    )

    if profiler:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=False, show_all=False))
        _save_profile(profiler, outdir, f"profile_phase3_{tag}")
        profiler.reset()

    print(f"  Soma traces: {time_out.soma.shape}")
    if time_out.dend is not None:
        print(f"  Dend traces: {time_out.dend.shape}")
    if time_out.bg is not None:
        print(f"  BG traces:   {time_out.bg.shape}")

    # ==================================================================
    # Phase 4: Volume Scanning
    # ==================================================================
    print("\n" + "-" * 60)
    print("PHASE 4: Volume Scanning")
    print("-" * 60)

    scan_params = ScanParams(motion=True, scan_avg=2)
    noise_params = NoiseParams()
    tpm_params = TpmParams()

    if profiler:
        profiler.start()

    scan_out = scan_volume(
        vol_out=vol_out,
        opt_out=opt_out,
        time_out=time_out,
        scan_params=scan_params,
        noise_params=noise_params,
        tpm_params=tpm_params,
        spike_params=spike_params,
        seed=seed,
    )

    if profiler:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=False, show_all=False))
        _save_profile(profiler, outdir, f"profile_phase4_{tag}")
        profiler.reset()

    print(f"  Movie shape: {scan_out.mov.shape}  (H x W x Nt)")
    print(f"  Noisy  range: [{scan_out.mov.min():.1f}, "
          f"{scan_out.mov.max():.1f}]")
    print(f"  Clean  range: [{scan_out.mov_raw.min():.1f}, "
          f"{scan_out.mov_raw.max():.1f}]")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print(f"  Movie:      {scan_out.mov.shape[0]}x{scan_out.mov.shape[1]}"
          f" pixels, {scan_out.mov.shape[2]} frames")
    print(f"  Duration:   {scan_out.mov.shape[2] * dt:.1f} s "
          f"at {1/dt:.0f} Hz")
    print("=" * 60)

    # ==================================================================
    # Save outputs
    # ==================================================================
    if not args.no_tiff:
        print("\nSaving TIFF stacks...")
        save_tiff(scan_out.mov,
                  os.path.join(outdir, f"movie_noisy_{tag}.tif"))
        save_tiff(scan_out.mov_raw,
                  os.path.join(outdir, f"movie_clean_{tag}.tif"))

    if not args.no_video:
        print("\nGenerating GIF video (this may take a moment)...")
        make_video(
            scan_out.mov, scan_out.mov_raw,
            os.path.join(outdir, f"movie_{tag}.gif"),
            dt=dt, fps=min(30, int(1 / dt)),
        )

    # ==================================================================
    # Save full intermediate data for comparison
    # ==================================================================
    print("\nSaving full intermediate data...")
    _save_all_phases(outdir, tag, vol_out, opt_out, time_out, scan_out)

    print(f"\nAll outputs saved to: {outdir}")
    print("Done!")


def _save_all_phases(outdir, tag, vol_out, opt_out, time_out, scan_out):
    """Save all intermediate phase data for MATLAB comparison."""
    import scipy.io as sio

    mat_path = os.path.join(outdir, f"phases_{tag}.mat")

    d = {}

    # --- Phase 1 ---
    d["neur_vol"] = vol_out.neur_vol
    d["neur_num"] = vol_out.neur_num
    d["neur_num_ad"] = vol_out.neur_num_ad
    d["locs"] = vol_out.locs
    if vol_out.neur_ves is not None:
        d["neur_ves"] = vol_out.neur_ves
    d["n_gp_vals"] = len(vol_out.gp_vals)
    d["n_gp_bgvals"] = len(vol_out.gp_bgvals)
    d["n_bg_proc"] = len(vol_out.bg_proc)

    # gp_vals per-component sizes and fluorescence stats
    gp_sizes = np.array([len(g.indices) for g in vol_out.gp_vals])
    d["gp_vals_sizes"] = gp_sizes

    # --- Phase 2 ---
    d["psf"] = opt_out.psf
    d["mask"] = opt_out.mask
    d["col_mask"] = opt_out.col_mask
    if opt_out.psf_top is not None:
        d["psf_top_weights"] = opt_out.psf_top.weights
        d["psf_top_mask"] = opt_out.psf_top.mask
    if opt_out.psf_bot is not None:
        d["psf_bot_weights"] = opt_out.psf_bot.weights
        d["psf_bot_mask"] = opt_out.psf_bot.mask

    # --- Phase 3 ---
    d["soma"] = time_out.soma
    if time_out.dend is not None:
        d["dend"] = time_out.dend
    if time_out.bg is not None:
        d["bg"] = time_out.bg
    if time_out.spikes is not None:
        d["spikes"] = time_out.spikes
    d["mod_vals"] = time_out.mod_vals

    # --- Phase 4 ---
    d["mov"] = scan_out.mov
    d["mov_raw"] = scan_out.mov_raw
    d["mot_hist"] = scan_out.mot_hist

    sio.savemat(mat_path, d, do_compression=True)
    sz_mb = os.path.getsize(mat_path) / 1024 / 1024
    print(f"  Full data saved: {mat_path} ({sz_mb:.1f} MB)")


def _save_profile(profiler, outdir, name):
    """Save pyinstrument profile as HTML."""
    html_path = os.path.join(outdir, f"{name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(profiler.output_html())
    print(f"  Profile saved: {html_path}")


if __name__ == "__main__":
    import _instrument; _instrument.start()  # run log + pyinstrument (mandated)
    main()

"""Demo: Run the full NAOMi pipeline with widefield (single-photon) imaging.

Counterpart to ``demo_full_pipeline.py`` but for the single-photon /
widefield path:

  Phase 1: simulate_neural_volume          - optics-independent
  Phase 2: simulate_optical_propagation    - widefield PSF (emission lambda)
  Phase 3: generate_time_traces            - optics-independent
  Phase 4: scan_volume -> scan_widefield   - camera-based image formation

Usage::

    conda run -n calcia python examples/demo_widefield_pipeline.py
    conda run -n calcia python examples/demo_widefield_pipeline.py --large
    conda run -n calcia python examples/demo_widefield_pipeline.py --match-matlab
    # Reuse a previously generated Phase 1 volume (e.g. from demo_full_pipeline):
    conda run -n calcia python examples/demo_widefield_pipeline.py --large \\
        --reuse-vol examples/output/phase12_cache_large_seed42.pkl

Because Phase 1 (tissue geometry) and Phase 3 (calcium dynamics) are
optics-independent, running two-photon then widefield on the same neurons
is a fair side-by-side comparison: Phase 2 and Phase 4 are the only
sources of divergence.
"""

import argparse
import os
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="NAOMi widefield pipeline demo (calcia)")
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--small", action="store_true",
        help="Small volume for quick testing (~1 min)")
    size.add_argument(
        "--large", action="store_true",
        help="Full MATLAB-default size (250x250x100 um)")
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
        "--outdir", type=str, default=None,
        help="Output directory (default: examples/output)")
    parser.add_argument(
        "--reuse-vol", type=str, default=None,
        help="Path to a Phase 1-2 cache (pickle) from demo_full_pipeline.py. "
             "The Phase 1 volume is reused; Phase 2 is re-run in widefield "
             "mode (two-photon PSF is discarded).")
    parser.add_argument(
        "--cache", action="store_true",
        help="Cache Phase 1 to disk; reuse on subsequent runs")
    parser.add_argument(
        "--no-video", action="store_true", help="Skip GIF video generation")
    parser.add_argument(
        "--no-tiff", action="store_true", help="Skip TIFF stack saving")
    return parser.parse_args()


def make_video(mov_noisy, mov_clean, path, dt, contrast=0.995, fps=30):
    """Side-by-side GIF video (noisy | clean)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    nt = mov_noisy.shape[2]
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
    ax1.set_title("Noisy (widefield)", fontsize=10)
    ax2.set_title("Clean (widefield)", fontsize=10)
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
    """Save (H, W, Nt) movie as a 16-bit TIFF stack."""
    import tifffile
    arr = mov.astype(np.float64)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr * 65535, 0, 65535).astype(np.uint16)
    arr = np.transpose(arr, (2, 0, 1))
    tifffile.imwrite(path, arr, imagej=True)
    print(f"  TIFF saved: {path}")


def main():
    args = parse_args()

    from calcia import (
        simulate_neural_volume,
        simulate_optical_propagation,
        generate_time_traces,
        scan_volume,
    )
    from calcia.config.params import (
        CameraNoiseParams,
        CalciumParams,
        PsfParams,
        ScanParams,
        SpikeParams,
        VolumeParams,
        WidefieldParams,
    )

    # ---- Preset ----
    if args.match_matlab:
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
        nt = args.nt or 300
        preset = "large"
    else:
        vol_sz = (100, 100, 50)
        vol_depth = 50
        nt = args.nt or 300
        preset = "medium"

    seed = args.seed
    dt = 1 / 30
    outdir = args.outdir or os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    tag = f"widefield_{preset}_seed{seed}"

    print("=" * 60)
    print("NAOMi Widefield Pipeline Demo (calcia)")
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

    # ==================================================================
    # Phase 1: Neural volume (reuse if provided or cached)
    # ==================================================================
    phase1_cache = os.path.join(outdir, f"phase1_cache_{preset}_seed{seed}.pkl")
    vol_out = None

    if args.reuse_vol and os.path.exists(args.reuse_vol):
        print(f"\nLoading Phase 1 from shared cache: {args.reuse_vol}")
        with open(args.reuse_vol, "rb") as f:
            payload = pickle.load(f)
        # demo_full_pipeline caches (vol_out, vol_params, opt_out)
        if isinstance(payload, tuple) and len(payload) >= 2:
            vol_out, vol_params = payload[0], payload[1]
        else:
            vol_out = payload
        print(f"  Neurons: {vol_params.N_neur}, "
              f"Volume grid: {vol_out.neur_vol.shape}")
    elif args.cache and os.path.exists(phase1_cache):
        print(f"\nLoading Phase 1 from cache: {phase1_cache}")
        with open(phase1_cache, "rb") as f:
            vol_out, vol_params = pickle.load(f)
        print(f"  Neurons: {vol_params.N_neur}, "
              f"Volume grid: {vol_out.neur_vol.shape}")

    if vol_out is None:
        print("\n" + "-" * 60)
        print("PHASE 1: Neural Volume Simulation")
        print("-" * 60)
        vol_out = simulate_neural_volume(
            vol_params=vol_params,
            seed=seed,
            verbose=1,
        )
        vol_params = vol_out.params["vol_params"]
        print(f"  Neurons: {vol_params.N_neur}, "
              f"Volume grid: {vol_out.neur_vol.shape}")
        if args.cache:
            print(f"  Caching Phase 1 to: {phase1_cache}")
            with open(phase1_cache, "wb") as f:
                pickle.dump((vol_out, vol_params), f)

    # ==================================================================
    # Phase 2: Optical Propagation (widefield path)
    # ==================================================================
    print("\n" + "-" * 60)
    print("PHASE 2: Optical Propagation (widefield emission PSF)")
    print("-" * 60)

    psf_params = PsfParams(
        imaging_mode="widefield",
        psf_type="gaussian_analytical",  # avoid the slow Fresnel path
        lambda_em_um=0.52,               # GFP emission ~509 nm
        obj_na=0.8,
        n=1.35,
        psf_sz=(12.0, 12.0, 20.0),       # XY in um; Z is replaced by vol depth
    )

    opt_out = simulate_optical_propagation(
        vol_params=vol_params,
        psf_params=psf_params,
        vol_out=vol_out,
        verbose=1,
    )
    print(f"  Widefield PSF shape: {opt_out.psf.shape}  "
          f"(Np1, Np2, Nz_vol)")

    # ==================================================================
    # Phase 3: Time Traces
    # ==================================================================
    print("\n" + "-" * 60)
    print("PHASE 3: Time Traces (Spikes + Calcium)")
    print("-" * 60)

    K = len(vol_out.gp_vals)
    has_axons = len(vol_out.bg_proc) > 0

    spike_params = SpikeParams(
        K=K,
        nt=nt,
        dt=dt,
        N_bg=0,
        axonflag=has_axons,
        rate=0.25,
        prot="GCaMP6f",
    )
    cal_params = CalciumParams(prot_type="gcamp6f")

    time_out = generate_time_traces(
        spike_params=spike_params,
        cal_params=cal_params,
        n_locs=vol_out.locs,
        verbose=1,
    )
    print(f"  Soma traces: {time_out.soma.shape}")

    # ==================================================================
    # Phase 4: Widefield (camera) scanning
    # ==================================================================
    print("\n" + "-" * 60)
    print("PHASE 4: Widefield scanning (camera-based)")
    print("-" * 60)

    scan_params = ScanParams(
        scan_buff=10,
        motion=True,
        sfrac=2,
        verbose=1,
    )
    wf_params = WidefieldParams(
        pavg=2.0,           # mW/mm^2 Koehler illumination
        lambda_ex_um=0.488,
        qe_det=0.8,
    )
    cam_params = CameraNoiseParams(
        qe=1.0,              # QE already folded into widefield_signal_scale
        dark_rate=0.3,
        t_exp=dt,
        read_noise=1.6,
        gain_e_per_adu=1.0,
    )

    # scan_volume auto-dispatches to scan_widefield because
    # opt_out.params['psf_params'].imaging_mode == 'widefield'.
    # We call scan_widefield directly so we can pass wf_params / cam_params.
    from calcia.scanning import scan_widefield
    scan_out = scan_widefield(
        vol_out=vol_out,
        opt_out=opt_out,
        time_out=time_out,
        scan_params=scan_params,
        cam_params=cam_params,
        wf_params=wf_params,
        spike_params=spike_params,
        seed=seed,
    )

    print(f"  Movie shape: {scan_out.mov.shape}  (H x W x Nt)")
    print(f"  Noisy  range: [{scan_out.mov.min():.1f}, "
          f"{scan_out.mov.max():.1f}]")
    print(f"  Clean  range: [{scan_out.mov_raw.min():.3g}, "
          f"{scan_out.mov_raw.max():.3g}]")

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
        print("\nGenerating GIF video...")
        make_video(
            scan_out.mov, scan_out.mov_raw,
            os.path.join(outdir, f"movie_{tag}.gif"),
            dt=dt, fps=min(30, int(1 / dt)),
        )

    # Intermediate .mat dump for comparison with two-photon
    print("\nSaving intermediate .mat for comparison...")
    _save_all_phases(outdir, tag, vol_out, opt_out, time_out, scan_out)

    print("\n" + "=" * 60)
    print("Widefield simulation complete.")
    print(f"  Movie:    {scan_out.mov.shape[0]}x{scan_out.mov.shape[1]}"
          f", {scan_out.mov.shape[2]} frames")
    print(f"  Duration: {scan_out.mov.shape[2] * dt:.1f} s at {1/dt:.0f} Hz")
    print(f"  Output:   {outdir}")
    print("=" * 60)


def _save_all_phases(outdir, tag, vol_out, opt_out, time_out, scan_out):
    """Dump key intermediates so widefield & two-photon runs can be compared."""
    import scipy.io as sio

    mat_path = os.path.join(outdir, f"phases_{tag}.mat")
    d = {}

    # Phase 1
    d["neur_vol"] = vol_out.neur_vol
    d["locs"] = vol_out.locs

    # Phase 2 (widefield)
    d["psf"] = opt_out.psf
    d["mask"] = opt_out.mask
    d["col_mask"] = opt_out.col_mask

    # Phase 3
    d["soma"] = time_out.soma
    if time_out.dend is not None:
        d["dend"] = time_out.dend
    if time_out.bg is not None:
        d["bg"] = time_out.bg

    # Phase 4 (widefield)
    d["mov"] = scan_out.mov
    d["mov_raw"] = scan_out.mov_raw
    d["mot_hist"] = scan_out.mot_hist

    sio.savemat(mat_path, d, do_compression=True)
    sz_mb = os.path.getsize(mat_path) / 1024 / 1024
    print(f"  Full data saved: {mat_path} ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()

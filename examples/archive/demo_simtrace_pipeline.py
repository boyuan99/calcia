"""Demo: NAOMi pipeline driven by a *sim-trace* coupling design.

Same end-to-end pipeline as ``demo_full_pipeline.py`` (Phase 1 volume -> Phase 2
optics -> Phase 3 traces -> Phase 4 scan -> movie), but Phase 3 sources its spike
trains from a `sim-trace` ``IntensityModel`` instead of calcia's built-in spatial
Hawkes / burst-Poisson.

sim-trace organises spike generation by *inter-neuron coupling structure*
(independent / shared-drive / pairwise / higher-order / hierarchical-latent).
Its own spike->calcium converter is intentionally minimal; it defers biophysical
calcium dynamics to calcia. This demo closes the loop: sim-trace picks *how the
population fires together*, calcia turns that into realistic fluorescence + a
scanned movie. See ``calcia/traces/simtrace_bridge.py``.

Usage:
    conda run -n calcia python examples/demo_simtrace_pipeline.py
    conda run -n calcia python examples/demo_simtrace_pipeline.py --design hmm_gated_hawkes
    conda run -n calcia python examples/demo_simtrace_pipeline.py --small --cache
    conda run -n calcia python examples/demo_simtrace_pipeline.py --design hawkes_scale_free --nt 400
"""

import os as _os, sys as _sys  # archived: find sibling helpers in ../
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import pickle

import numpy as np

# Reuse the movie / tiff writers from the native demo.
from demo_full_pipeline import make_video, save_tiff  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="NAOMi + sim-trace pipeline demo")
    size = p.add_mutually_exclusive_group()
    size.add_argument("--small", action="store_true",
                      help="Small volume for quick testing (~1-2 min)")
    size.add_argument("--large", action="store_true",
                      help="Full MATLAB-default size (250x250x100 um)")
    p.add_argument("--design", default="hmm_gated_hawkes",
                   choices=["hawkes_smallworld", "hawkes_scale_free",
                            "hmm_gated_hawkes"],
                   help="sim-trace coupling design driving Phase 3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nt", type=int, default=None,
                   help="Number of movie frames (overrides preset)")
    p.add_argument("--rate", type=float, default=0.25,
                   help="sim-trace baseline rate scale")
    p.add_argument("--bg-scale", type=float, default=1.0,
                   help="Neuropil/background brightness scale (1.0=NAOMi default)")
    p.add_argument("--outdir", default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-tiff", action="store_true")
    p.add_argument("--cache", action="store_true",
                   help="Cache Phase 1-2 to disk; reuse on subsequent runs")
    return p.parse_args()


def build_design(name, rate):
    """Instantiate a sim-trace model factory by name with a chosen rate."""
    from calcia.traces.simtrace_bridge import (
        hawkes_smallworld, hawkes_scale_free, hmm_gated_hawkes,
    )
    if name == "hawkes_smallworld":
        return hawkes_smallworld(rate=rate)
    if name == "hawkes_scale_free":
        return hawkes_scale_free(rate=rate)
    if name == "hmm_gated_hawkes":
        return hmm_gated_hawkes(rate=rate)
    raise ValueError(name)


def main():
    args = parse_args()

    from calcia import (
        simulate_neural_volume,
        simulate_optical_propagation,
        scan_volume,
    )
    from calcia.config.params import (
        VolumeParams, PsfParams, SpikeParams, CalciumParams,
        ScanParams, NoiseParams, TpmParams,
    )
    from calcia.traces.simtrace_bridge import generate_time_traces_simtrace

    # ---- preset ----
    if args.large:
        vol_sz, vol_depth, nt, preset = (250, 250, 100), 100, args.nt or 1500, "large"
    elif args.small:
        vol_sz, vol_depth, nt, preset = (80, 80, 50), 50, args.nt or 300, "small"
    else:
        vol_sz, vol_depth, nt, preset = (100, 100, 50), 50, args.nt or 500, "medium"

    seed = args.seed
    dt = 1 / 30  # 30 Hz frame rate

    outdir = args.outdir or os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    tag = f"simtrace_{args.design}_{preset}_seed{seed}"

    print("=" * 60)
    print("NAOMi + sim-trace Pipeline Demo")
    print(f"  Design:  {args.design}  (sim-trace coupling structure)")
    print(f"  Preset:  {preset}   Volume: {vol_sz} um, depth={vol_depth} um")
    print(f"  Frames:  {nt} ({nt * dt:.1f} s at {1/dt:.0f} Hz)")
    print(f"  Seed:    {seed}")
    print("=" * 60)

    vol_params = VolumeParams(vol_sz=vol_sz, vres=2, vol_depth=vol_depth)
    # Fast analytical Gaussian PSF (no full Fresnel wave-optics): Phase 2 is
    # only a fixed optics stage here, and the analytical path keeps this demo
    # light/quick so the focus stays on the sim-trace-driven Phase 3 + rescan.
    psf_params = PsfParams(psf_type="gaussian_analytical")

    # ---- Phase 1-2 (cacheable: independent of the trace design) ----
    cache_path = os.path.join(outdir, f"phase12_cache_{preset}_seed{seed}.pkl")
    if args.cache and os.path.exists(cache_path):
        print(f"\nLoading Phase 1-2 from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            vol_out, vol_params, opt_out = pickle.load(f)
    else:
        print("\n" + "-" * 60)
        print("PHASE 1: Neural Volume Simulation")
        print("-" * 60)
        vol_out = simulate_neural_volume(vol_params=vol_params, seed=seed, verbose=1)
        vol_params = vol_out.params["vol_params"]
        print(f"  Neurons: {vol_params.N_neur}, grid: {vol_out.neur_vol.shape}")

        print("\n" + "-" * 60)
        print("PHASE 2: Optical Propagation (PSF)")
        print("-" * 60)
        opt_out = simulate_optical_propagation(
            vol_params=vol_params, psf_params=psf_params, vol_out=vol_out, verbose=1)
        print(f"  PSF shape: {opt_out.psf.shape}")

        if args.cache:
            with open(cache_path, "wb") as f:
                pickle.dump((vol_out, vol_params, opt_out), f)
            print(f"  Cached Phase 1-2 to: {cache_path}")

    # ==================================================================
    # Phase 3: Time Traces  <-- sim-trace coupling design drives the spikes
    # ==================================================================
    print("\n" + "-" * 60)
    print("PHASE 3: Time Traces (sim-trace spikes -> calcia calcium)")
    print("-" * 60)

    K = len(vol_out.gp_vals)
    has_axons = len(vol_out.bg_proc) > 0

    spike_params = SpikeParams(
        K=K, nt=nt, dt=dt,
        N_bg=0, axonflag=has_axons,
        rate=args.rate, prot="GCaMP6f", bg_scale=args.bg_scale,
    )
    cal_params = CalciumParams(prot_type="gcamp6f")
    model_factory = build_design(args.design, args.rate)

    print(f"  K components: {K}   coupling: {args.design}")
    time_out = generate_time_traces_simtrace(
        spike_params=spike_params,
        model_factory=model_factory,
        cal_params=cal_params,
        n_locs=vol_out.locs,
        seed=seed,
        verbose=1,
    )
    print(f"  Soma traces: {time_out.soma.shape}")
    if time_out.spikes is not None:
        act = int((time_out.spikes.sum(1) > 0).sum())
        print(f"  Active soma: {act}/{K}, total spikes {int(time_out.spikes.sum())}")
    if time_out.bg is not None:
        print(f"  BG traces:   {time_out.bg.shape}")

    # ==================================================================
    # Phase 4: Volume Scanning -> movie
    # ==================================================================
    print("\n" + "-" * 60)
    print("PHASE 4: Volume Scanning")
    print("-" * 60)
    scan_out = scan_volume(
        vol_out=vol_out, opt_out=opt_out, time_out=time_out,
        scan_params=ScanParams(motion=True, scan_avg=2),
        noise_params=NoiseParams(), tpm_params=TpmParams(),
        spike_params=spike_params, seed=seed,
    )
    print(f"  Movie: {scan_out.mov.shape}  (H x W x Nt)")
    print(f"  Noisy range [{scan_out.mov.min():.1f}, {scan_out.mov.max():.1f}]")

    # ---- outputs ----
    if not args.no_tiff:
        print("\nSaving TIFF stacks...")
        save_tiff(scan_out.mov, os.path.join(outdir, f"movie_noisy_{tag}.tif"))
        save_tiff(scan_out.mov_raw, os.path.join(outdir, f"movie_clean_{tag}.tif"))

    if not args.no_video:
        print("\nGenerating GIF video...")
        make_video(scan_out.mov, scan_out.mov_raw,
                   os.path.join(outdir, f"movie_{tag}.gif"),
                   dt=dt, fps=min(30, int(1 / dt)))

    print(f"\nAll outputs saved to: {outdir}")
    print("Done!")


if __name__ == "__main__":
    import _instrument; _instrument.start()  # run log + pyinstrument (mandated)
    main()

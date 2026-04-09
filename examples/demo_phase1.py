"""Demo: Run the full Phase 1 neural volume simulation pipeline.

This script demonstrates the complete ``simulate_neural_volume`` function,
which executes all 7 steps of the NAOMi neural volume generation:

  1. Blood vessel simulation
  2. Neuron sampling and shape generation
  3. Neural volume voxelization
  4. Basal dendrite growth
  5. Through-volume apical dendrite growth
  6. Cell fluorescence distribution
  7. Background dendrites and axon generation

Usage:
    conda run -n calcia python examples/demo_phase1.py
"""

import os
import time

import numpy as np

from calcia import simulate_neural_volume, export_pipeline_output, import_pipeline_output
from calcia.config.params import VolumeParams


def main():
    print("Phase 1 Demo: Neural Volume Simulation")
    print("=" * 60)

    # --- Configure parameters ---
    # Match MATLAB NAOMi TPM_Simulation_Script.m defaults:
    #   vol_sz = [250, 250, 100], vol_depth = 100, vres = 2
    # Grid = 500x500x200 = 50M voxels, ~625 neurons by density
    vol_params = VolumeParams(
        vol_sz=(250, 250, 100),  # 250x250x100 um
        vres=2,                  # 2 voxels/um -> 500x500x200 grid
        vol_depth=100,           # 100 um below surface
    )

    # --- Run pipeline ---
    t0 = time.perf_counter()
    result = simulate_neural_volume(
        vol_params=vol_params,
        seed=42,
        verbose=1,
    )
    elapsed = time.perf_counter() - t0

    # --- Print summary ---
    print(f"\nCompleted in {elapsed:.1f} seconds.\n")

    grid = result.neur_vol.shape
    p = result.params["vol_params"]

    print("Output summary:")
    print(f"  neur_vol shape:  {grid} (dtype={result.neur_vol.dtype})")
    print(f"  neur_num shape:  {result.neur_num.shape} (dtype={result.neur_num.dtype})")
    print(f"  Neuron positions: {result.locs.shape[0]} neurons")
    print(f"  gp_vals:         {len(result.gp_vals)} components")
    print(f"  gp_bgvals:       {len(result.gp_bgvals)} axon processes")
    print(f"  bg_proc:         {len(result.bg_proc)} sorted bg processes")
    if result.neur_ves is not None:
        print(f"  neur_ves:        {result.neur_ves.shape} "
              f"({np.sum(result.neur_ves > 0):,} vessel voxels)")
    else:
        print("  neur_ves:        None (vessels disabled)")

    # Voxel breakdown
    nn = result.neur_num
    total = int(np.prod(grid))
    n_neuron = int(np.sum((nn >= 1) & (nn <= p.N_neur + p.N_den)))
    n_bg = int(np.sum(nn > p.N_neur + p.N_den))
    n_empty = int(np.sum(nn == 0))

    print(f"\nVoxel breakdown ({total:,} total):")
    print(f"  Neurons + dendrites: {n_neuron:,} ({100*n_neuron/total:.1f}%)")
    print(f"  Background neuropil: {n_bg:,} ({100*n_bg/total:.1f}%)")
    print(f"  Empty:               {n_empty:,} ({100*n_empty/total:.1f}%)")

    fl = result.neur_vol
    fl_nz = fl[fl > 0]
    if len(fl_nz) > 0:
        print(f"\nFluorescence (non-zero voxels: {len(fl_nz):,}):")
        print(f"  min={fl_nz.min():.4f}, max={fl_nz.max():.4f}, "
              f"mean={fl_nz.mean():.4f}, std={fl_nz.std():.4f}")

    print("\nAll parameter objects available in result.params:")
    for name in result.params:
        print(f"  result.params['{name}']")

    # --- Save / Load demo ---
    print("\n" + "=" * 60)
    print("Save / Load Demo")
    print("=" * 60)

    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    npz_path = os.path.join(outdir, "output.npz")

    # Save NPZ
    t0 = time.perf_counter()
    export_pipeline_output(npz_path, result, random_seed=42)
    t_npz = time.perf_counter() - t0
    npz_size = os.path.getsize(npz_path) / 1024 / 1024

    print(f"\nSaved to {outdir}")
    print(f"  NPZ:  {npz_size:.2f} MB ({t_npz:.2f}s)")

    # Load NPZ and verify round-trip
    t0 = time.perf_counter()
    loaded = import_pipeline_output(npz_path)
    t_load = time.perf_counter() - t0

    print(f"\nLoaded NPZ in {t_load:.2f}s")
    print(f"  neur_vol match: {np.allclose(result.neur_vol, loaded.neur_vol)}")
    print(f"  neur_num match: {np.array_equal(result.neur_num, loaded.neur_num)}")
    print(f"  gp_vals count:  {len(loaded.gp_vals)} (original: {len(result.gp_vals)})")
    print(f"  gp_bgvals count: {len(loaded.gp_bgvals)} (original: {len(result.gp_bgvals)})")
    print(f"  bg_proc count:  {len(loaded.bg_proc)} (original: {len(result.bg_proc)})")


if __name__ == "__main__":
    main()

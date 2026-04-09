"""Inspect Python vessel distribution to diagnose flat-slab appearance."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calcia import import_pipeline_output

out = import_pipeline_output('examples/output/output.npz')
ves = out.neur_ves
print(f"neur_ves shape: {ves.shape}, dtype: {ves.dtype}")
print(f"Nonzero voxels: {np.sum(ves > 0):,}")

# Per-axis distribution
print("\nVessel voxels per Z-slice (dim2) — first 20 and last 20:")
z_counts = np.sum(ves > 0, axis=(0, 1))
print(f"  z[0..19]:   {z_counts[:20].tolist()}")
print(f"  z[180..199]: {z_counts[180:].tolist()}")
print(f"  z min={z_counts.min()}, max={z_counts.max()}, nonzero slices={np.sum(z_counts>0)}")

print("\nVessel voxels per X-slice (dim0) — min/max:")
x_counts = np.sum(ves > 0, axis=(1, 2))
print(f"  x min={x_counts.min()}, max={x_counts.max()}, nonzero slices={np.sum(x_counts>0)}")

print("\nVessel voxels per Y-slice (dim1):")
y_counts = np.sum(ves > 0, axis=(0, 2))
print(f"  y min={y_counts.min()}, max={y_counts.max()}, nonzero slices={np.sum(y_counts>0)}")

# Bounding box of vessels
nz = np.argwhere(ves > 0)
print(f"\nVessel bounding box:")
print(f"  dim0 (X): {nz[:,0].min()} .. {nz[:,0].max()}")
print(f"  dim1 (Y): {nz[:,1].min()} .. {nz[:,1].max()}")
print(f"  dim2 (Z): {nz[:,2].min()} .. {nz[:,2].max()}")

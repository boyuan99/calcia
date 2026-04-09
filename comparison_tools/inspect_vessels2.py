"""Inspect vessel network node positions to understand coordinate system."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calcia.config.params import VolumeParams, VascParams
from calcia.volume.vasculature import simulate_blood_vessels

np.random.seed(42)
vol_params = VolumeParams(vol_sz=(250, 250, 100), vol_depth=100, vres=2)
vasc_params = VascParams()

print(f"vol_sz={vol_params.vol_sz}, vol_depth={vol_params.vol_depth}, vres={vol_params.vres}")
print(f"Imaging volume voxel shape: {tuple(s*vol_params.vres for s in vol_params.vol_sz)}")

net = simulate_blood_vessels(vol_params, vasc_params, verbose=0)

# nodes is a list
all_z = [n.pos[2] for n in net.nodes]
all_x = [n.pos[0] for n in net.nodes]
all_y = [n.pos[1] for n in net.nodes]

print(f"\nNode count: {len(net.nodes)}")
print(f"Node pos[2] (depth z, µm): min={min(all_z):.1f}, max={max(all_z):.1f}, mean={np.mean(all_z):.1f}")
print(f"Node pos[0] (x, µm):       min={min(all_x):.1f}, max={max(all_x):.1f}")
print(f"Node pos[1] (y, µm):       min={min(all_y):.1f}, max={max(all_y):.1f}")

# Node types distribution
from collections import Counter
type_counts = Counter(n.type for n in net.nodes)
print(f"\nNode types: {dict(type_counts)}")

# Per-type z range
for t in sorted(type_counts):
    zs = [n.pos[2] for n in net.nodes if n.type == t]
    print(f"  type {t}: z=[{min(zs):.1f}, {max(zs):.1f}]  (n={len(zs)})")

# Connection locs z range
all_conn_z = []
for conn in net.connections:
    for pos in conn.locs:
        all_conn_z.append(pos[2])
print(f"\nConnection locs pos[2]: min={min(all_conn_z):.1f}, max={max(all_conn_z):.1f}")
print(f"  * vol_depth={vol_params.vol_depth}µm, vol_sz[2]={vol_params.vol_sz[2]}µm")
print(f"  * Expected imaging range: {vol_params.vol_depth} .. {vol_params.vol_depth + vol_params.vol_sz[2]} µm")
print(f"  * In imaging range: {sum(vol_params.vol_depth <= z <= vol_params.vol_depth + vol_params.vol_sz[2] for z in all_conn_z)} / {len(all_conn_z)}")
print(f"  * Above surface (< vol_depth={vol_params.vol_depth}): {sum(z < vol_params.vol_depth for z in all_conn_z)}")
print(f"  * Below imaging vol (> {vol_params.vol_depth + vol_params.vol_sz[2]}): {sum(z > vol_params.vol_depth + vol_params.vol_sz[2] for z in all_conn_z)}")

# Capillary connectivity
cap_nodes = [n for n in net.nodes if n.type == 4]
vtcp_caps = [n for n in cap_nodes if n.root >= 0]
free_caps = [n for n in cap_nodes if n.root < 0]
conn_counts = [len(n.conn) for n in cap_nodes]
print(f"\nCapillary connectivity:")
print(f"  count: {len(cap_nodes)}  (vtcp={len(vtcp_caps)}, free={len(free_caps)})")
print(f"  connections/node: min={min(conn_counts)}, max={max(conn_counts)}, mean={np.mean(conn_counts):.2f}")
print(f"  nodes with 0 conn: {sum(c == 0 for c in conn_counts)}")
print(f"  nodes with 1 conn: {sum(c == 1 for c in conn_counts)}")
print(f"  nodes with ≥2 conn: {sum(c >= 2 for c in conn_counts)}")

# vessel_volume shape and voxel counts
print(f"\nVessel volume:")
vv = net.vessel_volume
if vv is not None:
    z_off = int(vol_params.vol_depth * vol_params.vres)
    print(f"  shape: {vv.shape}  (expected: (500, 500, 400))")
    print(f"  full total voxels: {int(np.sum(vv > 0)):,}  (MATLAB: ~4,549,012)")
    print(f"  imaging region (z≥{z_off}): {int(np.sum(vv[:,:,z_off:] > 0)):,}  (MATLAB: ~682,504)")
    print(f"  surface region (z< {z_off}): {int(np.sum(vv[:,:,:z_off] > 0)):,}  (MATLAB: ~3,866,508)")
else:
    print("  vessel_volume is None")

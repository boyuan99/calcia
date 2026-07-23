"""Generate a DEEP thin-vessel striatum volume for a GIVEN seed, and write a
FLAT-illumination stub for it — the end-to-end building block of the two-colour
DIVERSITY series.

Unlike the fixed-seed gen scripts, this takes ``--seed`` so every invocation
produces a GENUINELY DIFFERENT volume: different neuron sampling (positions) AND
a different vessel Dijkstra layout. That is what makes the resulting GCaMP/tdt
videos differ in geometry, not just in activity.

The stub it writes has ``illum_grad:false`` (flat illumination) so the GCaMP demo
reads flat and the tdt demo pairs with ``--no-illum`` — the design-pure BEST
recipe (two-scale PSF, composite OFF, flat illum).

Thin-visible vessels are injected by monkeypatching ``_STRIATUM_VASC`` in THIS
process only (no source edit; original values in
calcia/config/region_presets_backup/striatum_vasc_original.md).

Run:
    conda run -n calcia python examples/gen_deep_thinvessels_seed.py --seed 7
    conda run -n calcia python examples/gen_deep_thinvessels_seed.py --seed 7 --vol-um 500
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import pickle
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

import _striatum_common as _C

import calcia.config.params as _P
# thin-visible vessels (visible dark vasculature), striatum topology kept
_P._STRIATUM_VASC = {"depth_surf": 0.0, "vesSize": (4.0, 3.0, 1.5),
                     "vesFreq": (300.0, 350.0, 100.0), "distsc": 4.0}

from calcia import simulate_neural_volume
from calcia.config.params import VolumeParams

OUT = os.path.join(os.path.dirname(__file__), "output")
SHARED = os.path.join(OUT, "_shared")


def parse_args():
    p = argparse.ArgumentParser(description="Deep thin-vessel volume for a seed (flat stub)")
    p.add_argument("--seed", type=int, required=True,
                   help="Phase-1 seed: changes neuron sampling AND vessel layout.")
    p.add_argument("--vol-um", type=float, default=500.0, dest="vol_um",
                   help="Lateral FOV in um (square). 500 ~54 min; 1000 much longer.")
    p.add_argument("--depth-um", type=float, default=180.0, dest="depth_um")
    p.add_argument("--focal-um", type=float, default=45.0, dest="focal_um")
    p.add_argument("--nt", type=int, default=200)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(SHARED, exist_ok=True)
    _C.tee_stdout(f"gen_deepthinves_s{a.seed}_{int(a.vol_um)}um")

    vol_sz = (a.vol_um, a.vol_um, a.depth_um)
    vres = 1
    sig = hashlib.sha1(f"deepthinves_{vol_sz}_{vres}_{a.seed}".encode()).hexdigest()[:10]
    cache = os.path.join(SHARED, f"phase1_deepthinves_s{a.seed}_{sig}.pkl")

    if os.path.exists(cache):
        print(f"cache already exists -> {cache} (reusing)")
        with open(cache, "rb") as f:
            vol_out, vp = pickle.load(f)
    else:
        print(f"generating DEEP+THIN-VESSELS volume seed={a.seed} {vol_sz} -> {cache}")
        t0 = time.time()
        vp = VolumeParams(vol_sz=vol_sz, vres=vres, vol_depth=0, region="striatum",
                          N_neur=None)
        vol_out = simulate_neural_volume(vol_params=vp, seed=a.seed, verbose=1)
        vp = vol_out.params["vol_params"]
        nves = int((np.asarray(vol_out.neur_ves) > 0).sum())
        print(f"done in {time.time()-t0:.0f}s  N_neur={vp.N_neur}  "
              f"vessel voxels={nves} ({100*nves/np.prod(vol_out.neur_vol.shape):.2f}%)")
        with open(cache, "wb") as f:
            pickle.dump((vol_out, vp), f)
        print(f"cached ({os.path.getsize(cache)/1e9:.1f} GB)")

    ves = np.asarray(vol_out.neur_ves)
    nves = int((ves > 0).sum())
    # FLAT stub (illum_grad:false) — the design-pure BEST recipe.
    stub = os.path.join(OUT, f"deepthinves_s{a.seed}_{int(a.vol_um)}um_flat_stub")
    os.makedirs(stub, exist_ok=True)

    # Lightweight volume-stats sidecar so the diversity-verifier need not reload the
    # multi-GB Phase-1 pickle. Soma centres + vessel projection/depth-profile are all
    # it needs to quantify geometric diversity across volumes.
    soma_ids = np.array([i for i, cfd in enumerate(vol_out.gp_vals)
                         if np.any(np.asarray(cfd.soma_mask))], dtype=np.int64)
    soma_locs = np.asarray(vol_out.locs)[soma_ids].astype(np.float32)
    ves_bin = (ves > 0)
    np.savez_compressed(
        os.path.join(stub, "volume_stats.npz"),
        seed=a.seed, N_neur=int(vp.N_neur), n_soma=len(soma_ids),
        soma_locs=soma_locs, vol_sz=np.array([a.vol_um, a.vol_um, a.depth_um]),
        vessel_voxels=nves, vessel_frac=float(nves / ves.size),
        vessel_proj=ves_bin.any(axis=2).astype(np.uint8),          # XY footprint
        vessel_depth=ves_bin.reshape(-1, ves.shape[2]).sum(0).astype(np.int64))
    meta = dict(tag="deepthinves_seed_flat", phase1_cache=cache, seed=a.seed,
                nt=a.nt, vres=vres, vol_sz=[a.vol_um, a.vol_um, a.depth_um],
                region="striatum", illum_grad=False, deep=True,
                vessels="thin-visible", focal_depth_um=a.focal_um,
                vessel_voxels=nves, N_neur=int(vp.N_neur),
                note="deep thin-vessel volume, per-seed, flat illum",
                timestamp=_dt.datetime.now().isoformat())
    with open(os.path.join(stub, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"STUB: {os.path.basename(stub)}  N_neur={vp.N_neur}  vessel_vox={nves}")


if __name__ == "__main__":
    main()

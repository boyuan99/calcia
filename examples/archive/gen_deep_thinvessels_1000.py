"""Generate a DEEP striatum volume (500x500x180) WITH thin-visible (visible) blood
vessels, for the realistic two-colour videos. Depth gives the OOF wash; the dark
vessels are the dominant texture in the real tdt/GCaMP data.

The thin-visible vessel values are injected by monkeypatching ``_STRIATUM_VASC`` in
THIS process only (no source-file edit; nothing to revert). Original values live in
calcia/config/region_presets_backup/striatum_vasc_original.md.
"""
import datetime as _dt
import hashlib
import json
import os
import pickle
import sys
import time

# Unbuffered stdout so the [1/7]..[7/7] progress streams live in background runs
# (Python block-buffers stdout when not a tty). Run with `python -u` too.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
from pyinstrument import Profiler

import calcia.config.params as _P

_profiler = Profiler(); _profiler.start()

# archived: add examples/ (parent dir) so the sibling import resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _striatum_common as _C
_C.tee_stdout(os.path.splitext(os.path.basename(__file__))[0])

# thin-visible vessels (visible dark vasculature), striatum topology kept
_P._STRIATUM_VASC = {"depth_surf": 0.0, "vesSize": (4.0, 3.0, 1.5),
                     "vesFreq": (300.0, 350.0, 100.0), "distsc": 4.0}

from calcia import simulate_neural_volume
from calcia.config.params import VolumeParams

OUT = os.path.join(os.path.dirname(__file__), "output")
SHARED = os.path.join(OUT, "_shared")
os.makedirs(SHARED, exist_ok=True)

VOL_SZ = (1000, 1000, 180); VRES = 1; SEED = 42; NT = 200; FOCAL_UM = 45.0
sig = hashlib.sha1(f"deepthinves1k_{VOL_SZ}_{VRES}_{SEED}".encode()).hexdigest()[:10]
cache = os.path.join(SHARED, f"phase1_deepthinves1k_{sig}.pkl")

print(f"generating DEEP+THIN-VESSELS striatum volume {VOL_SZ} -> {cache}")
t0 = time.time()
vp = VolumeParams(vol_sz=VOL_SZ, vres=VRES, vol_depth=0, region="striatum",
                  N_neur=None)
vol_out = simulate_neural_volume(vol_params=vp, seed=SEED, verbose=1)
vp = vol_out.params["vol_params"]
nves = int((np.asarray(vol_out.neur_ves) > 0).sum())
print(f"done in {time.time()-t0:.0f}s  N_neur={vp.N_neur}  "
      f"vessel voxels={nves} ({100*nves/np.prod(vol_out.neur_vol.shape):.2f}%)")

with open(cache, "wb") as f:
    pickle.dump((vol_out, vp), f)
print(f"cached ({os.path.getsize(cache)/1e9:.1f} GB)")

run_dir = os.path.join(OUT, f"deepthinves1k_volume_1000um_d180_{_dt.datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(run_dir, exist_ok=True)
meta = dict(tag="deepthinves1k_volume", phase1_cache=cache, seed=SEED, nt=NT, vres=VRES,
            vol_sz=list(VOL_SZ), region="striatum", illum_grad=True, deep=True,
            vessels="thin-visible", focal_depth_um=FOCAL_UM, vessel_voxels=nves,
            config=dict(illum=dict(enable=True, cx=0.48, cy=0.45, sx=0.40,
                                   sy=0.45, floor=0.05)),
            N_neur=int(vp.N_neur), note="deep + visible vessels")
with open(os.path.join(run_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

# pyinstrument profile of the whole Phase-1 generation (where the hours go)
_profiler.stop()
with open(os.path.join(run_dir, "profile_phase1.html"), "w", encoding="utf-8") as f:
    f.write(_profiler.output_html())
print(_profiler.output_text(unicode=True, color=False, show_all=False))
print(f"stub run dir: {os.path.basename(run_dir)}")

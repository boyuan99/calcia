"""Generate a DEEP striatum volume (500 x 500 x 180 um) and cache it, with a
match-run-compatible metadata stub. Depth is the physically-correct lever for the
real 1P wash: out-of-focus fluorescence from the full tissue column forms a smooth
bright haze that dilutes cell contrast + dF/F to real levels (validated: 200 um ->
dff_p99 0.24, flatCV 0.11), while the in-focus slab stays SHARP. NO artificial
blur/scatter needed.

500 x 500 um FOV keeps generation tractable (~1-2 h); the full 1.7 mm FOV at this
depth would be ~10 h (dendrite growth scales with the ~5x10^4 neurons).
"""
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
from pyinstrument import Profiler

_profiler = Profiler(); _profiler.start()

import _striatum_common as _C
_C.tee_stdout(os.path.splitext(os.path.basename(__file__))[0])

from calcia import simulate_neural_volume
from calcia.config.params import VolumeParams

OUT = os.path.join(os.path.dirname(__file__), "output")
SHARED = os.path.join(OUT, "_shared")
os.makedirs(SHARED, exist_ok=True)

VOL_SZ = (500, 500, 180)
VRES = 1
SEED = 42
NT = 200
FOCAL_UM = 45.0   # focus in the upper tissue; deep tissue -> OOF smooth haze

sig = hashlib.sha1(f"deep_{VOL_SZ}_{VRES}_{SEED}".encode()).hexdigest()[:10]
cache = os.path.join(SHARED, f"phase1_deep_{sig}.pkl")

print(f"generating DEEP striatum volume {VOL_SZ} -> {cache}")
t0 = time.time()
vp = VolumeParams(vol_sz=VOL_SZ, vres=VRES, vol_depth=0, region="striatum",
                  N_neur=None)
vol_out = simulate_neural_volume(vol_params=vp, seed=SEED, verbose=1)
vp = vol_out.params["vol_params"]
print(f"done in {time.time()-t0:.0f}s  N_neur={vp.N_neur}  grid={vol_out.neur_vol.shape}")

with open(cache, "wb") as f:
    pickle.dump((vol_out, vp), f)
print(f"cached ({os.path.getsize(cache)/1e9:.1f} GB)")

run_dir = os.path.join(OUT, f"deep_volume_{VOL_SZ[0]}um_d{VOL_SZ[2]}_"
                       f"{_dt.datetime.now():%Y%m%d_%H%M%S}")
os.makedirs(run_dir, exist_ok=True)
meta = dict(tag="deep_volume", phase1_cache=cache, seed=SEED, nt=NT, vres=VRES,
            vol_sz=list(VOL_SZ), region="striatum", illum_grad=True, deep=True,
            focal_depth_um=FOCAL_UM,
            config=dict(illum=dict(enable=True, cx=0.48, cy=0.45, sx=0.40,
                                   sy=0.45, floor=0.05)),
            N_neur=int(vp.N_neur), note="deep tissue column for realistic OOF haze")
with open(os.path.join(run_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

_profiler.stop()
with open(os.path.join(run_dir, "profile_phase1.html"), "w", encoding="utf-8") as f:
    f.write(_profiler.output_html())
print(_profiler.output_text(unicode=True, color=False, show_all=False))
print(f"stub run dir: {os.path.basename(run_dir)}")
print(f"\nNext (NO scatter — depth does the work):\n"
      f"  python examples/demo_gcamp_realistic_matched.py --match-run "
      f"{os.path.basename(run_dir)} --scatter-um 0\n"
      f"  python examples/demo_static_tdtomato_matched.py --match-run "
      f"{os.path.basename(run_dir)} --scatter-um 0")

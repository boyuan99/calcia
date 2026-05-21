# Calcia — Known Issues & Deferred Work

This document records confirmed deviations from MATLAB NAOMi, their root causes,
estimated impact, and recommended fix priority.

---

## Phase 1 — Blood Vessel Simulation (`vasculature.py`)

### Issue V1: Surface vessel radius is uniform (not per-connection)

**Status:** Deferred
**Priority:** Medium — fix before Phase 2 validation

**Symptom:**
- Python surface region voxels: ~5.4M (MATLAB: ~3.9M, **+39%**)
- Python imaging region voxels: ~436K (MATLAB: ~682K, **-36%**)

**Root cause:**
`connections_to_volume()` assigns radius based on node type alone:
```python
radius_um = type_radii_um[start_type]  # same for all surface connections
```
`connToVol.m` uses `conn.weight` — a per-connection radius computed via
Dijkstra path propagation in `nodesToConn.m`. Surface connections have highly
variable radii depending on their position in the vascular tree; using a fixed
`vesSize[0] = 15µm` for all of them over-fills the surface region and leaves
the imaging region under-filled.

**What needs to be done:**
1. In `nodes_to_connections()`: propagate per-connection weight (radius) following
   the MATLAB algorithm in `nodesToConn.m` — end nodes get `gamma(3, (vesSize[1]-vesSize[2])/3) + vesSize[2]`;
   interior nodes inherit from neighbors via weighted interpolation.
2. In `connections_to_volume()`: use `conn.weight` (in µm) as the sphere radius
   instead of the node-type lookup table.

**Impact if left unfixed:**
- Imaging region vessel density ~1% lower than MATLAB NAOMi (436K vs 682K out of 25M voxels)
- Phase 2 TPM images will show slightly fewer/thinner dark vessel stripes
- No structural/algorithmic error; purely a quantitative rendering difference

---

### Issue V2: Interior surface node sampling missing (nsurf step)

**Status:** Deferred
**Priority:** Low — contributes to V1 but secondary

**Symptom:** Linked to V1 (surface +39%)

**Root cause:**
MATLAB `growMajorVessels.m` runs `pseudoRandSample2D` to place `nsurf` additional
interior surface nodes after growing edge branches, then connects them via Dijkstra.
Python only grows branches from edge source nodes; no interior sampling step exists.
This means Python generates fewer but thicker-rendered surface vessels.

**What needs to be done:**
After branch growth in `grow_major_vessels()`, add an interior surface node sampling
step using `pseudo_rand_sample_2d()` with `n_surf = round(vol_sz[0]*vol_sz[1] / vesFreq[0]²)`
nodes, then connect them via the existing `connect_vessel_nodes()`.

**Impact if left unfixed:** Minor; already partially compensated by V1 fix.

---

## Phase 1 — Biological Model Limitations

These are limitations of the underlying NAOMi modelling approach (inherited
from the original MATLAB implementation), not MATLAB-parity deviations. They
affect the biological realism of the simulation rather than its fidelity to
the reference port.

### Issue M1: No blood flow simulation inside vessels

**Status:** Deferred — requires substantial model extension
**Priority:** Low for most use cases; **High** if the simulator is used to
study hemodynamic imaging, optical absorption modulation, or flow-induced
motion artifacts.

**Current behavior:**
The vasculature is modeled as a **static geometric scaffold**. Vessels
contribute to the image only through:
- Fluorescence exclusion (vessels are dark holes in the neural volume)
- Hemoglobin absorption as a fixed attenuation
  (`exp(-x/vres * hemo_abs)` in [calcia/optics/signal.py](../calcia/optics/signal.py))

There is no notion of:
- Blood velocity / flow direction along vessel segments
- Pulsatile variation (heartbeat, respiration-synchronized modulation)
- Red blood cell transit (which produces measurable shadow flicker in
  real two-photon recordings — the basis of line-scan blood flow imaging)
- Blood oxygenation changes (HbO/HbR ratio) tied to neural activity — so
  the simulator cannot generate intrinsic optical signal (IOS) or
  hemodynamic response function (HRF) phenomena

**What a fix would entail:**
1. **Flow graph**: assign a directed flow rate to every vessel segment
   (consistent with Murray's law and boundary conditions at surface sources/drains)
2. **Time-varying absorption**: modulate `hemo_abs` per segment over time
   based on local velocity × cross-section + pulsatile component
3. **RBC shot noise**: optionally simulate discrete RBC passage for line-scan
   blood flow applications (stochastic transits modulating local absorption)
4. **Neurovascular coupling**: optional link between neural activity and
   local flow changes to produce HRF-like signals

**Impact if left unfixed:**
- Perfectly acceptable for calcium imaging studies of neural dynamics
  (the dominant NAOMi use case)
- Unsuitable for studies of blood flow imaging, BOLD/IOS signal generation,
  or vascular artifact characterization in two-photon data

---

### Issue M2: Neural volume component overlap may be too permissive

**Status:** Needs verification — observation reported during pipeline runs,
root cause not yet pinned down
**Priority:** Medium — affects biological realism of the neural volume

**Observed symptom:**
When rendering the combined neural volume, components (soma, dendrites,
apical dendrites, background dendrites, axons) appear to overlap with each
other at a level that does not match real cortical tissue, where
cells physically occupy disjoint volumes.

**Suspected contributing factors (to investigate):**
- `sample_dense_neurons` enforces a minimum inter-soma distance, but the
  check is on **soma centers** — dendrites from adjacent neurons may pass
  through each other's soma shells without being rejected
- `grow_neuron_dendrites` (Dijkstra-based) uses a cost volume that discourages
  growing into existing structures but does not make it hard-forbidden; the
  random-weight perturbation can push paths through occupied voxels
- `generate_bg_dendrites` and `generate_axons` random walks likely do not
  read back from the running neural volume — so they can trace through
  anything already placed

**What needs to be done:**
1. **Quantify the problem**: add a diagnostic that measures, per voxel type,
   the fraction of voxels occupied by more than one component — report as a
   histogram (0, 1, 2, 3+ owners per voxel)
2. **Compare against MATLAB**: run the same measurement on MATLAB output to
   determine whether this is a port-introduced regression or a shared NAOMi
   limitation
3. **If port-introduced**: tighten cost_volume penalties / add occupancy checks
   in dendrite / background / axon generators
4. **If shared with MATLAB**: document as a model limitation; a principled fix
   would use hard occupancy volumes (similar to vessel masks) rather than
   soft cost penalties

**Impact if left unfixed:**
- Fluorescence values in heavily-overlapped voxels are artificially elevated
- Spatial crosstalk in downstream Phase 2 PSF convolution is exaggerated —
  demixing / source-separation benchmarks run on calcia data will look
  easier than on real two-photon recordings
- Quantitative comparisons of "what fraction of a pixel is contributed by
  which cell" will be skewed

---

## Phase 1 — General

### Issue G1: Full pipeline runtime is very slow (~2.7 hours)

**Status:** Deferred
**Priority:** Medium — needed for iteration speed

**Symptom:**
`demo_phase1.py` (vol_sz=250×250×100, vres=2) takes ~9600 seconds.

**Likely bottlenecks (from pyinstrument profile):**
- `connections_to_volume()`: sphere-drawing loop in pure Python
- `grow_capillaries()`: O(n²) distance computations
- `generate_bg_dendrites()` / `generate_axons()`: random walk loops

**What needs to be done:**
- `connections_to_volume()`: replace Python sphere loop with `scipy.ndimage` binary dilation or a vectorized approach
- Distance computations: use `scipy.spatial.cKDTree` for nearest-neighbor queries
- Random walk loops: already have numba path; verify it is being used

---

## Phase 2 — TPM Scanning Simulation (Not yet implemented)

### Issue P2-1: Phase 2 not implemented

**Status:** Not started
**Priority:** High — this is the next major milestone

**What Phase 2 covers (MATLAB files):**
| MATLAB file | Description |
|-------------|-------------|
| `scan_ideal.m` | Ideal scanning: PSF convolution + raster sampling |
| `PoissonGaussNoiseModel.m` | Photon shot noise + readout noise |
| `applyNoiseModel.m` | Apply noise to scanned frames |
| `psf_fft.m` | FFT-based PSF application |
| `simulate_optical_propagation2.m` | Depth-dependent PSF broadening |
| `check_psf_params.m` | PSF parameter defaults |
| `check_scan_params.m` | Scanning parameter defaults |
| `check_tpm_params.m` | TPM parameter defaults |
| `check_noise_params.m` | Noise parameter defaults |

**Inputs from Phase 1:**
- `neur_vol` — 3D fluorescence volume (float32)
- `neur_ves` — full-depth vessel mask (uint8, shape 500×500×400)
- `gp_vals`, `gp_nuc`, `gp_soma` — per-neuron fluorescence data
- `bg_proc`, `gp_bgvals` — background/axon fluorescence

**Recommended implementation order:**
1. `check_psf_params.m` → `config/params.py::PSFParams`
2. `check_scan_params.m` → `config/params.py::ScanParams`
3. `check_noise_params.m` → `config/params.py::NoiseParams`
4. `gaussian_psf.m` → `optics/psf.py`
5. `scan_ideal.m` → `scanning/scan_ideal.py`
6. `PoissonGaussNoiseModel.m` → `scanning/noise.py`

---

## Comparison Tooling

### Issue C1: `compare_phase1.py` vessel metric uses old denominator

**Status:** Partially fixed
**Notes:**
Python side now uses full-depth `np.prod(neur_ves.shape)` as denominator.
MATLAB side (`run_phase1_for_comparison.m`) has been updated to use `numel(vol_out.neur_ves)`.
However, the previously saved `matlab_phase1_stats.mat` was generated before this fix —
re-run `run_phase1_for_comparison.m` in MATLAB to regenerate it.

---

## Summary Table

| ID | Area | Description | Priority | Effort |
|----|------|-------------|----------|--------|
| V1 | Vasculature | Per-connection radius in rendering | Medium | ~2 days |
| V2 | Vasculature | Interior surface node sampling | Low | ~0.5 days |
| M1 | Model | No blood flow / hemodynamics simulation | Low (use-case dependent) | ~2 weeks |
| M2 | Model | Neural volume component overlap (needs verification) | Medium | ~1 week |
| G1 | Performance | Full pipeline ~2.7 hrs | Medium | ~3 days |
| P2-1 | Phase 2 | TPM scanning not implemented | **High** | ~2 weeks |
| C1 | Tooling | Regenerate MATLAB stats .mat | Low | ~2 hrs (MATLAB run) |

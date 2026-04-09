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
| G1 | Performance | Full pipeline ~2.7 hrs | Medium | ~3 days |
| P2-1 | Phase 2 | TPM scanning not implemented | **High** | ~2 weeks |
| C1 | Tooling | Regenerate MATLAB stats .mat | Low | ~2 hrs (MATLAB run) |

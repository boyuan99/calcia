# NAOMi MATLAB to Python (Calcia) Migration Plan

This document tracks the migration progress from the original MATLAB NAOMi simulator to the Python Calcia implementation.

## Project Overview

- **Source**: `naomi_sim` (MATLAB)
- **Target**: `calcia` (Python)
- **Goal**: Complete Python reimplementation of the NAOMi two-photon microscopy simulator

---

## Migration Status Summary

| Phase | Description | Status | Progress |
|-------|-------------|--------|----------|
| Phase 1 | Infrastructure & Config | ✅ Complete | 100% |
| Phase 2 | Geometry & Algorithms | ✅ Complete | 100% |
| Phase 3 | Volume Generation | 🔄 Partial | ~60% |
| Phase 4 | Time Trace Generation | ❌ Not Started | 0% |
| Phase 5 | Optical Propagation | ❌ Not Started | 0% |
| Phase 6 | Scanning Simulation | ❌ Not Started | 0% |

---

## Phase 1: Infrastructure & Configuration (✅ Complete)

### Parameter Classes

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `check_vol_params.m` | `config/params.py::VolumeParams` | ✅ |
| `check_neur_params.m` | `config/params.py::NeuronParams` | ✅ |
| `check_vasc_params.m` | `config/params.py::VascParams` | ✅ |
| `check_dend_params.m` | `config/params.py::DendParams` | ✅ |
| `check_bg_params.m` | `config/params.py::BgParams` | ✅ |
| `check_axon_params.m` | `config/params.py::AxonParams` | ✅ |
| `check_spike_opts.m` | - | ❌ Not needed yet |
| `check_noise_params.m` | - | ❌ Not needed yet |
| `check_psf_params.m` | - | ❌ Not needed yet |
| `check_scan_params.m` | - | ❌ Not needed yet |
| `check_tpm_params.m` | - | ❌ Not needed yet |
| `check_cal_params.m` | - | ❌ Not needed yet |

### Utilities

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| Index conversion utilities | `utils/indexing.py` | ✅ |

---

## Phase 2: Geometry & Algorithms (✅ Complete)

### Geometry

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `SpiralSampleSphere.m` | `geometry/sphere_sampling.py::spiral_sample_sphere` | ✅ |
| `IcosahedronMesh.m` | `geometry/sphere_sampling.py::icosahedron_vertices` | ✅ |
| `SubdivideSphericalMesh.m` | `geometry/sphere_sampling.py::subdivide_sphere_mesh` | ✅ |
| `intriangulation.m` | `geometry/triangulation.py::in_triangulation` | ✅ |

### Algorithms

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `vessel_dijkstra.m` | `algorithms/dijkstra.py::vessel_dijkstra` | ✅ |
| `dendrite_dijkstra2.m` | `algorithms/dijkstra.py::dendrite_dijkstra` | ✅ |
| `dendrite_dijkstra_cpp.cpp` | `algorithms/dijkstra.py` (numba) | ✅ |
| GP sampling (in `generateNeuralBody.m`) | `algorithms/gaussian_process.py` | ✅ |
| `teardrop_poj.m` | `algorithms/gaussian_process.py::teardrop_projection` | ✅ |

---

## Phase 3: Volume Generation (🔄 Partial)

### Neural Volume - Main Functions

| MATLAB File | Python File | Status | Notes |
|-------------|-------------|--------|-------|
| `simulate_neural_volume.m` | - | ❌ | Main orchestrator |
| `simulate_neural_volume_with_checkpoints.m` | - | ❌ | With checkpoint support |

### Blood Vessels

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `simulatebloodvessels.m` | `volume/vasculature.py::simulate_blood_vessels` | ✅ |
| `growMajorVessels.m` | `volume/vasculature.py::grow_major_vessels` | ✅ |
| `growCapillaries.m` | `volume/vasculature.py::grow_capillaries` | ✅ |
| `branchGrowNodes.m` | `volume/vasculature.py::branch_grow_nodes` | ✅ |
| `gennode.m` | `volume/vasculature.py::VesselNode` | ✅ |
| `genconn.m` | `volume/vasculature.py::VesselConnection` | ✅ |
| `nodesToConn.m` | `volume/vasculature.py::nodes_to_connections` | ✅ |
| `connToVol.m` | `volume/vasculature.py::connections_to_volume` | ✅ |
| `pseudoRandSample2D.m` | `volume/vasculature.py::pseudo_rand_sample_2d` | ✅ |
| `pseudoRandSample3D.m` | `volume/vasculature.py::pseudo_rand_sample_3d` | ✅ |
| `delnode.m` | - | ❌ |

### Neurons

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `generateNeuralBody.m` | `volume/neurons.py::generate_neural_body` | ✅ |
| `sampleDenseNeurons.m` | - | ❌ |
| `generateNeuralVolume.m` | - | ❌ |
| `smoothCellBody.m` | - | ❌ |

### Dendrites

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `growNeuronDendrites.m` | - | ❌ |
| `growApicalDendrites.m` | - | ❌ |
| `getDendritePath2.m` | - | ❌ |
| `dilateDendritePathAll.m` | - | ❌ |
| `dendrite_randomwalk2.m` | - | ❌ |
| `dendrite_randomwalk_cpp.cpp` | - | ❌ |

### Axons & Background

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `generate_axons.m` | - | ❌ |
| `generate_bgdendrites.m` | - | ❌ |
| `sort_axons.m` | - | ❌ |

### Fluorescence

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `setCellFluoresence.m` | - | ❌ |

---

## Phase 4: Time Trace Generation (❌ Not Started)

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `generateTimeTraces.m` | - | ❌ |
| `genCorrelatedSpikeTrains2.m` | - | ❌ |
| `gen_burst_spike_times.m` | - | ❌ |
| `calcium_dynamics.m` | - | ❌ |
| `make_calcium_impulse.m` | - | ❌ |
| `markpointproc.m` | - | ❌ |
| `binSpikeTrains.m` | - | ❌ |
| `expression_variation.m` | - | ❌ |
| `generateNextTimePoint.m` | - | ❌ |
| `genNextCalciumDynamics.m` | - | ❌ |
| `genNextSpikeTimepoint.m` | - | ❌ |

---

## Phase 5: Optical Propagation (❌ Not Started)

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `simulate_optical_propagation.m` | - | ❌ |
| `simulate_optical_propagation2.m` | - | ❌ |
| `genCorticalLightPath.m` | - | ❌ |
| `genCorticalLightPathLite.m` | - | ❌ |
| `fresnel_propagation_multi.m` | - | ❌ |
| `generateBA.m` | - | ❌ |
| `generateBesselBA.m` | - | ❌ |
| `generateCylindricalBA.m` | - | ❌ |
| `generateGaussianProfile.m` | - | ❌ |
| `generateBesselProfile.m` | - | ❌ |
| `generateVtwinsBA.m` | - | ❌ |
| `generateZernike.m` | - | ❌ |
| `applyZernike.m` | - | ❌ |
| `applyTemporalFocusing.m` | - | ❌ |
| `gaussian_psf.m` | - | ❌ |
| `gaussian_psf_na.m` | - | ❌ |
| `gaussianBeamSize.m` | - | ❌ |
| `zernike.m` | - | ❌ |
| `setOpticalParams.m` | - | ❌ |
| `getDefaultPSFParams.m` | - | ❌ |
| `widthestimate.m` | - | ❌ |
| `widthestimate3D.m` | - | ❌ |
| `tpmSignalscale.m` | - | ❌ |
| `groupzproject.m` | - | ❌ |

---

## Phase 6: Scanning Simulation (❌ Not Started)

| MATLAB File | Python File | Status |
|-------------|-------------|--------|
| `scan_volume.m` | - | ❌ |
| `scan_volume_frame.m` | - | ❌ |
| `setup_scan_volume_frame.m` | - | ❌ |
| `single_scan.m` | - | ❌ |
| `scan_ideal.m` | - | ❌ |
| `calculateIdealComps.m` | - | ❌ |
| `PoissonGaussNoiseModel.m` | - | ❌ |
| `applyNoiseModel.m` | - | ❌ |
| `pixel_bleed.m` | - | ❌ |
| `blurredBackComp2.m` | - | ❌ |
| `imgSubRowShift.m` | - | ❌ |
| `psf_fft.m` | - | ❌ |

---

## MEX Files to Port

These C++ MEX files need Python equivalents (using NumPy/Numba):

| MEX File | Python Equivalent | Status |
|----------|-------------------|--------|
| `dendrite_dijkstra_cpp.cpp` | `algorithms/dijkstra.py` (numba) | ✅ |
| `dendrite_randomwalk_cpp.cpp` | - | ❌ |
| `array_SubMod.cpp` | NumPy operations | ❌ |
| `array_SubSub.cpp` | NumPy operations | ❌ |
| `array_SubModTest.cpp` | NumPy operations | ❌ |
| `array_SubSubTest.cpp` | NumPy operations | ❌ |
| `locate_neighbors.cpp` | - | ❌ |

---

## Visualization (✅ Complete)

| Feature | Python File | Status |
|---------|-------------|--------|
| 3D mesh plotting | `visualization/viewer3d.py` | ✅ |
| Sphere points plotting | `visualization/viewer3d.py` | ✅ |
| Neuron shape plotting | `visualization/viewer3d.py` | ✅ |
| Volume slice plotting | `visualization/viewer3d.py` | ✅ |

---

## External Dependencies

### MATLAB External Packages
- `inpaint_nans.m` - NaN interpolation
- `intriangulation.m` - Point-in-mesh (replaced by trimesh)
- `S2_Sampling_Suite/` - Sphere sampling (reimplemented)
- `MatlabProgressBar` - Progress display

### Python Dependencies
- `numpy` - Core numerical operations
- `scipy` - Scientific computing (interpolation, linear algebra)
- `trimesh` - Mesh operations (replaces intriangulation)
- `matplotlib` - Visualization
- `numba` (optional) - JIT compilation for performance

---

## Priority Order for Next Implementation

1. **High Priority** (Core Volume Generation)
   - `sampleDenseNeurons.m` → Place neurons in volume
   - `generateNeuralVolume.m` → Create neural volume grid
   - `growNeuronDendrites.m` → Dendrite growth
   - `growApicalDendrites.m` → Apical dendrite growth

2. **Medium Priority** (Activity & Optics)
   - `generateTimeTraces.m` → Time trace generation
   - `calcium_dynamics.m` → Calcium kinetics
   - `simulate_optical_propagation.m` → PSF generation

3. **Lower Priority** (Scanning & Analysis)
   - `scan_volume.m` → Scanning simulation
   - `PoissonGaussNoiseModel.m` → Noise model
   - Analysis utilities

---

## Validation Strategy

1. **Unit Tests**: Compare Python output to MATLAB output for identical inputs
2. **Integration Tests**: Run full simulation pipeline and compare statistics
3. **Visual Validation**: Compare generated volumes and images visually
4. **Performance Benchmarks**: Ensure Python version is reasonably performant

Validation scripts should be placed in `calcia/validation/`.

---

## Notes

- All Python implementations should maintain MATLAB-compatible behavior where possible
- Use `numpy.random` with seeds for reproducibility matching MATLAB's `rng(seed)`
- Document any intentional deviations from MATLAB behavior
- Prefer clarity over performance initially; optimize later

---

## Last Updated

2026-02-03

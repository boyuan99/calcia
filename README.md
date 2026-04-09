# Calcia

Python implementation of neural volume simulation.

## Installation

```bash
pip install -e .
```

### Dependencies

- numpy
- scipy
- trimesh
- matplotlib
- numba (optional, for Dijkstra acceleration)

## Project Structure

```
calcia/
├── calcia/
│   ├── config/          # Parameter dataclasses
│   ├── volume/          # Core volume generation (neurons, vasculature)
│   ├── geometry/        # Geometry utilities (sphere sampling, triangulation)
│   ├── algorithms/      # Core algorithms (Dijkstra, Gaussian Process)
│   ├── visualization/   # 3D visualization utilities
│   └── utils/           # Utility functions (indexing, compatibility)
├── examples/            # Visualization and usage examples
├── tests/               # Unit tests
└── validation/          # Comparison tools
```

## Usage

### Neuron Shape Generation

```python
import numpy as np
from calcia.config.params import NeuronParams
from calcia.volume.neurons import generate_neural_body, compute_neuron_statistics

np.random.seed(42)

# Configure neuron parameters
params = NeuronParams(
    n_samps=200,          # Number of sampling points
    avg_rad=5.9,          # Average radius (um)
    neur_type='pyramidal' # Neuron type: pyramidal/spherical/stellate/fusiform
)

# Generate neuron shape
Vcell, Vnuc, faces, angles = generate_neural_body(params)

# Compute statistics
stats = compute_neuron_statistics(Vcell, Vnuc)
print(f"Average radius: {stats['avg_radius']:.2f} um")
print(f"Volume: {stats['volume']:.1f} um³")
```

### Blood Vessel Simulation

```python
import numpy as np
from calcia.config.params import VolumeParams, VascParams
from calcia.volume.vasculature import simulate_blood_vessels

np.random.seed(42)

# Configure parameters
vol_params = VolumeParams(vol_sz=(100, 100, 200))
vasc_params = VascParams(
    depth_surf=15.0,
    depth_vasc=180.0,
    vesFreq=(125.0, 200.0, 50.0),
)

# Generate vessel network
network = simulate_blood_vessels(vol_params, vasc_params, verbose=1)

print(f"Total nodes: {len(network.nodes)}")
print(f"Total connections: {len(network.connections)}")
```

### Sphere Sampling

```python
from calcia.geometry.sphere_sampling import spiral_sample_sphere

# Generate uniform sampling on unit sphere
V, Tri = spiral_sample_sphere(n_samples=200)
print(f"Vertices: {V.shape}, Faces: {Tri.shape}")
```

## Examples

Run example scripts in the `examples/` directory:

```bash
python examples/viz_neuron_shape.py      # Neuron generation demo
python examples/viz_vasculature.py       # Blood vessel demo
python examples/viz_sphere_sampling.py   # Sphere sampling demo
python examples/viz_dijkstra.py          # Dijkstra algorithm demo
```

## Development Status

This project is in active development (Phase 2: Core Simulation).

### Completed
- [x] Project structure
- [x] Parameter dataclasses (`config/params.py`)
- [x] Index conversion utilities (`utils/indexing.py`)
- [x] Sphere sampling (`geometry/sphere_sampling.py`)
- [x] Triangulation utilities (`geometry/triangulation.py`)
- [x] Gaussian process sampling (`algorithms/gaussian_process.py`)
- [x] Dijkstra path planning (`algorithms/dijkstra.py`)
- [x] Neuron shape generation (`volume/neurons.py`)
- [x] Blood vessel simulation (`volume/vasculature.py`)
- [x] 3D visualization (`visualization/viewer3d.py`)

### Planned
- [ ] Dendrite growth
- [ ] Axon generation
- [ ] Fluorescence distribution
- [ ] Background generation
- [ ] Full volume composition

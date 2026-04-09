"""
Export neuron meshes to JSON format for Three.js visualization.
"""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from calcia.config.params import NeuronParams
from calcia.volume.neurons import generate_neural_body, generate_multiple_neurons


def export_single_neuron(output_path: str = "neuron_data.json"):
    """Export a single neuron mesh to JSON."""
    print("Generating neuron...")

    np.random.seed(42)
    params = NeuronParams(n_samps=200, avg_rad=5.9, neur_type='pyramidal')

    Vcell, Vnuc, faces, angles = generate_neural_body(params)

    data = {
        "neurons": [{
            "soma": {
                "vertices": Vcell.tolist(),
                "faces": faces.tolist(),
            },
            "nucleus": {
                "vertices": Vnuc.tolist(),
                "faces": faces.tolist(),
            },
            "position": [0, 0, 0],
            "rotation": angles.tolist(),
        }]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported to {output_path}")
    return data


def export_multiple_neurons(n: int = 5, output_path: str = "neurons_data.json"):
    """Export multiple neurons to JSON."""
    print(f"Generating {n} neurons...")

    np.random.seed(42)
    params = NeuronParams(n_samps=150, avg_rad=5.9, neur_type='pyramidal')

    # Create grid positions
    spacing = 25.0
    positions = []
    cols = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        x = (i % cols) * spacing
        y = (i // cols) * spacing
        positions.append([x, y, 0])
    positions = np.array(positions)

    neurons, angles_list, _ = generate_multiple_neurons(n, params, positions)

    data = {"neurons": []}

    for i, ((Vcell, Vnuc, faces), angles) in enumerate(zip(neurons, angles_list)):
        data["neurons"].append({
            "soma": {
                "vertices": Vcell.tolist(),
                "faces": faces.tolist(),
            },
            "nucleus": {
                "vertices": Vnuc.tolist(),
                "faces": faces.tolist(),
            },
            "position": positions[i].tolist(),
            "rotation": angles.tolist(),
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported {n} neurons to {output_path}")
    return data


def export_neuron_types(output_path: str = "neuron_types_data.json"):
    """Export different neuron types for comparison."""
    print("Generating different neuron types...")

    # Supported neuron types:
    # - pyramidal: teardrop shape for cortical pyramidal neurons
    # - spherical: spherical shape for granule cells, MSN neurons
    # - stellate: highly spherical for cortical layer IV stellate cells
    # - fusiform: elongated spindle shape for layer VI, von Economo neurons
    types = ['pyramidal', 'spherical', 'stellate', 'fusiform']
    type_names = ['Pyramidal', 'Spherical', 'Stellate', 'Fusiform']

    data = {"neurons": []}

    for i, (ntype, name) in enumerate(zip(types, type_names)):
        np.random.seed(42)  # Same seed for fair comparison
        params = NeuronParams(n_samps=200, avg_rad=5.9, neur_type=ntype, max_ang=0)

        Vcell, Vnuc, faces, angles = generate_neural_body(params)

        # Position them side by side
        offset = np.array([i * 20.0, 0, 0])

        data["neurons"].append({
            "name": name,
            "type": ntype,
            "soma": {
                "vertices": (Vcell + offset).tolist(),
                "faces": faces.tolist(),
            },
            "nucleus": {
                "vertices": (Vnuc + offset).tolist(),
                "faces": faces.tolist(),
            },
            "position": offset.tolist(),
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported neuron types to {output_path}")
    return data


if __name__ == "__main__":
    output_dir = Path(__file__).parent

    export_single_neuron(str(output_dir / "neuron_data.json"))
    export_multiple_neurons(9, str(output_dir / "neurons_data.json"))
    export_neuron_types(str(output_dir / "neuron_types_data.json"))

    print("\nDone! Open viewer.html in a browser to view.")

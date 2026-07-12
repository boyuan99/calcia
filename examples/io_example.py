"""
Example: Export and import volume simulation data.

Demonstrates the unified JSON format for saving and loading
complete volume simulations.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calcia.config.params import VolumeParams, NeuronParams, VascParams
from calcia.volume.neurons import generate_multiple_neurons
from calcia.volume.vasculature import simulate_blood_vessels
from calcia.io import export_volume, import_volume, export_neurons, export_vessels


def example_full_export():
    """Export a complete simulation with neurons and vessels."""
    print("=" * 60)
    print("Example: Full Volume Export")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Configure parameters
    vol_params = VolumeParams(vol_sz=(100, 100, 200))
    neur_params = NeuronParams(n_samps=150, avg_rad=5.9, neur_type='pyramidal')
    vasc_params = VascParams(depth_surf=15.0, depth_vasc=180.0)

    # Generate neurons
    print("\nGenerating neurons...")
    n_neurons = 5
    positions = np.array([
        [25, 25, 50],
        [75, 25, 50],
        [50, 50, 100],
        [25, 75, 150],
        [75, 75, 150],
    ], dtype=float)
    neurons, angles, _ = generate_multiple_neurons(n_neurons, neur_params, positions)
    print(f"  Generated {len(neurons)} neurons")

    # Generate vessel network
    print("\nGenerating vessel network...")
    network = simulate_blood_vessels(vol_params, vasc_params, verbose=0)
    print(f"  Generated {len(network.nodes)} nodes, {len(network.connections)} connections")

    # Export to unified format
    output_path = Path(__file__).parent / "output" / "full_simulation.json"
    output_path.parent.mkdir(exist_ok=True)

    print(f"\nExporting to {output_path}...")
    data = export_volume(
        str(output_path),
        neurons=neurons,
        neuron_positions=positions,
        neuron_angles=angles,
        vessel_network=network,
        vol_params=vol_params,
        neur_params=neur_params,
        vasc_params=vasc_params,
        random_seed=42,
        description="Example full volume simulation",
    )

    print(f"  Format version: {data['format_version']}")
    print(f"  Components: {data['metadata']['components_included']}")

    return str(output_path)


def example_import(input_path: str):
    """Import and verify the exported data."""
    print("\n" + "=" * 60)
    print("Example: Import and Verify")
    print("=" * 60)

    print(f"\nImporting from {input_path}...")
    data = import_volume(input_path)

    # Check metadata
    print(f"\nMetadata:")
    print(f"  Created: {data['metadata'].get('created_at', 'N/A')}")
    print(f"  Version: {data['metadata'].get('format_version', 'N/A')}")
    print(f"  Components: {data['metadata'].get('components_included', [])}")

    # Check parameters
    if "volume" in data["parameters"]:
        vol_params = data["parameters"]["volume"]
        if hasattr(vol_params, 'vol_sz'):
            print(f"\nVolume params: vol_sz={vol_params.vol_sz}")
        else:
            print(f"\nVolume params: {vol_params}")

    # Check neurons
    if "neurons" in data:
        neurons = data["neurons"]
        positions = data["neuron_positions"]
        print(f"\nNeurons: {len(neurons)} loaded")
        for i, (Vcell, Vnuc, faces, angles) in enumerate(neurons):
            print(f"  Neuron {i}: soma {Vcell.shape}, nucleus {Vnuc.shape}, "
                  f"faces {faces.shape}, pos {positions[i]}")

    # Check vessels
    if "vessel_network" in data:
        network = data["vessel_network"]
        print(f"\nVessels: {len(network.nodes)} nodes, "
              f"{len(network.connections)} connections")

    return data


def example_neurons_only():
    """Export neurons only."""
    print("\n" + "=" * 60)
    print("Example: Neurons Only Export")
    print("=" * 60)

    np.random.seed(123)

    neur_params = NeuronParams(n_samps=100, neur_type='spherical')
    positions = np.array([[0, 0, 0], [20, 0, 0], [40, 0, 0]], dtype=float)

    neurons, angles, _ = generate_multiple_neurons(3, neur_params, positions)

    output_path = Path(__file__).parent / "output" / "neurons_only.json"
    output_path.parent.mkdir(exist_ok=True)

    data = export_neurons(
        str(output_path),
        neurons=neurons,
        positions=positions,
        angles=angles,
        neur_params=neur_params,
        random_seed=123,
    )

    print(f"Exported {len(neurons)} neurons to {output_path}")
    return str(output_path)


def example_vessels_only():
    """Export vessels only."""
    print("\n" + "=" * 60)
    print("Example: Vessels Only Export")
    print("=" * 60)

    np.random.seed(456)

    vol_params = VolumeParams(vol_sz=(50, 50, 100))
    vasc_params = VascParams(depth_surf=10.0, depth_vasc=90.0)

    network = simulate_blood_vessels(vol_params, vasc_params, verbose=0)

    output_path = Path(__file__).parent / "output" / "vessels_only.json"
    output_path.parent.mkdir(exist_ok=True)

    data = export_vessels(
        str(output_path),
        vessel_network=network,
        vol_params=vol_params,
        vasc_params=vasc_params,
        random_seed=456,
    )

    print(f"Exported {len(network.nodes)} nodes to {output_path}")
    return str(output_path)


def main():
    """Run all examples."""
    print("Calcia I/O Examples")
    print("=" * 60)

    # Full export
    full_path = example_full_export()

    # Import and verify
    example_import(full_path)

    # Neurons only
    neurons_path = example_neurons_only()

    # Vessels only
    vessels_path = example_vessels_only()

    # Verify neurons import
    print("\n" + "=" * 60)
    print("Verifying neurons-only import...")
    from calcia.io import import_neurons
    neurons, positions = import_neurons(neurons_path)
    print(f"  Loaded {len(neurons)} neurons")

    # Verify vessels import
    print("\nVerifying vessels-only import...")
    from calcia.io import import_vessels
    network = import_vessels(vessels_path)
    print(f"  Loaded {len(network.nodes)} nodes")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import _instrument; _instrument.start()  # run log + pyinstrument (mandated)
    main()

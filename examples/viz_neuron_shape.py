"""
Visualization example for neuron shape generation.

Demonstrates the GP-based neuron shape generation and validates
the statistical properties of generated neurons.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calcia.config.params import NeuronParams
from calcia.volume.neurons import (
    generate_neural_body,
    generate_multiple_neurons,
    compute_neuron_statistics,
)


def plot_single_neuron():
    """Generate and plot a single neuron with soma and nucleus."""
    print("Generating single neuron...")

    params = NeuronParams(
        n_samps=200,
        avg_rad=5.9,
        neur_type='pyr',
    )

    Vcell, Vnuc, faces, angles = generate_neural_body(params)

    # Compute statistics
    stats = compute_neuron_statistics(Vcell, Vnuc)
    print(f"  Average radius: {stats['avg_radius']:.2f} um")
    print(f"  Radius range: [{stats['min_radius']:.2f}, {stats['max_radius']:.2f}] um")
    if stats['volume'] is not None:
        print(f"  Volume: {stats['volume']:.1f} um^3")
        print(f"  Sphericity: {stats['sphericity']:.3f}")
    print(f"  Rotation angles: [{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}] deg")

    # Create figure
    fig = plt.figure(figsize=(12, 5))

    # Plot 1: Soma only
    ax1 = fig.add_subplot(131, projection='3d')
    cell_polys = Vcell[faces]
    cell_coll = Poly3DCollection(
        cell_polys,
        facecolors='lightblue',
        edgecolors='gray',
        alpha=0.6,
        linewidths=0.2,
    )
    ax1.add_collection3d(cell_coll)
    _set_equal_aspect(ax1, Vcell)
    ax1.set_title('Soma (Cell Body)')
    ax1.set_xlabel('X (um)')
    ax1.set_ylabel('Y (um)')
    ax1.set_zlabel('Z (um)')

    # Plot 2: Nucleus only
    ax2 = fig.add_subplot(132, projection='3d')
    nuc_polys = Vnuc[faces]
    nuc_coll = Poly3DCollection(
        nuc_polys,
        facecolors='darkblue',
        edgecolors='black',
        alpha=0.8,
        linewidths=0.2,
    )
    ax2.add_collection3d(nuc_coll)
    _set_equal_aspect(ax2, Vnuc)
    ax2.set_title('Nucleus')
    ax2.set_xlabel('X (um)')
    ax2.set_ylabel('Y (um)')
    ax2.set_zlabel('Z (um)')

    # Plot 3: Both together
    ax3 = fig.add_subplot(133, projection='3d')
    cell_coll2 = Poly3DCollection(
        cell_polys,
        facecolors='lightblue',
        edgecolors='gray',
        alpha=0.3,
        linewidths=0.1,
    )
    nuc_coll2 = Poly3DCollection(
        nuc_polys,
        facecolors='darkblue',
        edgecolors='black',
        alpha=0.9,
        linewidths=0.2,
    )
    ax3.add_collection3d(cell_coll2)
    ax3.add_collection3d(nuc_coll2)
    _set_equal_aspect(ax3, Vcell)
    ax3.set_title('Complete Neuron')
    ax3.set_xlabel('X (um)')
    ax3.set_ylabel('Y (um)')
    ax3.set_zlabel('Z (um)')

    plt.tight_layout()
    plt.savefig('neuron_single.png', dpi=150, bbox_inches='tight')
    print("Saved: neuron_single.png")
    plt.close()


def plot_multiple_neurons():
    """Generate and plot multiple neurons with different shapes."""
    print("\nGenerating multiple neurons...")

    params = NeuronParams(
        n_samps=150,
        avg_rad=5.9,
        neur_type='pyr',
    )

    # Generate 4 neurons at different positions
    positions = np.array([
        [0, 0, 0],
        [20, 0, 0],
        [0, 20, 0],
        [20, 20, 0],
    ], dtype=float)

    neurons, angles, _ = generate_multiple_neurons(4, params, positions)

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    all_vertices = []
    for i, (Vcell, Vnuc, faces) in enumerate(neurons):
        # Plot soma
        cell_polys = Vcell[faces]
        cell_coll = Poly3DCollection(
            cell_polys,
            facecolors=colors[i],
            edgecolors='gray',
            alpha=0.5,
            linewidths=0.1,
        )
        ax.add_collection3d(cell_coll)

        # Plot nucleus (darker version of same color)
        nuc_polys = Vnuc[faces]
        nuc_coll = Poly3DCollection(
            nuc_polys,
            facecolors='navy',
            edgecolors='black',
            alpha=0.8,
            linewidths=0.1,
        )
        ax.add_collection3d(nuc_coll)

        all_vertices.append(Vcell)

        # Print statistics
        stats = compute_neuron_statistics(Vcell, Vnuc)
        print(f"  Neuron {i+1}: radius={stats['avg_radius']:.2f} um, "
              f"volume={stats['volume']:.0f} um^3" if stats['volume'] else "")

    all_vertices = np.vstack(all_vertices)
    _set_equal_aspect(ax, all_vertices)
    ax.set_title('Multiple Pyramidal Neurons')
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (um)')

    plt.savefig('neurons_multiple.png', dpi=150, bbox_inches='tight')
    print("Saved: neurons_multiple.png")
    plt.close()


def plot_neuron_types():
    """Compare different neuron types."""
    print("\nComparing neuron types...")

    fig = plt.figure(figsize=(12, 5))

    neuron_types = ['pyr', 'other', 'peanut']
    titles = ['Pyramidal (teardrop)', 'Spherical', 'Peanut']

    for i, (ntype, title) in enumerate(zip(neuron_types, titles)):
        params = NeuronParams(
            n_samps=200,
            avg_rad=5.9,
            neur_type=ntype,
            max_ang=0,  # No rotation for comparison
        )

        Vcell, Vnuc, faces, _ = generate_neural_body(params)

        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        cell_polys = Vcell[faces]
        cell_coll = Poly3DCollection(
            cell_polys,
            facecolors='lightblue',
            edgecolors='gray',
            alpha=0.5,
            linewidths=0.2,
        )
        nuc_polys = Vnuc[faces]
        nuc_coll = Poly3DCollection(
            nuc_polys,
            facecolors='darkblue',
            edgecolors='black',
            alpha=0.8,
            linewidths=0.2,
        )

        ax.add_collection3d(cell_coll)
        ax.add_collection3d(nuc_coll)
        _set_equal_aspect(ax, Vcell)
        ax.set_title(title)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_zlabel('Z (um)')

        stats = compute_neuron_statistics(Vcell)
        print(f"  {title}: sphericity={stats['sphericity']:.3f}"
              if stats['sphericity'] else f"  {title}")

    plt.tight_layout()
    plt.savefig('neuron_types.png', dpi=150, bbox_inches='tight')
    print("Saved: neuron_types.png")
    plt.close()


def validate_statistics():
    """Validate statistical properties of neuron generation."""
    print("\nValidating neuron generation statistics...")

    params = NeuronParams(
        n_samps=200,
        avg_rad=5.9,
        neur_type='pyr',
    )

    n_samples = 50
    radii = []
    volumes = []
    sphericities = []

    print(f"  Generating {n_samples} neurons...")
    for _ in range(n_samples):
        Vcell, Vnuc, _, _ = generate_neural_body(params)
        stats = compute_neuron_statistics(Vcell, Vnuc)
        radii.append(stats['avg_radius'])
        if stats['volume'] is not None:
            volumes.append(stats['volume'])
        if stats['sphericity'] is not None:
            sphericities.append(stats['sphericity'])

    radii = np.array(radii)
    volumes = np.array(volumes)
    sphericities = np.array(sphericities)

    print("\n  Statistical summary:")
    print(f"    Average radius: {np.mean(radii):.2f} +/- {np.std(radii):.2f} um")
    print(f"    Target avg_rad: {params.avg_rad} um")
    print(f"    Volume: {np.mean(volumes):.0f} +/- {np.std(volumes):.0f} um^3")
    print(f"    Sphericity: {np.mean(sphericities):.3f} +/- {np.std(sphericities):.3f}")

    # Create histogram
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].hist(radii, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(params.avg_rad, color='red', linestyle='--', label=f'Target: {params.avg_rad}')
    axes[0].axvline(np.mean(radii), color='green', linestyle='-', label=f'Mean: {np.mean(radii):.2f}')
    axes[0].set_xlabel('Average Radius (um)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Radius Distribution')
    axes[0].legend()

    axes[1].hist(volumes, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Volume (um^3)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Volume Distribution')

    axes[2].hist(sphericities, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Sphericity')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Sphericity Distribution')

    plt.tight_layout()
    plt.savefig('neuron_statistics.png', dpi=150, bbox_inches='tight')
    print("\nSaved: neuron_statistics.png")
    plt.close()

    # Validation checks
    print("\n  Validation checks:")
    radius_error = abs(np.mean(radii) - params.avg_rad) / params.avg_rad
    print(f"    Radius error: {radius_error*100:.1f}% (should be < 10%)")

    if radius_error < 0.10:
        print("    [PASS] Average radius close to target")
    else:
        print("    [WARN] Average radius differs from target")

    if np.std(radii) > 0:
        print("    [PASS] Shape variation observed")
    else:
        print("    [FAIL] No shape variation")


def _set_equal_aspect(ax, points):
    """Set equal aspect ratio for 3D plot."""
    max_range = np.max([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]) / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
    mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
    mid_z = (points[:, 2].max() + points[:, 2].min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility

    plot_single_neuron()
    plot_multiple_neurons()
    plot_neuron_types()
    validate_statistics()

    print("\nDone! Check the generated PNG files.")

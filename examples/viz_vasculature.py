"""
Visualize blood vessel network generation.

This script demonstrates the vasculature simulation pipeline and
provides 3D visualization of the resulting vessel network.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calcia.config.params import VolumeParams, VascParams
from calcia.volume.vasculature import (
    simulate_blood_vessels,
    VesselNetwork,
)


def plot_vessel_network_3d(
    network: VesselNetwork,
    title: str = "Blood Vessel Network",
    show_nodes: bool = True,
    show_connections: bool = True,
    figsize: tuple = (12, 10),
):
    """
    Create 3D visualization of vessel network.

    Args:
        network: VesselNetwork to visualize.
        title: Plot title.
        show_nodes: Whether to show nodes as points.
        show_connections: Whether to show connection lines.
        figsize: Figure size.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Color map for different node types
    type_colors = {
        0: 'gray',      # Internal nodes
        1: 'green',     # Source nodes
        2: 'blue',      # Branch points
        3: 'red',       # Diving vessels
        4: 'cyan',      # Capillaries
    }
    type_names = {
        0: 'Internal',
        1: 'Source',
        2: 'Branch',
        3: 'Diving',
        4: 'Capillary',
    }
    type_sizes = {
        0: 20,
        1: 100,
        2: 50,
        3: 40,
        4: 10,
    }

    # Plot nodes by type
    if show_nodes and network.nodes:
        for node_type in sorted(set(n.type for n in network.nodes)):
            nodes_of_type = [n for n in network.nodes if n.type == node_type]
            positions = np.array([n.pos for n in nodes_of_type])

            if len(positions) > 0:
                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    positions[:, 2],
                    c=type_colors.get(node_type, 'gray'),
                    s=type_sizes.get(node_type, 20),
                    label=f'{type_names.get(node_type, "Unknown")} ({len(nodes_of_type)})',
                    alpha=0.7,
                )

    # Plot connections
    if show_connections and network.connections:
        for conn in network.connections:
            start_node = network.nodes[conn.start]
            end_node = network.nodes[conn.ends]

            # Color based on node types
            if start_node.type == 3 or end_node.type == 3:
                color = 'red'
                alpha = 0.8
                linewidth = 1.5
            elif start_node.type == 4 or end_node.type == 4:
                color = 'cyan'
                alpha = 0.5
                linewidth = 0.5
            elif start_node.type == 1 or end_node.type == 1:
                color = 'green'
                alpha = 0.9
                linewidth = 2
            else:
                color = 'gray'
                alpha = 0.6
                linewidth = 1

            ax.plot(
                [start_node.pos[0], end_node.pos[0]],
                [start_node.pos[1], end_node.pos[1]],
                [start_node.pos[2], end_node.pos[2]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (depth, um)')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)

    # Invert Z axis to show depth going down
    ax.invert_zaxis()

    return fig, ax


def plot_vessel_volume_slices(
    network: VesselNetwork,
    n_slices: int = 6,
    figsize: tuple = (15, 10),
):
    """
    Plot slices through the vessel volume.

    Args:
        network: VesselNetwork with rendered volume.
        n_slices: Number of Z-slices to show.
        figsize: Figure size.
    """
    if network.vessel_volume is None:
        print("No vessel volume to display")
        return None, None

    vol = network.vessel_volume
    z_indices = np.linspace(0, vol.shape[2] - 1, n_slices, dtype=int)

    rows = 2
    cols = (n_slices + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, z_idx in enumerate(z_indices):
        axes[i].imshow(vol[:, :, z_idx].T, cmap='Reds', origin='lower')
        axes[i].set_title(f'Z = {z_idx}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')

    # Hide unused axes
    for i in range(n_slices, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Vessel Volume Slices (Z-depth)')
    plt.tight_layout()

    return fig, axes


def plot_vessel_statistics(network: VesselNetwork, figsize: tuple = (12, 8)):
    """
    Plot statistics about the vessel network.

    Args:
        network: VesselNetwork to analyze.
        figsize: Figure size.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Node type distribution
    ax = axes[0, 0]
    type_counts = {}
    type_names = {
        0: 'Internal',
        1: 'Source',
        2: 'Branch',
        3: 'Diving',
        4: 'Capillary',
    }
    for n in network.nodes:
        name = type_names.get(n.type, f'Type {n.type}')
        type_counts[name] = type_counts.get(name, 0) + 1

    if type_counts:
        ax.bar(type_counts.keys(), type_counts.values(), color=['gray', 'green', 'blue', 'red', 'cyan'][:len(type_counts)])
        ax.set_title('Node Type Distribution')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

    # 2. Node depth distribution
    ax = axes[0, 1]
    depths = [n.pos[2] for n in network.nodes]
    if depths:
        ax.hist(depths, bins=20, color='steelblue', edgecolor='black')
        ax.set_title('Node Depth Distribution')
        ax.set_xlabel('Z (depth, um)')
        ax.set_ylabel('Count')

    # 3. Connection length distribution
    ax = axes[1, 0]
    lengths = [conn.weight for conn in network.connections]
    if lengths:
        ax.hist(lengths, bins=20, color='coral', edgecolor='black')
        ax.set_title('Connection Length Distribution')
        ax.set_xlabel('Length (um)')
        ax.set_ylabel('Count')

    # 4. Volume fill by depth
    ax = axes[1, 1]
    if network.vessel_volume is not None:
        vol = network.vessel_volume
        fill_by_z = np.sum(vol, axis=(0, 1)) / (vol.shape[0] * vol.shape[1])
        ax.plot(fill_by_z, color='purple')
        ax.set_title('Vessel Fill Fraction by Depth')
        ax.set_xlabel('Z (depth, voxels)')
        ax.set_ylabel('Fill Fraction')
    else:
        ax.text(0.5, 0.5, 'No volume data', ha='center', va='center')

    plt.tight_layout()
    return fig, axes


def main():
    """Main function to demonstrate vasculature visualization."""
    print("=" * 60)
    print("Blood Vessel Network Visualization")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create parameters
    vol_params = VolumeParams(vol_sz=(100, 100, 200))
    vasc_params = VascParams(
        depth_surf=15.0,
        depth_vasc=180.0,
        vesFreq=(125.0, 200.0, 50.0),
    )

    print("\nSimulating blood vessel network...")
    print(f"  Volume size: {vol_params.vol_sz} um")
    print(f"  Surface depth: {vasc_params.depth_surf} um")
    print(f"  Maximum depth: {vasc_params.depth_vasc} um")

    # Generate vessel network
    network = simulate_blood_vessels(vol_params, vasc_params, verbose=1)

    # Print summary
    print("\n" + "=" * 60)
    print("Network Summary:")
    print("=" * 60)

    type_counts = {}
    type_names = {0: 'Internal', 1: 'Source', 2: 'Branch', 3: 'Diving', 4: 'Capillary'}
    for n in network.nodes:
        name = type_names.get(n.type, f'Type {n.type}')
        type_counts[name] = type_counts.get(name, 0) + 1

    for name, count in sorted(type_counts.items()):
        print(f"  {name} nodes: {count}")

    print(f"  Total connections: {len(network.connections)}")

    if network.vessel_volume is not None:
        fill_frac = np.sum(network.vessel_volume) / np.prod(network.vessel_volume.shape)
        print(f"  Volume fill fraction: {fill_frac:.4f} ({fill_frac*100:.2f}%)")

    # Create visualizations
    print("\nCreating visualizations...")

    # 3D network plot
    fig1, ax1 = plot_vessel_network_3d(network, title="Blood Vessel Network (3D)")

    # Volume slices
    fig2, axes2 = plot_vessel_volume_slices(network, n_slices=6)

    # Statistics
    fig3, axes3 = plot_vessel_statistics(network)

    plt.show()


if __name__ == "__main__":
    main()

"""
Visualization example: Sphere sampling.

This script demonstrates the sphere sampling algorithm and provides
visual verification that points are uniformly distributed on the sphere.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calcia.geometry.sphere_sampling import (
    spiral_sample_sphere,
    geodesic_distance_matrix,
)
from calcia.visualization.viewer3d import plot_sphere_points, plot_mesh_surface


def visualize_sphere_sampling(n_samples: int = 200):
    """
    Visualize sphere sampling with multiple views.

    Args:
        n_samples: Number of points to sample.
    """
    print(f"Generating {n_samples} points on unit sphere...")
    V, Tri = spiral_sample_sphere(n_samples)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. 3D scatter plot of points
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(V[:, 0], V[:, 1], V[:, 2], c='blue', s=20, alpha=0.7)
    ax1.set_title(f'Sphere Sampling (n={n_samples})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 2. Triangulated mesh
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_mesh_surface(V, Tri, title='Triangulated Mesh', alpha=0.3, ax=ax2)

    # 3. Z-coordinate distribution (should be uniform)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(V[:, 2], bins=20, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Z coordinate')
    ax3.set_ylabel('Count')
    ax3.set_title('Z Distribution (should be uniform)')
    ax3.axhline(n_samples / 20, color='red', linestyle='--', label='Expected')
    ax3.legend()

    # 4. Nearest neighbor distance distribution
    ax4 = fig.add_subplot(2, 3, 4)
    tree = cKDTree(V)
    nn_dists, _ = tree.query(V, k=2)
    nn_dists = nn_dists[:, 1]  # Exclude self

    ax4.hist(nn_dists, bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(nn_dists), color='red', linestyle='--',
                label=f'Mean: {np.mean(nn_dists):.3f}')
    ax4.set_xlabel('Nearest Neighbor Distance')
    ax4.set_ylabel('Count')
    ax4.set_title('NN Distance Distribution')
    ax4.legend()

    # 5. Geodesic distance heatmap (subset for performance)
    ax5 = fig.add_subplot(2, 3, 5)
    subset_size = min(50, n_samples)
    V_subset = V[:subset_size]
    D = geodesic_distance_matrix(V_subset)
    im = ax5.imshow(D, cmap='viridis')
    plt.colorbar(im, ax=ax5)
    ax5.set_title(f'Geodesic Distance Matrix (first {subset_size} points)')
    ax5.set_xlabel('Point Index')
    ax5.set_ylabel('Point Index')

    # 6. Statistics summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    cv = np.std(nn_dists) / np.mean(nn_dists)
    stats_text = f"""
    Sphere Sampling Statistics
    ==========================

    Number of points: {n_samples}
    Number of triangles: {len(Tri)}

    Point coordinates:
      X range: [{V[:, 0].min():.3f}, {V[:, 0].max():.3f}]
      Y range: [{V[:, 1].min():.3f}, {V[:, 1].max():.3f}]
      Z range: [{V[:, 2].min():.3f}, {V[:, 2].max():.3f}]

    Distance from origin:
      Mean: {np.mean(np.linalg.norm(V, axis=1)):.6f}
      Std:  {np.std(np.linalg.norm(V, axis=1)):.2e}

    Nearest neighbor distances:
      Mean: {np.mean(nn_dists):.4f}
      Std:  {np.std(nn_dists):.4f}
      CV:   {cv:.4f} {'(GOOD)' if cv < 0.3 else '(POOR)'}

    Uniformity test: {'PASSED' if cv < 0.3 else 'FAILED'}
    """
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('sphere_sampling_visualization.png', dpi=150)
    print("Saved: sphere_sampling_visualization.png")
    plt.show()


def compare_sample_sizes():
    """Compare sphere sampling with different sample sizes."""
    sample_sizes = [50, 100, 200, 500, 1000]

    fig, axes = plt.subplots(2, len(sample_sizes), figsize=(20, 8))

    for i, n in enumerate(sample_sizes):
        V, Tri = spiral_sample_sphere(n)

        # 3D view
        ax_3d = fig.add_subplot(2, len(sample_sizes), i + 1, projection='3d')
        ax_3d.scatter(V[:, 0], V[:, 1], V[:, 2], s=max(5, 50 - n // 20), alpha=0.7)
        ax_3d.set_title(f'n = {n}')

        # Uniformity metric
        tree = cKDTree(V)
        nn_dists, _ = tree.query(V, k=2)
        cv = np.std(nn_dists[:, 1]) / np.mean(nn_dists[:, 1])

        ax_hist = axes[1, i]
        ax_hist.hist(nn_dists[:, 1], bins=15, edgecolor='black', alpha=0.7)
        ax_hist.set_title(f'CV = {cv:.3f}')
        ax_hist.set_xlabel('NN Distance')

    plt.tight_layout()
    plt.savefig('sphere_sampling_comparison.png', dpi=150)
    print("Saved: sphere_sampling_comparison.png")
    plt.show()


if __name__ == '__main__':
    print("=" * 50)
    print("Sphere Sampling Visualization")
    print("=" * 50)

    # Basic visualization
    visualize_sphere_sampling(n_samples=200)

    # Compare different sample sizes
    print("\nComparing different sample sizes...")
    compare_sample_sizes()

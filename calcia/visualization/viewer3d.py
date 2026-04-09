"""
Basic 3D visualization utilities using matplotlib.

This module provides simple 3D visualization functions for:
- Sphere sampling points
- Mesh surfaces
- Neuron shapes
- Volume slices

Uses matplotlib for maximum compatibility. Can be extended with pyvista
for better performance with large datasets.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_sphere_points(
    points: np.ndarray,
    title: str = "Sphere Sampling",
    color: str = 'blue',
    size: float = 20,
    alpha: float = 0.8,
    show_axes: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot points on a 3D sphere.

    Args:
        points: (N, 3) array of 3D points.
        title: Plot title.
        color: Point color.
        size: Point size.
        alpha: Point transparency.
        show_axes: Whether to show axis labels.
        ax: Existing axes to plot on (creates new if None).
        figsize: Figure size if creating new figure.

    Returns:
        matplotlib Figure object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=color, s=size, alpha=alpha
    )

    ax.set_title(title)

    if show_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Set equal aspect ratio
    _set_equal_aspect_3d(ax, points)

    return fig


def plot_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    title: str = "Mesh Surface",
    color: str = 'cyan',
    alpha: float = 0.5,
    edge_color: str = 'black',
    edge_alpha: float = 0.3,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot a triangulated mesh surface.

    Args:
        vertices: (N, 3) array of vertex coordinates.
        faces: (M, 3) array of face-vertex connectivity.
        title: Plot title.
        color: Surface color.
        alpha: Surface transparency.
        edge_color: Edge color.
        edge_alpha: Edge transparency.
        ax: Existing axes to plot on.
        figsize: Figure size if creating new figure.

    Returns:
        matplotlib Figure object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Create polygon collection
    polygons = vertices[faces]
    collection = Poly3DCollection(
        polygons,
        facecolors=color,
        edgecolors=edge_color,
        alpha=alpha,
        linewidths=0.5,
    )

    ax.add_collection3d(collection)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    _set_equal_aspect_3d(ax, vertices)

    return fig


def plot_neuron_shape(
    cell_vertices: np.ndarray,
    cell_faces: np.ndarray,
    nucleus_vertices: Optional[np.ndarray] = None,
    nucleus_faces: Optional[np.ndarray] = None,
    title: str = "Neuron Shape",
    cell_color: str = 'lightblue',
    nucleus_color: str = 'darkblue',
    cell_alpha: float = 0.3,
    nucleus_alpha: float = 0.8,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Plot a neuron shape with cell body and optional nucleus.

    Args:
        cell_vertices: (N, 3) cell body vertex coordinates.
        cell_faces: (M, 3) cell body face connectivity.
        nucleus_vertices: (N', 3) nucleus vertex coordinates.
        nucleus_faces: (M', 3) nucleus face connectivity.
        title: Plot title.
        cell_color: Cell body color.
        nucleus_color: Nucleus color.
        cell_alpha: Cell body transparency.
        nucleus_alpha: Nucleus transparency.
        ax: Existing axes.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Plot cell body
    cell_polygons = cell_vertices[cell_faces]
    cell_collection = Poly3DCollection(
        cell_polygons,
        facecolors=cell_color,
        edgecolors='gray',
        alpha=cell_alpha,
        linewidths=0.2,
    )
    ax.add_collection3d(cell_collection)

    # Plot nucleus if provided
    if nucleus_vertices is not None and nucleus_faces is not None:
        nuc_polygons = nucleus_vertices[nucleus_faces]
        nuc_collection = Poly3DCollection(
            nuc_polygons,
            facecolors=nucleus_color,
            edgecolors='black',
            alpha=nucleus_alpha,
            linewidths=0.2,
        )
        ax.add_collection3d(nuc_collection)

    ax.set_title(title)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (um)')

    _set_equal_aspect_3d(ax, cell_vertices)

    return fig


def plot_volume_slice(
    volume: np.ndarray,
    slice_idx: int,
    axis: int = 2,
    title: str = "Volume Slice",
    cmap: str = 'viridis',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot a 2D slice of a 3D volume.

    Args:
        volume: 3D array.
        slice_idx: Index of slice to display.
        axis: Axis to slice along (0, 1, or 2).
        title: Plot title.
        cmap: Colormap.
        ax: Existing axes.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Extract slice
    if axis == 0:
        slice_data = volume[slice_idx, :, :]
        xlabel, ylabel = 'Y', 'Z'
    elif axis == 1:
        slice_data = volume[:, slice_idx, :]
        xlabel, ylabel = 'X', 'Z'
    else:
        slice_data = volume[:, :, slice_idx]
        xlabel, ylabel = 'X', 'Y'

    im = ax.imshow(slice_data.T, origin='lower', cmap=cmap, aspect='equal')
    plt.colorbar(im, ax=ax)

    ax.set_title(f"{title} (axis={axis}, idx={slice_idx})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig


def plot_multiple_neurons(
    neurons: List[Tuple[np.ndarray, np.ndarray]],
    colors: Optional[List[str]] = None,
    title: str = "Multiple Neurons",
    alpha: float = 0.4,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 12),
) -> plt.Figure:
    """
    Plot multiple neuron shapes in the same figure.

    Args:
        neurons: List of (vertices, faces) tuples.
        colors: List of colors for each neuron.
        title: Plot title.
        alpha: Transparency.
        ax: Existing axes.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    if colors is None:
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(neurons))]

    all_vertices = []
    for i, (vertices, faces) in enumerate(neurons):
        polygons = vertices[faces]
        collection = Poly3DCollection(
            polygons,
            facecolors=colors[i],
            edgecolors='gray',
            alpha=alpha,
            linewidths=0.1,
        )
        ax.add_collection3d(collection)
        all_vertices.append(vertices)

    ax.set_title(title)
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (um)')

    # Set bounds based on all vertices
    all_vertices = np.vstack(all_vertices)
    _set_equal_aspect_3d(ax, all_vertices)

    return fig


def _set_equal_aspect_3d(ax: plt.Axes, points: np.ndarray) -> None:
    """
    Set equal aspect ratio for 3D plot.

    Args:
        ax: matplotlib 3D axes.
        points: (N, 3) array of points to determine bounds.
    """
    # Compute bounds
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

"""
Sphere sampling utilities.

Provides uniform sampling of unit sphere using spiral method.
Port of SpiralSampleSphere.m by Anton Semechko.

Reference:
    Christopher Carlson, 'How I Made Wine Glasses from Sunflowers',
    July 8, 2011. http://blog.wolfram.com/2011/07/28/how-i-made-wine-glasses-from-sunflowers/
"""

import numpy as np
from scipy.spatial import ConvexHull
from typing import Tuple, Optional


def spiral_sample_sphere(
    n_samples: int = 200,
    return_triangulation: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Produce approximately uniform sampling of a unit sphere using spiral method.

    Uses the golden angle to distribute points uniformly on a sphere surface.
    Latitudes are assigned to ensure uniform sampling density.

    Args:
        n_samples: Desired number of sample points. Default is 200.
                   Sampling becomes more uniform with increasing n_samples.
        return_triangulation: Whether to compute and return the triangulation.

    Returns:
        V: (n_samples, 3) array of vertex coordinates on unit sphere.
        Tri: (M, 3) array of face-vertex connectivity indices (0-based).
             Returns None if return_triangulation is False.

    Example:
        >>> V, Tri = spiral_sample_sphere(200)
        >>> V.shape
        (200, 3)
        >>> np.allclose(np.linalg.norm(V, axis=1), 1.0)
        True
    """
    n_samples = int(round(n_samples))

    # Golden ratio and golden angle
    golden_ratio = (1 + np.sqrt(5)) / 2
    golden_angle = 2 * np.pi * (1 - 1 / golden_ratio)

    # Particle index (0 to N-1)
    i = np.arange(n_samples)

    # Latitude: defined so particle index is proportional to surface area
    # between 0 and lat
    lat = np.arccos(1 - 2 * i / (n_samples - 1))

    # Longitude: position particles at even intervals along longitude
    lon = i * golden_angle

    # Convert from spherical to Cartesian coordinates
    x = np.sin(lat) * np.cos(lon)
    y = np.sin(lat) * np.sin(lon)
    z = np.cos(lat)

    V = np.column_stack([x, y, z])

    # Compute triangulation if requested
    Tri = None
    if return_triangulation:
        # Use convex hull to get triangulation
        # fliplr for counterclockwise winding
        hull = ConvexHull(V)
        Tri = hull.simplices[:, ::-1]  # Flip for counterclockwise winding (fliplr(convhulln(V)))

    return V, Tri


def fibonacci_sphere(n_samples: int = 200) -> np.ndarray:
    """
    Alternative sphere sampling using Fibonacci lattice.

    This is mathematically equivalent to spiral_sample_sphere but
    uses a slightly different formulation.

    Args:
        n_samples: Number of sample points.

    Returns:
        (n_samples, 3) array of vertex coordinates on unit sphere.
    """
    indices = np.arange(n_samples, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / n_samples)
    theta = np.pi * (1 + np.sqrt(5)) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.column_stack([x, y, z])


def icosahedron_vertices() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate vertices and faces of a regular icosahedron.

    Returns:
        V: (12, 3) array of vertex coordinates.
        F: (20, 3) array of face indices.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # 12 vertices of icosahedron
    V = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)

    # Normalize to unit sphere
    V = V / np.linalg.norm(V[0])

    # 20 faces (triangles)
    F = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)

    return V, F


def subdivide_sphere_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_subdivisions: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subdivide a spherical triangle mesh.

    Each triangle is split into 4 smaller triangles by adding
    vertices at edge midpoints, then projecting to unit sphere.

    Args:
        vertices: (N, 3) array of vertex coordinates.
        faces: (M, 3) array of face indices.
        n_subdivisions: Number of subdivision iterations.

    Returns:
        new_vertices: Subdivided vertex array.
        new_faces: Subdivided face array.
    """
    V = vertices.copy()
    F = faces.copy()

    for _ in range(n_subdivisions):
        # Dictionary to cache edge midpoints
        edge_cache = {}
        new_faces = []

        for face in F:
            v0, v1, v2 = face

            # Get or create midpoints for each edge
            mid_verts = []
            for i, j in [(v0, v1), (v1, v2), (v2, v0)]:
                edge_key = tuple(sorted([i, j]))
                if edge_key not in edge_cache:
                    # Create midpoint and project to sphere
                    midpoint = (V[i] + V[j]) / 2
                    midpoint = midpoint / np.linalg.norm(midpoint)
                    edge_cache[edge_key] = len(V)
                    V = np.vstack([V, midpoint])
                mid_verts.append(edge_cache[edge_key])

            m01, m12, m20 = mid_verts

            # Create 4 new faces
            new_faces.extend([
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20]
            ])

        F = np.array(new_faces, dtype=np.int32)

    return V, F


def geodesic_distance_matrix(vertices: np.ndarray) -> np.ndarray:
    """
    Compute geodesic (arc-length) distance matrix between all vertices.

    For points on a unit sphere, geodesic distance is the arc length,
    which equals the angle between the vectors.

    Args:
        vertices: (N, 3) array of points on unit sphere.

    Returns:
        (N, N) distance matrix where D[i,j] is geodesic distance from i to j.
    """
    # Normalize to ensure unit sphere
    V = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Compute dot products (cosine of angle)
    cos_angles = np.clip(V @ V.T, -1.0, 1.0)

    # Arc-length distance = arccos(dot product)
    return np.arccos(cos_angles)

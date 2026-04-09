"""
Triangulation utilities for point-in-mesh testing.

Provides functionality to test whether points are inside a triangulated mesh.
Replaces intriangulation.m using the trimesh library.
"""

import numpy as np
from typing import Union, Optional

# Try to import trimesh, provide fallback if not available
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


def in_triangulation(
    vertices: np.ndarray,
    faces: np.ndarray,
    test_points: np.ndarray
) -> np.ndarray:
    """
    Test whether points are inside a triangulated mesh.

    Replaces the intriangulation function.
    Uses ray casting to determine if points are inside the closed mesh.

    Args:
        vertices: (N, 3) array of mesh vertex coordinates.
        faces: (M, 3) array of face-vertex connectivity (0-based indices).
        test_points: (P, 3) array of points to test.

    Returns:
        (P,) boolean array, True if point is inside the mesh.

    Raises:
        ImportError: If trimesh is not installed.

    Example:
        >>> # Create a simple cube
        >>> vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0],
        ...                      [0,0,1], [1,0,1], [1,1,1], [0,1,1]], dtype=float)
        >>> faces = np.array([[0,1,2], [0,2,3], ...])  # cube faces
        >>> points = np.array([[0.5, 0.5, 0.5], [2.0, 0.0, 0.0]])
        >>> inside = in_triangulation(vertices, faces, points)
        >>> inside
        array([ True, False])
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError(
            "trimesh is required for in_triangulation. "
            "Install with: pip install trimesh"
        )

    # Create trimesh mesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Use trimesh's contains method (ray casting)
    inside = mesh.contains(test_points)

    return inside


def in_triangulation_batch(
    vertices: np.ndarray,
    faces: np.ndarray,
    test_points: np.ndarray,
    batch_size: int = 100000
) -> np.ndarray:
    """
    Test whether points are inside a mesh, processing in batches.

    For very large point sets, this reduces memory usage.

    Args:
        vertices: (N, 3) array of mesh vertex coordinates.
        faces: (M, 3) array of face-vertex connectivity.
        test_points: (P, 3) array of points to test.
        batch_size: Number of points to process at once.

    Returns:
        (P,) boolean array.
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    n_points = len(test_points)
    inside = np.zeros(n_points, dtype=bool)

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        inside[start:end] = mesh.contains(test_points[start:end])

    return inside


def create_mesh_from_vertices_faces(
    vertices: np.ndarray,
    faces: np.ndarray
) -> 'trimesh.Trimesh':
    """
    Create a trimesh mesh object from vertices and faces.

    Args:
        vertices: (N, 3) array of mesh vertex coordinates.
        faces: (M, 3) array of face-vertex connectivity.

    Returns:
        trimesh.Trimesh object.
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required")

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def compute_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Compute the volume of a closed triangulated mesh.

    Args:
        vertices: (N, 3) array of mesh vertex coordinates.
        faces: (M, 3) array of face-vertex connectivity.

    Returns:
        Volume of the mesh (signed, negative if faces are wound clockwise).
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh.volume


def compute_mesh_centroid(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute the centroid of a triangulated mesh.

    Args:
        vertices: (N, 3) array of mesh vertex coordinates.
        faces: (M, 3) array of face-vertex connectivity.

    Returns:
        (3,) array with centroid coordinates.
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh.centroid


def ray_mesh_intersection(
    vertices: np.ndarray,
    faces: np.ndarray,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray
) -> np.ndarray:
    """
    Compute ray-mesh intersections.

    Args:
        vertices: (N, 3) array of mesh vertex coordinates.
        faces: (M, 3) array of face-vertex connectivity.
        ray_origins: (R, 3) array of ray origin points.
        ray_directions: (R, 3) array of ray direction vectors.

    Returns:
        (R,) array of intersection distances (np.inf if no intersection).
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Normalize ray directions
    ray_directions = ray_directions / np.linalg.norm(
        ray_directions, axis=1, keepdims=True
    )

    # Use trimesh's ray casting
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )

    # Compute distances for each ray
    n_rays = len(ray_origins)
    distances = np.full(n_rays, np.inf)

    if len(locations) > 0:
        for i, (loc, ray_idx) in enumerate(zip(locations, index_ray)):
            dist = np.linalg.norm(loc - ray_origins[ray_idx])
            if dist < distances[ray_idx]:
                distances[ray_idx] = dist

    return distances


# Fallback implementation without trimesh
def _in_triangulation_fallback(
    vertices: np.ndarray,
    faces: np.ndarray,
    test_points: np.ndarray
) -> np.ndarray:
    """
    Fallback point-in-mesh test using simple ray casting.

    This is slower than trimesh but doesn't require external dependencies.
    Only works for watertight (closed) meshes.

    Args:
        vertices: (N, 3) array of mesh vertex coordinates.
        faces: (M, 3) array of face-vertex connectivity.
        test_points: (P, 3) array of points to test.

    Returns:
        (P,) boolean array.
    """
    # Simple ray casting in +z direction
    # Count intersections - odd count means inside

    n_points = len(test_points)
    inside = np.zeros(n_points, dtype=bool)

    for i, point in enumerate(test_points):
        count = 0
        for face in faces:
            v0, v1, v2 = vertices[face]

            # Check if ray from point in +z direction intersects triangle
            if _ray_triangle_intersection(point, np.array([0, 0, 1]), v0, v1, v2):
                count += 1

        inside[i] = (count % 2) == 1

    return inside


def _ray_triangle_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    epsilon: float = 1e-10
) -> bool:
    """
    Moller-Trumbore ray-triangle intersection algorithm.

    Args:
        ray_origin: Ray origin point.
        ray_dir: Ray direction (normalized).
        v0, v1, v2: Triangle vertices.
        epsilon: Numerical tolerance.

    Returns:
        True if ray intersects triangle with t > 0.
    """
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    if abs(a) < epsilon:
        return False

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)

    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(edge2, q)

    return t > epsilon

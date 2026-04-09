"""
Index conversion utilities for MATLAB to Python compatibility.

MATLAB uses 1-based indexing and column-major (Fortran) order.
Python/NumPy uses 0-based indexing and row-major (C) order by default.

This module provides utilities to handle these differences.
"""

import numpy as np
from typing import Tuple, Union, Sequence


def ind2sub_matlab(
    shape: Tuple[int, ...],
    indices: Union[int, np.ndarray]
) -> Tuple[np.ndarray, ...]:
    """
    Convert linear indices to subscripts using MATLAB conventions.

    MATLAB uses column-major (Fortran) order and 1-based indexing.
    This function uses Fortran order but returns 0-based indices for Python.

    Args:
        shape: Shape of the array (as tuple).
        indices: Linear indices (0-based for Python).

    Returns:
        Tuple of coordinate arrays, one per dimension.

    Example:
        >>> shape = (3, 4, 5)
        >>> idx = 10
        >>> ind2sub_matlab(shape, idx)
        (1, 0, 0)  # Fortran order decomposition

    Note:
        To match MATLAB output exactly, add 1 to all returned indices.
    """
    return np.unravel_index(indices, shape, order='F')


def sub2ind_matlab(
    shape: Tuple[int, ...],
    *subscripts: np.ndarray
) -> np.ndarray:
    """
    Convert subscripts to linear indices using MATLAB conventions.

    MATLAB uses column-major (Fortran) order.
    This function expects 0-based subscripts (Python convention).

    Args:
        shape: Shape of the array (as tuple).
        *subscripts: Coordinate arrays, one per dimension (0-based).

    Returns:
        Linear indices (0-based).

    Example:
        >>> shape = (3, 4, 5)
        >>> sub2ind_matlab(shape, 1, 0, 0)
        10
    """
    return np.ravel_multi_index(subscripts, shape, order='F')


def matlab_to_python_index(idx: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """
    Convert MATLAB 1-based index to Python 0-based index.

    Args:
        idx: MATLAB index (1-based).

    Returns:
        Python index (0-based).
    """
    return idx - 1


def python_to_matlab_index(idx: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """
    Convert Python 0-based index to MATLAB 1-based index.

    Args:
        idx: Python index (0-based).

    Returns:
        MATLAB index (1-based).
    """
    return idx + 1


def ensure_fortran_order(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is in Fortran (column-major) order.

    This is useful when interfacing with MATLAB data or algorithms
    that assume column-major memory layout.

    Args:
        arr: Input array.

    Returns:
        Array in Fortran order (may be a copy if reordering needed).
    """
    if arr.flags['F_CONTIGUOUS']:
        return arr
    return np.asfortranarray(arr)


def ensure_c_order(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is in C (row-major) order.

    This is the default NumPy order.

    Args:
        arr: Input array.

    Returns:
        Array in C order (may be a copy if reordering needed).
    """
    if arr.flags['C_CONTIGUOUS']:
        return arr
    return np.ascontiguousarray(arr)


def meshgrid_matlab(*xi: np.ndarray, indexing: str = 'ij') -> Tuple[np.ndarray, ...]:
    """
    Create coordinate matrices matching MATLAB's ndgrid behavior.

    MATLAB's ndgrid uses 'ij' indexing (first dimension varies along first axis).
    MATLAB's meshgrid uses 'xy' indexing (first dimension varies along second axis).

    For most volume generation code, we want ndgrid behavior.

    Args:
        *xi: 1-D arrays representing coordinates.
        indexing: 'ij' for ndgrid behavior (default), 'xy' for meshgrid behavior.

    Returns:
        Tuple of coordinate matrices.

    Example:
        >>> x = np.arange(3)
        >>> y = np.arange(4)
        >>> X, Y = meshgrid_matlab(x, y)
        >>> X.shape
        (3, 4)
    """
    return np.meshgrid(*xi, indexing=indexing)


def create_3d_indices(shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 3D coordinate grids for a volume.

    Args:
        shape: (nx, ny, nz) shape of the volume.

    Returns:
        Tuple of (X, Y, Z) coordinate arrays, each with shape=shape.
    """
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    return np.meshgrid(x, y, z, indexing='ij')


def get_neighbors_6(
    idx: int,
    shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Get 6-connected neighbors of a voxel in 3D.

    Returns indices of the 6 face-adjacent neighbors (up, down, left, right, front, back).

    Args:
        idx: Linear index of the voxel (0-based).
        shape: (nx, ny, nz) shape of the volume.

    Returns:
        Array of up to 6 neighbor indices. Invalid neighbors (outside bounds) are excluded.
    """
    nx, ny, nz = shape
    x, y, z = np.unravel_index(idx, shape, order='F')

    # 6-connectivity offsets: +x, -x, +y, -y, +z, -z
    offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]

    neighbors = []
    for dx, dy, dz in offsets:
        nx_new, ny_new, nz_new = x + dx, y + dy, z + dz
        if 0 <= nx_new < nx and 0 <= ny_new < ny and 0 <= nz_new < nz:
            neighbor_idx = np.ravel_multi_index((nx_new, ny_new, nz_new), shape, order='F')
            neighbors.append(neighbor_idx)

    return np.array(neighbors, dtype=np.int64)


def linear_offsets_6(shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Get linear index offsets for 6-connected neighbors.

    This is useful for fast neighbor access in Dijkstra-like algorithms.
    Matches the convention used in dendrite_dijkstra_cpp.cpp.

    Args:
        shape: (nx, ny, nz) shape of the volume.

    Returns:
        Array of 6 offsets: [+x, -x, +y, -y, +z, -z] in Fortran order.
    """
    nx, ny, nz = shape
    # In Fortran order:
    # +x: +1
    # -x: -1
    # +y: +nx
    # -y: -nx
    # +z: +nx*ny
    # -z: -nx*ny
    return np.array([1, -1, nx, -nx, nx * ny, -nx * ny], dtype=np.int64)

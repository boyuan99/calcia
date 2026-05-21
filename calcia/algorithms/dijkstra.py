"""
Dijkstra's algorithm implementations for neural volume simulation.

Provides two variants:
1. vessel_dijkstra: For graph-based vessel network with distance matrix
2. dendrite_dijkstra: For 3D grid-based dendrite growth with obstacles
"""

import numpy as np
from typing import Tuple, Optional
import heapq


def vessel_dijkstra(
    dist_matrix: np.ndarray,
    root: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dijkstra's algorithm for vessel network growth.

    Finds shortest paths from a root node to all other nodes in a graph
    defined by a distance matrix.

    Args:
        dist_matrix: (N, N) symmetric distance matrix between nodes.
                    dist_matrix[i,j] is the edge weight between nodes i and j.
                    Use inf for non-connected nodes.
        root: Index of the root node (0-based).

    Returns:
        distance: (N,) array of minimum distances from root to each node.
        path_from: (N,) array of parent node indices (-1 for unreachable).

    Example:
        >>> dist = np.array([[0, 1, 4], [1, 0, 2], [4, 2, 0]], dtype=float)
        >>> dists, parents = vessel_dijkstra(dist, root=0)
        >>> dists  # [0, 1, 3] - shortest path to node 2 is 0->1->2 = 3
    """
    n_nodes = dist_matrix.shape[0]

    # Initialize
    distance = np.full(n_nodes, np.inf)
    distance[root] = 0.0
    path_from = np.full(n_nodes, -1, dtype=np.int32)
    visited = np.zeros(n_nodes, dtype=bool)

    # Priority queue: (distance, node)
    heap = [(0.0, root)]

    while heap:
        dist_u, u = heapq.heappop(heap)

        if visited[u]:
            continue
        visited[u] = True

        # Explore neighbors
        for v in range(n_nodes):
            if visited[v]:
                continue

            edge_weight = dist_matrix[u, v]
            if np.isinf(edge_weight):
                continue

            new_dist = dist_u + edge_weight
            if new_dist < distance[v]:
                distance[v] = new_dist
                path_from[v] = u
                heapq.heappush(heap, (new_dist, v))

    return distance, path_from


def dendrite_dijkstra(
    cost_volume: np.ndarray,
    root: Tuple[int, int, int],
    use_numba: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dijkstra's algorithm for 3D dendrite growth through a volume.

    Finds shortest paths from a root voxel to all other voxels,
    using 6-connectivity (face neighbors). The cost volume defines
    the traversal cost at each voxel (high cost = obstacle).

    Args:
        cost_volume: (X, Y, Z) array of traversal costs.
                    Use high values (e.g., inf) for blocked voxels.
        root: (x, y, z) coordinates of the root voxel (0-based).
        use_numba: Whether to use numba acceleration if available.

    Returns:
        distance: (X, Y, Z) array of minimum distances from root.
        path_from: (X, Y, Z, 3) array of parent voxel coordinates.
                  Values of -1 indicate no parent (unreachable or root).

    Note:
        This implements the same algorithm as the MATLAB MEX function
        `dendrite_dijkstra_cpp`, but in pure Python with optional
        numba acceleration.
    """
    # Ensure float32 for consistency with MATLAB
    cost_volume = cost_volume.astype(np.float32)
    dims = cost_volume.shape

    # Try numba-accelerated version first
    if use_numba:
        try:
            return _dendrite_dijkstra_numba(cost_volume, root)
        except ImportError:
            pass

    # Fall back to pure Python implementation
    return _dendrite_dijkstra_python(cost_volume, root)


def _dendrite_dijkstra_python(
    cost_volume: np.ndarray,
    root: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure Python implementation of 3D Dijkstra."""
    dims = cost_volume.shape
    n_voxels = np.prod(dims)

    # Initialize distance and path arrays
    distance = np.full(dims, np.inf, dtype=np.float32)
    distance[root] = 0.0

    path_from = np.full((*dims, 3), -1, dtype=np.int32)
    visited = np.zeros(dims, dtype=bool)

    # 6-connected neighbors: +x, -x, +y, -y, +z, -z
    neighbors = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=np.int32)

    # Priority queue: (distance, (x, y, z))
    heap = [(0.0, root)]

    while heap:
        dist_u, (x, y, z) = heapq.heappop(heap)

        if visited[x, y, z]:
            continue
        visited[x, y, z] = True

        # Explore 6-connected neighbors
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz

            # Check bounds
            if not (0 <= nx < dims[0] and 0 <= ny < dims[1] and 0 <= nz < dims[2]):
                continue

            if visited[nx, ny, nz]:
                continue

            # Cost to reach neighbor
            edge_cost = cost_volume[nx, ny, nz]
            new_dist = dist_u + edge_cost

            if new_dist < distance[nx, ny, nz]:
                distance[nx, ny, nz] = new_dist
                path_from[nx, ny, nz] = [x, y, z]
                heapq.heappush(heap, (new_dist, (nx, ny, nz)))

    return distance, path_from


def _dendrite_dijkstra_numba(
    cost_volume: np.ndarray,
    root: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated implementation of 3D Dijkstra."""
    from numba import njit
    import numba

    # We need to use linear indexing with numba for the heap
    dims = cost_volume.shape
    n_voxels = dims[0] * dims[1] * dims[2]

    # Convert root to linear index
    root_idx = root[0] + root[1] * dims[0] + root[2] * dims[0] * dims[1]

    # Neighbor offsets in linear indexing
    neighbor_offsets = np.array([
        1,                          # +x
        -1,                         # -x
        dims[0],                    # +y
        -dims[0],                   # -y
        dims[0] * dims[1],          # +z
        -dims[0] * dims[1],         # -z
    ], dtype=np.int64)

    # Flatten cost volume for linear indexing (Fortran order to match
    # the x + y*X + z*X*Y indexing used in the numba core)
    cost_flat = cost_volume.ravel(order='F')

    distance, path_from_flat = _dijkstra_core_numba(
        cost_flat, neighbor_offsets, root_idx, dims
    )

    # Reshape outputs (Fortran order to match the flat indexing)
    distance = distance.reshape(dims, order='F')
    path_from = np.full((*dims, 3), -1, dtype=np.int32)

    # Convert linear indices back to subscripts
    valid_mask = path_from_flat >= 0
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        parent_indices = path_from_flat[valid_indices]

        # Convert linear to subscript
        px = parent_indices % dims[0]
        py = (parent_indices // dims[0]) % dims[1]
        pz = parent_indices // (dims[0] * dims[1])

        # Convert current linear index to subscript
        cx = valid_indices % dims[0]
        cy = (valid_indices // dims[0]) % dims[1]
        cz = valid_indices // (dims[0] * dims[1])

        path_from[cx, cy, cz, 0] = px
        path_from[cx, cy, cz, 1] = py
        path_from[cx, cy, cz, 2] = pz

    return distance, path_from


def dendrite_dijkstra_6dir(
    cost_6dir: np.ndarray,
    dims: Tuple[int, int, int],
    root: Tuple[int, int, int],
    use_numba: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dijkstra with 6-directional costs, matching MATLAB dendrite_dijkstra2.

    Each voxel has 6 independent costs, one per neighbor direction.
    Uses Fortran-order (column-major) linear indexing internally,
    matching the MATLAB MEX implementation exactly.

    Directions: +x(0), -x(1), +y(2), -y(3), +z(4), -z(5)

    Args:
        cost_6dir: (prod(dims), 6) float32 array. Column-major spatial
                   flattening. cost_6dir[nn, i] = cost of arriving at
                   voxel nn via direction i.
        dims: (3,) volume dimensions (dx, dy, dz).
        root: (3,) 0-based root coordinates.
        use_numba: Whether to use numba acceleration.

    Returns:
        distance: (dims) array of minimum distances from root.
        path_from: (dims, 3) array of parent voxel coordinates.
    """
    dims = tuple(int(d) for d in dims)
    n_voxels = dims[0] * dims[1] * dims[2]

    if cost_6dir.shape != (n_voxels, 6):
        raise ValueError(
            f"cost_6dir shape {cost_6dir.shape} doesn't match "
            f"(prod(dims)={n_voxels}, 6)"
        )

    if cost_6dir.dtype != np.float32:
        cost_6dir = cost_6dir.astype(np.float32)

    # Fortran-order linear index for root
    root_idx = root[0] + root[1] * dims[0] + root[2] * dims[0] * dims[1]

    # Neighbor offsets in Fortran-order linear indexing (matching MATLAB)
    # Direction 0: +x -> +1
    # Direction 1: -x -> -1
    # Direction 2: +y -> +dims[0]
    # Direction 3: -y -> -dims[0]
    # Direction 4: +z -> +dims[0]*dims[1]
    # Direction 5: -z -> -dims[0]*dims[1]
    pe = np.array([
        1, -1,
        dims[0], -dims[0],
        dims[0] * dims[1], -dims[0] * dims[1],
    ], dtype=np.int64)

    if use_numba:
        try:
            dist_flat, pf_flat = _dijkstra_6dir_core_numba(
                cost_6dir, pe, root_idx, dims
            )
        except (ImportError, TypeError):
            dist_flat, pf_flat = _dijkstra_6dir_core_python(
                cost_6dir, pe, root_idx, n_voxels, dims
            )
    else:
        dist_flat, pf_flat = _dijkstra_6dir_core_python(
            cost_6dir, pe, root_idx, n_voxels, dims
        )

    # Reshape distance (Fortran order to match flat indexing)
    distance = dist_flat.reshape(dims, order='F')

    # Convert path_from linear indices to 3D subscripts
    path_from = np.full((*dims, 3), -1, dtype=np.int32)
    valid_mask = pf_flat >= 0
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        parent_indices = pf_flat[valid_indices]

        # Fortran-order linear -> subscript
        px = parent_indices % dims[0]
        py = (parent_indices // dims[0]) % dims[1]
        pz = parent_indices // (dims[0] * dims[1])

        cx = valid_indices % dims[0]
        cy = (valid_indices // dims[0]) % dims[1]
        cz = valid_indices // (dims[0] * dims[1])

        path_from[cx, cy, cz, 0] = px
        path_from[cx, cy, cz, 1] = py
        path_from[cx, cy, cz, 2] = pz

    return distance, path_from


def _dijkstra_6dir_core_python(cost_6dir, pe, root_idx, n_voxels, dims):
    """Pure Python 6-directional Dijkstra core."""
    distance = np.full(n_voxels, np.inf, dtype=np.float32)
    distance[root_idx] = 0.0
    path_from = np.full(n_voxels, -1, dtype=np.int64)
    visited = np.zeros(n_voxels, dtype=bool)

    heap = [(np.float32(0.0), int(root_idx))]
    dims_x, dims_y = dims[0], dims[1]

    while heap:
        dist_u, u = heapq.heappop(heap)

        if visited[u]:
            continue
        visited[u] = True

        for i in range(6):
            nn = u + int(pe[i])
            if nn < 0 or nn >= n_voxels:
                continue

            # Wrap-around check
            ux = u % dims_x
            uy = (u // dims_x) % dims_y
            vx = nn % dims_x
            vy = (nn // dims_x) % dims_y
            if abs(vx - ux) > 1 or abs(vy - uy) > 1:
                continue

            if visited[nn]:
                continue

            ndist = dist_u + cost_6dir[nn, i]
            if ndist < distance[nn]:
                distance[nn] = ndist
                path_from[nn] = u
                heapq.heappush(heap, (ndist, nn))

    return distance, path_from


# Try to compile numba function at import time
try:
    from numba import njit

    @njit(cache=True)
    def _dijkstra_core_numba(cost_flat, neighbor_offsets, root_idx, dims):
        """Numba-compiled core Dijkstra algorithm."""
        n_voxels = len(cost_flat)
        dims_x, dims_y, dims_z = dims[0], dims[1], dims[2]

        # Initialize
        distance = np.full(n_voxels, np.float32(np.inf), dtype=np.float32)
        distance[root_idx] = 0.0
        path_from = np.full(n_voxels, -1, dtype=np.int64)
        visited = np.zeros(n_voxels, dtype=np.bool_)

        # Simple heap using arrays (numba doesn't support heapq)
        # Use a large array and track size
        heap_dist = np.zeros(n_voxels * 2, dtype=np.float32)
        heap_idx = np.zeros(n_voxels * 2, dtype=np.int64)
        heap_size = 1
        heap_dist[0] = 0.0
        heap_idx[0] = root_idx

        while heap_size > 0:
            # Pop minimum from heap
            min_i = 0
            for i in range(1, heap_size):
                if heap_dist[i] < heap_dist[min_i]:
                    min_i = i

            dist_u = heap_dist[min_i]
            u = heap_idx[min_i]

            # Remove from heap by moving last element
            heap_size -= 1
            heap_dist[min_i] = heap_dist[heap_size]
            heap_idx[min_i] = heap_idx[heap_size]

            if visited[u]:
                continue
            visited[u] = True

            # Get current subscript indices
            ux = u % dims_x
            uy = (u // dims_x) % dims_y
            uz = u // (dims_x * dims_y)

            # Explore neighbors
            for i in range(6):
                v = u + neighbor_offsets[i]

                if v < 0 or v >= n_voxels:
                    continue

                # Get neighbor subscript and check bounds
                vx = v % dims_x
                vy = (v // dims_x) % dims_y
                vz = v // (dims_x * dims_y)

                # Check if we wrapped around (invalid neighbor)
                dx = abs(vx - ux)
                dy = abs(vy - uy)
                dz = abs(vz - uz)
                if dx > 1 or dy > 1 or dz > 1:
                    continue

                if visited[v]:
                    continue

                new_dist = dist_u + cost_flat[v]
                if new_dist < distance[v]:
                    distance[v] = new_dist
                    path_from[v] = u

                    # Add to heap
                    heap_dist[heap_size] = new_dist
                    heap_idx[heap_size] = v
                    heap_size += 1

        return distance, path_from

    @njit(cache=True)
    def _dijkstra_6dir_core_numba(cost_6dir, pe, root_idx, dims):
        """Numba-compiled 6-directional Dijkstra matching MATLAB MEX.

        Args:
            cost_6dir: (n_voxels, 6) float32 cost array.
            pe: (6,) int64 neighbor offsets in Fortran-order linear index.
            root_idx: int, Fortran-order linear index of root.
            dims: (3,) tuple of volume dimensions.
        """
        n_voxels = cost_6dir.shape[0]
        dims_x, dims_y = dims[0], dims[1]

        distance = np.full(n_voxels, np.float32(np.inf), dtype=np.float32)
        distance[root_idx] = np.float32(0.0)
        path_from = np.full(n_voxels, -1, dtype=np.int64)
        visited = np.zeros(n_voxels, dtype=np.bool_)

        heap_dist = np.zeros(n_voxels * 2, dtype=np.float32)
        heap_idx = np.zeros(n_voxels * 2, dtype=np.int64)
        heap_size = 1
        heap_dist[0] = np.float32(0.0)
        heap_idx[0] = root_idx

        while heap_size > 0:
            # Pop minimum
            min_i = 0
            for i in range(1, heap_size):
                if heap_dist[i] < heap_dist[min_i]:
                    min_i = i

            dist_u = heap_dist[min_i]
            u = heap_idx[min_i]
            heap_size -= 1
            heap_dist[min_i] = heap_dist[heap_size]
            heap_idx[min_i] = heap_idx[heap_size]

            if visited[u]:
                continue
            visited[u] = True

            ux = u % dims_x
            uy = (u // dims_x) % dims_y

            for i in range(6):
                nn = u + pe[i]
                if nn < 0 or nn >= n_voxels:
                    continue

                # Wrap-around check
                vx = nn % dims_x
                vy = (nn // dims_x) % dims_y
                if abs(vx - ux) > 1 or abs(vy - uy) > 1:
                    continue

                if visited[nn]:
                    continue

                ndist = dist_u + cost_6dir[nn, i]
                if ndist < distance[nn]:
                    distance[nn] = ndist
                    path_from[nn] = u
                    heap_dist[heap_size] = ndist
                    heap_idx[heap_size] = nn
                    heap_size += 1

        return distance, path_from

except ImportError:
    _dijkstra_core_numba = None
    _dijkstra_6dir_core_numba = None


def reconstruct_path(
    path_from: np.ndarray,
    target: Tuple[int, int, int],
) -> np.ndarray:
    """
    Reconstruct path from root to target using path_from array.

    Args:
        path_from: (X, Y, Z, 3) array of parent coordinates.
        target: (x, y, z) coordinates of target voxel.

    Returns:
        path: (N, 3) array of voxel coordinates from root to target.
              Returns empty array if target is unreachable.
    """
    path = []
    current = np.array(target, dtype=np.int32)

    # Check if target is reachable
    if path_from[target[0], target[1], target[2], 0] < 0:
        # Check if target is the root (all parents are -1 but distance is 0)
        # Otherwise unreachable
        return np.array(path)

    # Trace back to root
    max_steps = np.prod(path_from.shape[:3])  # Prevent infinite loop
    for _ in range(max_steps):
        path.append(current.copy())
        parent = path_from[current[0], current[1], current[2]]

        if parent[0] < 0:  # Reached root
            break

        current = parent

    # Reverse to get path from root to target
    path = path[::-1]
    return np.array(path, dtype=np.int32)


def compute_distance_matrix(
    points: np.ndarray,
    metric: str = 'euclidean',
) -> np.ndarray:
    """
    Compute distance matrix between points.

    Args:
        points: (N, D) array of N points in D dimensions.
        metric: Distance metric ('euclidean' or 'geodesic').

    Returns:
        (N, N) symmetric distance matrix.
    """
    n = len(points)

    if metric == 'euclidean':
        # Efficient pairwise distance computation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    elif metric == 'geodesic':
        # For points on unit sphere
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        euclidean = np.sqrt(np.sum(diff ** 2, axis=2))
        # Geodesic = arc length = 2 * arcsin(chord/2)
        dist_matrix = 2 * np.arcsin(np.clip(euclidean / 2, -1, 1))

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return dist_matrix

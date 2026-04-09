"""Directed random walk for background dendrite and axon simulation.

Port of ``dendrite_randomwalk_cpp.cpp`` (C++ MEX function).

The algorithm performs a greedy, direction-biased walk through a 3D cost
volume.  At each step the cheapest of the 6 face-connected neighbours is
chosen, with a directional bias toward the target endpoint.  The cost
matrix **M** is mutated in-place: accepted paths increase visited-voxel
costs by *fillweight*, while rejected (too short) paths restore original
costs.
"""

from typing import Tuple

import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

FLT_MAX = np.finfo(np.float32).max


# ------------------------------------------------------------------
# Pure-Python implementation
# ------------------------------------------------------------------

def _dendrite_random_walk_python(
    M: np.ndarray,
    root: np.ndarray,
    ends: np.ndarray,
    distsc: float,
    maxlength: int,
    fillweight: float,
    maxel: int,
    minlength: int,
) -> np.ndarray:
    """Pure-Python directed random walk.  See :func:`dendrite_random_walk`."""
    sx, sy, sz = M.shape
    maxfill = maxel * fillweight

    cx, cy, cz = int(root[0]), int(root[1]), int(root[2])
    ex, ey, ez = int(ends[0]), int(ends[1]), int(ends[2])

    path = np.empty((maxlength, 3), dtype=np.int32)
    mvals = np.empty(maxlength, dtype=np.float32)
    bglength = 0

    # 6-connected neighbour offsets: +x, +y, +z, -x, -y, -z
    offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
               (-1, 0, 0), (0, -1, 0), (0, 0, -1))

    for i in range(maxlength):
        # Direction bias toward endpoint
        dx = float(cx - ex)
        dy = float(cy - ey)
        dz = float(cz - ez)
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5 / distsc
        if dist < 1e-10:
            dist = 1e-10
        dvx = dx / dist
        dvy = dy / dist
        dvz = dz / dist
        distvec = (dvx, dvy, dvz)

        jmin = 6  # sentinel
        minmat = FLT_MAX

        for j in range(6):
            ox, oy, oz = offsets[j]
            nx, ny, nz = cx + ox, cy + oy, cz + oz
            # Bounds check
            if nx < 0 or nx >= sx:
                continue
            if ny < 0 or ny >= sy:
                continue
            if nz < 0 or nz >= sz:
                continue

            # Cost = M[neighbour] + direction bias
            # +axis directions: add distvec (penalise moving away from ends)
            # -axis directions: subtract distvec (reward moving toward ends)
            # NOTE: C++ original line 55 had matvals[1] instead of matvals[0]
            # (bug). We use the corrected version here.
            if j < 3:
                cost = float(M[nx, ny, nz]) + distvec[j]
            else:
                cost = float(M[nx, ny, nz]) - distvec[j - 3]

            if cost < minmat:
                jmin = j
                minmat = cost

        if jmin < 6 and minmat < maxfill:
            ox, oy, oz = offsets[jmin]
            cx += ox
            cy += oy
            cz += oz
            path[i, 0] = cx
            path[i, 1] = cy
            path[i, 2] = cz
            mvals[i] = minmat
            M[cx, cy, cz] = FLT_MAX  # block revisit

            # Stop at volume boundary
            if (cx == 0 or cx == sx - 1 or
                    cy == 0 or cy == sy - 1 or
                    cz == 0 or cz == sz - 1):
                bglength = i + 1
                break
        else:
            bglength = i
            break

        # Reached endpoint
        if cx == ex and cy == ey and cz == ez:
            bglength = i + 1
            break
    else:
        bglength = maxlength

    if bglength == 0 and i > 0:
        bglength = i

    # Post-processing: update cost matrix
    if bglength >= minlength:
        for k in range(bglength):
            px, py, pz = int(path[k, 0]), int(path[k, 1]), int(path[k, 2])
            if mvals[k] < maxfill:
                M[px, py, pz] = mvals[k] + fillweight
        return path[:bglength].copy()
    else:
        # Reject: restore original costs
        for k in range(bglength):
            px, py, pz = int(path[k, 0]), int(path[k, 1]), int(path[k, 2])
            M[px, py, pz] = mvals[k]
        return np.zeros((0, 3), dtype=np.int32)


# ------------------------------------------------------------------
# Numba-accelerated implementation
# ------------------------------------------------------------------

if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _dendrite_random_walk_numba(
        M, root_x, root_y, root_z, end_x, end_y, end_z,
        distsc, maxlength, fillweight, maxel, minlength,
    ):
        """Numba-accelerated directed random walk."""
        sx, sy, sz = M.shape[0], M.shape[1], M.shape[2]
        maxfill = maxel * fillweight
        flt_max = np.float32(3.4028235e+38)

        cx, cy, cz = root_x, root_y, root_z

        path_x = np.empty(maxlength, dtype=np.int32)
        path_y = np.empty(maxlength, dtype=np.int32)
        path_z = np.empty(maxlength, dtype=np.int32)
        mvals = np.empty(maxlength, dtype=np.float32)

        bglength = 0
        last_i = 0

        for i in range(maxlength):
            last_i = i
            dx = float(cx - end_x)
            dy = float(cy - end_y)
            dz = float(cz - end_z)
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5 / distsc
            if dist < 1e-10:
                dist = 1e-10
            dvx = np.float32(dx / dist)
            dvy = np.float32(dy / dist)
            dvz = np.float32(dz / dist)

            jmin = 6
            minmat = flt_max

            # +x
            if cx < sx - 1:
                cost = M[cx + 1, cy, cz] + dvx
                if cost < minmat:
                    jmin = 0
                    minmat = cost
            # +y
            if cy < sy - 1:
                cost = M[cx, cy + 1, cz] + dvy
                if cost < minmat:
                    jmin = 1
                    minmat = cost
            # +z
            if cz < sz - 1:
                cost = M[cx, cy, cz + 1] + dvz
                if cost < minmat:
                    jmin = 2
                    minmat = cost
            # -x
            if cx > 0:
                cost = M[cx - 1, cy, cz] - dvx
                if cost < minmat:
                    jmin = 3
                    minmat = cost
            # -y
            if cy > 0:
                cost = M[cx, cy - 1, cz] - dvy
                if cost < minmat:
                    jmin = 4
                    minmat = cost
            # -z
            if cz > 0:
                cost = M[cx, cy, cz - 1] - dvz
                if cost < minmat:
                    jmin = 5
                    minmat = cost

            if jmin < 6 and minmat < maxfill:
                if jmin == 0:
                    cx += 1
                elif jmin == 1:
                    cy += 1
                elif jmin == 2:
                    cz += 1
                elif jmin == 3:
                    cx -= 1
                elif jmin == 4:
                    cy -= 1
                else:
                    cz -= 1

                path_x[i] = cx
                path_y[i] = cy
                path_z[i] = cz
                mvals[i] = minmat
                M[cx, cy, cz] = flt_max

                if (cx == 0 or cx == sx - 1 or
                        cy == 0 or cy == sy - 1 or
                        cz == 0 or cz == sz - 1):
                    bglength = i + 1
                    break
            else:
                bglength = i
                break

            if cx == end_x and cy == end_y and cz == end_z:
                bglength = i + 1
                break
        else:
            bglength = maxlength

        if bglength == 0 and last_i > 0:
            bglength = last_i

        if bglength >= minlength:
            for k in range(bglength):
                if mvals[k] < maxfill:
                    M[path_x[k], path_y[k], path_z[k]] = mvals[k] + fillweight
            return path_x[:bglength], path_y[:bglength], path_z[:bglength]
        else:
            for k in range(bglength):
                M[path_x[k], path_y[k], path_z[k]] = mvals[k]
            empty = np.empty(0, dtype=np.int32)
            return empty, empty, empty


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def dendrite_random_walk(
    M: np.ndarray,
    root: np.ndarray,
    ends: np.ndarray,
    distsc: float,
    maxlength: int,
    fillweight: float,
    maxel: int,
    minlength: int,
) -> np.ndarray:
    """Directed random walk through a 3D cost volume.

    Port of ``dendrite_randomwalk_cpp.cpp``.  At each step the algorithm
    evaluates the 6 face-connected neighbours of the current position and
    moves to the one with lowest effective cost (material cost plus a
    directional bias toward *ends*).

    **M is mutated in-place.**  Accepted walks (length >= *minlength*)
    increase visited-voxel costs by *fillweight*; rejected walks restore
    original costs.

    Args:
        M: 3D float32 cost matrix.  Higher values = harder to traverse.
            Occupied / boundary voxels should be set to ``FLT_MAX``.
        root: (3,) int start position (0-based).
        ends: (3,) int target endpoint (0-based).
        distsc: Direction bias divisor (higher = stronger bias toward *ends*).
        maxlength: Maximum number of walk steps.
        fillweight: Cost increment added to each visited voxel on accept.
        maxel: Maximum number of accepted walks through a single voxel.
            ``maxfill = maxel * fillweight``.
        minlength: Minimum path length; shorter walks are rejected.

    Returns:
        (N, 3) int32 array of 0-based path coordinates, or empty (0, 3)
        array if the walk was rejected.
    """
    M = np.ascontiguousarray(M, dtype=np.float32)
    root = np.asarray(root, dtype=np.int32).ravel()
    ends = np.asarray(ends, dtype=np.int32).ravel()

    if _HAS_NUMBA:
        px, py, pz = _dendrite_random_walk_numba(
            M,
            int(root[0]), int(root[1]), int(root[2]),
            int(ends[0]), int(ends[1]), int(ends[2]),
            float(distsc), int(maxlength), float(fillweight),
            int(maxel), int(minlength),
        )
        if len(px) == 0:
            return np.zeros((0, 3), dtype=np.int32)
        return np.column_stack((px, py, pz))
    else:
        return _dendrite_random_walk_python(
            M, root, ends, distsc, maxlength, fillweight, maxel, minlength,
        )

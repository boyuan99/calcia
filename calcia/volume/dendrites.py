"""
Dendrite growth module.

Grows dendrites from neuron somas using two-level Dijkstra path planning
(coarse grid + fine grid) through a 3D volume, avoiding obstacles.

- Step 4: grow_neuron_dendrites (per-neuron basal + apical dendrites)
- Step 5: grow_apical_dendrites (through-volume apical dendrites)

Corresponds to MATLAB: growNeuronDendrites.m, growApicalDendrites.m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from ..config.params import VolumeParams, DendParams, NeuronParams
from ..algorithms.dijkstra import dendrite_dijkstra_6dir, reconstruct_path
from .neural_volume import NeuralVolumeResult


@dataclass
class DendriteResult:
    """Result of dendrite growth.

    Attributes:
        neur_num: 3D uint16 array. Each voxel contains the 1-based neuron ID
                  occupying it (soma + dendrites), or 0 if empty.
        dendrite_ad: 3D uint16 array. Apical dendrite map after dilation.
        dend_params: Updated dendrite parameters.
        gp_soma: Updated soma data. List of tuples (soma_indices, smoothed_body)
                 per neuron.
    """
    neur_num: np.ndarray
    dendrite_ad: np.ndarray
    dend_params: DendParams
    gp_soma: list


@dataclass
class ApicalDendriteResult:
    """Result of through-volume apical dendrite growth (Step 5).

    Attributes:
        neur_num: 3D uint16 array. Updated volume with through-volume apical
                  dendrites added.
        neur_num_ad: 3D uint16 array. Apical-dendrite-only voxels (for Step 6
                     fluorescence setup). Soma and nucleus regions cleared.
        dend_params: Updated dendrite parameters.
    """
    neur_num: np.ndarray
    neur_num_ad: np.ndarray
    dend_params: DendParams


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_subvolume(cell_volume, center, fdims, fulldims, small_z):
    """Extract a subvolume centered on a neuron, handling boundary clipping.

    Args:
        cell_volume: Full-resolution 3D volume.
        center: (3,) array of neuron center in voxel coords (0-based).
        fdims: (3,) fine grid dimensions.
        fulldims: (3,) full volume dimensions.
        small_z: If True, z-axis is smaller than fdims[2], use full z range.

    Returns:
        obstruction: (fdims) subvolume array.
        root_local: (3,) root position in local coordinates.
        border_flag: True if boundary clipping was needed.
        offsets: (3,) global offset for converting local->global coords.
    """
    fdims = np.asarray(fdims, dtype=int)
    center = np.asarray(center, dtype=int)
    fulldims = np.asarray(fulldims, dtype=int)

    half = fdims // 2

    if small_z:
        root_local = np.array([half[0], half[1], center[2]], dtype=int)
    else:
        root_local = np.array([half[0], half[1], half[2]], dtype=int)

    # Compute source and destination ranges for each axis
    border_flag = False
    obstruction = np.zeros(tuple(fdims), dtype=np.float32)

    axes_to_clip = [0, 1] if small_z else [0, 1, 2]
    src_slices = []
    dst_slices = []

    for ax in range(3):
        if small_z and ax == 2:
            # Use full z range
            src_start = 0
            src_end = fulldims[2]
            dst_start = 0
            dst_end = fulldims[2]
        else:
            src_start = center[ax] - half[ax]
            src_end = center[ax] + half[ax]

            dst_start = 0
            dst_end = fdims[ax]

            # Clip source to volume bounds
            if src_start < 0:
                dst_start = -src_start
                src_start = 0
                border_flag = True
            if src_end > fulldims[ax]:
                dst_end = fdims[ax] - (src_end - fulldims[ax])
                src_end = fulldims[ax]
                border_flag = True

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    try:
        obstruction[dst_slices[0], dst_slices[1], dst_slices[2]] = \
            cell_volume[src_slices[0], src_slices[1], src_slices[2]]
    except (IndexError, ValueError):
        border_flag = True

    # Compute offsets for local->global conversion
    if not border_flag:
        if small_z:
            offsets = np.array([
                center[0] - half[0],
                center[1] - half[1],
                0
            ], dtype=int)
        else:
            offsets = np.array([
                center[0] - half[0],
                center[1] - half[1],
                center[2] - half[2],
            ], dtype=int)
    else:
        offsets = np.array([
            src_slices[0].start - dst_slices[0].start,
            src_slices[1].start - dst_slices[1].start,
            src_slices[2].start - dst_slices[2].start,
        ], dtype=int)

    return obstruction, root_local, border_flag, offsets


def _compute_fill_fraction(obstruction, dims, dimsSS):
    """Compute fraction of occupied voxels in each coarse grid cell.

    Args:
        obstruction: Fine-resolution obstruction volume.
        dims: Coarse grid dimensions (3,).
        dimsSS: Subsampling factor (3,).

    Returns:
        fillfrac: (dims) array of fill fractions in [0, 1].
    """
    dims = np.asarray(dims, dtype=int)
    dimsSS = np.asarray(dimsSS, dtype=int)
    binary = (obstruction > 0).astype(np.float32)

    # Reshape into blocks and sum
    # obstruction shape is (dims[0]*dimsSS[0], dims[1]*dimsSS[1], dims[2]*dimsSS[2])
    reshaped = binary.reshape(
        dims[0], dimsSS[0], dims[1], dimsSS[1], dims[2], dimsSS[2]
    )
    fillfrac = reshaped.sum(axis=(1, 3, 5)) / float(np.prod(dimsSS))
    return fillfrac


def _set_apical_path_cost_6dir(cost_volume_6dir, root, aproot, dims):
    """Set zero cost along the apical path in specific directions.

    Matches MATLAB exactly: sets cost to 0 only in the direction of
    travel along the L-shaped path from root to apical root. This creates
    a directional "highway" — cheap to travel along but normal cost to
    travel against or across.

    Args:
        cost_volume_6dir: (dims[0], dims[1], dims[2], 6) cost array, modified in-place.
                         Directions: +x(0), -x(1), +y(2), -y(3), +z(4), -z(5)
        root: (3,) coarse root position (0-based).
        aproot: (3,) coarse apical root position (0-based).
        dims: (3,) coarse grid dimensions.
    """
    r = np.array(root, dtype=int)
    a = np.array(aproot, dtype=int)

    # X segment: set direction-specific cost to 0
    if a[0] > r[0]:
        # Moving in +x direction (direction 0)
        cost_volume_6dir[r[0]:a[0]+1, r[1], r[2], 0] = 0
    elif a[0] < r[0]:
        # Moving in -x direction (direction 1)
        cost_volume_6dir[a[0]:r[0]+1, r[1], r[2], 1] = 0

    # Y segment (from aproot x position)
    if a[1] > r[1]:
        # Moving in +y direction (direction 2)
        cost_volume_6dir[a[0], r[1]:a[1]+1, r[2], 2] = 0
    elif a[1] < r[1]:
        # Moving in -y direction (direction 3)
        cost_volume_6dir[a[0], a[1]:r[1]+1, r[2], 3] = 0

    # Z segment (from aproot x,y position)
    if a[2] > r[2]:
        # Moving in +z direction (direction 4)
        cost_volume_6dir[a[0], a[1], r[2]:a[2]+1, 4] = 0
    elif a[2] < r[2]:
        # Moving in -z direction (direction 5)
        cost_volume_6dir[a[0], a[1], a[2]:r[2]+1, 5] = 0



def _generate_basal_endpoints(num_dt, root_local, dt_params, fdims, obstruction):
    """Generate basal dendrite endpoints.

    Args:
        num_dt: Number of basal dendrites.
        root_local: (3,) root position in local coords.
        dt_params: Dendritic tree parameters tuple.
        fdims: Fine grid dimensions.
        obstruction: Fine-resolution obstruction volume.

    Returns:
        ends: (num_dt, 3) array of endpoint positions (0-based).
    """
    fdims = np.asarray(fdims, dtype=int)
    ends = np.zeros((num_dt, 3), dtype=int)

    for i in range(num_dt):
        flag = True
        dist_sc = 1.0
        num_it = 0
        while flag and num_it < 100:
            theta = np.random.rand() * 2 * np.pi
            r = np.sqrt(np.random.rand()) * dt_params[1] * dist_sc
            end_pt = np.floor([
                r * np.cos(theta) + root_local[0],
                r * np.sin(theta) + root_local[1],
                2 * dt_params[2] * (np.random.rand() - 0.5) + root_local[2],
            ]).astype(int)

            # Clamp to volume bounds (0-based)
            end_pt = np.clip(end_pt, 0, fdims - 1)

            if obstruction[end_pt[0], end_pt[1], end_pt[2]] == 0:
                ends[i] = end_pt
                flag = False
            dist_sc *= 1.01
            num_it += 1

    return ends


def _generate_apical_endpoints(num_at, root_local, at_params, fdims,
                                obstruction, rot_ang=None):
    """Generate apical dendrite endpoints.

    Args:
        num_at: Number of apical dendrites.
        root_local: (3,) root position in local coords.
        at_params: Apical dendrite parameters tuple.
        fdims: Fine grid dimensions.
        obstruction: Fine-resolution obstruction volume.
        rot_ang: Optional (3,) rotation angles for this neuron.

    Returns:
        ends: (num_at, 3) array of endpoint positions (0-based).
        root_a: (3,) apical root position.
    """
    fdims = np.asarray(fdims, dtype=int)

    # Apical root: offset from soma root
    root_a = np.floor([
        root_local[0] + 2 * at_params[3] * (np.random.rand() - 0.5),
        root_local[1] + 2 * at_params[3] * (np.random.rand() - 0.5),
        at_params[2],
    ]).astype(int)

    # Apply rotation if available
    if rot_ang is not None:
        root_a2 = root_a + root_local[2] * np.sin(np.radians([
            rot_ang[1], -rot_ang[0], 0
        ]))
        root_a2 = root_a2.astype(int)
    else:
        root_a2 = root_a.copy()

    ends = np.zeros((num_at, 3), dtype=int)
    for i in range(num_at):
        flag = True
        dist_sc = 1.0
        num_it = 0
        while flag and num_it < 100:
            theta = np.random.rand() * 2 * np.pi
            r = np.sqrt(np.random.rand()) * at_params[1] * dist_sc
            end_pt = np.floor([
                r * np.cos(theta) + root_a2[0],
                r * np.sin(theta) + root_a2[1],
                2 * at_params[2] * (np.random.rand() - 0.5) + root_a2[2],
            ]).astype(int)

            end_pt = np.clip(end_pt, 0, fdims - 1)

            if obstruction[end_pt[0], end_pt[1], end_pt[2]] == 0:
                ends[i] = end_pt
                flag = False
            dist_sc *= 1.01
            num_it += 1

    return ends, root_a


def _get_dendrite_path(path_from, target, root):
    """Reconstruct dendrite path from target to root.

    Wrapper around reconstruct_path that handles edge cases.

    Args:
        path_from: (X, Y, Z, 3) path parent array.
        target: (3,) target coordinates.
        root: (3,) root coordinates.

    Returns:
        path: (N, 3) array of path coordinates, or empty array.
    """
    target = np.array(target, dtype=int)
    root = np.array(root, dtype=int)

    # If target equals root, return just the root
    if np.array_equal(target, root):
        return np.array([root], dtype=np.int32)

    # Check bounds
    shape = path_from.shape[:3]
    for ax in range(3):
        if target[ax] < 0 or target[ax] >= shape[ax]:
            return np.array([], dtype=np.int32).reshape(0, 3)

    path = reconstruct_path(path_from, tuple(target))
    if len(path) == 0:
        return np.array([], dtype=np.int32).reshape(0, 3)

    return path


def _compute_path_weights(path, dend_var):
    """Compute dendrite thickness weights along a path using curvature.

    Uses second-order finite differences to penalize sharp turns.

    Args:
        path: (N, 3) path coordinates.
        dend_var: Variance for dendrite size randomization.

    Returns:
        path_w: (N,) weight array.
    """
    n = len(path)
    if n <= 2:
        return np.ones(n, dtype=np.float32)

    dend_sz = max(0, np.random.normal(1, dend_var)) ** 2

    # Second-order differences (curvature proxy)
    d1 = np.diff(path.astype(np.float32), axis=0)
    d2 = np.diff(np.abs(d1), axis=0)
    curvature = np.sum(np.abs(d2), axis=1) / 2.0

    # Weight: reduce at high-curvature points
    path_w = np.zeros(n, dtype=np.float32)
    path_w[0] = 1.0
    path_w[1:-1] = 1.0 - (1.0 - 1.0 / np.sqrt(2)) * curvature
    path_w[-1] = 1.0

    path_w *= dend_sz
    return path_w


def _compute_apical_path_weights(path, dend_var):
    """Compute apical dendrite thickness weights (no squaring of dendSz).

    Args:
        path: (N, 3) path coordinates.
        dend_var: Variance for dendrite size randomization.

    Returns:
        path_w: (N,) weight array.
    """
    n = len(path)
    if n <= 2:
        return np.ones(n, dtype=np.float32)

    dend_sz = max(0, np.random.normal(1, dend_var))

    d1 = np.diff(path.astype(np.float32), axis=0)
    d2 = np.diff(np.abs(d1), axis=0)
    curvature = np.sum(np.abs(d2), axis=1) / 2.0

    path_w = np.zeros(n, dtype=np.float32)
    path_w[0] = 1.0
    path_w[1:-1] = 1.0 - (1.0 - 1.0 / np.sqrt(2)) * curvature
    path_w[-1] = 1.0

    path_w *= dend_sz
    return path_w


def _smooth_cell_body(allpaths, cell_body, fdims):
    """Smooth dendrite-soma connections using spline interpolation.

    Faithful translation of MATLAB smoothCellBody.m. Creates smooth
    transitions between dendrite paths and soma boundary using:
    1. Group dendrites by their soma connection point
    2. Cubic spline arcs from dendrite entry through offset to border
    3. Iterative morphological filling (>=4/6 neighbors -> fill)

    Args:
        allpaths: List of (N,3) path arrays (one per dendrite).
        cell_body: 1D array of linear indices for soma voxels (0-based).
        fdims: (3,) fine grid dimensions.

    Returns:
        smoothed: 1D array of linear indices for smoothed cell body region.
    """
    from scipy.interpolate import CubicSpline

    fdims = tuple(int(d) for d in fdims)
    n_paths = len(allpaths)

    if len(cell_body) == 0 or n_paths == 0:
        return cell_body.copy()

    cell_body_set = set(cell_body.tolist())

    # Step 1: Find where each dendrite path first enters the cell body
    # NOTE: Python paths are root→endpoint (reversed vs MATLAB endpoint→root).
    # Search from endpoint side backward to find the soma entry point.
    conn_idx_root = np.zeros((n_paths, 3), dtype=np.float64)
    empty_idxs = np.ones(n_paths, dtype=bool)

    for i in range(n_paths):
        path = allpaths[i]
        if path is None or len(path) == 0:
            continue
        for k in range(len(path) - 1, -1, -1):
            idx = int(np.ravel_multi_index(
                (path[k, 0], path[k, 1], path[k, 2]), fdims
            ))
            if idx in cell_body_set:
                conn_idx_root[i] = path[k]
                empty_idxs[i] = False
                break

    if np.all(empty_idxs):
        return cell_body.copy()

    # Step 2: Group dendrites by shared connection point (distance == 0)
    dx = conn_idx_root[:, 0:1] - conn_idx_root[:, 0:1].T
    dy = conn_idx_root[:, 1:2] - conn_idx_root[:, 1:2].T
    dz = conn_idx_root[:, 2:3] - conn_idx_root[:, 2:3].T
    dist_mat = np.sqrt(dx**2 + dy**2 + dz**2)
    dist_mat = (dist_mat == 0).astype(float)
    if np.any(empty_idxs):
        dist_mat[empty_idxs, :] = np.nan

    dend_groups = []
    for i in range(n_paths):
        if np.isnan(dist_mat[i, i]):
            continue
        group = np.where(dist_mat[i, :] > 0)[0].tolist()
        dist_mat[group, :] = np.nan
        dend_groups.append(group)

    if len(dend_groups) == 0:
        return cell_body.copy()

    # Step 3: For each group, find offset connection point and root
    offset_val = 2
    n_groups = len(dend_groups)
    conn_idx = np.zeros((n_groups, 3), dtype=np.float64)
    conn_roots = np.zeros((n_groups, 3), dtype=np.float64)

    for i, group in enumerate(dend_groups):
        path = allpaths[group[0]]
        if path is None or len(path) == 0:
            continue
        # Search from endpoint side (Python paths: root→endpoint)
        for k in range(len(path) - 1, -1, -1):
            idx = int(np.ravel_multi_index(
                (path[k, 0], path[k, 1], path[k, 2]), fdims
            ))
            if idx in cell_body_set:
                hit_idx = k
                back_offset = round(offset_val * np.sqrt(len(group)))
                # Offset TOWARD endpoint (higher index in root→endpoint path)
                offset_idx = min(len(path) - 1, hit_idx + back_offset)
                conn_idx[i] = path[offset_idx]
                conn_roots[i] = path[hit_idx]
                break

    # Step 4: Find cell body boundary voxels
    cb_coords = np.array(np.unravel_index(cell_body, fdims)).T
    cb_min = cb_coords.min(axis=0)
    cb_max = cb_coords.max(axis=0)

    cell_mat = np.zeros(fdims, dtype=bool)
    cell_mat.ravel()[cell_body] = True

    # Crop to bounding box
    s = tuple(slice(int(cb_min[ax]), int(cb_max[ax]) + 1) for ax in range(3))
    cell_crop = cell_mat[s].copy()

    # Find boundary: voxels with >0 and <6 neighbors inside soma
    cell_borders = cell_crop.copy()
    if all(d >= 3 for d in cell_crop.shape):
        cell_diff = (
            cell_crop[:-2, 1:-1, 1:-1].astype(np.int32) +
            cell_crop[2:,  1:-1, 1:-1].astype(np.int32) +
            cell_crop[1:-1, :-2, 1:-1].astype(np.int32) +
            cell_crop[1:-1, 2:,  1:-1].astype(np.int32) +
            cell_crop[1:-1, 1:-1, :-2].astype(np.int32) +
            cell_crop[1:-1, 1:-1, 2:].astype(np.int32)
        )
        cell_borders[1:-1, 1:-1, 1:-1] = (
            (cell_diff > 0) & (cell_diff < 6) &
            cell_borders[1:-1, 1:-1, 1:-1]
        )

    cell_borders_full = np.zeros(fdims, dtype=bool)
    cell_borders_full[s] = cell_borders

    borders_sub = np.array(np.where(cell_borders_full)).T.astype(np.float64)

    if len(borders_sub) == 0:
        return cell_body.copy()

    # Step 5: Spline interpolation + morphological filling per group
    test_dist = [0, 4, 10]
    numsamp = 20
    cell_processed = np.zeros(fdims, dtype=bool)

    for j_g in range(n_groups):
        group = dend_groups[j_g]
        dist_off = min(
            max(test_dist[1], round(offset_val * np.sqrt(len(group)))),
            test_dist[2]
        )

        # Find border voxels near this group's connection root
        border_dist = np.linalg.norm(
            conn_roots[j_g] - borders_sub, axis=1
        )
        test_idx = np.where(
            (border_dist < dist_off) & (border_dist > test_dist[0])
        )[0]

        test_sub_list = []
        for bi in test_idx:
            # 3 control points: connRoots -> connIdx -> border voxel
            pts = np.array([
                conn_roots[j_g],
                conn_idx[j_g],
                borders_sub[bi],
            ])

            # Chord-length parameterization
            seg_lens = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
            t_knots = np.concatenate([[0], np.cumsum(seg_lens)])

            if t_knots[-1] < 1e-10:
                continue

            try:
                t_eval = np.linspace(t_knots[0], t_knots[-1], numsamp)
                dpts = np.zeros((numsamp, 3))
                for dim in range(3):
                    cs = CubicSpline(
                        t_knots, pts[:, dim], bc_type='not-a-knot'
                    )
                    dpts[:, dim] = cs(t_eval)
                dpts = np.round(dpts).astype(int)
                test_sub_list.append(dpts)
            except Exception:
                continue

        if len(test_sub_list) == 0:
            continue

        test_sub = np.vstack(test_sub_list)
        # Clip to valid range [0, fdims-1]
        test_sub = np.clip(test_sub, 0, np.array(fdims) - 1)
        test_ind = np.ravel_multi_index(
            (test_sub[:, 0], test_sub[:, 1], test_sub[:, 2]), fdims
        )

        # Build initial mask: cell borders + cell body + spline points
        cell_bump = cell_borders_full.copy()
        cell_bump.ravel()[cell_body] = True
        cell_bump.ravel()[test_ind] = True

        # Bounding box for filling
        bump_where = np.where(cell_bump)
        bump_min = np.array([w.min() for w in bump_where])
        bump_max = np.array([w.max() for w in bump_where])

        # Iterative morphological filling: >=4/6 neighbors -> fill
        # (MATLAB stores logical: addition → OR; converges when no new fills)
        while True:
            s2 = tuple(
                slice(int(bump_min[ax]), int(bump_max[ax]) + 1)
                for ax in range(3)
            )
            crop2 = cell_bump[s2].copy()

            if any(d < 3 for d in crop2.shape):
                break

            diff2 = (
                crop2[:-2, 1:-1, 1:-1].astype(np.int32) +
                crop2[2:,  1:-1, 1:-1].astype(np.int32) +
                crop2[1:-1, :-2, 1:-1].astype(np.int32) +
                crop2[1:-1, 2:,  1:-1].astype(np.int32) +
                crop2[1:-1, 1:-1, :-2].astype(np.int32) +
                crop2[1:-1, 1:-1, 2:].astype(np.int32)
            )
            new_fill = (diff2 >= 4) & ~crop2[1:-1, 1:-1, 1:-1]
            if not new_fill.any():
                break
            crop2[1:-1, 1:-1, 1:-1] |= new_fill
            cell_bump[s2] = crop2

        cell_processed |= cell_bump

    output = np.where(cell_processed.ravel())[0].astype(np.int32)
    return output


def _dilate_dendrite_paths(path_vals, path_ids, neur_num, fulldims):
    """Dilate dendrite center-line paths by redistributing thickness.

    Each center-line voxel with thickness > 1 distributes excess thickness
    to nearby empty voxels (1 voxel per unit of thickness). New voxels must
    be 6-connected to existing voxels of the same neuron.

    Corresponds to MATLAB: dilateDendritePathAll.m

    Args:
        path_vals: 3D uint16 array of dendrite thickness values.
        path_ids: 3D uint16 array of neuron IDs at dendrite locations.
        neur_num: 3D uint16 array of current volume occupancy.
        fulldims: Tuple of volume dimensions.

    Returns:
        dilated_ids: 3D uint16 array of dilated neuron ID assignments.
    """
    fulldims = tuple(int(d) for d in fulldims)
    dilated_ids = neur_num.copy()

    # Working copy of thickness - obstruction blocks expansion
    paths = path_vals.astype(np.float32).copy()
    paths[neur_num > 0] = np.nan  # occupied voxels blocked
    # Restore dendrite centerline values (override NaN from neur_num)
    dend_mask = path_vals > 0
    paths[dend_mask] = path_vals[dend_mask].astype(np.float32)

    # Include all centerline voxels in the output (MATLAB's pathnums
    # already contains these, and the function returns pathnums directly)
    dilated_ids[dend_mask] = path_ids[dend_mask]

    pathnums = path_ids.copy()
    pdims = int(np.prod(fulldims))

    # 6-connected neighbor offsets (C-order linear index)
    # C-order strides: axis 0 -> d2*d3, axis 1 -> d3, axis 2 -> 1
    d1, d2, d3 = fulldims
    stride_x = d2 * d3
    stride_y = d3
    stride_z = 1
    shifts_6 = np.array([
        -stride_x, stride_x,  # ±x
        -stride_y, stride_y,  # ±y
        -stride_z, stride_z,  # ±z
    ], dtype=np.int64)

    # Precompute distance shells
    max_dist = 20
    r = np.arange(-max_dist, max_dist + 1)
    gx, gy, gz = np.meshgrid(r, r, r, indexing='ij')
    dist_sq = (gx**2 + gy**2 + gz**2).ravel()
    shell_size = 2 * max_dist + 1

    # Sort offsets by distance
    sorted_idx = np.argsort(dist_sq)
    sorted_dist = dist_sq[sorted_idx]
    # Find shell boundaries (groups of equal distance)
    dpos = np.where(np.diff(sorted_dist))[0]  # indices where distance changes

    # Precompute 3D offsets for each sorted entry
    all_offsets = np.column_stack([
        gx.ravel()[sorted_idx],
        gy.ravel()[sorted_idx],
        gz.ravel()[sorted_idx],
    ])

    paths_flat = paths.ravel()
    pathnums_flat = pathnums.ravel()

    # Main redistribution loop: iterate through distance shells
    idxs = np.flatnonzero(paths_flat > 1)
    shell_idx = 0

    while shell_idx < len(dpos) and len(idxs) > 0:
        # Get offset entries for this distance shell
        start = dpos[shell_idx] + 1 if shell_idx > 0 else 1  # skip d=0
        if shell_idx + 1 < len(dpos):
            end = dpos[shell_idx + 1] + 1
        else:
            end = len(sorted_idx)

        if sorted_dist[start] > max_dist * max_dist:
            break

        # 3D offsets for this shell
        shell_offsets = all_offsets[start:end]  # (K, 3)

        # Convert to C-order linear index offsets
        lin_offsets = (shell_offsets[:, 0] * stride_x
                       + shell_offsets[:, 1] * stride_y
                       + shell_offsets[:, 2] * stride_z).astype(np.int64)

        # Process each center voxel
        for j_idx in range(len(idxs)):
            center = idxs[j_idx]
            if paths_flat[center] <= 1:
                continue

            nid = pathnums_flat[center]
            if nid == 0:
                continue

            # Candidate neighbors at this distance
            pidxs = center + lin_offsets
            # Bounds check
            valid = (pidxs >= 0) & (pidxs < pdims)
            pidxs = pidxs[valid]
            # Must be empty
            pidxs = pidxs[paths_flat[pidxs] == 0]

            if len(pidxs) == 0:
                continue

            # Connectivity check: each candidate must have a 6-neighbor
            # belonging to the same neuron
            connected = np.zeros(len(pidxs), dtype=bool)
            for k in range(len(pidxs)):
                neighbors = pidxs[k] + shifts_6
                neighbors = neighbors[(neighbors >= 0) & (neighbors < pdims)]
                if np.any(pathnums_flat[neighbors] == nid):
                    connected[k] = True
            pidxs = pidxs[connected]

            if len(pidxs) == 0:
                continue

            # Redistribute: move thickness from center to random neighbors
            while paths_flat[center] > 1 and len(pidxs) > 0:
                ridx = np.random.randint(len(pidxs))
                pidx = pidxs[ridx]
                pidxs = np.delete(pidxs, ridx)

                paths_flat[center] -= 1
                paths_flat[pidx] = 1
                pathnums_flat[pidx] = nid
                dilated_ids.ravel()[pidx] = nid

        # Refresh candidates for next shell
        idxs = np.flatnonzero(paths_flat > 1)
        shell_idx += 1

    return dilated_ids


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def grow_neuron_dendrites(
    vol_params: VolumeParams,
    dend_params: Optional[DendParams] = None,
    neural_volume: Optional[NeuralVolumeResult] = None,
    vessel_mask: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    rotation_angles: Optional[List[np.ndarray]] = None,
    verbose: Optional[int] = None,
) -> DendriteResult:
    """
    Grow dendrites from neuron somas using two-level Dijkstra path planning.

    Uses coarse-grid Dijkstra for global routing followed by fine-grid
    Dijkstra for detailed path refinement. Generates both basal and apical
    dendrite branches, then dilates center-line paths to target thickness.

    Corresponds to MATLAB: growNeuronDendrites.m

    Args:
        vol_params: Volume parameters.
        dend_params: Dendrite parameters. Uses defaults if None.
        neural_volume: Result from generate_neural_volume (Step 3).
        vessel_mask: Optional 3D vessel mask (same shape as neur_soma).
        positions: (N, 3) neuron center positions in micrometers.
        rotation_angles: Optional list of per-neuron (3,) rotation angles.
        verbose: Verbosity override. If None, uses vol_params.verbose.

    Returns:
        DendriteResult containing neur_num, dendrite_ad, updated params and
        soma data.
    """
    if dend_params is None:
        dend_params = DendParams()

    vres = vol_params.vres
    vol_sz = np.array(vol_params.vol_sz)
    N_neur = vol_params.N_neur
    verbosity = verbose if verbose is not None else vol_params.verbose

    # Extract parameters (scale by vres as MATLAB does)
    dtParams = list(dend_params.dtParams)
    atParams = list(dend_params.atParams)
    dweight = dend_params.dweight
    bweight = dend_params.bweight
    thicknessScale = dend_params.thicknessScale
    dims = np.array(dend_params.dims, dtype=int)
    dimsSS = np.array(dend_params.dimsSS, dtype=int)
    rallexp = dend_params.rallexp

    # Scale parameters by vres (matching MATLAB lines 100-105)
    fulldims = (vol_sz * vres).astype(int)
    dims = np.minimum(dims, (vol_sz / dimsSS).astype(int))
    dims = (dims * vres).astype(int)
    dtParams[1] = dtParams[1] * vres  # horizontal radius
    dtParams[2] = dtParams[2] * vres  # vertical radius
    atParams[1] = atParams[1] * vres
    atParams[2] = atParams[2] * vres
    atParams[3] = atParams[3] * vres
    thicknessScale = thicknessScale * vres * vres

    fdims = (dims * dimsSS).astype(int)
    fdims = np.minimum(fdims, fulldims)

    dendVar = dend_params.dendVar if dend_params.dendVar is not None else 0.25

    if verbosity >= 1:
        print("Growing out dendrites...")
        if verbosity > 1:
            print(f"      Number of neurons: {N_neur}")
            print(f"      Dendrite tree branches: {int(dtParams[0])} "
                  f"(+/-{int(dtParams[4])})")

    # Initialize cell volume from neural volume result
    neur_soma = neural_volume.neur_soma
    gp_nuc = neural_volume.gp_nuc
    gp_soma_input = neural_volume.gp_soma

    # Build cellVolume: soma IDs + vessel obstruction
    cell_volume = neur_soma.astype(np.float32)

    # Add vessel mask as high-value obstruction
    if vessel_mask is not None:
        obstruction_id = N_neur + vol_params.N_den + vol_params.N_bg + 1
        # Handle vol_depth slicing if vessel_mask is larger
        if vessel_mask.shape[2] > fulldims[2]:
            vol_depth = vol_params.vol_depth * vres
            vessel_slice = vessel_mask[:, :, vol_depth:vol_depth + fulldims[2]]
        else:
            vessel_slice = vessel_mask
        cell_volume += obstruction_id * vessel_slice.astype(np.float32)

    # Fill nucleus voxels with neuron ID
    for kk in range(N_neur):
        nuc_idx = gp_nuc[kk][0]
        if len(nuc_idx) > 0:
            cell_volume.ravel()[nuc_idx] = kk + 1

    neur_num = neur_soma.copy()

    # Initialize tracking volumes
    cell_volume_idx = np.zeros(tuple(fulldims), dtype=np.float32)
    cell_volume_val = np.zeros(tuple(fulldims), dtype=np.float32)
    cell_volume_ad = np.zeros(tuple(fulldims), dtype=bool)

    # Neuron roots in voxel coordinates (0-based)
    # MATLAB: ceil(max(vres*neur_locs, 1e-4)) gives 1-based coords
    # Python: subtract 1 for 0-based indexing
    allroots = np.maximum(np.ceil(vres * positions).astype(int) - 1, 0)
    # Clip to volume bounds
    for ax in range(3):
        allroots[:, ax] = np.clip(allroots[:, ax], 0, fulldims[ax] - 1)

    small_z = fulldims[2] <= fdims[2]

    # Per-neuron soma output
    gp_soma_out = []
    for kk in range(N_neur):
        gp_soma_out.append((gp_soma_input[kk], np.array([], dtype=np.int32)))

    # --- Grow dendrites for each neuron ---
    for j in range(N_neur):
        if verbosity > 1:
            print(f"    Processing neuron {j+1}/{N_neur}...")

        # Find apical root: voxel with minimum linear index in soma
        soma_idx = gp_soma_input[j]
        if len(soma_idx) == 0:
            continue
        aproot_flat = np.min(soma_idx)
        aproot = np.array(np.unravel_index(aproot_flat, tuple(fulldims)))

        # Number of basal dendrites (with random variation)
        numdt = max(1, int(dtParams[0] + round(dtParams[4] * np.random.randn())))

        # --- Extract local subvolume ---
        obstruction, root_local, border_flag, offsets = _extract_subvolume(
            cell_volume, allroots[j], fdims, fulldims, small_z
        )

        # Find and clear own cell body from obstruction
        cell_body = np.flatnonzero(obstruction == (j + 1))
        obstruction.ravel()[cell_body] = 0

        # --- Coarse grid: build 6-directional cost and run Dijkstra ---
        # Floor division: MATLAB uses ceil() with 1-based indices,
        # equivalent to floor division with 0-based indices.
        root_coarse = (root_local // dimsSS).astype(int)
        root_coarse = np.clip(root_coarse, 0, dims - 1)

        # Coarse cost volume: 6 independent random costs per direction
        # Directions: +x(0), -x(1), +y(2), -y(3), +z(4), -z(5)
        cost_6dir = (1 + dweight * np.random.rand(
            *dims, 6
        )).astype(np.float32)

        # Apical root in coarse coordinates
        aproot_local = root_local + (aproot - allroots[j])
        aproot_coarse = root_coarse + np.round(
            (aproot - allroots[j]) / dimsSS
        ).astype(int)
        aproot_coarse = np.clip(aproot_coarse, 0, dims - 1)

        # Set directional apical path cost (zero cost only in travel direction)
        _set_apical_path_cost_6dir(cost_6dir, root_coarse, aproot_coarse, dims)

        # Directional boundary inf (prevents linear-index wrap-around)
        cost_6dir[0, :, :, 0] = np.inf      # +x at x=0
        cost_6dir[-1, :, :, 1] = np.inf     # -x at x=end
        cost_6dir[:, 0, :, 2] = np.inf      # +y at y=0
        cost_6dir[:, -1, :, 3] = np.inf     # -y at y=end
        cost_6dir[:, :, 0, 4] = np.inf      # +z at z=0
        cost_6dir[:, :, -1, 5] = np.inf     # -z at z=end

        # Add obstruction penalty (same for all 6 directions)
        try:
            fillfrac = _compute_fill_fraction(obstruction, dims, dimsSS)
            penalty = -bweight * np.log(
                np.maximum(1e-10, 1 - 2 * np.maximum(0, fillfrac - 0.5))
            )
            cost_6dir += penalty[..., np.newaxis]
        except ValueError:
            pass  # Shape mismatch edge case at volume borders

        # Reshape to (prod(dims), 6) with Fortran-order spatial flattening.
        # F-order spatial flatten == transpose spatial axes to (z, y, x) then
        # C-order reshape; one contiguous copy instead of six strided ones.
        n_coarse = int(np.prod(dims))
        cost_flat = np.ascontiguousarray(
            cost_6dir.transpose(2, 1, 0, 3)
        ).reshape(n_coarse, 6)

        # Run coarse Dijkstra (6-directional)
        _, path_from_c = dendrite_dijkstra_6dir(
            cost_flat,
            tuple(dims),
            tuple(root_coarse),
            use_numba=True,
        )

        # --- Generate dendrite endpoints ---
        ends_basal = _generate_basal_endpoints(
            numdt, root_local, dtParams, fdims, obstruction
        )

        num_at = int(atParams[0])
        rot_ang = None
        if rotation_angles is not None and j < len(rotation_angles):
            rot_ang = rotation_angles[j]
        ends_apical, root_a = _generate_apical_endpoints(
            num_at, root_local, atParams, fdims, obstruction, rot_ang
        )

        all_ends = np.vstack([ends_basal, ends_apical]) if num_at > 0 else ends_basal
        all_ends_local = all_ends.copy()
        nends = numdt + num_at

        # Convert to coarse coords for path retrieval
        ends_coarse = (all_ends.astype(int) // dimsSS).astype(int)
        ends_coarse = np.clip(ends_coarse, 0, dims - 1)

        # --- Retrieve coarse paths ---
        paths_coarse = np.zeros(tuple(dims), dtype=bool)
        for i in range(nends):
            path = _get_dendrite_path(
                path_from_c, ends_coarse[i], root_coarse
            )
            if len(path) > 0:
                valid = np.all((path >= 0) & (path < dims), axis=1)
                path = path[valid]
                if len(path) > 0:
                    idx = np.ravel_multi_index(
                        (path[:, 0], path[:, 1], path[:, 2]),
                        tuple(dims)
                    )
                    paths_coarse.ravel()[idx] = True

        # --- Fine grid: build 6-directional cost from coarse paths ---
        # Ensure root coarse cell is always in the path
        paths_coarse[tuple(root_coarse)] = True

        cost_fine_6dir = np.full((*tuple(fdims), 6), np.inf, dtype=np.float32)

        den_locs = np.flatnonzero(paths_coarse)
        for loc in den_locs:
            lx, ly, lz = np.unravel_index(loc, tuple(dims))
            # Fine grid region for this coarse cell
            sx = slice(lx * dimsSS[0], min((lx + 1) * dimsSS[0], fdims[0]))
            sy = slice(ly * dimsSS[1], min((ly + 1) * dimsSS[1], fdims[1]))
            sz = slice(lz * dimsSS[2], min((lz + 1) * dimsSS[2], fdims[2]))

            # 6-directional random cost
            cell_shape = (
                sx.stop - sx.start, sy.stop - sy.start, sz.stop - sz.start
            )
            local_cost_6 = (1 + dweight * np.random.rand(
                *cell_shape, 6
            )).astype(np.float32)

            # Add obstruction (inf where obstacles, same for all directions)
            obs_region = obstruction[sx, sy, sz]
            # MATLAB: filled = obstruction*inf; filled(isnan(filled))=0;
            # Then temp = temp + filled (broadcast to 6 dirs)
            obs_mask = obs_region > 0
            local_cost_6[obs_mask] = np.inf

            cost_fine_6dir[sx, sy, sz, :] = local_cost_6

        # Fine root: center of the coarse root cell
        root_fine = (root_coarse * dimsSS + dimsSS // 2).astype(int)
        root_fine = np.clip(root_fine, 0, np.array(fdims) - 1)
        # Ensure root_fine cost is not inf
        if np.any(np.isinf(cost_fine_6dir[tuple(root_fine)])):
            cost_fine_6dir[tuple(root_fine)] = 1.0

        aproot_fine = root_fine + (aproot - allroots[j])
        aproot_fine = np.clip(aproot_fine, 0, np.array(fdims) - 1)
        _set_apical_path_cost_6dir(
            cost_fine_6dir, root_fine, aproot_fine, fdims
        )

        # Directional boundary inf
        cost_fine_6dir[0, :, :, 0] = np.inf
        cost_fine_6dir[-1, :, :, 1] = np.inf
        cost_fine_6dir[:, 0, :, 2] = np.inf
        cost_fine_6dir[:, -1, :, 3] = np.inf
        cost_fine_6dir[:, :, 0, 4] = np.inf
        cost_fine_6dir[:, :, -1, 5] = np.inf

        # Reshape to (prod(fdims), 6) with Fortran-order spatial flattening
        # (transpose spatial axes + single contiguous copy; see coarse grid).
        n_fine = int(np.prod(fdims))
        cost_fine_flat = np.ascontiguousarray(
            cost_fine_6dir.transpose(2, 1, 0, 3)
        ).reshape(n_fine, 6)

        # Run fine Dijkstra (6-directional)
        _, path_from_f = dendrite_dijkstra_6dir(
            cost_fine_flat,
            tuple(fdims),
            tuple(root_fine),
            use_numba=True,
        )

        # --- Retrieve fine paths and compute thickness ---
        fine_paths_idx = np.zeros(tuple(fdims), dtype=np.float32)
        fine_idxs = []
        allpaths = [None] * nends

        # Basal dendrites
        for i in range(numdt):
            path = _get_dendrite_path(
                path_from_f, all_ends_local[i], root_fine
            )
            if len(path) == 0:
                allpaths[i] = np.array([], dtype=np.int32).reshape(0, 3)
                continue

            # Pad short paths (MATLAB lines 350-351)
            if path.shape[0] == 1:
                path = np.vstack([path[0], path[0], path])
            elif path.shape[0] == 2:
                path = np.vstack([path[0], path])

            allpaths[i] = path
            path_w = _compute_path_weights(path, dendVar)

            # Accumulate thickness at path voxels
            valid = np.all((path >= 0) & (path < fdims), axis=1)
            path = path[valid]
            path_w = path_w[valid[:len(path_w)]] if len(path_w) > len(valid) else path_w[valid]
            if len(path) > 0:
                idx = np.ravel_multi_index(
                    (path[:, 0], path[:, 1], path[:, 2]), tuple(fdims)
                )
                fine_paths_idx.ravel()[idx] += path_w
                fine_idxs.extend(idx.tolist())

        # Apply Rall's law scaling to basal dendrites
        fine_idxs_unique = np.unique(fine_idxs).astype(int)
        if len(fine_idxs_unique) > 0:
            fine_paths_idx.ravel()[fine_idxs_unique] = (
                thicknessScale * dtParams[3]
                * (fine_paths_idx.ravel()[fine_idxs_unique] ** (1.0 / rallexp))
            )

        # Apical dendrites
        fine_paths_val = np.zeros(tuple(fdims), dtype=np.float32)
        fine_idxs2 = []

        for i in range(num_at):
            path = _get_dendrite_path(
                path_from_f, all_ends_local[numdt + i], root_fine
            )
            if len(path) == 0:
                allpaths[numdt + i] = np.array([], dtype=np.int32).reshape(0, 3)
                continue

            allpaths[numdt + i] = path
            path_w = _compute_apical_path_weights(path, dendVar)

            valid = np.all((path >= 0) & (path < fdims), axis=1)
            path = path[valid]
            path_w = path_w[valid[:len(path_w)]] if len(path_w) > len(valid) else path_w[valid]
            if len(path) > 0:
                idx = np.ravel_multi_index(
                    (path[:, 0], path[:, 1], path[:, 2]), tuple(fdims)
                )
                fine_paths_val.ravel()[idx] += path_w
                fine_idxs2.extend(idx.tolist())

        # Apply Rall's law to apical dendrites
        fine_idxs2_unique = np.unique(fine_idxs2).astype(int)
        fine_paths_ad = np.zeros(tuple(fdims), dtype=bool)
        if len(fine_idxs2_unique) > 0:
            fine_paths_ad.ravel()[fine_idxs2_unique] = True
            fine_paths_val.ravel()[fine_idxs2_unique] = (
                thicknessScale * atParams[4]
                * (fine_paths_val.ravel()[fine_idxs2_unique] ** (1.0 / rallexp))
            )

        # Combine basal and apical thickness
        if len(fine_idxs_unique) > 0:
            fine_paths_val.ravel()[fine_idxs_unique] += \
                fine_paths_idx.ravel()[fine_idxs_unique]

        # --- Smooth cell body ---
        root_fine_idx = np.ravel_multi_index(tuple(root_fine), tuple(fdims))
        cell_body = np.append(cell_body, root_fine_idx)

        valid_paths = [p for p in allpaths if p is not None and len(p) > 0]
        if len(valid_paths) > 0:
            cell_body_smoothed = _smooth_cell_body(valid_paths, cell_body, fdims)
            # Only keep newly added voxels
            cell_body_set = set(cell_body.tolist())
            new_voxels = np.array(
                [v for v in cell_body_smoothed if v not in cell_body_set],
                dtype=np.int32,
            )
        else:
            new_voxels = np.array([], dtype=np.int32)

        # Combine all fine indices
        all_fine = fine_idxs_unique.tolist() + fine_idxs2_unique.tolist() + \
            new_voxels.tolist()
        fine_idxs3 = np.unique(all_fine).astype(int)

        # Create local neuron ID volume
        fine_paths_neuron_id = np.zeros(tuple(fdims), dtype=np.float32)
        if len(fine_idxs3) > 0:
            fine_paths_neuron_id.ravel()[fine_idxs3] = j + 1
        if len(new_voxels) > 0:
            fine_paths_neuron_id.ravel()[new_voxels] = j + 1
            fine_paths_val.ravel()[new_voxels] += 1

        # Clear cell body interior
        fine_paths_neuron_id.ravel()[cell_body] = 0
        fine_paths_val.ravel()[cell_body] = 0
        fine_paths_ad.ravel()[cell_body] = False

        # --- Write back to global volume ---
        if len(fine_idxs3) == 0:
            gp_soma_out[j] = (gp_soma_input[j], new_voxels)
            continue

        # Convert local fine coords to global coords
        local_coords = np.array(np.unravel_index(fine_idxs3, tuple(fdims))).T
        global_coords = local_coords + offsets

        # Filter to valid global bounds
        valid_mask = np.ones(len(global_coords), dtype=bool)
        for ax in range(3):
            valid_mask &= (global_coords[:, ax] >= 0)
            valid_mask &= (global_coords[:, ax] < fulldims[ax])

        if np.any(valid_mask):
            gc = global_coords[valid_mask]
            fi = fine_idxs3[valid_mask]
            global_flat = np.ravel_multi_index(
                (gc[:, 0], gc[:, 1], gc[:, 2]), tuple(fulldims)
            )

            cell_volume.ravel()[global_flat] += fine_paths_neuron_id.ravel()[fi]
            cell_volume_idx.ravel()[global_flat] += fine_paths_neuron_id.ravel()[fi]
            cell_volume_val.ravel()[global_flat] += fine_paths_val.ravel()[fi]
            cell_volume_ad.ravel()[global_flat] |= fine_paths_ad.ravel()[fi]

        gp_soma_out[j] = (gp_soma_input[j], new_voxels)

        if verbosity > 1:
            n_valid = sum(1 for p in allpaths if p is not None and len(p) > 0)
            print(f"        Neuron {j+1}: {n_valid}/{nends} paths")

    # --- Post-processing: dilate and finalize ---
    if verbosity >= 1:
        print("Dilating dendrite paths...")

    # Stochastic rounding of thickness values
    cell_volume_val_uint = np.floor(cell_volume_val).astype(np.uint16)
    frac = cell_volume_val - np.floor(cell_volume_val)
    cell_volume_val_uint += (frac > np.random.rand(*frac.shape)).astype(np.uint16)

    cell_volume_idx_uint = cell_volume_idx.astype(np.uint16)
    cell_volume_ad_uint = cell_volume_ad.astype(np.uint16)

    # Dilate apical dendrites
    ad_vals = cell_volume_val_uint * cell_volume_ad_uint
    ad_ids = cell_volume_idx_uint * cell_volume_ad_uint
    dendnum_ad = _dilate_dendrite_paths(ad_vals, ad_ids, neur_num, fulldims)

    # Dilate basal dendrites
    bd_mask = (~cell_volume_ad).astype(np.uint16)
    bd_vals = cell_volume_val_uint * bd_mask
    bd_ids = cell_volume_idx_uint * bd_mask
    dendnum_bd = _dilate_dendrite_paths(bd_vals, bd_ids, neur_num, fulldims)

    # Remove nucleus and restore soma in both maps
    for kk in range(N_neur):
        nuc_idx = gp_nuc[kk][0]
        soma_idx = gp_soma_input[kk]
        if len(nuc_idx) > 0:
            dendnum_ad.ravel()[nuc_idx] = 0
            dendnum_bd.ravel()[nuc_idx] = 0
        if len(soma_idx) > 0:
            dendnum_ad.ravel()[soma_idx] = 0
            dendnum_bd.ravel()[soma_idx] = 0

    # Merge: apical overrides basal
    dendnum_bd[dendnum_ad > 0] = dendnum_ad[dendnum_ad > 0]
    neur_num[dendnum_bd > 0] = dendnum_bd[dendnum_bd > 0]

    # Final cleanup: ensure nuclei cleared, somas restored
    for kk in range(N_neur):
        nuc_idx = gp_nuc[kk][0]
        soma_idx = gp_soma_input[kk]
        if len(nuc_idx) > 0:
            neur_num.ravel()[nuc_idx] = 0
        if len(soma_idx) > 0:
            neur_num.ravel()[soma_idx] = np.uint16(kk + 1)

    if verbosity >= 1:
        total_dend = int(np.sum((neur_num > 0) & (neur_soma == 0)))
        print(f"done. Total dendrite voxels: {total_dend}")

    return DendriteResult(
        neur_num=neur_num,
        dendrite_ad=dendnum_ad,
        dend_params=dend_params,
        gp_soma=gp_soma_out,
    )


# ---------------------------------------------------------------------------
# Step 5: Through-volume apical dendrites
# ---------------------------------------------------------------------------


def grow_apical_dendrites(
    vol_params: VolumeParams,
    dend_params: Optional[DendParams] = None,
    dend_result: Optional[DendriteResult] = None,
    neural_volume: Optional[NeuralVolumeResult] = None,
    verbose: Optional[int] = None,
) -> ApicalDendriteResult:
    """Grow through-volume apical dendrites (Step 5).

    Generates apical dendrites that traverse the full volume depth,
    independent of specific neuron somas. These represent Layer 5
    pyramidal neuron apical tufts passing through the imaging field.

    Corresponds to MATLAB: growApicalDendrites.m

    Args:
        vol_params: Volume parameters (N_den, vol_sz, vres, N_neur).
        dend_params: Dendrite parameters. Uses defaults if None.
        dend_result: Output from Step 4 (grow_neuron_dendrites).
        neural_volume: Output from Step 3 (generate_neural_volume).
        verbose: Verbosity override.

    Returns:
        ApicalDendriteResult with updated neur_num, neur_num_ad, and params.
    """
    if dend_params is None:
        dend_params = DendParams()

    vres = vol_params.vres
    vol_sz = np.array(vol_params.vol_sz)
    N_neur = vol_params.N_neur
    N_den = vol_params.N_den
    verbosity = verbose if verbose is not None else vol_params.verbose

    # Early return if no through-volume dendrites
    if N_den == 0:
        if verbosity >= 1:
            print("No through-volume apical dendrites to grow (N_den=0).")
        return ApicalDendriteResult(
            neur_num=dend_result.neur_num.copy(),
            neur_num_ad=dend_result.dendrite_ad.copy(),
            dend_params=dend_params,
        )

    # --- Parameter setup (MATLAB lines 91-108) ---
    atParams = list(dend_params.atParams2)
    dweight = dend_params.dweight
    bweight = dend_params.bweight
    thicknessScale = dend_params.thicknessScale
    dims = np.array(dend_params.dims, dtype=int)
    dimsSS = np.array(dend_params.dimsSS, dtype=int)
    rallexp = dend_params.rallexp

    fulldims = (vol_sz * vres).astype(int)
    dims = np.minimum(dims, (vol_sz / dimsSS).astype(int))
    dims = (dims * vres).astype(int)
    atParams[1] *= vres   # xy_radius
    atParams[2] *= vres   # z_radius
    atParams[3] *= vres   # offset
    thicknessScale *= vres * vres

    fdims = (dims * dimsSS).astype(int)
    fdims = np.minimum(fdims, fulldims)
    # Through-volume dendrites must span full z
    fdims[2] = fulldims[2]

    # dendVar: prefer apicalVar, then dendVar, default 0.35 (MATLAB lines 129-137)
    if dend_params.apicalVar is not None:
        dendVar = dend_params.apicalVar
    elif dend_params.dendVar is not None:
        dendVar = dend_params.dendVar
    else:
        dendVar = 0.35

    if verbosity >= 1:
        print("Growing out apical dendrites...")
        if verbosity > 1:
            print(f"      Number of through-volume dendrites: {N_den}")
            print(f"      Apical dendrite radius (xy): "
                  f"{atParams[1]/vres:.1f} microns")

    # --- Cell volume initialization (MATLAB lines 110-113) ---
    cell_volume = dend_result.neur_num.astype(np.float32).copy()
    gp_nuc = neural_volume.gp_nuc
    gp_soma = dend_result.gp_soma

    for kk in range(N_neur):
        nuc_idx = gp_nuc[kk][0]
        if len(nuc_idx) > 0:
            cell_volume.ravel()[nuc_idx] = float(kk + 1)

    # --- Root selection (MATLAB lines 116-123) ---
    # Roots at bottom of volume (z=fulldims[2]-1), random XY in empty space
    root_den = np.zeros((N_den, 3), dtype=int)
    for j in range(N_den):
        attempts = 0
        while root_den[j, 2] == 0 and attempts < 10000:
            # MATLAB: ceil(fulldims(1:2).*rand(1,2)) -> 1-based
            root_xy = np.ceil(fulldims[:2] * np.random.rand(2)).astype(int) - 1
            root_xy = np.clip(root_xy, 0, fulldims[:2] - 1)
            # MATLAB checks z=1 (top of volume, 1-based) -> Python z=0
            if cell_volume[root_xy[0], root_xy[1], 0] == 0:
                root_den[j] = [root_xy[0], root_xy[1], fulldims[2] - 1]
            attempts += 1

    # --- Global tracking volumes ---
    cell_volume_idx = np.zeros(tuple(fulldims), dtype=np.float32)
    cell_volume_val = np.zeros(tuple(fulldims), dtype=np.float32)

    # Pre-allocate fine cost matrix (reused across iterations)
    ML = np.full((*tuple(fdims), 6), np.inf, dtype=np.float32)

    # --- Main loop: grow each through-volume dendrite (MATLAB lines 147-293) ---
    for j in range(N_den):
        # 3a. Extract subvolume centered on root_den[j] in XY, full z
        center = np.array([root_den[j, 0], root_den[j, 1], 0], dtype=int)
        obstruction, root_local, border_flag, offsets = _extract_subvolume(
            cell_volume, center, fdims, fulldims, small_z=True
        )
        # root_local from _extract_subvolume has z=center[2]=0, but we need z=bottom
        root_local = np.array([fdims[0] // 2, fdims[1] // 2, fdims[2] - 1],
                              dtype=int)

        # Cell body clearing (defensive, usually no-op on first pass)
        # MATLAB: cellBody = (obstruction == j+N_neur)
        cell_body = (obstruction == (j + 1 + N_neur))
        obstruction[cell_body] = 0

        # 3b. Coarse Dijkstra (MATLAB lines 168-177)
        root_coarse = (root_local // dimsSS).astype(int)
        root_coarse = np.clip(root_coarse, 0, dims - 1)

        # 6-directional random cost
        cost_6dir = (1 + dweight * np.random.rand(
            dims[0], dims[1], dims[2], 6
        )).astype(np.float32)

        # Boundary inf
        cost_6dir[0, :, :, 0] = np.inf
        cost_6dir[-1, :, :, 1] = np.inf
        cost_6dir[:, 0, :, 2] = np.inf
        cost_6dir[:, -1, :, 3] = np.inf
        cost_6dir[:, :, 0, 4] = np.inf
        cost_6dir[:, :, -1, 5] = np.inf

        # Fill fraction penalty (MATLAB lines 172-174)
        fillfrac = _compute_fill_fraction(obstruction, dims, dimsSS)
        penalty = -bweight * np.log(
            np.maximum(1e-10, 1 - 2 * np.maximum(0, fillfrac - 0.5))
        )
        cost_6dir += penalty[..., np.newaxis]

        # Flatten with Fortran-order and run Dijkstra (transpose spatial axes
        # + single contiguous copy; see grow_neuron_dendrites coarse grid).
        n_coarse = int(np.prod(dims))
        cost_flat = np.ascontiguousarray(
            cost_6dir.transpose(2, 1, 0, 3)
        ).reshape(n_coarse, 6)

        _, path_from_c = dendrite_dijkstra_6dir(
            cost_flat, tuple(dims), tuple(root_coarse)
        )

        # 3c. Endpoint generation (MATLAB lines 180-206)
        num_endpoints = int(atParams[0])
        endsA = np.zeros((num_endpoints, 3), dtype=int)

        # Offset root (MATLAB line 181)
        rootA = np.floor([
            root_local[0] + 2 * atParams[3] * (np.random.rand() - 0.5),
            root_local[1] + 2 * atParams[3] * (np.random.rand() - 0.5),
            fdims[2] - 1  # bottom
        ]).astype(int)

        for i in range(num_endpoints):
            flag = True
            dist_sc = 1.0
            num_it = 0
            while flag and num_it < 100:
                theta = np.random.rand() * 2 * np.pi
                r = np.sqrt(np.random.rand()) * atParams[1] * dist_sc
                endA = np.floor([
                    r * np.cos(theta) + rootA[0],
                    r * np.sin(theta) + rootA[1],
                    0  # top of volume (MATLAB z=1, Python z=0)
                ]).astype(int)
                endA = np.clip(endA, 0, np.array(fdims) - 1)
                if obstruction[endA[0], endA[1], endA[2]] == 0:
                    endsA[i] = endA
                    flag = False
                dist_sc *= 1.01
                num_it += 1

        # Coarse endpoints
        endsAC = (endsA // dimsSS).astype(int)
        endsAC = np.clip(endsAC, 0, dims - 1)

        # 3d. Coarse path retrieval (MATLAB lines 208-212)
        paths_coarse = np.zeros(tuple(dims), dtype=np.float32)
        for i in range(num_endpoints):
            path = _get_dendrite_path(path_from_c, endsAC[i], root_coarse)
            if len(path) > 0:
                valid = np.all((path >= 0) & (path < dims), axis=1)
                path = path[valid]
                if len(path) > 0:
                    idx = np.ravel_multi_index(
                        (path[:, 0], path[:, 1], path[:, 2]), tuple(dims)
                    )
                    paths_coarse.ravel()[idx] += 1

        # 3e. Fine Dijkstra refinement (MATLAB lines 214-230)
        ML[:] = np.inf

        # Fine root: center of coarse root cell
        rootL_fine = (root_coarse * dimsSS + dimsSS // 2).astype(int)
        rootL_fine = np.clip(rootL_fine, 0, np.array(fdims) - 1)

        # Fill coarse path cells with random costs
        den_locs = np.flatnonzero(paths_coarse)
        for loc in den_locs:
            lx, ly, lz = np.unravel_index(loc, tuple(dims))
            sx = slice(lx * dimsSS[0], min((lx + 1) * dimsSS[0], fdims[0]))
            sy = slice(ly * dimsSS[1], min((ly + 1) * dimsSS[1], fdims[1]))
            sz = slice(lz * dimsSS[2], min((lz + 1) * dimsSS[2], fdims[2]))
            cell_shape = (
                sx.stop - sx.start, sy.stop - sy.start, sz.stop - sz.start
            )
            ML[sx, sy, sz, :] = (1 + dweight * np.random.rand(
                *cell_shape, 6
            )).astype(np.float32)

        # Boundary inf
        ML[0, :, :, 0] = np.inf
        ML[-1, :, :, 1] = np.inf
        ML[:, 0, :, 2] = np.inf
        ML[:, -1, :, 3] = np.inf
        ML[:, :, 0, 4] = np.inf
        ML[:, :, -1, 5] = np.inf

        # Obstruction: use boolean mask (avoid 0*inf=NaN)
        obs_mask = obstruction > 0
        for di in range(6):
            ML[:, :, :, di][obs_mask] = np.inf

        # Ensure root cost is not inf
        if np.any(np.isinf(ML[tuple(rootL_fine)])):
            ML[tuple(rootL_fine)] = 1.0

        # Flatten and run fine Dijkstra (transpose spatial axes + single
        # contiguous copy; see grow_neuron_dendrites coarse grid).
        n_fine = int(np.prod(fdims))
        cost_fine_flat = np.ascontiguousarray(
            ML.transpose(2, 1, 0, 3)
        ).reshape(n_fine, 6)

        _, path_from_f = dendrite_dijkstra_6dir(
            cost_fine_flat, tuple(fdims), tuple(rootL_fine), use_numba=True
        )

        # 3f. Fine path weights (MATLAB lines 232-244)
        finepathsVal = np.zeros(tuple(fdims), dtype=np.float32)
        for i in range(num_endpoints):
            path = _get_dendrite_path(path_from_f, endsA[i], rootL_fine)
            if len(path) == 0:
                continue

            path_w = _compute_apical_path_weights(path, dendVar)

            valid = np.all((path >= 0) & (path < fdims), axis=1)
            path = path[valid]
            path_w = path_w[valid[:len(path_w)]] if len(path_w) > len(valid) \
                else path_w[valid]
            if len(path) > 0:
                idx = np.ravel_multi_index(
                    (path[:, 0], path[:, 1], path[:, 2]), tuple(fdims)
                )
                finepathsVal.ravel()[idx] += path_w

        # Apply Rall's law (MATLAB line 242)
        mask = finepathsVal > 0
        finepathsVal[mask] = (
            thicknessScale * atParams[4]
            * (finepathsVal[mask] ** (1.0 / rallexp))
        )

        # Clear cell body (MATLAB line 243)
        finepathsVal[cell_body] = 0

        # Dendrite ID: j+1+N_neur (MATLAB: j+N_neur with 1-based j)
        dendrite_id = float(j + 1 + N_neur)
        finepathsIdx = dendrite_id * (finepathsVal > 0).astype(np.float32)

        # 3g. Write back to global volume (MATLAB lines 248-283)
        local_indices = np.flatnonzero(finepathsIdx > 0)
        if len(local_indices) > 0:
            local_coords = np.array(
                np.unravel_index(local_indices, tuple(fdims))
            ).T
            global_coords = local_coords + offsets

            valid_mask = np.ones(len(global_coords), dtype=bool)
            for ax in range(3):
                valid_mask &= (global_coords[:, ax] >= 0)
                valid_mask &= (global_coords[:, ax] < fulldims[ax])

            if np.any(valid_mask):
                gc = global_coords[valid_mask]
                li = local_indices[valid_mask]
                gf = np.ravel_multi_index(
                    (gc[:, 0], gc[:, 1], gc[:, 2]), tuple(fulldims)
                )
                cell_volume.ravel()[gf] += finepathsIdx.ravel()[li]
                cell_volume_val.ravel()[gf] += finepathsVal.ravel()[li]
                cell_volume_idx.ravel()[gf] += finepathsIdx.ravel()[li]

        if verbosity > 1:
            n_voxels = int(np.sum(finepathsIdx > 0))
            print(f"        Apical dendrite {j+1}: {n_voxels} path voxels")

    # --- Post-processing: dilation (MATLAB lines 301-304) ---
    if verbosity >= 1:
        print("Dilating apical dendrite paths...")

    # Stochastic rounding (matching Step 4 pattern)
    cell_volume_val = np.ceil(cell_volume_val).astype(np.float32)
    cell_volume_val_uint = cell_volume_val.astype(np.uint16)
    cell_volume_idx_uint = cell_volume_idx.astype(np.uint16)

    dendnum = _dilate_dendrite_paths(
        cell_volume_val_uint, cell_volume_idx_uint,
        dend_result.neur_num, fulldims
    )

    # --- Merge and cleanup (MATLAB lines 307-324) ---
    # MATLAB: neur_num += dendnum; cellVolumeAD += dendnum
    # In MATLAB, dendnum is a delta (0 at existing neuron voxels, through-vol
    # IDs at new voxels).  In Python, _dilate_dendrite_paths returns the full
    # merged volume (neur_num + dilated through-vol IDs).  So we use overwrite
    # semantics (like Step 4) instead of addition to avoid double-counting.
    neur_num_out = dendnum.copy().astype(np.uint16)

    # dendrite_ad: accumulate Step 4 apical + Step 5 through-volume
    # Only include through-volume apical IDs (N_neur < ID <= N_neur+N_den),
    # not BG dendrite IDs (>N_neur+N_den) that may exist in dendnum.
    dendrite_ad = dend_result.dendrite_ad.copy().astype(np.uint16)
    through_mask = (dendnum > N_neur) & (dendnum <= N_neur + N_den)
    dendrite_ad[through_mask] = dendnum[through_mask].astype(np.uint16)

    # Clear nuclei from neur_num
    for kk in range(N_neur):
        nuc_idx = gp_nuc[kk][0]
        if len(nuc_idx) > 0:
            neur_num_out.ravel()[nuc_idx] = np.uint16(0)

    # Restore somas in neur_num
    for kk in range(N_neur):
        soma_idx = gp_soma[kk][0]
        if len(soma_idx) > 0:
            neur_num_out.ravel()[soma_idx] = np.uint16(kk + 1)

    # Build neur_num_ad (MATLAB lines 318-324)
    neur_num_ad = dendrite_ad.copy()
    for kk in range(N_neur):
        nuc_idx = gp_nuc[kk][0]
        soma_idx = gp_soma[kk][0]
        if len(nuc_idx) > 0:
            neur_num_ad.ravel()[nuc_idx] = np.uint16(0)
        if len(soma_idx) > 0:
            neur_num_ad.ravel()[soma_idx] = np.uint16(0)

    # MATLAB line 324: neur_num_AD((neur_num_AD - neur_num) > 0) = 0
    overflow = neur_num_ad.astype(np.int32) - neur_num_out.astype(np.int32)
    neur_num_ad[overflow > 0] = np.uint16(0)

    if verbosity >= 1:
        total_apical = int(np.sum(dendnum > 0))
        print(f"done. Through-volume apical voxels: {total_apical}")

    return ApicalDendriteResult(
        neur_num=neur_num_out,
        neur_num_ad=neur_num_ad,
        dend_params=dend_params,
    )

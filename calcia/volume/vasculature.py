"""
Blood vessel generation module.

Generates realistic blood vessel networks including surface vessels,
diving vessels, and capillary networks for neural volume simulation.

Based on MATLAB NAOMi simulatebloodvessels.m and related functions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_dilation, generate_binary_structure

from ..config.params import VascParams, VolumeParams
from ..algorithms.dijkstra import vessel_dijkstra


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class VesselNode:
    """
    A single node in the vessel network.

    Corresponds to MATLAB gennode() structure.

    Attributes:
        num: Node index (1-based in MATLAB, 0-based here).
        root: Index of the root node this connects to (-1 if is root).
        conn: List of connected node indices.
        pos: 3D position [x, y, z] in micrometers.
        type: Node type:
            0 = internal node
            1 = source node (edge of volume)
            2 = branch point
            3 = diving vessel origin
            4 = capillary node
        misc: Additional miscellaneous data.
    """
    num: int
    root: int = -1
    conn: List[int] = field(default_factory=list)
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    type: int = 0
    misc: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure pos is a numpy array."""
        self.pos = np.asarray(self.pos, dtype=np.float64)


@dataclass
class VesselConnection:
    """
    A connection (edge) in the vessel network.

    Corresponds to MATLAB connection struct in nodesToConn.m.

    Attributes:
        start: Starting node index.
        ends: Ending node index.
        weight: Connection weight (for Dijkstra).
        locs: Interpolated 3D positions along the vessel segment.
    """
    start: int
    ends: int
    weight: float = 1.0
    locs: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class VesselNetwork:
    """
    Complete vessel network with nodes and connections.

    Attributes:
        nodes: List of VesselNode objects.
        connections: List of VesselConnection objects.
        vessel_volume: 3D binary volume of vessel locations.
        vessel_ids: 3D volume with vessel ID at each voxel.
    """
    nodes: List[VesselNode] = field(default_factory=list)
    connections: List[VesselConnection] = field(default_factory=list)
    vessel_volume: Optional[np.ndarray] = None
    vessel_ids: Optional[np.ndarray] = None


# =============================================================================
# Pseudo-Random Sampling with Gaussian Exclusion
# =============================================================================

def pseudo_rand_sample_2d(
    n_samples: int,
    bounds: Tuple[float, float, float, float],
    exclusion_sigma: float = 10.0,
    max_iter: int = 1000,
    existing_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate 2D points with Gaussian exclusion sampling.

    Each new point is sampled from regions where existing points
    have low influence (Gaussian-weighted exclusion).

    Corresponds to MATLAB pseudoRandSample2D.m.

    Args:
        n_samples: Number of points to generate.
        bounds: (x_min, x_max, y_min, y_max) sampling bounds.
        exclusion_sigma: Standard deviation of exclusion Gaussians.
        max_iter: Maximum iterations per sample.
        existing_points: (M, 2) existing points to avoid.

    Returns:
        (n_samples, 2) array of sampled points.
    """
    x_min, x_max, y_min, y_max = bounds
    points = []

    if existing_points is not None and len(existing_points) > 0:
        all_points = list(existing_points)
    else:
        all_points = []

    for _ in range(n_samples):
        best_point = None
        best_score = -np.inf

        for _ in range(max_iter):
            # Sample random candidate
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            candidate = np.array([x, y])

            if len(all_points) == 0:
                best_point = candidate
                break

            # Compute exclusion score (lower = more excluded)
            all_pts = np.array(all_points)
            dists = np.linalg.norm(all_pts - candidate, axis=1)
            # Score is minimum distance (we want max min distance)
            score = np.min(dists)

            if score > best_score:
                best_score = score
                best_point = candidate

            # Early termination if score is good enough
            if score > exclusion_sigma:
                break

        if best_point is not None:
            points.append(best_point)
            all_points.append(best_point)

    return np.array(points)


def pseudo_rand_sample_3d(
    n_samples: int,
    bounds: Tuple[float, float, float, float, float, float],
    exclusion_sigma: float = 10.0,
    max_iter: int = 1000,
    existing_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate 3D points with Gaussian exclusion sampling.

    Each new point is sampled from regions where existing points
    have low influence (Gaussian-weighted exclusion).

    Corresponds to MATLAB pseudoRandSample3D.m.

    Args:
        n_samples: Number of points to generate.
        bounds: (x_min, x_max, y_min, y_max, z_min, z_max) sampling bounds.
        exclusion_sigma: Standard deviation of exclusion Gaussians.
        max_iter: Maximum iterations per sample.
        existing_points: (M, 3) existing points to avoid.

    Returns:
        (n_samples, 3) array of sampled points.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    points = []

    if existing_points is not None and len(existing_points) > 0:
        all_points = list(existing_points)
    else:
        all_points = []

    for _ in range(n_samples):
        best_point = None
        best_score = -np.inf

        for _ in range(max_iter):
            # Sample random candidate
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = np.random.uniform(z_min, z_max)
            candidate = np.array([x, y, z])

            if len(all_points) == 0:
                best_point = candidate
                break

            # Compute exclusion score
            all_pts = np.array(all_points)
            dists = np.linalg.norm(all_pts - candidate, axis=1)
            score = np.min(dists)

            if score > best_score:
                best_score = score
                best_point = candidate

            if score > exclusion_sigma:
                break

        if best_point is not None:
            points.append(best_point)
            all_points.append(best_point)

    return np.array(points)


# =============================================================================
# Branch Growth Algorithm
# =============================================================================

def branch_grow_nodes(
    start_pos: np.ndarray,
    bounds: Tuple[float, float, float, float],
    n_steps: int,
    step_size: float,
    step_var: float,
    direction: np.ndarray,
    direction_var: float = np.pi / 8,
    branch_prob: float = 0.02,
    min_dist: float = 10.0,
    existing_nodes: Optional[List[VesselNode]] = None,
) -> List[VesselNode]:
    """
    Grow vessel branches using constrained random walk.

    Implements a random walk with directional bias, occasional branching,
    and collision avoidance with existing nodes.

    Corresponds to MATLAB branchGrowNodes.m.

    Args:
        start_pos: (2,) or (3,) starting position.
        bounds: (x_min, x_max, y_min, y_max) or 6-element for 3D.
        n_steps: Number of growth steps.
        step_size: Mean step length in micrometers.
        step_var: Standard deviation of step length.
        direction: (2,) or (3,) initial growth direction (unit vector).
        direction_var: Maximum angular deviation per step (radians).
        branch_prob: Probability of branching at each step.
        min_dist: Minimum distance to existing nodes.
        existing_nodes: List of existing nodes to avoid.

    Returns:
        List of new VesselNode objects (including branches).
    """
    is_3d = len(start_pos) == 3
    nodes = []

    # Initialize with existing nodes for collision checking
    # Always use consistent dimensionality (2D or 3D)
    if existing_nodes is not None:
        if is_3d:
            existing_positions = np.array([n.pos for n in existing_nodes])
        else:
            # Use only XY coordinates for 2D comparison
            existing_positions = np.array([n.pos[:2] for n in existing_nodes])
    else:
        existing_positions = np.zeros((0, 3 if is_3d else 2))

    # Queue of active growth fronts: (position, direction, parent_idx, steps_remaining)
    fronts = [(start_pos.copy(), direction.copy(), -1, n_steps)]

    node_idx = 0

    while fronts:
        pos, dir_vec, parent_idx, steps_left = fronts.pop(0)

        if steps_left <= 0:
            continue

        # Normalize direction
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-10)

        # Random step size
        step = max(0.1, step_size + step_var * np.random.randn())

        # Random direction perturbation
        if is_3d:
            # 3D rotation using random axis
            perturb = np.random.randn(3)
            perturb = perturb - np.dot(perturb, dir_vec) * dir_vec  # Orthogonal component
            if np.linalg.norm(perturb) > 1e-10:
                perturb = perturb / np.linalg.norm(perturb)
                angle = direction_var * (np.random.rand() - 0.5) * 2
                # Rodrigues rotation
                dir_vec = (dir_vec * np.cos(angle) +
                          perturb * np.sin(angle))
        else:
            # 2D rotation
            angle = direction_var * (np.random.rand() - 0.5) * 2
            c, s = np.cos(angle), np.sin(angle)
            dir_vec = np.array([c * dir_vec[0] - s * dir_vec[1],
                               s * dir_vec[0] + c * dir_vec[1]])

        # New position
        new_pos = pos + step * dir_vec

        # Check bounds
        if is_3d:
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            if not (x_min <= new_pos[0] <= x_max and
                    y_min <= new_pos[1] <= y_max and
                    z_min <= new_pos[2] <= z_max):
                continue
        else:
            x_min, x_max, y_min, y_max = bounds
            if not (x_min <= new_pos[0] <= x_max and
                    y_min <= new_pos[1] <= y_max):
                continue

        # Check minimum distance to existing nodes
        if len(existing_positions) > 0:
            dists = np.linalg.norm(existing_positions - new_pos, axis=1)
            if np.min(dists) < min_dist:
                continue

        # Also check against newly created nodes
        if nodes:
            if is_3d:
                new_node_positions = np.array([n.pos for n in nodes])
            else:
                # Use only XY coordinates for 2D comparison
                new_node_positions = np.array([n.pos[:2] for n in nodes])
            dists = np.linalg.norm(new_node_positions - new_pos, axis=1)
            if np.min(dists) < min_dist * 0.5:  # Slightly relaxed for own branch
                continue

        # Create new node
        node = VesselNode(
            num=node_idx,
            root=parent_idx,
            pos=new_pos if is_3d else np.array([new_pos[0], new_pos[1], 0]),
            type=0,  # Internal node
        )

        if parent_idx >= 0:
            node.conn.append(parent_idx)
            nodes[parent_idx].conn.append(node_idx)

        nodes.append(node)

        # Continue growth
        fronts.append((new_pos, dir_vec, node_idx, steps_left - 1))

        # Possible branching
        if np.random.rand() < branch_prob:
            # Create branch with perpendicular direction
            if is_3d:
                branch_dir = np.cross(dir_vec, np.random.randn(3))
                if np.linalg.norm(branch_dir) > 1e-10:
                    branch_dir = branch_dir / np.linalg.norm(branch_dir)
                else:
                    branch_dir = np.random.randn(3)
                    branch_dir = branch_dir / np.linalg.norm(branch_dir)
            else:
                # 90 degree rotation
                branch_dir = np.array([-dir_vec[1], dir_vec[0]])
                if np.random.rand() > 0.5:
                    branch_dir = -branch_dir

            # Add branch to queue (shorter than main branch)
            branch_steps = int(steps_left * 0.5 * np.random.rand())
            if branch_steps > 0:
                fronts.append((new_pos, branch_dir, node_idx, branch_steps))

        node_idx += 1

    return nodes


# =============================================================================
# Source Node Generation
# =============================================================================

def generate_source_nodes(
    vol_params: VolumeParams,
    vasc_params: VascParams,
) -> List[VesselNode]:
    """
    Generate source nodes at edges of the volume.

    Source nodes are placed on the lateral edges of the volume
    where major vessels enter the simulated region.

    Corresponds to MATLAB growMajorVessels.m source node generation.

    Args:
        vol_params: Volume parameters.
        vasc_params: Vasculature parameters.

    Returns:
        List of source VesselNode objects.
    """
    vol_sz = np.array(vol_params.vol_sz)
    node_params = vasc_params.node_params

    # Calculate number of source nodes per edge
    n_sources_x = max(1, int(np.ceil(vol_sz[0] / vasc_params.sourceFreq)))
    n_sources_y = max(1, int(np.ceil(vol_sz[1] / vasc_params.sourceFreq)))

    nodes = []
    node_idx = 0

    # Surface depth for source nodes
    z_surface = vasc_params.depth_surf

    # Generate source nodes on each edge
    edges = [
        ('x_min', 0, vol_sz[1]),      # Left edge
        ('x_max', 0, vol_sz[1]),      # Right edge
        ('y_min', 0, vol_sz[0]),      # Bottom edge
        ('y_max', 0, vol_sz[0]),      # Top edge
    ]

    for edge_name, coord_min, coord_max in edges:
        if 'x' in edge_name:
            n_sources = n_sources_y
        else:
            n_sources = n_sources_x

        for i in range(n_sources):
            # Position along edge with some randomness
            t = (i + 0.5 + 0.3 * (np.random.rand() - 0.5)) / n_sources
            coord = coord_min + t * (coord_max - coord_min)

            if edge_name == 'x_min':
                pos = np.array([0, coord, z_surface])
                direction = np.array([1, 0, 0])
            elif edge_name == 'x_max':
                pos = np.array([vol_sz[0], coord, z_surface])
                direction = np.array([-1, 0, 0])
            elif edge_name == 'y_min':
                pos = np.array([coord, 0, z_surface])
                direction = np.array([0, 1, 0])
            else:  # y_max
                pos = np.array([coord, vol_sz[1], z_surface])
                direction = np.array([0, -1, 0])

            node = VesselNode(
                num=node_idx,
                root=-1,  # Source nodes have no parent
                pos=pos,
                type=1,   # Source node type
                misc={'direction': direction, 'edge': edge_name},
            )
            nodes.append(node)
            node_idx += 1

    return nodes


# =============================================================================
# Major Vessel Growth
# =============================================================================

def grow_major_vessels(
    vol_params: VolumeParams,
    vasc_params: VascParams,
    verbose: int = 1,
) -> VesselNetwork:
    """
    Grow major (surface) blood vessels.

    Implements the surface vessel growth algorithm:
    1. Generate source nodes at volume edges
    2. Grow branches from each source
    3. Connect branches using Dijkstra's algorithm

    Corresponds to MATLAB growMajorVessels.m.

    Args:
        vol_params: Volume parameters.
        vasc_params: Vasculature parameters.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).

    Returns:
        VesselNetwork with surface vessel nodes and connections.
    """
    vol_sz = np.array(vol_params.vol_sz)
    node_params = vasc_params.node_params

    if verbose >= 1:
        print("Growing major blood vessels...")

    # Step 1: Generate source nodes
    if verbose >= 2:
        print("  Generating source nodes...")
    source_nodes = generate_source_nodes(vol_params, vasc_params)

    if verbose >= 2:
        print(f"  Generated {len(source_nodes)} source nodes")

    # Step 2: Grow branches from each source
    if verbose >= 2:
        print("  Growing vessel branches...")

    all_nodes = list(source_nodes)

    # 2D bounds for surface vessel growth
    bounds_2d = (0, vol_sz[0], 0, vol_sz[1])

    for source in source_nodes:
        direction = source.misc.get('direction', np.array([1, 0, 0]))[:2]

        # Grow branch from this source
        branch_nodes = branch_grow_nodes(
            start_pos=source.pos[:2],
            bounds=bounds_2d,
            n_steps=int(node_params.lensc * 2),
            step_size=node_params.lensc,
            step_var=node_params.varsc,
            direction=direction,
            direction_var=node_params.dirvar,
            branch_prob=node_params.branchp,
            min_dist=node_params.mindist,
            existing_nodes=all_nodes,
        )

        # Add z-coordinate (surface depth) and renumber nodes
        start_idx = len(all_nodes)
        for i, node in enumerate(branch_nodes):
            node.num = start_idx + i
            if node.root >= 0:
                node.root = start_idx + node.root
            node.conn = [start_idx + c for c in node.conn]
            node.pos[2] = vasc_params.depth_surf  # Surface depth

            # Connect first node to source
            if i == 0:
                node.root = source.num
                node.conn.append(source.num)
                source.conn.append(node.num)

        all_nodes.extend(branch_nodes)

    if verbose >= 2:
        print(f"  Total nodes after branch growth: {len(all_nodes)}")

    # Step 3: Connect branches using Dijkstra
    if verbose >= 2:
        print("  Connecting branches with Dijkstra...")

    all_nodes = connect_vessel_nodes(all_nodes, vasc_params)

    if verbose >= 1:
        print(f"  Major vessels complete: {len(all_nodes)} nodes")

    return VesselNetwork(nodes=all_nodes)


def connect_vessel_nodes(
    nodes: List[VesselNode],
    vasc_params: VascParams,
    max_connection_dist: Optional[float] = None,
) -> List[VesselNode]:
    """
    Connect vessel nodes using Dijkstra's algorithm.

    Creates a minimum spanning tree connecting all nodes based on
    distance-weighted costs.

    Args:
        nodes: List of VesselNode objects.
        vasc_params: Vasculature parameters.
        max_connection_dist: Maximum distance for connections (default: auto).

    Returns:
        Updated nodes with new connections.
    """
    if len(nodes) < 2:
        return nodes

    # Build position matrix
    positions = np.array([n.pos for n in nodes])
    n_nodes = len(nodes)

    # Compute distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    # Set maximum connection distance
    if max_connection_dist is None:
        max_connection_dist = vasc_params.node_params.lensc * 3

    # Create weighted distance matrix for Dijkstra
    # Weight by distance with random scaling
    weight_matrix = dist_matrix.copy()
    weight_matrix += vasc_params.randWeightScale * np.random.rand(n_nodes, n_nodes) * dist_matrix
    weight_matrix = (weight_matrix + weight_matrix.T) / 2  # Symmetrize

    # Remove connections that are too far
    weight_matrix[dist_matrix > max_connection_dist] = np.inf
    np.fill_diagonal(weight_matrix, 0)

    # Find source nodes (type 1) as roots
    source_indices = [i for i, n in enumerate(nodes) if n.type == 1]

    if not source_indices:
        # No source nodes, use first node
        source_indices = [0]

    # Run Dijkstra from each source and collect connections
    all_parents = np.full((n_nodes, len(source_indices)), -1, dtype=np.int32)
    all_distances = np.full((n_nodes, len(source_indices)), np.inf)

    for i, source_idx in enumerate(source_indices):
        distances, parents = vessel_dijkstra(weight_matrix, source_idx)
        all_parents[:, i] = parents
        all_distances[:, i] = distances

    # For each node, connect to the closest source tree
    for node_idx in range(n_nodes):
        if node_idx in source_indices:
            continue

        # Find which source tree gives minimum distance
        min_source = np.argmin(all_distances[node_idx])
        parent_idx = all_parents[node_idx, min_source]

        if parent_idx >= 0 and parent_idx not in nodes[node_idx].conn:
            nodes[node_idx].conn.append(parent_idx)
            nodes[node_idx].root = parent_idx
            if node_idx not in nodes[parent_idx].conn:
                nodes[parent_idx].conn.append(node_idx)

    return nodes


# =============================================================================
# Diving Vessels
# =============================================================================

def grow_diving_vessels(
    network: VesselNetwork,
    vol_params: VolumeParams,
    vasc_params: VascParams,
    verbose: int = 1,
) -> VesselNetwork:
    """
    Grow diving vessels from surface into the volume depth.

    Diving vessels connect surface vasculature to the capillary
    network deeper in the tissue.

    Corresponds to MATLAB growMajorVessels.m diving vessel section.

    Args:
        network: Existing VesselNetwork with surface vessels.
        vol_params: Volume parameters.
        vasc_params: Vasculature parameters.
        verbose: Verbosity level.

    Returns:
        Updated VesselNetwork with diving vessels.
    """
    vol_sz = np.array(vol_params.vol_sz)
    # Full tissue depth includes vol_depth (above imaging volume) + vol_sz[2]
    full_depth = vol_params.vol_depth + vol_sz[2]

    # Calculate number of diving vessels
    n_diving = max(1, int(np.ceil(
        vol_sz[0] * vol_sz[1] / (vasc_params.vesFreq[1] ** 2)
    )))

    # Add noise to count
    n_diving = max(1, int(n_diving * (1 + vasc_params.vesNumScale * np.random.randn())))

    if verbose >= 1:
        print(f"  Growing {n_diving} diving vessels...")

    # Sample positions for diving vessels
    existing_surface = np.array([n.pos[:2] for n in network.nodes if n.pos[2] <= vasc_params.depth_surf + 1])

    dive_positions = pseudo_rand_sample_2d(
        n_samples=n_diving,
        bounds=(0, vol_sz[0], 0, vol_sz[1]),
        exclusion_sigma=vasc_params.vesFreq[1] * 0.5,
        existing_points=existing_surface if len(existing_surface) > 0 else None,
    )

    # For each diving vessel position, find closest surface node and grow downward
    surface_positions = np.array([n.pos for n in network.nodes if n.pos[2] <= vasc_params.depth_surf + 1])
    surface_indices = [i for i, n in enumerate(network.nodes) if n.pos[2] <= vasc_params.depth_surf + 1]

    nodes = list(network.nodes)

    for dive_pos in dive_positions:
        # Find closest surface node
        if len(surface_positions) > 0:
            dists = np.linalg.norm(surface_positions[:, :2] - dive_pos, axis=1)
            closest_idx = surface_indices[np.argmin(dists)]
            start_pos = nodes[closest_idx].pos.copy()
        else:
            closest_idx = 0
            start_pos = np.array([dive_pos[0], dive_pos[1], vasc_params.depth_surf])

        # Grow vessel downward until it reaches full_depth (matches MATLAB while loop).
        # Use enough steps to ensure we can reach the bottom.
        n_steps = max(50, int(np.ceil((full_depth - vasc_params.depth_surf)
                                     / vasc_params.node_params.lensc)) + 5)
        current_pos = start_pos.copy()
        parent_idx = closest_idx

        for step in range(n_steps):
            # Move downward with some wobble
            wobble = vasc_params.ves_shift * (np.random.rand(3) - 0.5)
            new_z = current_pos[2] + vasc_params.node_params.lensc

            # Clamp to full_depth (MATLAB: min(node_pos, nv.size))
            reached_bottom = new_z >= full_depth
            new_z = min(new_z, full_depth)

            new_pos = np.array([
                current_pos[0] + wobble[0],
                current_pos[1] + wobble[1],
                new_z
            ])

            # Clamp to volume bounds
            new_pos[0] = np.clip(new_pos[0], 0, vol_sz[0])
            new_pos[1] = np.clip(new_pos[1], 0, vol_sz[1])

            # Create node
            node = VesselNode(
                num=len(nodes),
                root=parent_idx,
                conn=[parent_idx],
                pos=new_pos,
                type=3,  # Diving vessel node
            )
            nodes[parent_idx].conn.append(node.num)
            nodes.append(node)

            parent_idx = node.num
            current_pos = new_pos

            if reached_bottom:
                break

    network.nodes = nodes

    if verbose >= 1:
        print(f"  Diving vessels complete: {len(nodes)} total nodes")

    return network


# =============================================================================
# Capillary Network
# =============================================================================

def grow_capillaries(
    network: VesselNetwork,
    vol_params: VolumeParams,
    vasc_params: VascParams,
    verbose: int = 1,
) -> VesselNetwork:
    """
    Generate capillary network between diving vessels.

    Capillaries form a dense network connecting the larger vessels
    and filling the tissue volume.

    Corresponds to MATLAB growCapillaries.m.

    Args:
        network: Existing VesselNetwork with major vessels.
        vol_params: Volume parameters.
        vasc_params: Vasculature parameters.
        verbose: Verbosity level.

    Returns:
        Updated VesselNetwork with capillaries.
    """
    vol_sz = np.array(vol_params.vol_sz)
    full_depth = vol_params.vol_depth + vol_sz[2]

    # Calculate capillary density (using full depth volume)
    n_capillary = max(1, int(np.ceil(
        vol_sz[0] * vol_sz[1] * full_depth / (vasc_params.vesFreq[2] ** 3)
    )))

    if verbose >= 1:
        print(f"  Growing {n_capillary} capillary nodes...")

    nodes = list(network.nodes)

    # Get positions of deep vessel nodes (below surface)
    deep_nodes = [(i, n) for i, n in enumerate(nodes) if n.pos[2] > vasc_params.depth_surf]

    if len(deep_nodes) == 0:
        if verbose >= 1:
            print("  Warning: No deep nodes for capillary attachment")
        return network

    # Sample capillary positions throughout the volume
    existing_positions = np.array([n.pos for n in nodes])

    capillary_bounds = (
        0, vol_sz[0],
        0, vol_sz[1],
        vasc_params.depth_surf, full_depth
    )

    cap_positions = pseudo_rand_sample_3d(
        n_samples=n_capillary,
        bounds=capillary_bounds,
        exclusion_sigma=vasc_params.vesFreq[2] * 0.3,
        existing_points=existing_positions,
    )

    # ----------------------------------------------------------------
    # vtcp: connect diving vessels to nearest capillary positions
    # Mirrors MATLAB growCapillaries.m lines 70-115.
    # Each diving vessel chain claims 1..max_vtcp capillary positions
    # (the nearest unclaimed ones), creating vtcp-type connections.
    # Claimed positions get root set; remaining are free capillaries.
    # ----------------------------------------------------------------

    # sfvt equivalent: type=3 nodes whose parent is not type=3
    sfvt_indices = [
        i for i, n in enumerate(nodes)
        if n.type == 3 and (n.root < 0 or nodes[n.root].type != 3)
    ]

    # max vtcp connections per diving vessel: ceil(full_depth / vesFreq[2])
    max_vtcp = max(1, int(np.ceil(full_depth / vasc_params.vesFreq[2])))

    # Use NaN to mark claimed positions (MATLAB: capppos(TMP,:) = nan)
    cap_pos_arr = np.array(cap_positions, dtype=float)
    vtcp_nodes_info = []  # list of (dive_node_idx, cap_pos)

    for sfvt_idx in sfvt_indices:
        # Traverse down the chain collecting all type=3 nodes (DFS)
        chain = []
        visited = {sfvt_idx}
        stack = [sfvt_idx]
        while stack:
            cur = stack.pop()
            chain.append(cur)
            for nb in nodes[cur].conn:
                if nb not in visited and nodes[nb].type == 3:
                    visited.add(nb)
                    stack.append(nb)

        n_vtcp = np.random.randint(1, max_vtcp + 1)  # randi(max_vtcp)

        for _ in range(n_vtcp):
            valid_mask = ~np.isnan(cap_pos_arr[:, 0])
            valid_caps = np.where(valid_mask)[0]
            if len(valid_caps) == 0:
                break
            # Pick random node along the diving vessel chain
            dive_node_idx = chain[np.random.randint(len(chain))]
            dive_pos = nodes[dive_node_idx].pos
            # Find nearest unclaimed capillary position
            dists = np.linalg.norm(cap_pos_arr[valid_caps] - dive_pos, axis=1)
            nearest = valid_caps[np.argmin(dists)]
            vtcp_pos = cap_pos_arr[nearest].copy()
            cap_pos_arr[nearest] = np.nan  # mark as claimed
            vtcp_nodes_info.append((dive_node_idx, vtcp_pos))

    if verbose >= 2:
        print(f"  vtcp: {len(vtcp_nodes_info)} diving-vessel→capillary connections"
              f" from {len(sfvt_indices)} diving vessel chains")

    # Create vtcp capillary nodes (have root → connected to diving vessel)
    vtcp_cap_indices = []
    for dive_node_idx, vtcp_pos in vtcp_nodes_info:
        node = VesselNode(
            num=len(nodes),
            root=dive_node_idx,
            pos=vtcp_pos,
            type=4,
        )
        nodes[dive_node_idx].conn.append(node.num)
        nodes.append(node)
        vtcp_cap_indices.append(node.num)

    # Create free capillary nodes (no root → unclaimed positions)
    free_cap_indices = []
    for cap_pos in cap_pos_arr:
        if not np.isnan(cap_pos[0]):
            node = VesselNode(
                num=len(nodes),
                root=-1,
                pos=cap_pos,
                type=4,
            )
            nodes.append(node)
            free_cap_indices.append(node.num)

    # Connect capillaries using distance-weighted graph
    if verbose >= 2:
        print("  Connecting capillary network...")

    # vtcp capillaries must not connect to each other (MATLAB:
    # cappmat(1:nv.nvert_sum, 1:nv.nvert_sum) = inf)
    vtcp_cap_set = set(vtcp_cap_indices)

    capillary_indices = vtcp_cap_indices + free_cap_indices
    deep_indices = [i for i, n in enumerate(nodes) if n.pos[2] > vasc_params.depth_surf and n.type != 4]

    # Connect each capillary to nearby vessels and other capillaries.
    # max_cap_dist matches MATLAB: 2 * vesFreq(3) * vres converted to µm = 2 * vesFreq[2]
    positions = np.array([n.pos for n in nodes])
    max_cap_dist = vasc_params.vesFreq[2] * 2  # = 100 µm (MATLAB: 2*vesFreq(3)*vres voxels)
    distsc = vasc_params.distsc

    def _add_connection(a, b):
        if b not in nodes[a].conn:
            nodes[a].conn.append(b)
            nodes[b].conn.append(a)
            if nodes[a].root < 0:
                nodes[a].root = b

    # Initial pass: connect each capillary to its 3 closest neighbors.
    # vtcp capillaries cannot connect to other vtcp capillaries.
    for cap_idx in capillary_indices:
        cap_pos = nodes[cap_idx].pos
        if cap_idx in vtcp_cap_set:
            candidate_indices = deep_indices + free_cap_indices
        else:
            candidate_indices = deep_indices + [c for c in capillary_indices if c != cap_idx]
        if not candidate_indices:
            continue
        candidate_positions = positions[candidate_indices]
        dists = np.linalg.norm(candidate_positions - cap_pos, axis=1)
        n_connections = min(3, len(candidate_indices))
        for c in np.argsort(dists)[:n_connections]:
            if dists[c] < max_cap_dist:
                _add_connection(cap_idx, candidate_indices[c])

    # Supplemental pass: ensure every capillary has ≥2 connections (mirrors MATLAB
    # growCapillaries.m lines 159-191 probabilistic loop).
    # vtcp capillaries use restricted candidate set.
    for _ in range(n_capillary * 5):
        under = [i for i in capillary_indices if len(nodes[i].conn) < 2]
        if not under:
            break
        src = under[np.random.randint(len(under))]
        src_pos = nodes[src].pos
        if src in vtcp_cap_set:
            candidates = deep_indices + free_cap_indices
        else:
            candidates = deep_indices + [c for c in capillary_indices if c != src]
        if not candidates:
            continue
        cand_pos = positions[candidates]
        dists = np.linalg.norm(cand_pos - src_pos, axis=1)
        valid = dists < max_cap_dist
        if not valid.any():
            continue
        # Distance-weighted probability: 1 / d^distsc (MATLAB line 171)
        w = 1.0 / (dists[valid] ** distsc + 1e-9)
        chosen = candidates[np.where(valid)[0][np.random.choice(len(w), p=w / w.sum())]]
        _add_connection(src, chosen)

    network.nodes = nodes

    if verbose >= 1:
        n_conn = sum(len(nodes[i].conn) for i in capillary_indices)
        print(f"  Capillaries complete: {len(nodes)} total nodes,"
              f" {len(vtcp_cap_indices)} vtcp + {len(free_cap_indices)} free capillaries,"
              f" {n_conn} connections (avg {n_conn/max(1,len(capillary_indices)):.1f}/node)")

    return network


# =============================================================================
# Volume Rendering
# =============================================================================

def nodes_to_connections(
    network: VesselNetwork,
    vasc_params: VascParams,
) -> VesselNetwork:
    """
    Convert node-based network to connection-based representation.

    Creates smooth spline interpolations along each vessel segment,
    using neighboring nodes for tangent estimation (like MATLAB's cscvn).

    Corresponds to MATLAB nodesToConn.m and connToVol.m spline logic.

    Args:
        network: VesselNetwork with nodes.
        vasc_params: Vasculature parameters.

    Returns:
        VesselNetwork with populated connections.
    """
    from scipy.interpolate import CubicSpline

    nodes = network.nodes
    connections = []

    # Build connection list from node connectivity
    visited_edges = set()

    for node in nodes:
        for neighbor_idx in node.conn:
            # Create edge key (sorted to avoid duplicates)
            edge = tuple(sorted([node.num, neighbor_idx]))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)

            start_node = node
            end_node = nodes[neighbor_idx]

            # Find neighboring nodes for tangent estimation (MATLAB cscvn style)
            # Get a neighbor of start that's not end
            start_neighbors = [n for n in start_node.conn if n != neighbor_idx]
            # Get a neighbor of end that's not start
            end_neighbors = [n for n in end_node.conn if n != node.num]

            # Build control points for spline
            control_points = []

            # Add preceding neighbor if available (for tangent at start)
            if start_neighbors:
                prev_idx = start_neighbors[np.random.randint(len(start_neighbors))]
                control_points.append(nodes[prev_idx].pos)

            # Add main segment points
            control_points.append(start_node.pos)
            control_points.append(end_node.pos)

            # Add following neighbor if available (for tangent at end)
            if end_neighbors:
                next_idx = end_neighbors[np.random.randint(len(end_neighbors))]
                control_points.append(nodes[next_idx].pos)

            control_points = np.array(control_points)

            # Create smooth path using spline interpolation
            n_control = len(control_points)
            segment_length = np.linalg.norm(end_node.pos - start_node.pos)
            n_interp = max(4, int(segment_length * 2))

            if n_control >= 3:
                # Use cubic spline for smooth curves
                # Parametric spline: t -> (x, y, z)
                t_control = np.linspace(0, 1, n_control)

                try:
                    # Create splines for each coordinate
                    spline_x = CubicSpline(t_control, control_points[:, 0], bc_type='natural')
                    spline_y = CubicSpline(t_control, control_points[:, 1], bc_type='natural')
                    spline_z = CubicSpline(t_control, control_points[:, 2], bc_type='natural')

                    # Sample along the segment between control points 1 and 2
                    # (or 0 and 1 if no preceding neighbor)
                    if len(start_neighbors) > 0:
                        t_start = t_control[1]
                        t_end = t_control[2]
                    else:
                        t_start = t_control[0]
                        t_end = t_control[1]

                    t_interp = np.linspace(t_start, t_end, n_interp)
                    locs = np.column_stack([
                        spline_x(t_interp),
                        spline_y(t_interp),
                        spline_z(t_interp)
                    ])
                except Exception:
                    # Fallback to linear interpolation
                    t = np.linspace(0, 1, n_interp)
                    locs = start_node.pos + np.outer(t, end_node.pos - start_node.pos)
            else:
                # Not enough points for spline, use linear
                t = np.linspace(0, 1, n_interp)
                locs = start_node.pos + np.outer(t, end_node.pos - start_node.pos)

            conn = VesselConnection(
                start=node.num,
                ends=neighbor_idx,
                weight=segment_length,
                locs=locs,
            )
            connections.append(conn)

    network.connections = connections
    return network


def connections_to_volume(
    network: VesselNetwork,
    vol_params: VolumeParams,
    vasc_params: VascParams,
    verbose: int = 1,
) -> VesselNetwork:
    """
    Render vessel connections to a 3D volume.

    Uses per-connection spherical dilation to create vessel tubes,
    matching MATLAB connToVol.m behavior.

    Each connection gets its own radius based on node types, and
    dilation is performed locally for efficiency.

    Args:
        network: VesselNetwork with connections.
        vol_params: Volume parameters.
        vasc_params: Vasculature parameters.
        verbose: Verbosity level.

    Returns:
        VesselNetwork with vessel_volume populated.
    """
    img_vol_sz = tuple(int(s * vol_params.vres) for s in vol_params.vol_sz)
    vres = vol_params.vres
    # Render into a full-depth volume (surface → bottom of imaging volume).
    # Vessel positions use absolute depth (z=0 = tissue surface).
    z_offset_vox = int(vol_params.vol_depth * vres)
    full_z_vox   = z_offset_vox + img_vol_sz[2]
    full_vol_sz  = (img_vol_sz[0], img_vol_sz[1], full_z_vox)

    if verbose >= 1:
        print(f"  Rendering vessels to full-depth volume {full_vol_sz}, then cropping to imaging region...")

    # Initialize full-depth volume
    vessel_volume = np.zeros(full_vol_sz, dtype=np.uint8)

    # Vessel radii for different types (in micrometers, will convert to voxels)
    type_radii_um = {
        1: vasc_params.vesSize[0],  # Surface vessels
        2: vasc_params.vesSize[0],  # Branch points
        3: vasc_params.vesSize[1],  # Diving vessels
        4: vasc_params.vesSize[2],  # Capillaries
        0: vasc_params.vesSize[2],  # Default/Internal
    }

    # For large radii, we'll draw spheres directly instead of using dilation
    def draw_sphere_at(volume, center, radius, vol_shape):
        """Draw a sphere directly into the volume."""
        cx, cy, cz = center
        r = int(np.ceil(radius))

        # Compute bounding box
        x_min = max(0, cx - r)
        x_max = min(vol_shape[0], cx + r + 1)
        y_min = max(0, cy - r)
        y_max = min(vol_shape[1], cy + r + 1)
        z_min = max(0, cz - r)
        z_max = min(vol_shape[2], cz + r + 1)

        # Create coordinate grids for the local region
        xs = np.arange(x_min, x_max)
        ys = np.arange(y_min, y_max)
        zs = np.arange(z_min, z_max)

        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            return

        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
        dist_sq = (xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2

        mask = dist_sq <= radius**2
        volume[x_min:x_max, y_min:y_max, z_min:z_max] = np.maximum(
            volume[x_min:x_max, y_min:y_max, z_min:z_max],
            mask.astype(np.uint8)
        )

    # Draw each connection with its own radius
    for conn in network.connections:
        # Determine vessel radius based on node types
        start_type = network.nodes[conn.start].type
        end_type = network.nodes[conn.ends].type
        start_z = network.nodes[conn.start].pos[2]
        end_z = network.nodes[conn.ends].pos[2]

        # Determine vessel radius based on depth and type
        # Surface vessels (near depth_surf) use large radius
        # Diving vessels use medium radius
        # Capillaries use small radius
        if start_type == 4 or end_type == 4:
            # Capillary connection
            radius_um = type_radii_um[4]
        elif start_type == 3 or end_type == 3:
            # Diving vessel connection
            radius_um = type_radii_um[3]
        elif max(start_z, end_z) <= vasc_params.depth_surf + 5:
            # Surface vessel connection (at or near surface)
            radius_um = type_radii_um[1]
        else:
            # Use the larger of the two radii for other connections
            radius_um = max(type_radii_um.get(start_type, 2), type_radii_um.get(end_type, 2))

        radius_voxels = max(1, int(np.ceil(radius_um * vres)))

        # Get voxel locations along the connection
        locs_voxels = []
        for pos in conn.locs:
            vx = int(np.round(pos[0] * vres))
            vy = int(np.round(pos[1] * vres))
            vz = int(np.round(pos[2] * vres))   # absolute z, no offset
            if 0 <= vx < full_vol_sz[0] and 0 <= vy < full_vol_sz[1] and 0 <= vz < full_vol_sz[2]:
                locs_voxels.append((vx, vy, vz))

        if not locs_voxels:
            continue

        # Draw spheres at each point along the vessel
        # This is more memory-efficient than dilation for large radii
        for loc in locs_voxels:
            draw_sphere_at(vessel_volume, loc, radius_voxels, full_vol_sz)

    # Keep full-depth volume (matches MATLAB vol_out.neur_ves which includes surface region)
    network.vessel_volume = vessel_volume

    if verbose >= 1:
        fill_pct = 100 * np.sum(vessel_volume) / np.prod(full_vol_sz)
        imaging_vox = int(np.sum(vessel_volume[:, :, z_offset_vox:]))
        print(f"  Volume rendered: {fill_pct:.2f}% filled (full depth),"
              f" imaging region: {imaging_vox:,} voxels")

    return network


# =============================================================================
# Main Entry Point
# =============================================================================

def simulate_blood_vessels(
    vol_params: Optional[VolumeParams] = None,
    vasc_params: Optional[VascParams] = None,
    verbose: int = 1,
) -> VesselNetwork:
    """
    Simulate complete blood vessel network.

    This is the main entry point for vasculature generation.
    Generates surface vessels, diving vessels, and capillary network.

    Corresponds to MATLAB simulatebloodvessels.m.

    Args:
        vol_params: Volume parameters. Uses defaults if None.
        vasc_params: Vasculature parameters. Uses defaults if None.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).

    Returns:
        VesselNetwork containing nodes, connections, and rendered volume.

    Example:
        >>> from calcia.config.params import VolumeParams, VascParams
        >>> vol_params = VolumeParams(vol_sz=(100, 100, 50))
        >>> vasc_params = VascParams()
        >>> network = simulate_blood_vessels(vol_params, vasc_params)
        >>> print(f"Generated {len(network.nodes)} vessel nodes")
    """
    if vol_params is None:
        vol_params = VolumeParams()
    if vasc_params is None:
        vasc_params = VascParams()

    if not vasc_params.flag:
        if verbose >= 1:
            print("Vasculature simulation disabled (vasc_params.flag=False)")
        return VesselNetwork()

    if verbose >= 1:
        print("=" * 50)
        print("Simulating blood vessel network...")
        print("=" * 50)

    # Step 1: Grow major (surface) vessels
    network = grow_major_vessels(vol_params, vasc_params, verbose)

    # Step 2: Grow diving vessels
    network = grow_diving_vessels(network, vol_params, vasc_params, verbose)

    # Step 3: Grow capillary network
    network = grow_capillaries(network, vol_params, vasc_params, verbose)

    # Step 4: Convert to connections
    if verbose >= 1:
        print("  Converting to connections...")
    network = nodes_to_connections(network, vasc_params)

    # Step 5: Render to volume
    network = connections_to_volume(network, vol_params, vasc_params, verbose)

    if verbose >= 1:
        print("=" * 50)
        print(f"Blood vessel simulation complete!")
        print(f"  Total nodes: {len(network.nodes)}")
        print(f"  Total connections: {len(network.connections)}")
        print("=" * 50)

    return network

"""
Validate blood vessel network generation.

Checks statistical properties and structural correctness of generated vessels.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from calcia.config.params import VolumeParams, VascParams
from calcia.volume.vasculature import simulate_blood_vessels


def validate_vessel_network(network, vol_params, vasc_params):
    """
    Validate vessel network properties.

    Returns dict with validation results.
    """
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
        'stats': {}
    }

    nodes = network.nodes
    connections = network.connections
    vol_sz = np.array(vol_params.vol_sz)

    # === Basic Statistics ===
    results['stats']['total_nodes'] = len(nodes)
    results['stats']['total_connections'] = len(connections)

    # Count by type
    type_counts = {}
    type_names = {0: 'internal', 1: 'source', 2: 'branch', 3: 'diving', 4: 'capillary'}
    for n in nodes:
        name = type_names.get(n.type, f'type_{n.type}')
        type_counts[name] = type_counts.get(name, 0) + 1
    results['stats']['type_counts'] = type_counts

    # === Test 1: Has all required vessel types ===
    if type_counts.get('source', 0) > 0:
        results['passed'].append("Has source nodes")
    else:
        results['failed'].append("Missing source nodes")

    if type_counts.get('diving', 0) > 0:
        results['passed'].append("Has diving vessels")
    else:
        results['warnings'].append("No diving vessels (may be ok for shallow volumes)")

    if type_counts.get('capillary', 0) > 0:
        results['passed'].append("Has capillary nodes")
    else:
        results['warnings'].append("No capillary nodes")

    # === Test 2: Nodes within volume bounds ===
    out_of_bounds = 0
    for n in nodes:
        if (n.pos[0] < -1 or n.pos[0] > vol_sz[0] + 1 or
            n.pos[1] < -1 or n.pos[1] > vol_sz[1] + 1 or
            n.pos[2] < -1 or n.pos[2] > vol_sz[2] + 1):
            out_of_bounds += 1

    if out_of_bounds == 0:
        results['passed'].append("All nodes within volume bounds")
    else:
        results['failed'].append(f"{out_of_bounds} nodes out of bounds")

    # === Test 3: Source nodes at correct depth ===
    source_depths = [n.pos[2] for n in nodes if n.type == 1]
    if source_depths:
        avg_source_depth = np.mean(source_depths)
        expected_depth = vasc_params.depth_surf
        if abs(avg_source_depth - expected_depth) < 5:
            results['passed'].append(f"Source nodes at correct depth (~{expected_depth}um)")
        else:
            results['warnings'].append(
                f"Source depth {avg_source_depth:.1f}um vs expected {expected_depth}um"
            )
        results['stats']['avg_source_depth'] = avg_source_depth

    # === Test 4: Diving vessels go deeper ===
    diving_depths = [n.pos[2] for n in nodes if n.type == 3]
    if diving_depths:
        max_diving_depth = max(diving_depths)
        min_diving_depth = min(diving_depths)
        if max_diving_depth > vasc_params.depth_surf + 10:
            results['passed'].append(f"Diving vessels reach depth {max_diving_depth:.1f}um")
        else:
            results['warnings'].append("Diving vessels don't go deep enough")
        results['stats']['diving_depth_range'] = (min_diving_depth, max_diving_depth)

    # === Test 5: Capillaries distributed in volume ===
    cap_positions = np.array([n.pos for n in nodes if n.type == 4])
    if len(cap_positions) > 0:
        cap_z_range = (np.min(cap_positions[:, 2]), np.max(cap_positions[:, 2]))
        if cap_z_range[1] > vasc_params.depth_surf:
            results['passed'].append("Capillaries distributed in deep tissue")
        results['stats']['capillary_z_range'] = cap_z_range

    # === Test 6: Connectivity check ===
    isolated_nodes = sum(1 for n in nodes if len(n.conn) == 0 and n.type != 1)
    if isolated_nodes == 0:
        results['passed'].append("No isolated non-source nodes")
    else:
        results['warnings'].append(f"{isolated_nodes} isolated nodes (may be ok for capillaries)")
    results['stats']['isolated_nodes'] = isolated_nodes

    # === Test 7: Connection lengths reasonable ===
    if connections:
        lengths = [c.weight for c in connections]
        avg_length = np.mean(lengths)
        max_length = np.max(lengths)
        results['stats']['avg_connection_length'] = avg_length
        results['stats']['max_connection_length'] = max_length

        if max_length < vol_sz[0]:  # No connection longer than volume width
            results['passed'].append(f"Connection lengths reasonable (avg={avg_length:.1f}um)")
        else:
            results['warnings'].append(f"Some connections very long ({max_length:.1f}um)")

    # === Test 8: Volume rendering ===
    if network.vessel_volume is not None:
        vol = network.vessel_volume
        fill_fraction = np.sum(vol) / np.prod(vol.shape)
        results['stats']['volume_fill_fraction'] = fill_fraction

        if 0.0001 < fill_fraction < 0.5:
            results['passed'].append(f"Volume fill reasonable ({fill_fraction*100:.2f}%)")
        elif fill_fraction < 0.0001:
            results['warnings'].append("Volume fill very low")
        else:
            results['warnings'].append("Volume fill very high")

    return results


def print_validation_results(results):
    """Print validation results in a formatted way."""
    print("\n" + "=" * 60)
    print("VESSEL NETWORK VALIDATION RESULTS")
    print("=" * 60)

    print("\n--- Statistics ---")
    for key, value in results['stats'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif isinstance(value, tuple):
            print(f"  {key}: {value[0]:.1f} - {value[1]:.1f}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n--- Passed Tests ---")
    for p in results['passed']:
        print(f"  [OK] {p}")

    if results['warnings']:
        print("\n--- Warnings ---")
        for w in results['warnings']:
            print(f"  [!] {w}")

    if results['failed']:
        print("\n--- Failed Tests ---")
        for f in results['failed']:
            print(f"  [X] {f}")

    print("\n" + "=" * 60)
    total = len(results['passed']) + len(results['failed'])
    passed = len(results['passed'])
    print(f"SUMMARY: {passed}/{total} tests passed, {len(results['warnings'])} warnings")
    print("=" * 60)

    return len(results['failed']) == 0


def main():
    """Run validation on vessel network."""
    print("Generating vessel network for validation...")

    np.random.seed(42)

    # Use reasonable volume size
    vol_params = VolumeParams(vol_sz=(100, 100, 200))
    vasc_params = VascParams(
        depth_surf=15.0,
        depth_vasc=180.0,
    )

    print(f"Volume size: {vol_params.vol_sz}")
    print(f"Surface depth: {vasc_params.depth_surf}um")
    print(f"Max vessel depth: {vasc_params.depth_vasc}um")

    # Generate network
    network = simulate_blood_vessels(vol_params, vasc_params, verbose=1)

    # Validate
    results = validate_vessel_network(network, vol_params, vasc_params)

    # Print results
    success = print_validation_results(results)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

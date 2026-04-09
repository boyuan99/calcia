"""
Export blood vessel network to JSON format for Three.js visualization.
"""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from calcia.config.params import VolumeParams, VascParams
from calcia.volume.vasculature import simulate_blood_vessels


def export_vessel_network(output_path: str = "vessels_data.json"):
    """Export vessel network to JSON for Three.js."""
    print("Generating blood vessel network...")

    np.random.seed(42)

    # Parameters for a moderate-sized volume
    vol_params = VolumeParams(vol_sz=(100, 100, 200))
    vasc_params = VascParams(
        depth_surf=15.0,
        depth_vasc=180.0,
    )

    network = simulate_blood_vessels(vol_params, vasc_params, verbose=1)

    # Convert to JSON-serializable format
    data = {
        "metadata": {
            "volume_size": list(vol_params.vol_sz),
            "depth_surf": vasc_params.depth_surf,
            "depth_vasc": vasc_params.depth_vasc,
            "n_nodes": len(network.nodes),
            "n_connections": len(network.connections),
        },
        "nodes": [],
        "connections": [],
    }

    # Type names for display
    type_names = {
        0: 'Internal',
        1: 'Source',
        2: 'Branch',
        3: 'Diving',
        4: 'Capillary',
    }

    # Export nodes
    for node in network.nodes:
        data["nodes"].append({
            "id": node.num,
            "type": node.type,
            "type_name": type_names.get(node.type, 'Unknown'),
            "position": node.pos.tolist(),
            "connections": [int(c) for c in node.conn],
        })

    # Export connections (as line segments for rendering)
    for conn in network.connections:
        # Use interpolated points for smooth curves
        if len(conn.locs) > 0:
            points = conn.locs.tolist()
        else:
            # Fallback to straight line
            start_pos = network.nodes[conn.start].pos.tolist()
            end_pos = network.nodes[conn.ends].pos.tolist()
            points = [start_pos, end_pos]

        data["connections"].append({
            "start": conn.start,
            "end": conn.ends,
            "weight": conn.weight,
            "points": points,
            "start_type": network.nodes[conn.start].type,
            "end_type": network.nodes[conn.ends].type,
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Exported vessel network to {output_path}")
    print(f"  Nodes: {len(data['nodes'])}")
    print(f"  Connections: {len(data['connections'])}")

    return data


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    export_vessel_network(str(output_dir / "vessels_data.json"))
    print("\nDone! You can load this file in a Three.js viewer.")

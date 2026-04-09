"""
Legacy format support for backwards compatibility.

Handles detection and upgrade of older JSON formats to the current schema.
"""

import json
from typing import Dict, Any

from .schema import SCHEMA_VERSION


def detect_format_version(data: dict) -> str:
    """
    Detect the format version of JSON data.

    Args:
        data: Loaded JSON data dictionary.

    Returns:
        Version string or "unknown" if unrecognized.
    """
    # Check for explicit version
    if "format_version" in data:
        return data["format_version"]

    # Legacy neuron format (from export_mesh.py)
    if "neurons" in data and isinstance(data["neurons"], list):
        if data["neurons"] and "soma" in data["neurons"][0]:
            return "0.1.0"  # Legacy visualization format

    # Legacy vessel format (from export_vessels.py)
    if "metadata" in data and "nodes" in data:
        return "0.1.0"  # Legacy vessel format

    # Very old format with just raw arrays
    if "neurons" in data and isinstance(data["neurons"], list):
        return "0.0.1"

    return "unknown"


def upgrade_legacy_format(data: dict) -> dict:
    """
    Upgrade legacy JSON format to current schema.

    Args:
        data: Legacy format data.

    Returns:
        Data in current schema format.
    """
    version = detect_format_version(data)

    if version == "unknown":
        # Return as-is with minimal wrapper
        return {
            "format_version": SCHEMA_VERSION,
            "metadata": {
                "created_at": None,
                "calcia_version": "unknown",
                "format_version": SCHEMA_VERSION,
                "random_seed": None,
                "description": "Imported from unknown format",
                "components_included": [],
            },
            "parameters": {},
            "components": {},
        }

    if version.startswith("0."):
        # Detect if it's neurons or vessels
        if "neurons" in data and isinstance(data["neurons"], list):
            if data["neurons"] and "soma" in data["neurons"][0]:
                return _upgrade_legacy_neurons(data)

        if "nodes" in data:
            return _upgrade_legacy_vessels(data)

    # Already current version
    return data


def _upgrade_legacy_neurons(legacy_data: dict) -> dict:
    """Convert legacy neuron JSON to new schema."""
    neurons = legacy_data.get("neurons", [])

    # Check if neurons have faces embedded
    has_shared_faces = False
    shared_faces = None
    if neurons and "soma" in neurons[0]:
        if "faces" in neurons[0]["soma"]:
            shared_faces = neurons[0]["soma"]["faces"]
            has_shared_faces = True

    items = []
    for i, neuron in enumerate(neurons):
        item = {
            "id": i,
            "type": neuron.get("type", neuron.get("name", "pyramidal")),
            "position": neuron.get("position", [0, 0, 0]),
            "rotation": neuron.get("rotation", [0, 0, 0]),
            "soma": {
                "vertices": neuron["soma"]["vertices"],
                "faces_ref": "shared" if has_shared_faces else "embedded",
            },
            "nucleus": {
                "vertices": neuron["nucleus"]["vertices"],
                "faces_ref": "shared" if has_shared_faces else "embedded",
            },
        }

        # Keep embedded faces if not using shared
        if not has_shared_faces and "faces" in neuron["soma"]:
            item["soma"]["faces"] = neuron["soma"]["faces"]
        if not has_shared_faces and "faces" in neuron["nucleus"]:
            item["nucleus"]["faces"] = neuron["nucleus"]["faces"]

        items.append(item)

    result = {
        "format_version": SCHEMA_VERSION,
        "metadata": {
            "created_at": None,
            "calcia_version": "unknown",
            "format_version": SCHEMA_VERSION,
            "random_seed": None,
            "description": "Imported from legacy neuron format",
            "components_included": ["neurons"],
        },
        "parameters": {},
        "components": {
            "neurons": {
                "count": len(items),
                "mesh_format": "indexed_triangle_list",
                "coordinate_system": "micrometers",
                "shared_triangulation": {
                    "enabled": has_shared_faces,
                    "faces": shared_faces or [],
                },
                "items": items,
            }
        },
    }

    return result


def _upgrade_legacy_vessels(legacy_data: dict) -> dict:
    """Convert legacy vessel JSON to new schema."""
    nodes = []
    for node in legacy_data.get("nodes", []):
        nodes.append({
            "id": node["id"],
            "type": node.get("type", 0),
            "position": node["position"],
            "connections": node.get("connections", []),
            "root": -1,
            "misc": {},
        })

    connections = []
    for i, conn in enumerate(legacy_data.get("connections", [])):
        connections.append({
            "id": i,
            "start": conn["start"],
            "end": conn["end"],
            "weight": conn.get("weight", 1.0),
            "points": conn.get("points", []),
        })

    # Extract metadata if available
    metadata_src = legacy_data.get("metadata", {})

    result = {
        "format_version": SCHEMA_VERSION,
        "metadata": {
            "created_at": None,
            "calcia_version": "unknown",
            "format_version": SCHEMA_VERSION,
            "random_seed": None,
            "description": "Imported from legacy vessel format",
            "components_included": ["vessels"],
        },
        "parameters": {
            "volume": {
                "vol_sz": metadata_src.get("volume_size", [100, 100, 200]),
            },
            "vasculature": {
                "depth_surf": metadata_src.get("depth_surf", 15.0),
                "depth_vasc": metadata_src.get("depth_vasc", 200.0),
            },
        },
        "components": {
            "vessels": {
                "count_nodes": len(nodes),
                "count_connections": len(connections),
                "coordinate_system": "micrometers",
                "node_types": {
                    "0": "internal",
                    "1": "source",
                    "2": "branch",
                    "3": "diving",
                    "4": "capillary",
                },
                "nodes": nodes,
                "connections": connections,
            }
        },
    }

    return result

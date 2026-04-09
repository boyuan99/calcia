"""
Export volume simulation data.

Provides functions to export neurons, vessels, and complete pipeline
output to JSON or NPZ format.
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from pathlib import Path

from .schema import SCHEMA_VERSION, params_to_dict, create_metadata

if TYPE_CHECKING:
    from ..pipeline import NeuralVolumeOutput


def export_volume(
    output_path: str,
    neurons: Optional[List[tuple]] = None,
    neuron_positions: Optional[np.ndarray] = None,
    neuron_angles: Optional[List[np.ndarray]] = None,
    vessel_network: Optional[Any] = None,
    vol_params: Optional[Any] = None,
    neur_params: Optional[Any] = None,
    vasc_params: Optional[Any] = None,
    random_seed: Optional[int] = None,
    description: str = "Neural volume simulation output",
    indent: int = 2,
) -> dict:
    """
    Export complete volume simulation to unified JSON format.

    Args:
        output_path: Path to output JSON file.
        neurons: List of (Vcell, Vnuc, faces) tuples from generate_multiple_neurons.
        neuron_positions: (N, 3) array of neuron center positions.
        neuron_angles: List of rotation angle arrays.
        vessel_network: VesselNetwork object from simulate_blood_vessels.
        vol_params: VolumeParams used for simulation.
        neur_params: NeuronParams used for simulation.
        vasc_params: VascParams used for simulation.
        random_seed: Random seed used for reproducibility.
        description: Description of the export.
        indent: JSON indentation (None for compact).

    Returns:
        The exported data dictionary.
    """
    # Determine which components are included
    components_included = []
    if neurons:
        components_included.append("neurons")
    if vessel_network:
        components_included.append("vessels")

    # Build the export data structure
    data = {
        "format_version": SCHEMA_VERSION,
        "metadata": create_metadata(
            random_seed=random_seed,
            components_included=components_included,
            description=description,
        ),
        "parameters": {},
        "components": {},
    }

    # Export parameters
    if vol_params:
        data["parameters"]["volume"] = params_to_dict(vol_params)
    if neur_params:
        data["parameters"]["neuron"] = params_to_dict(neur_params)
    if vasc_params:
        data["parameters"]["vasculature"] = params_to_dict(vasc_params)

    # Export neurons
    if neurons:
        data["components"]["neurons"] = _export_neurons(
            neurons, neuron_positions, neuron_angles, neur_params
        )

    # Export vessels
    if vessel_network:
        data["components"]["vessels"] = _export_vessels(vessel_network)

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)

    return data


def _export_neurons(
    neurons: List[tuple],
    positions: Optional[np.ndarray],
    angles: Optional[List[np.ndarray]],
    neur_params: Optional[Any],
) -> dict:
    """Export neurons component to dict format."""
    if not neurons:
        return {"count": 0, "items": []}

    # Extract shared triangulation from first neuron
    _, _, shared_faces = neurons[0]

    result = {
        "count": len(neurons),
        "mesh_format": "indexed_triangle_list",
        "coordinate_system": "micrometers",
        "shared_triangulation": {
            "enabled": True,
            "faces": shared_faces.tolist() if isinstance(shared_faces, np.ndarray) else shared_faces,
        },
        "items": [],
    }

    # Get neuron type from params
    neur_type = "pyramidal"
    if neur_params and hasattr(neur_params, 'neur_type'):
        neur_type = neur_params.neur_type

    for i, neuron_data in enumerate(neurons):
        Vcell, Vnuc, faces = neuron_data

        # Get position and rotation
        pos = [0.0, 0.0, 0.0]
        if positions is not None and i < len(positions):
            pos = positions[i].tolist() if isinstance(positions[i], np.ndarray) else list(positions[i])

        rot = [0.0, 0.0, 0.0]
        if angles is not None and i < len(angles):
            rot = angles[i].tolist() if isinstance(angles[i], np.ndarray) else list(angles[i])

        item = {
            "id": i,
            "type": neur_type,
            "position": pos,
            "rotation": rot,
            "soma": {
                "vertices": Vcell.tolist() if isinstance(Vcell, np.ndarray) else Vcell,
                "faces_ref": "shared",
            },
            "nucleus": {
                "vertices": Vnuc.tolist() if isinstance(Vnuc, np.ndarray) else Vnuc,
                "faces_ref": "shared",
            },
        }

        # Add statistics if available
        try:
            from ..volume.neurons import compute_neuron_statistics
            stats = compute_neuron_statistics(Vcell, Vnuc)
            item["statistics"] = {
                k: float(v) if v is not None else None
                for k, v in stats.items()
            }
        except Exception:
            pass

        result["items"].append(item)

    return result


def _export_vessels(network: Any) -> dict:
    """Export vessel network to dict format."""
    type_names = {
        0: "internal",
        1: "source",
        2: "branch",
        3: "diving",
        4: "capillary",
    }

    result = {
        "count_nodes": len(network.nodes),
        "count_connections": len(network.connections),
        "coordinate_system": "micrometers",
        "node_types": type_names,
        "nodes": [],
        "connections": [],
    }

    # Export nodes
    for node in network.nodes:
        node_data = {
            "id": int(node.num),
            "type": int(node.type),
            "position": node.pos.tolist() if isinstance(node.pos, np.ndarray) else list(node.pos),
            "connections": [int(c) for c in node.conn],
            "root": int(node.root),
        }

        # Handle misc field
        if node.misc:
            misc_data = {}
            for k, v in node.misc.items():
                if isinstance(v, np.ndarray):
                    misc_data[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    misc_data[k] = v.item()
                else:
                    misc_data[k] = v
            node_data["misc"] = misc_data

        result["nodes"].append(node_data)

    # Export connections
    for i, conn in enumerate(network.connections):
        conn_data = {
            "id": i,
            "start": int(conn.start),
            "end": int(conn.ends),
            "weight": float(conn.weight),
        }

        # Export interpolated points
        if hasattr(conn, 'locs') and len(conn.locs) > 0:
            conn_data["points"] = conn.locs.tolist() if isinstance(conn.locs, np.ndarray) else conn.locs
        else:
            conn_data["points"] = []

        result["connections"].append(conn_data)

    return result


def export_neurons(
    output_path: str,
    neurons: List[tuple],
    positions: Optional[np.ndarray] = None,
    angles: Optional[List[np.ndarray]] = None,
    neur_params: Optional[Any] = None,
    random_seed: Optional[int] = None,
    indent: int = 2,
) -> dict:
    """
    Export neurons only to JSON format.

    Args:
        output_path: Path to output JSON file.
        neurons: List of (Vcell, Vnuc, faces) tuples.
        positions: (N, 3) array of neuron positions.
        angles: List of rotation angle arrays.
        neur_params: NeuronParams used.
        random_seed: Random seed used.
        indent: JSON indentation.

    Returns:
        The exported data dictionary.
    """
    return export_volume(
        output_path,
        neurons=neurons,
        neuron_positions=positions,
        neuron_angles=angles,
        neur_params=neur_params,
        random_seed=random_seed,
        description="Neuron shapes export",
        indent=indent,
    )


def export_vessels(
    output_path: str,
    vessel_network: Any,
    vol_params: Optional[Any] = None,
    vasc_params: Optional[Any] = None,
    random_seed: Optional[int] = None,
    indent: int = 2,
) -> dict:
    """
    Export vessel network only to JSON format.

    Args:
        output_path: Path to output JSON file.
        vessel_network: VesselNetwork object.
        vol_params: VolumeParams used.
        vasc_params: VascParams used.
        random_seed: Random seed used.
        indent: JSON indentation.

    Returns:
        The exported data dictionary.
    """
    return export_volume(
        output_path,
        vessel_network=vessel_network,
        vol_params=vol_params,
        vasc_params=vasc_params,
        random_seed=random_seed,
        description="Vessel network export",
        indent=indent,
    )


# ---------------------------------------------------------------------------
# Pipeline output export (NeuralVolumeOutput)
# ---------------------------------------------------------------------------

def export_pipeline_output(
    output_path: str,
    output: "NeuralVolumeOutput",
    random_seed: Optional[int] = None,
) -> None:
    """Export a complete NeuralVolumeOutput to file.

    Format is auto-detected from the file extension:
    - ``.npz``: Compressed numpy archive (recommended, small & fast).
    - ``.json``: Human-readable JSON (large, slow for big volumes).

    Args:
        output_path: Destination file path (.npz or .json).
        output: The NeuralVolumeOutput from simulate_neural_volume().
        random_seed: Optional seed used for the simulation.

    Raises:
        ValueError: If the file extension is not .npz or .json.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".npz":
        _export_pipeline_npz(path, output, random_seed)
    elif ext == ".json":
        _export_pipeline_json(path, output, random_seed)
    else:
        raise ValueError(
            f"Unsupported extension '{ext}'. Use .npz or .json."
        )


def _build_metadata(output: "NeuralVolumeOutput",
                    random_seed: Optional[int]) -> dict:
    """Build the shared metadata dict for both formats."""
    params_dict = {}
    for name, obj in output.params.items():
        params_dict[name] = params_to_dict(obj)

    return {
        "format_version": SCHEMA_VERSION,
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "calcia_version": "0.0.1",
        "random_seed": random_seed,
        "parameters": params_dict,
    }


def _pack_variable_lists(arrays):
    """Concatenate a list of 1-D arrays and return (data, offsets).

    ``offsets`` has length ``len(arrays) + 1`` so that element *i* spans
    ``data[offsets[i]:offsets[i+1]]``.  Empty input returns empty arrays.
    """
    if not arrays:
        return np.array([], dtype=np.int32), np.array([0], dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for i, a in enumerate(arrays):
        offsets[i + 1] = offsets[i] + len(a)
    data = np.concatenate(arrays) if offsets[-1] > 0 else np.array([])
    return data, offsets


# ---- NPZ format ----------------------------------------------------------

def _export_pipeline_npz(path: Path, output: "NeuralVolumeOutput",
                         random_seed: Optional[int]) -> None:
    meta = _build_metadata(output, random_seed)
    meta_bytes = json.dumps(meta).encode("utf-8")

    arrays: Dict[str, np.ndarray] = {
        "_metadata": np.frombuffer(meta_bytes, dtype=np.uint8),
        "neur_vol": output.neur_vol,
        "neur_num": output.neur_num,
        "neur_num_ad": output.neur_num_ad,
        "locs": output.locs,
    }

    if output.neur_ves is not None:
        arrays["neur_ves"] = output.neur_ves

    # gp_vals: List[CellFluorescenceData]
    gv_indices, gv_idx_off = _pack_variable_lists(
        [g.indices for g in output.gp_vals])
    gv_fluor, _ = _pack_variable_lists(
        [g.fluorescence for g in output.gp_vals])
    gv_soma, _ = _pack_variable_lists(
        [g.soma_mask.view(np.uint8) for g in output.gp_vals])
    arrays["gp_vals_indices"] = gv_indices
    arrays["gp_vals_fluorescence"] = gv_fluor
    arrays["gp_vals_soma_mask"] = gv_soma
    arrays["gp_vals_offsets"] = gv_idx_off

    # gp_nuc: List[Tuple[ndarray, float]]
    nuc_indices, nuc_off = _pack_variable_lists(
        [n[0] for n in output.gp_nuc])
    nuc_values = np.array([n[1] for n in output.gp_nuc], dtype=np.float64)
    arrays["gp_nuc_indices"] = nuc_indices
    arrays["gp_nuc_offsets"] = nuc_off
    arrays["gp_nuc_values"] = nuc_values

    # gp_soma: list of (soma_indices, smoothed_body) tuples
    soma_idx_data, soma_idx_off = _pack_variable_lists(
        [s[0] for s in output.gp_soma])
    soma_body_data, soma_body_off = _pack_variable_lists(
        [s[1] for s in output.gp_soma])
    arrays["gp_soma_indices"] = soma_idx_data
    arrays["gp_soma_indices_offsets"] = soma_idx_off
    arrays["gp_soma_body"] = soma_body_data
    arrays["gp_soma_body_offsets"] = soma_body_off

    # gp_bgvals: List[Tuple[ndarray, ndarray]]
    bg_indices, bg_off = _pack_variable_lists(
        [b[0] for b in output.gp_bgvals])
    bg_fluor, _ = _pack_variable_lists(
        [b[1] for b in output.gp_bgvals])
    arrays["gp_bgvals_indices"] = bg_indices
    arrays["gp_bgvals_fluorescence"] = bg_fluor
    arrays["gp_bgvals_offsets"] = bg_off

    # bg_proc: List[BgProcessData]
    bp_indices, bp_off = _pack_variable_lists(
        [p.indices for p in output.bg_proc])
    bp_fluor, _ = _pack_variable_lists(
        [p.fluorescence for p in output.bg_proc])
    arrays["bg_proc_indices"] = bp_indices
    arrays["bg_proc_fluorescence"] = bp_fluor
    arrays["bg_proc_offsets"] = bp_off

    np.savez_compressed(str(path), **arrays)


# ---- JSON format ----------------------------------------------------------

def _array_to_json(arr: np.ndarray) -> dict:
    """Serialize a numpy array as {shape, dtype, data}."""
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.ravel().tolist(),
    }


def _export_pipeline_json(path: Path, output: "NeuralVolumeOutput",
                          random_seed: Optional[int]) -> None:
    meta = _build_metadata(output, random_seed)

    data: Dict[str, Any] = {
        "format_version": meta["format_version"],
        "metadata": {
            "created_at": meta["created_at"],
            "calcia_version": meta["calcia_version"],
            "random_seed": meta["random_seed"],
        },
        "parameters": meta["parameters"],
        "volumes": {
            "neur_vol": _array_to_json(output.neur_vol),
            "neur_num": _array_to_json(output.neur_num),
            "neur_num_ad": _array_to_json(output.neur_num_ad),
            "neur_ves": _array_to_json(output.neur_ves) if output.neur_ves is not None else None,
            "locs": _array_to_json(output.locs),
        },
        "gp_vals": [
            {
                "indices": g.indices.tolist(),
                "fluorescence": g.fluorescence.tolist(),
                "soma_mask": g.soma_mask.tolist(),
            }
            for g in output.gp_vals
        ],
        "gp_nuc": [
            {"indices": n[0].tolist(), "value": float(n[1])}
            for n in output.gp_nuc
        ],
        "gp_soma": [
            {
                "indices": s[0].tolist() if isinstance(s[0], np.ndarray) else list(s[0]),
                "body": s[1].tolist() if isinstance(s[1], np.ndarray) else list(s[1]),
            }
            for s in output.gp_soma
        ],
        "gp_bgvals": [
            {"indices": b[0].tolist(), "fluorescence": b[1].tolist()}
            for b in output.gp_bgvals
        ],
        "bg_proc": [
            {"indices": p.indices.tolist(), "fluorescence": p.fluorescence.tolist()}
            for p in output.bg_proc
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f)

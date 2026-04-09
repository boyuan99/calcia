"""
Import volume simulation data from JSON or NPZ format.

Provides functions to load and reconstruct neurons, vessels, and
complete pipeline output from standardized formats.
"""

import json
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path

from .schema import SCHEMA_VERSION, validate_format_version, dict_to_params

if TYPE_CHECKING:
    from ..pipeline import NeuralVolumeOutput


def import_volume(input_path: str) -> dict:
    """
    Import complete volume from JSON file.

    Args:
        input_path: Path to JSON file.

    Returns:
        Dictionary with reconstructed objects:
        - 'metadata': Metadata dictionary
        - 'parameters': Dict of parameter dataclasses
        - 'neurons': List of (Vcell, Vnuc, faces, angles) tuples (if present)
        - 'neuron_positions': (N, 3) array (if neurons present)
        - 'vessel_network': VesselNetwork object (if present)

    Raises:
        ValueError: If format version is incompatible.
        FileNotFoundError: If input file doesn't exist.
    """
    with open(input_path) as f:
        data = json.load(f)

    # Check format version
    version = data.get("format_version", "0.0.0")
    if not validate_format_version(version):
        # Try legacy import
        from .legacy import upgrade_legacy_format
        data = upgrade_legacy_format(data)

    result = {
        "metadata": data.get("metadata", {}),
        "parameters": _import_parameters(data.get("parameters", {})),
    }

    # Import neurons
    if "neurons" in data.get("components", {}):
        neurons, positions = _import_neurons(data["components"]["neurons"])
        result["neurons"] = neurons
        result["neuron_positions"] = positions

    # Import vessels
    if "vessels" in data.get("components", {}):
        result["vessel_network"] = _import_vessels(data["components"]["vessels"])

    return result


def _import_parameters(params_data: dict) -> dict:
    """Reconstruct parameter dataclasses from JSON."""
    from ..config.params import VolumeParams, NeuronParams, VascParams

    result = {}

    if "volume" in params_data:
        try:
            result["volume"] = dict_to_params(params_data["volume"], VolumeParams)
        except Exception:
            result["volume"] = params_data["volume"]

    if "neuron" in params_data:
        try:
            result["neuron"] = dict_to_params(params_data["neuron"], NeuronParams)
        except Exception:
            result["neuron"] = params_data["neuron"]

    if "vasculature" in params_data:
        try:
            result["vasculature"] = dict_to_params(params_data["vasculature"], VascParams)
        except Exception:
            result["vasculature"] = params_data["vasculature"]

    return result


def _import_neurons(neurons_data: dict) -> Tuple[List[tuple], np.ndarray]:
    """Reconstruct neuron meshes from JSON."""
    neurons = []
    positions = []

    # Get shared faces
    shared_faces = None
    if neurons_data.get("shared_triangulation", {}).get("enabled"):
        faces_data = neurons_data["shared_triangulation"]["faces"]
        shared_faces = np.array(faces_data, dtype=np.int32)

    for item in neurons_data.get("items", []):
        # Reconstruct vertices
        Vcell = np.array(item["soma"]["vertices"], dtype=np.float64)
        Vnuc = np.array(item["nucleus"]["vertices"], dtype=np.float64)

        # Get faces
        if item["soma"].get("faces_ref") == "shared" and shared_faces is not None:
            faces = shared_faces
        elif "faces" in item["soma"]:
            faces = np.array(item["soma"]["faces"], dtype=np.int32)
        else:
            faces = np.array([], dtype=np.int32)

        # Get rotation
        angles = np.array(item.get("rotation", [0, 0, 0]), dtype=np.float64)

        neurons.append((Vcell, Vnuc, faces, angles))
        positions.append(item.get("position", [0, 0, 0]))

    return neurons, np.array(positions, dtype=np.float64)


def _import_vessels(vessels_data: dict):
    """Reconstruct VesselNetwork from JSON."""
    from ..volume.vasculature import VesselNode, VesselConnection, VesselNetwork

    nodes = []
    connections = []

    # Import nodes
    for node_data in vessels_data.get("nodes", []):
        node = VesselNode(
            num=node_data["id"],
            root=node_data.get("root", -1),
            conn=node_data.get("connections", []),
            pos=np.array(node_data["position"], dtype=np.float64),
            type=node_data.get("type", 0),
            misc=node_data.get("misc", {}),
        )
        nodes.append(node)

    # Import connections
    for conn_data in vessels_data.get("connections", []):
        points = conn_data.get("points", [])
        locs = np.array(points, dtype=np.float64) if points else np.array([])

        conn = VesselConnection(
            start=conn_data["start"],
            ends=conn_data["end"],
            weight=conn_data.get("weight", 1.0),
            locs=locs,
        )
        connections.append(conn)

    return VesselNetwork(nodes=nodes, connections=connections)


def import_neurons(input_path: str) -> Tuple[List[tuple], np.ndarray]:
    """
    Import neurons only from JSON file.

    Args:
        input_path: Path to JSON file.

    Returns:
        Tuple of (neurons, positions) where:
        - neurons: List of (Vcell, Vnuc, faces, angles) tuples
        - positions: (N, 3) array of neuron positions
    """
    result = import_volume(input_path)
    return result.get("neurons", []), result.get("neuron_positions", np.array([]))


def import_vessels(input_path: str):
    """
    Import vessel network only from JSON file.

    Args:
        input_path: Path to JSON file.

    Returns:
        VesselNetwork object.
    """
    from ..volume.vasculature import VesselNetwork

    result = import_volume(input_path)
    return result.get("vessel_network", VesselNetwork())


# ---------------------------------------------------------------------------
# Pipeline output import (NeuralVolumeOutput)
# ---------------------------------------------------------------------------

def import_pipeline_output(input_path: str) -> "NeuralVolumeOutput":
    """Import a NeuralVolumeOutput from file.

    Format is auto-detected from the file extension:
    - ``.npz``: Compressed numpy archive.
    - ``.json``: Human-readable JSON.

    Args:
        input_path: Path to .npz or .json file.

    Returns:
        Reconstructed NeuralVolumeOutput.

    Raises:
        ValueError: If the file extension is not .npz or .json.
    """
    path = Path(input_path)
    ext = path.suffix.lower()

    if ext == ".npz":
        return _import_pipeline_npz(path)
    elif ext == ".json":
        return _import_pipeline_json(path)
    else:
        raise ValueError(
            f"Unsupported extension '{ext}'. Use .npz or .json."
        )


def _reconstruct_params(params_data: dict) -> dict:
    """Reconstruct all 6 parameter dataclasses from a dict."""
    from ..config.params import (
        VolumeParams, NeuronParams, VascParams,
        DendParams, BgParams, AxonParams,
    )
    _class_map = {
        "vol_params": VolumeParams,
        "neur_params": NeuronParams,
        "vasc_params": VascParams,
        "dend_params": DendParams,
        "bg_params": BgParams,
        "axon_params": AxonParams,
    }
    result = {}
    for key, cls in _class_map.items():
        if key in params_data:
            try:
                result[key] = dict_to_params(params_data[key], cls)
            except Exception:
                result[key] = params_data[key]
    return result


def _unpack_variable_lists(data, offsets):
    """Split a concatenated array back into a list of arrays using offsets."""
    result = []
    for i in range(len(offsets) - 1):
        start, end = int(offsets[i]), int(offsets[i + 1])
        result.append(data[start:end])
    return result


# ---- NPZ import -----------------------------------------------------------

def _import_pipeline_npz(path: Path) -> "NeuralVolumeOutput":
    from ..pipeline import NeuralVolumeOutput
    from ..volume.fluorescence import CellFluorescenceData
    from ..volume.background import BgProcessData

    npz = np.load(str(path), allow_pickle=False)

    # Metadata
    meta_bytes = npz["_metadata"].tobytes()
    meta = json.loads(meta_bytes)
    params = _reconstruct_params(meta.get("parameters", {}))

    # Direct arrays
    neur_vol = npz["neur_vol"]
    neur_num = npz["neur_num"]
    neur_num_ad = npz["neur_num_ad"]
    locs = npz["locs"]
    neur_ves = npz["neur_ves"] if "neur_ves" in npz else None

    # gp_vals
    gv_offsets = npz["gp_vals_offsets"]
    gv_indices_list = _unpack_variable_lists(
        npz["gp_vals_indices"], gv_offsets)
    gv_fluor_list = _unpack_variable_lists(
        npz["gp_vals_fluorescence"], gv_offsets)
    gv_soma_list = _unpack_variable_lists(
        npz["gp_vals_soma_mask"], gv_offsets)
    gp_vals = [
        CellFluorescenceData(
            indices=idx.astype(np.int32),
            fluorescence=fl.astype(np.float32),
            soma_mask=sm.astype(bool),
        )
        for idx, fl, sm in zip(gv_indices_list, gv_fluor_list, gv_soma_list)
    ]

    # gp_nuc
    nuc_offsets = npz["gp_nuc_offsets"]
    nuc_indices_list = _unpack_variable_lists(
        npz["gp_nuc_indices"], nuc_offsets)
    nuc_values = npz["gp_nuc_values"]
    gp_nuc = [
        (idx.astype(np.int32), float(val))
        for idx, val in zip(nuc_indices_list, nuc_values)
    ]

    # gp_soma: list of (soma_indices, smoothed_body) tuples
    soma_idx_list = _unpack_variable_lists(
        npz["gp_soma_indices"], npz["gp_soma_indices_offsets"])
    soma_body_list = _unpack_variable_lists(
        npz["gp_soma_body"], npz["gp_soma_body_offsets"])
    gp_soma = [
        (idx.astype(np.int32), body.astype(np.int32))
        for idx, body in zip(soma_idx_list, soma_body_list)
    ]

    # gp_bgvals
    bg_offsets = npz["gp_bgvals_offsets"]
    bg_indices_list = _unpack_variable_lists(
        npz["gp_bgvals_indices"], bg_offsets)
    bg_fluor_list = _unpack_variable_lists(
        npz["gp_bgvals_fluorescence"], bg_offsets)
    gp_bgvals = [
        (idx.astype(np.int32), fl.astype(np.float32))
        for idx, fl in zip(bg_indices_list, bg_fluor_list)
    ]

    # bg_proc
    bp_offsets = npz["bg_proc_offsets"]
    bp_indices_list = _unpack_variable_lists(
        npz["bg_proc_indices"], bp_offsets)
    bp_fluor_list = _unpack_variable_lists(
        npz["bg_proc_fluorescence"], bp_offsets)
    bg_proc = [
        BgProcessData(
            indices=idx.astype(np.int32),
            fluorescence=fl.astype(np.float32),
        )
        for idx, fl in zip(bp_indices_list, bp_fluor_list)
    ]

    return NeuralVolumeOutput(
        neur_vol=neur_vol,
        gp_nuc=gp_nuc,
        gp_soma=gp_soma,
        gp_vals=gp_vals,
        neur_ves=neur_ves,
        bg_proc=bg_proc,
        locs=locs,
        neur_num=neur_num,
        neur_num_ad=neur_num_ad,
        gp_bgvals=gp_bgvals,
        params=params,
    )


# ---- JSON import -----------------------------------------------------------

def _json_to_array(obj: dict) -> np.ndarray:
    """Reconstruct a numpy array from {shape, dtype, data}."""
    return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])


def _import_pipeline_json(path: Path) -> "NeuralVolumeOutput":
    from ..pipeline import NeuralVolumeOutput
    from ..volume.fluorescence import CellFluorescenceData
    from ..volume.background import BgProcessData

    with open(path) as f:
        data = json.load(f)

    params = _reconstruct_params(data.get("parameters", {}))

    vols = data["volumes"]
    neur_vol = _json_to_array(vols["neur_vol"])
    neur_num = _json_to_array(vols["neur_num"])
    neur_num_ad = _json_to_array(vols["neur_num_ad"])
    locs = _json_to_array(vols["locs"])
    neur_ves = _json_to_array(vols["neur_ves"]) if vols.get("neur_ves") else None

    gp_vals = [
        CellFluorescenceData(
            indices=np.array(g["indices"], dtype=np.int32),
            fluorescence=np.array(g["fluorescence"], dtype=np.float32),
            soma_mask=np.array(g["soma_mask"], dtype=bool),
        )
        for g in data["gp_vals"]
    ]

    gp_nuc = [
        (np.array(n["indices"], dtype=np.int32), float(n["value"]))
        for n in data["gp_nuc"]
    ]

    gp_soma = [
        (np.array(s["indices"], dtype=np.int32),
         np.array(s["body"], dtype=np.int32))
        for s in data["gp_soma"]
    ]

    gp_bgvals = [
        (np.array(b["indices"], dtype=np.int32),
         np.array(b["fluorescence"], dtype=np.float32))
        for b in data["gp_bgvals"]
    ]

    bg_proc = [
        BgProcessData(
            indices=np.array(p["indices"], dtype=np.int32),
            fluorescence=np.array(p["fluorescence"], dtype=np.float32),
        )
        for p in data["bg_proc"]
    ]

    return NeuralVolumeOutput(
        neur_vol=neur_vol,
        gp_nuc=gp_nuc,
        gp_soma=gp_soma,
        gp_vals=gp_vals,
        neur_ves=neur_ves,
        bg_proc=bg_proc,
        locs=locs,
        neur_num=neur_num,
        neur_num_ad=neur_num_ad,
        gp_bgvals=gp_bgvals,
        params=params,
    )

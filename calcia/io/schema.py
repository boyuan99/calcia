"""
JSON schema definitions and validation for volume data.

Defines the schema version and provides validation utilities for
the unified volume export format.
"""

from dataclasses import dataclass, asdict, fields
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

# Schema version (semver)
SCHEMA_VERSION = "1.0.0"


def params_to_dict(params) -> dict:
    """
    Convert a dataclass params object to JSON-serializable dict.

    Handles numpy arrays, tuples, and nested dataclasses.

    Args:
        params: A dataclass instance or dict.

    Returns:
        JSON-serializable dictionary.
    """
    import numpy as np

    if params is None:
        return {}

    if isinstance(params, dict):
        return {k: _serialize_value(v) for k, v in params.items()}

    if hasattr(params, '__dataclass_fields__'):
        result = {}
        for field in fields(params):
            value = getattr(params, field.name)
            result[field.name] = _serialize_value(value)
        return result

    return {}


def _serialize_value(value) -> Any:
    """Serialize a single value to JSON-compatible format."""
    import numpy as np

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if hasattr(value, '__dataclass_fields__'):
        return params_to_dict(value)
    return value


def dict_to_params(data: dict, param_class):
    """
    Convert a dictionary back to a dataclass params object.

    Args:
        data: Dictionary with parameter values.
        param_class: The dataclass type to instantiate.

    Returns:
        Instance of param_class.
    """
    if data is None:
        return None

    # Convert lists back to tuples where needed
    converted = {}
    for field in fields(param_class):
        if field.name in data:
            value = data[field.name]
            # Check if the field type is a tuple
            if hasattr(field.type, '__origin__') and field.type.__origin__ is tuple:
                if isinstance(value, list):
                    value = tuple(value)
            # Handle nested dataclasses
            elif hasattr(field.type, '__dataclass_fields__'):
                if isinstance(value, dict):
                    value = dict_to_params(value, field.type)
            converted[field.name] = value

    return param_class(**converted)


def create_metadata(
    random_seed: Optional[int] = None,
    components_included: Optional[List[str]] = None,
    description: str = "Neural volume simulation output",
) -> dict:
    """
    Create metadata dictionary for export.

    Args:
        random_seed: Random seed used for generation.
        components_included: List of component names included.
        description: Description of the export.

    Returns:
        Metadata dictionary.
    """
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "calcia_version": "0.0.1",
        "format_version": SCHEMA_VERSION,
        "random_seed": random_seed,
        "description": description,
        "components_included": components_included or [],
    }


def validate_format_version(version: str) -> bool:
    """
    Check if a format version is compatible with current schema.

    Uses semantic versioning - major version must match.

    Args:
        version: Version string to check.

    Returns:
        True if compatible.
    """
    try:
        file_major = int(version.split('.')[0])
        current_major = int(SCHEMA_VERSION.split('.')[0])
        return file_major == current_major
    except (ValueError, IndexError):
        return False

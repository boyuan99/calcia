"""
I/O utilities for volume data import/export.

Provides functions to save and load volume simulation data
in a unified JSON format.

Example usage:
    >>> from calcia.io import export_volume, import_volume
    >>> from calcia.volume.neurons import generate_multiple_neurons
    >>> from calcia.volume.vasculature import simulate_blood_vessels
    >>>
    >>> # Generate data
    >>> neurons, angles, positions = generate_multiple_neurons(10, neur_params)
    >>> network = simulate_blood_vessels(vol_params, vasc_params)
    >>>
    >>> # Export
    >>> export_volume("output.json", neurons=neurons, vessel_network=network)
    >>>
    >>> # Import
    >>> data = import_volume("output.json")
    >>> neurons = data["neurons"]
    >>> network = data["vessel_network"]
"""

from .schema import SCHEMA_VERSION, params_to_dict, create_metadata
from .export import (
    export_volume,
    export_neurons,
    export_vessels,
    export_pipeline_output,
)
from .import_ import (
    import_volume,
    import_neurons,
    import_vessels,
    import_pipeline_output,
)
from .legacy import (
    detect_format_version,
    upgrade_legacy_format,
)
from .render import (
    load_phase1,
    make_video,
    save_tif,
    save_tiff_normalized,
)
from .bundle import save_full_bundle, write_summary_report

__all__ = [
    # Schema
    "SCHEMA_VERSION",
    "params_to_dict",
    "create_metadata",
    # Export
    "export_volume",
    "export_neurons",
    "export_vessels",
    "export_pipeline_output",
    # Import
    "import_volume",
    "import_neurons",
    "import_vessels",
    "import_pipeline_output",
    # Legacy
    "detect_format_version",
    "upgrade_legacy_format",
    # Rendering / movie IO
    "load_phase1",
    "make_video",
    "save_tif",
    "save_tiff_normalized",
    # Full run bundle + report
    "save_full_bundle",
    "write_summary_report",
]

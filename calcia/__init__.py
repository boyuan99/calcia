"""
Calcia - Python implementation of neural volume simulation.

Python implementation of neural volume simulation.
"""

__version__ = "0.0.1"

# Convenience imports for I/O functions
from .io import export_volume, import_volume
from .io import export_pipeline_output, import_pipeline_output

# Main pipeline entry point
from .pipeline import NeuralVolumeOutput, simulate_neural_volume

# Phase 2: optical propagation
from .optics import OpticalPropagationResult, simulate_optical_propagation

# Phase 3: time-trace generation
from .traces import TimeTracesResult, generate_time_traces

# Phase 4: scanning simulation
from .scanning import ScanResult, scan_volume, scan_widefield

# Widefield (single-photon) parameter classes
from .config.params import CameraNoiseParams, MotionParams, WidefieldParams

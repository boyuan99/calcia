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

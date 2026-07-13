"""Configuration and parameter management."""

from .params import (
    VolumeParams,
    NeuronParams,
    VascParams,
    DendParams,
    BgParams,
    AxonParams,
    PsfParams,
    TpmParams,
    SpikeParams,
    CalciumParams,
)
from .indicator_presets import STATIC_PRESETS, REAL_TARGETS

__all__ = [
    "VolumeParams",
    "NeuronParams",
    "VascParams",
    "DendParams",
    "BgParams",
    "AxonParams",
    "PsfParams",
    "TpmParams",
    "SpikeParams",
    "CalciumParams",
    # Static-indicator presets (tdt / bfp)
    "STATIC_PRESETS",
    "REAL_TARGETS",
]

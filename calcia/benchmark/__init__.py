"""Spatio-temporal segmentation benchmark toolkit.

A system for evaluating cell-segmentation / demixing algorithms against a
calcia simulation ground truth.  It answers four questions:

1. **Which neurons are detectable?**  :mod:`detectability` scores every neuron
   by the physics that sets how many photons reach the detector (expression /
   AAV infection, depth attenuation, illumination & collection weighting) and
   bins it into ``uninfected .. easy``.
2. **Which neurons get confused / merged?**  :mod:`confusability` builds a
   graph of bright neighbours within one footprint and finds merge-prone groups.
3. **How well did an algorithm do?**  :mod:`loaders` + :mod:`matching` +
   :mod:`metrics` give unified, GT-anchored detection / trace-fidelity /
   separation metrics (spatial and temporally-gated), stratified by category.
4. **What does unclean signal cost downstream?**  :mod:`downstream` measures
   the damage to functional-correlation structure and event detection.

:func:`evaluate` (in :mod:`report`) runs the whole pipeline for one algorithm
result and emits figures + a markdown/json report.
"""

from __future__ import annotations

from .gt import GroundTruth
from .detectability import (
    Detectability, DetectabilityConfig, characterize, CATEGORIES,
)
from .confusability import Confusability, ConfusabilityConfig, analyze

__all__ = [
    "GroundTruth",
    "Detectability", "DetectabilityConfig", "characterize", "CATEGORIES",
    "Confusability", "ConfusabilityConfig", "analyze",
]

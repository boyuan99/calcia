"""Phase 4: scanning simulation."""

from .motion import describe_motion_gt, load_motion_gt
from .noise import camera_noise, pixel_bleed, poisson_gauss_noise
from .scanning import ScanResult, scan_volume
from .widefield import scan_widefield

__all__ = [
    "scan_volume",
    "scan_widefield",
    "ScanResult",
    # motion ground truth: motion_gt is the DEFAULT artifact, mot_hist is legacy
    "load_motion_gt",
    "describe_motion_gt",
    "camera_noise",
    "pixel_bleed",
    "poisson_gauss_noise",
]

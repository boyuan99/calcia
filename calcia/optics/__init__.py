"""Optical propagation simulation (Phase 2).

Port of MATLAB ``simulate_optical_propagation.m`` and supporting
functions from ``OpticsCode/``.
"""

from .mask import compute_collection_mask, compute_illumination_mask
from .propagation import OpticalPropagationResult, simulate_optical_propagation
from .psf import PsfTail, compute_psf_tails, gaussian_beam_size, gaussian_psf_na
from .signal import tpm_signal_scale, widefield_signal_scale
from .widefield import simulate_optical_propagation_widefield
from .fresnel import (
    fresnel_propagation_multi,
    generate_ba,
    generate_scatter_volume,
    group_z_project,
    zernike_polynomial,
)
from .fresnel_psf import decompose_vessel_volume, gen_cortical_light_path_lite

__all__ = [
    # Main entry point
    "simulate_optical_propagation",
    "simulate_optical_propagation_widefield",
    "OpticalPropagationResult",
    # PSF
    "gaussian_psf_na",
    "gaussian_beam_size",
    "PsfTail",
    "compute_psf_tails",
    # Signal scaling
    "tpm_signal_scale",
    "widefield_signal_scale",
    # Masks
    "compute_illumination_mask",
    "compute_collection_mask",
    # Fresnel propagation
    "fresnel_propagation_multi",
    "generate_ba",
    "generate_scatter_volume",
    "group_z_project",
    "zernike_polynomial",
    "decompose_vessel_volume",
    "gen_cortical_light_path_lite",
]

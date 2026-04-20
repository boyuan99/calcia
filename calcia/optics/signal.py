"""
Two-photon microscope signal scaling.

Port of MATLAB ``tpmSignalscale.m``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..config.params import PsfParams, TpmParams


def tpm_signal_scale(
    tpm_params: "TpmParams",
    psf_params: Optional["PsfParams"] = None,
) -> float:
    """Compute the average two-photon photon collection rate.

    Port of MATLAB ``tpmSignalscale.m``.

    Formula (Xu & Webb 1996, JOSA B)::

        Ftavg = phi * eta * conc * delta * gp * 8 * nidx * pavg^2
                / (2 * f * tau * pi * lambda)

    with SI unit conversions applied before evaluation.

    Args:
        tpm_params: TPM signal parameters.
        psf_params: If provided, overrides ``nidx``, ``nac``, and
            ``lambda_um`` from psf_params (matches MATLAB dual-input
            behaviour where psf_params fields shadow tpm_params).

    Returns:
        Ftavg: Average photon collection rate in photons / second.
    """
    # Local copies, potentially overridden by psf_params
    nidx = tpm_params.nidx
    phi = tpm_params.phi
    lambda_um = tpm_params.lambda_um

    if psf_params is not None:
        nidx = psf_params.n
        lambda_um = psf_params.lambda_um
        # Recompute phi from psf_params obj_na / nidx
        sa = (1.0 - math.sqrt(1.0 - (psf_params.obj_na / nidx) ** 2)) / 2.0
        phi = 0.8 * sa * 0.4

    # Unit conversions
    conc = tpm_params.conc * 1e-6 * 6.02e23 * 1e3      # uM → molecules/m^3
    delta = tpm_params.delta * 1e-58                    # GM → m^4·s/photon
    f = tpm_params.f * 1e6                              # MHz → Hz
    tau = tpm_params.tau * 1e-15                        # fs → s
    lambda_m = lambda_um * 1e-6                         # um → m
    h = 6.626e-34                                       # Planck constant (J·s)
    c = 3e8                                             # Speed of light (m/s)
    pavg = 1e-3 * tpm_params.pavg / (h * c / lambda_m)  # mW → photons/s

    eta = tpm_params.eta
    gp = tpm_params.gp

    ftavg = (
        phi * eta * conc * delta * gp * 8.0 * nidx * pavg ** 2
        / (2.0 * f * tau * math.pi * lambda_m)
    )
    return float(ftavg)

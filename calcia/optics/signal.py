"""
Microscope signal scaling (two-photon and widefield one-photon).

Port of MATLAB ``tpmSignalscale.m`` plus a widefield (single-photon)
counterpart used by the camera-based imaging path.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..config.params import PsfParams, TpmParams, WidefieldParams


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


def widefield_signal_scale(wf_params: "WidefieldParams") -> float:
    """Compute the per-fluorophore single-photon emission collection rate.

    Linear (one-photon) counterpart to :func:`tpm_signal_scale`. The rate
    scales linearly with excitation intensity (no I^2 term) and uses the
    one-photon absorption cross-section sigma_abs instead of the two-photon
    cross-section delta.

    Formula::

        rate_per_molecule = sigma_abs * I_exc                     # photons/s per fluorophore
        F = phi * qe_det * omega * t_optics * rate_per_molecule * conc_uM

    where I_exc is the excitation photon flux in photons / s / cm^2 and
    ``conc_uM`` is used as a dimensionless scale factor (the simulated
    fluorescence volume already encodes spatial molecule density, so we
    do not multiply by Avogadro's number here).

    Unit conventions:
      - sigma_abs in cm^2 (native units for one-photon absorption).
      - pavg in mW / mm^2 (Koehler illumination) -> photons / s / cm^2
        via ``I = pavg * 1e-1 / E_photon`` (1 mW/mm^2 = 0.1 W/cm^2).
      - conc in uM, used as a raw multiplier.

    Args:
        wf_params: Widefield parameters (see :class:`WidefieldParams`).

    Returns:
        Float signal scaling constant — photons per second per unit
        fluorescence (dimensionless in the volume's normalized units).
        Used as a multiplicative prefactor in
        :func:`calcia.scanning.scan_widefield`.
    """
    sigma_abs = wf_params.sigma_abs                  # cm^2, as given
    lambda_m = wf_params.lambda_ex_um * 1e-6         # um -> m
    h = 6.626e-34                                    # Planck constant (J*s)
    c = 3e8                                          # Speed of light (m/s)
    e_photon = h * c / lambda_m                      # J per excitation photon
    # mW/mm^2 -> W/cm^2: 1 mW/mm^2 = 1e-3 W / 1e-2 cm^2 = 1e-1 W/cm^2
    i_exc = wf_params.pavg * 1e-1 / e_photon         # photons / s / cm^2

    omega = wf_params.omega if wf_params.omega is not None else 0.0

    f_rate = (
        wf_params.phi
        * wf_params.qe_det
        * omega
        * wf_params.t_optics
        * sigma_abs
        * i_exc
        * wf_params.conc
    )
    return float(f_rate)

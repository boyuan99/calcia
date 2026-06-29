"""
Parameter dataclasses for neural volume simulation.

Converted from MATLAB check_*_params.m files.
Each dataclass corresponds to a MATLAB struct with default parameters.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
import math


@dataclass
class VolumeParams:
    """
    Volume parameters configuration.

    Corresponds to MATLAB: check_vol_params.m

    Attributes:
        vol_sz: 3-element tuple with the size (in um) of the volume to generate.
        min_dist: Minimum distance between neurons in um.
        vres: Resolution to simulate volume at (samples/um).
        N_bg: Number of background/neuropil components to simulate.
        vol_depth: Depth of the volume under the brain surface in um.
        dendrite_tau: Dendrite decay strength exponential distance.
        verbose: Level of verbosity (0=none, 1=some, 2=detailed).
        N_neur: Number of neurons to generate (computed from density if not set).
        neur_density: Neuron density in neurons/mm^3.
        N_den: Number of apical dendrites (computed from density if not set).
        AD_density: Apical dendrite density.
        region: Tissue region preset. 'cortex' (default) keeps the original
            pyramidal/pial-surface model. 'striatum' applies MSN + deep-
            perforator-vasculature defaults via apply_region_defaults().
    """
    vol_sz: Tuple[int, int, int] = (100, 100, 50)
    min_dist: float = 16.0
    vres: int = 2
    N_bg: int = 1_000_000
    vol_depth: int = 200
    dendrite_tau: float = 5.0
    verbose: int = 1
    N_neur: Optional[int] = None
    neur_density: float = 1e5  # neurons/mm^3
    N_den: Optional[int] = None
    AD_density: float = 2e3
    vasc_sz: Optional[Tuple[int, int, int]] = None
    region: str = "cortex"

    def __post_init__(self):
        """Initialize computed parameters after dataclass initialization."""
        # Ensure vol_sz[2] is a multiple of 10 (for indexing in dendrite code)
        vol_sz = list(self.vol_sz)
        if vol_sz[2] % 10 != 0:
            vol_sz[2] = 10 * math.ceil(vol_sz[2] / 10)
            self.vol_sz = tuple(vol_sz)

        # Calculate number of neurons from density if not specified
        vol_um3 = np.prod(self.vol_sz)
        if self.N_neur is None:
            self.N_neur = int(math.ceil(self.neur_density * vol_um3 / 1e9))

        # Calculate number of apical dendrites from density if not specified.
        # Striatal MSNs have no apical dendrites, so the striatum preset
        # forces N_den=0 here (the None sentinel is gone by the time the
        # pipeline runs, so this branch must live in __post_init__).
        if self.N_den is None:
            if self.region == "striatum":
                self.N_den = 0
            else:
                area_um2 = self.vol_sz[0] * self.vol_sz[1]
                self.N_den = int(self.AD_density * area_um2 / 1e6)


@dataclass
class NeuronParams:
    """
    Neuron generation parameters.

    Corresponds to MATLAB: check_neur_params.m

    Attributes:
        n_samps: Number of sphere samples for mesh generation.
        l_scale: Length-scale for isotropic GP of soma shapes (controls bumpiness).
        p_scale: Overall variance of the isotropic GP of soma shape.
        avg_rad: Average radius of each neuron in um.
        nuc_rad: Nuclear radius parameters.
        max_ang: Maximum angle tilt in degrees.
        plot_opt: Whether to plot during generation.
        dendrite_tau: Dendrite decay parameter.
        nuc_fluorsc: Nuclear fluorescence level (0-1).
        min_thic: Minimum cytoplasmic thickness.
        eccen: Maximum eccentricity of neuron.
        exts: Parameters for max/min of soma radii.
        nexts: Parameters for nucleus shrink and smooth.
        neur_type: Neuron type. Supported values:
            - 'pyramidal' or 'pyr': Teardrop-shaped pyramidal neurons (cortex, hippocampus)
            - 'spherical': Spherical neurons (granule cells, MSNs)
            - 'stellate': Highly spherical neurons (cortex layer IV, cerebellum)
            - 'fusiform': Elongated spindle-shaped neurons (cortex layer VI, von Economo)
        fluor_dist: Somatic neural fluorescence distribution.
    """
    n_samps: int = 200
    l_scale: float = 90.0
    p_scale: float = 1000.0
    avg_rad: float = 5.9
    nuc_rad: Tuple[float, float] = (5.65, 2.5)
    max_ang: float = 20.0
    plot_opt: bool = False
    dendrite_tau: float = 50.0
    nuc_fluorsc: float = 0.0
    min_thic: Tuple[float, float] = (0.4, 0.4)
    eccen: Tuple[float, float, float] = (0.35, 0.35, 0.5)
    exts: Tuple[float, float] = (0.75, 1.7)
    nexts: Tuple[float, float] = (0.5, 1.0)
    neur_type: str = "pyramidal"
    fluor_dist: Tuple[float, float] = (1.0, 0.2)


@dataclass
class VascNodeParams:
    """
    Vasculature node placement parameters.

    Sub-parameters for VascParams.
    """
    maxit: int = 25  # Maximum iteration to place nodes
    lensc: float = 50.0  # Average distance between branch points (um)
    varsc: float = 15.0  # Std dev of distances between branch points (um)
    mindist: float = 10.0  # Minimum inter-node distance (um)
    varpos: float = 5.0  # Std dev of vasculature placement (um)
    dirvar: float = math.pi / 8  # Maximum branching angle
    branchp: float = 0.02  # Probability of branching surface vasculature
    vesrad: float = 25.0  # Radius of surface vasculature (um)


@dataclass
class VascParams:
    """
    Vasculature simulation parameters.

    Corresponds to MATLAB: check_vasc_params.m

    Attributes:
        flag: On/off flag for vasculature simulation.
        ves_shift: 3-vector of wobble allowed for blood vessels (um).
        depth_vasc: Depth into tissue for vasculature simulation (um).
        depth_surf: Depth into tissue of surface vasculature (um).
        distWeightScale: Scaling factor for node distance weight.
        randWeightScale: Scaling factor for node weight variability.
        cappAmpScale: Scaling factor for capillary weights (lateral).
        cappAmpZscale: Scaling factor for capillary weights (axial).
        vesSize: Vessel radius (surface, axial, capillaries) in um.
        vesFreq: Blood vessel frequency in um.
        sourceFreq: Rate of generation of source nodes (um/node).
        vesNumScale: Vessel number random scaling factor.
        sepweight: Weight for vasculature node placement (0-1).
        distsc: How strongly local capillary connections are.
        node_params: Sub-parameters for node placement.
    """
    flag: bool = True
    ves_shift: Tuple[float, float, float] = (5.0, 15.0, 5.0)
    depth_vasc: float = 200.0
    depth_surf: float = 15.0
    distWeightScale: float = 2.0
    randWeightScale: float = 0.1
    cappAmpScale: float = 0.5
    cappAmpZscale: float = 0.5
    vesSize: Tuple[float, float, float] = (15.0, 9.0, 2.0)
    vesFreq: Tuple[float, float, float] = (125.0, 200.0, 50.0)
    sourceFreq: float = 1000.0
    vesNumScale: float = 0.2
    sepweight: float = 0.75
    distsc: float = 4.0
    node_params: VascNodeParams = field(default_factory=VascNodeParams)


@dataclass
class DendParams:
    """
    Dendrite simulation parameters.

    Corresponds to MATLAB: check_dend_params.m

    Attributes:
        dtParams: Dendritic tree parameters [number, horiz_radius, vert_radius,
                  width_scale, variation].
        atParams: Apical dendrite (L2/3) parameters.
        atParams2: Apical dendrite (L5) parameters.
        dweight: Weight for path planning randomness.
        bweight: Weight for obstruction.
        thicknessScale: Scaling for dendrite thickness in um^2.
        weightScale: Scaling for dendrite fluorescence [1/dist, dist_weight, variation].
        dims: Dims set at 10 um per space.
        dimsSS: Dims subsampling factor.
        rallexp: Rall exponent for size change at branching locations.
        dendVar: Dendrite size variation (default 0.25 for basal, 0.35 for through-volume apical).
        apicalVar: Through-volume apical dendrite size variation. Overrides dendVar for Step 5.
    """
    dtParams: Tuple[float, float, float, float, float] = (40.0, 150.0, 50.0, 1.0, 10.0)
    atParams: Tuple[float, float, float, float, float] = (6.0, 5.0, 5.0, 5.0, 1.0)
    atParams2: Tuple[float, float, float, float, float] = (1.0, 5.0, 5.0, 5.0, 4.0)
    dweight: float = 10.0
    bweight: float = 5.0
    thicknessScale: float = 0.5
    weightScale: Tuple[float, float, float] = (150.0, 1.0, 0.8)
    dims: Tuple[int, int, int] = (60, 60, 60)
    dimsSS: Tuple[int, int, int] = (5, 5, 5)
    rallexp: float = 1.5
    dendVar: Optional[float] = None
    apicalVar: Optional[float] = None


@dataclass
class BgParams:
    """
    Background dendrite generation parameters.

    Corresponds to MATLAB: check_bg_params.m

    Attributes:
        flag: Flag for generation of background dendrites.
        distsc: Parameter for random walk direction (higher = more directed).
        fillweight: Maximum length for a single process branch (um).
        maxlength: Maximum length for background processes (um).
        minlength: Minimum length for background processes (um).
        maxdist: Maximum distance to end of a process (um).
        maxel: Max number of axons per voxel.
    """
    flag: bool = True
    distsc: float = 0.5
    fillweight: float = 100.0
    maxlength: float = 200.0
    minlength: float = 10.0
    maxdist: float = 100.0
    maxel: int = 1


@dataclass
class AxonParams:
    """
    Axon generation parameters.

    Corresponds to MATLAB: check_axon_params.m

    Attributes:
        flag: Flag for generation of axons.
        distsc: Parameter for random walk direction.
        fillweight: Maximum length for a single process branch (um).
        maxlength: Maximum length for background processes (um).
        minlength: Minimum length for background processes (um).
        maxdist: Maximum distance to end of a process (um).
        maxel: Max number of axons per voxel.
        varfill: Variation in filling weight.
        maxvoxel: Maximum number of elements per voxel.
        padsize: Background padding size for smoothness.
        numbranches: Number of allowable branches per process.
        varbranches: Std dev of number of branches per process.
        maxfill: Fraction of volume to fill with background processes.
        N_proc: Number of background components.
        l: Gaussian process length scale.
        rho: Gaussian process variance parameter.
    """
    flag: bool = True
    distsc: float = 0.5
    fillweight: float = 100.0
    maxlength: float = 200.0
    minlength: float = 10.0
    maxdist: float = 100.0
    maxel: int = 8
    varfill: float = 0.3
    maxvoxel: int = 6
    padsize: int = 20
    numbranches: int = 20
    varbranches: float = 5.0
    maxfill: float = 0.5
    N_proc: int = 10
    l: float = 25.0
    rho: float = 0.1


@dataclass
class PsfParams:
    """
    Point-spread function and optical propagation parameters.

    Corresponds to MATLAB: check_psf_params.m

    Attributes:
        na: Numerical aperture of the excitation beam.
        obj_na: Numerical aperture of the objective lens (collection cone).
        n: Refractive index of the propagation medium (brain tissue).
        n_diff: Refractive index shift from blood vessels relative to tissue.
        lambda_um: Two-photon excitation wavelength in microns.
            Named lambda_um to avoid conflict with Python keyword ``lambda``.
        obj_fl: Objective focal length in mm.
        ss: Subsampling factor for Fresnel propagation relative to vres.
        sampling: Spatial sampling period for illumination mask (um).
        psf_sz: (X, Y, Z) size of the simulated PSF volume in um.
        prop_sz: Fresnel propagation chunk length outside PSF volume (um).
        blur: Lateral PSF blurring sigma in um (aberration model).
        scatter_sz: Scattering object sizes in um.
        scatter_wt: Scattering weights (same length as scatter_sz).
        zernike_wt: 11-element Zernike aberration weights (in wavelengths).
        tail_length: Distance from PSF edge to estimate tail weight (um).
        psf_type: PSF type. 'gaussian' is the only supported type in this port.
        scaling: Scaling type. 'two-photon' applies intensity^2.
        hemo_abs: Hemoglobin absorbance factor for collection mask.
            Stored as 0.00674 * log(10) to match MATLAB hemoabs field.
            Usage: ``np.power(10, -col / vres * hemo_abs)``.
        scatter_length_um_wf: Widefield-only. Effective ONE-WAY tissue
            scattering length at the widefield emission wavelength (um).
            The round-trip excitation + emission attenuation for a source
            at depth z BELOW THE VOLUME TOP is ``exp(-2*z / scatter_length_um_wf)``.
            Depth is measured from z=0 at the top of the imaging volume
            (where the imaging window sits flush against tissue); the
            ``VolumeParams.vol_depth`` offset is NOT added — that field
            is a 2P cranial-window concept for modelling extra overlying
            tissue. Default 70 um reflects cortical mu_s' ~ 13-20 cm^-1
            at 488-520 nm. Ignored when imaging_mode == "two-photon".
        hemo_abs_wf: Widefield-only. Hemoglobin absorbance factor used by
            the collection mask when imaging_mode == "widefield". Defaults
            to ~30x the two-photon value to reflect HbO2 absorption at
            520 nm vs 920 nm. Ignored when imaging_mode == "two-photon".
        prop_crop: If True, crop propagation volume to beam extent.
        fast_mask: If True, compute mask at lower resolution then upsample.
        fm_sampling: Fast-mask coarse spatial sampling (um).
        fm_fine_samp: Fast-mask fine subsampling factor.
        fm_ss: Fast-mask Fresnel propagation subsampling.
    """
    na: float = 0.6
    obj_na: float = 0.8
    n: float = 1.35
    n_diff: float = 0.02
    lambda_um: float = 0.92
    obj_fl: float = 4.5
    ss: int = 2
    sampling: float = 50.0
    psf_sz: Tuple[float, float, float] = (20.0, 20.0, 50.0)
    prop_sz: float = 10.0
    blur: float = 3.0
    scatter_sz: Tuple[float, ...] = (0.51, 1.56, 4.52, 14.78)
    scatter_wt: Tuple[float, ...] = (0.57, 0.29, 0.19, 0.15)
    zernike_wt: Tuple[float, ...] = (0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0.12)
    tail_length: float = 50.0
    psf_type: str = "gaussian"
    scaling: str = "two-photon"
    imaging_mode: str = "two-photon"
    lambda_em_um: float = 0.52
    hemo_abs: float = field(default_factory=lambda: 0.00674 * math.log(10))
    scatter_length_um_wf: float = 70.0
    hemo_abs_wf: float = field(
        default_factory=lambda: 0.00674 * math.log(10) * 30.0
    )
    prop_crop: bool = True
    fast_mask: bool = True
    fm_sampling: float = 10.0
    fm_fine_samp: int = 2
    fm_ss: int = 1


@dataclass
class TpmParams:
    """
    Two-photon microscope signal scaling parameters.

    Corresponds to MATLAB: check_tpm_params.m

    Attributes:
        nidx: Refractive index of the immersion medium (water = 1.33).
        nac: Objective collection numerical aperture.
        phi: Collection efficiency (solid angle * transmission * QE).
            Computed from nac/nidx if None.
        eta: Fluorophore quantum yield (eGFP estimate).
        conc: Fluorophore concentration in uM.
        delta: Two-photon absorption cross-section in GM (Goeppert-Mayer).
        gp: Pulse-shape temporal coherence factor (sech pulse = 0.588).
        f: Laser repetition rate in MHz.
        tau: Laser pulse width in fs.
        pavg: Average laser power in mW.
        lambda_um: Excitation wavelength in um.
            Named lambda_um to avoid conflict with Python keyword ``lambda``.
    """
    nidx: float = 1.33
    nac: float = 0.8
    phi: Optional[float] = None
    eta: float = 0.6
    conc: float = 10.0
    delta: float = 2.0
    gp: float = 0.588
    f: float = 80.0
    tau: float = 150.0
    pavg: float = 40.0
    lambda_um: float = 0.92

    def __post_init__(self):
        if self.phi is None:
            sa = (1.0 - math.sqrt(1.0 - (self.nac / self.nidx) ** 2)) / 2.0
            self.phi = 0.8 * sa * 0.4


@dataclass
class WidefieldParams:
    """
    Single-photon widefield signal-scaling parameters.

    Counterpart of TpmParams for linear (one-photon) excitation with
    camera-based detection.

    Attributes:
        phi: Fluorophore quantum yield (GFP ~ 0.6).
        sigma_abs: Absorption cross-section in cm^2 (EGFP @ 488 nm ~ 3.8e-16).
        conc: Fluorophore concentration in uM.
        lambda_ex_um: Excitation wavelength in um (GFP ~ 0.488).
        pavg: Excitation intensity in mW/mm^2 (Koehler uniform illumination).
        na_col: Objective collection numerical aperture.
        nidx: Refractive index of the immersion medium.
        t_optics: Optical path transmission fraction.
        qe_det: Camera quantum efficiency at emission wavelength.
        omega: Fractional collection solid angle. Auto-computed from
            na_col/nidx if None.
    """
    phi: float = 0.6
    sigma_abs: float = 3.8e-16
    conc: float = 10.0
    lambda_ex_um: float = 0.488
    pavg: float = 1.0
    na_col: float = 0.8
    nidx: float = 1.33
    t_optics: float = 0.7
    qe_det: float = 0.8
    omega: Optional[float] = None

    def __post_init__(self):
        if self.omega is None:
            self.omega = 0.5 * (
                1.0 - math.sqrt(1.0 - (self.na_col / self.nidx) ** 2)
            )


# ---------------------------------------------------------------------------
# Phase 3: Time-trace generation parameters
# ---------------------------------------------------------------------------

#: Protein-specific defaults for CalciumParams.
#: Keys are normalised protein names (lowercase, hyphens removed).
_PROT_DEFAULTS: dict = {
    "gcamp6f": {"ca_amp": 76.1251,  "t_on": 0.8535,   "t_off": 98.6173,  "ext_rate": 292.3},
    "gcamp6":  {"ca_amp": 76.1251,  "t_on": 0.8535,   "t_off": 98.6173,  "ext_rate": 292.3},
    "gcamp6s": {"ca_amp": 54.6943,  "t_on": 0.4526,   "t_off": 68.5461,  "ext_rate": 299.0833},
    "gcamp7":  {"ca_amp": 230.917,  "t_on": 0.020137, "t_off": 3.1295,   "ext_rate": 265.73},
    "gcamp3":  {"ca_amp": 0.05,     "t_on": 1.0,      "t_off": 1.0,      "ext_rate": 265.73},
}


@dataclass
class SpikeParams:
    """
    Spike-train generation parameters.

    Corresponds to MATLAB: check_spike_opts.m

    Attributes:
        K: Number of neurons to generate time traces for.
        nt: Number of time-steps to simulate (at the output frame-rate).
        rate: Average firing rate used as Gamma scale parameter.
        dt: Time-step in seconds (1/frame-rate).
        mu: Mean of the log-normal r.v. for AR spike amplitudes.
        sig: Std-dev of the log-normal r.v. for AR spike amplitudes.
        dyn_type: Calcium dynamics model.  Options: 'AR1', 'AR2', 'single',
            'Ca_DE', 'double'.
        rate_dist: Distribution for per-neuron firing rates.  Options:
            'gamma', 'uniform'.
        N_bg: Number of background/neuropil GP components (0 = none).
        prot: Fluorescent protein name (used to set CalciumParams defaults).
        alpha: Gamma shape parameter for the rate distribution.
        burst_mean: Mean spikes per burst (Poisson).  0 disables bursting.
        smod_flag: Spike model.  'hawkes' = correlated Hawkes process;
            'poisson' = independent Poisson bursts.
        p_off: Probability that a cell has zero expression.
        selfact: Self-excitation scaling for the Hawkes diagonal.
        min_mod: Modulation range (min_val, gamma_shape) for per-cell scaling.
        spikeflag: If True, store raw spike counts in TimeTracesResult.
        dendflag: If True, simulate dendrite fluorescence traces.
        axonflag: If True, simulate axon/background fluorescence traces.
        bg_scale: Multiplicative brightness scale applied to the background /
            neuropil traces (soma and dendrite traces are unaffected). 1.0
            reproduces the default NAOMi amplitude; lower values dim the
            diffuse neuropil (e.g. 0.2 -> 20%) to reduce whole-FOV wash-out
            without removing the background entirely.
        ensure_activity: If True, inject spikes into silent soma neurons
            when the spike generation step produces very little activity.
            This can happen when ``rate`` is low relative to the simulation
            duration (``nt * dt``).  Default is False.
        n_soma: Number of actual soma neurons among the K components.
            When set, ``ensure_activity`` only injects spikes into the
            first ``n_soma`` components (indices 0..n_soma-1), leaving
            dendrite and background components untouched.  When None,
            defaults to K (all components are treated as soma).
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
    """
    K: int = 30
    nt: int = 1000
    rate: float = 1e-3
    dt: float = field(default_factory=lambda: 1 / 30)
    mu: float = 0.0
    sig: float = 1.0
    dyn_type: str = "Ca_DE"
    rate_dist: str = "gamma"
    N_bg: int = 0
    prot: str = "GCaMP6f"
    alpha: float = 1.0
    burst_mean: int = 10
    smod_flag: str = "hawkes"
    p_off: float = 0.2
    selfact: float = 1.2
    min_mod: Tuple[float, float] = (0.4, 2.53)
    spikeflag: bool = True
    dendflag: bool = True
    axonflag: bool = True
    bg_scale: float = 1.0
    ensure_activity: bool = False
    n_soma: Optional[int] = None
    verbose: int = 1


@dataclass
class CalciumParams:
    """
    Calcium dynamics simulation parameters.

    Corresponds to MATLAB: check_cal_params.m

    Protein-dependent fields (``ca_amp``, ``t_on``, ``t_off``, ``ext_rate``)
    are resolved from ``_PROT_DEFAULTS`` in ``__post_init__`` when left as
    ``None``.

    Attributes:
        prot_type: Fluorescent protein name (case-insensitive, hyphens OK).
        ca_bind: Calcium binding ratio (dimensionless).
        ca_rest: Resting calcium concentration in M.
        ind_con: Indicator concentration in M.
        ca_dis: Calcium dissociation constant in M.
        ca_sat: Saturation parameter (0–1; 1 = no saturation).
        sat_type: Dynamics model: 'Ca_DE', 'single', or 'double'.
        dt: Internal simulation time-step in seconds (default 1/100 for 100 Hz).
        a_bind: Binding rate constant.
        a_ubind: Unbinding rate constant.
        ca_amp: Double-exponential kernel amplitude (protein-specific default).
        t_on: Rising time-constant of Ca²⁺ transient in s (protein-specific).
        t_off: Falling time-constant of Ca²⁺ transient in s (protein-specific).
        ext_rate: Ca²⁺ extrusion rate (protein-specific default).
    """
    prot_type: str = "gcamp6f"
    ca_bind: float = 110.0
    ca_rest: float = 50e-9
    ind_con: float = 200e-6
    ca_dis: float = 290e-9
    ca_sat: float = 1.0
    sat_type: str = "double"
    dt: float = field(default_factory=lambda: 1 / 100)
    a_bind: float = 3.5
    a_ubind: float = 7.0
    ca_amp: Optional[float] = None
    t_on: Optional[float] = None
    t_off: Optional[float] = None
    ext_rate: Optional[float] = None

    def __post_init__(self):
        key = self.prot_type.lower().replace("-", "")
        defaults = _PROT_DEFAULTS.get(key, _PROT_DEFAULTS["gcamp6f"])
        if self.ca_amp is None:
            self.ca_amp = defaults["ca_amp"]
        if self.t_on is None:
            self.t_on = defaults["t_on"]
        if self.t_off is None:
            self.t_off = defaults["t_off"]
        if self.ext_rate is None:
            self.ext_rate = defaults["ext_rate"]


# ---------------------------------------------------------------------------
# Phase 4: Scanning simulation parameters
# ---------------------------------------------------------------------------


@dataclass
class ScanParams:
    """
    Scanning simulation parameters.

    Corresponds to MATLAB: check_scan_params.m

    Attributes:
        scan_buff: Edge buffer in granular pixels (cropped from each side).
        motion: Enable tissue motion simulation.
        scan_avg: Z-axis pre-summing factor for PSF convolution.
        sfrac: Pixel binning / downsampling factor.
        verbose: Verbosity level (0=silent, 1=summary, 2=detailed).
        nuc_label: Nuclear label mode (0=off, >=1=on).
        zoffset: Z-axis offset from centre in voxels.
    """
    scan_buff: int = 10
    motion: bool = True
    scan_avg: int = 2
    sfrac: int = 2
    verbose: int = 1
    nuc_label: int = 0
    zoffset: int = 0


@dataclass
class NoiseParams:
    """
    PMT / electronics noise model parameters.

    Corresponds to MATLAB: check_noise_params.m

    Attributes:
        mu: Mean measurement increase per photon.
        mu0: Electronics DC offset.
        sigma: Variance increase per photon.
        sigma0: Electronics baseline variance.
        darkcount: Dark counts + autofluorescence rate.
        bleedp: Pixel bleed probability.
        bleedw: Pixel bleed max amplitude (fraction).
    """
    mu: float = 100.0
    mu0: float = 0.0
    sigma: float = 2300.0
    sigma0: float = 2.7
    darkcount: float = 0.05
    bleedp: float = 0.3
    bleedw: float = 0.4


@dataclass
class CameraNoiseParams:
    """
    sCMOS / CCD camera noise model parameters.

    Widefield counterpart of NoiseParams. Replaces the PMT lognormal-gain
    chain with a Poisson + read-noise model.

    Attributes:
        qe: Camera quantum efficiency. Set to 1.0 if QE is already folded
            into widefield_signal_scale.
        dark_rate: Dark current in electrons / pixel / second.
        t_exp: Exposure time in seconds (~0.033 for 30 fps).
        read_noise: Read noise in electrons rms.
        gain_e_per_adu: Conversion gain (electrons per ADU).
        bit_depth: ADC bit depth (output is clipped to [0, 2**bit_depth - 1]).
        pixel_gain_sigma: Fixed-pattern PRNU sigma. Set to 0 to disable.
    """
    qe: float = 0.8
    dark_rate: float = 0.3
    t_exp: float = 0.033
    read_noise: float = 1.6
    gain_e_per_adu: float = 1.0
    bit_depth: int = 16
    pixel_gain_sigma: float = 0.01


# ---------------------------------------------------------------------------
# Region presets
# ---------------------------------------------------------------------------
# Striatum (medium spiny neuron tissue) preset. Literature-backed:
#   - MSN soma 15-18 um diameter -> avg_rad ~8; ovoid/spherical shape.
#   - No apical dendrites; 5-6 primary dendrites in a ~250 um spherical field.
#   - Microvascular density comparable to cortex (so capillary spacing kept);
#     differences are topology: no pial surface layer (depth_surf=0), isotropic
#     penetrators (handled in vasculature.py via the depth_surf<=0 sentinel),
#     and fewer anastomoses (higher distsc -> more terminal vessels).
_STRIATUM_NEURON = {
    "neur_type": "spherical",
    "avg_rad": 8.0,
    "eccen": (0.15, 0.15, 0.15),
}
_STRIATUM_VASC = {
    "depth_surf": 0.0,
    "vesSize": (9.0, 9.0, 2.0),
    "vesFreq": (200.0, 200.0, 50.0),
    "distsc": 6.0,
}
_STRIATUM_DEND = {
    "dtParams": (5.0, 125.0, 125.0, 1.0, 1.0),
    "atParams": (0.0, 5.0, 5.0, 5.0, 1.0),
}


def apply_region_defaults(vol_params, neur_params, vasc_params, dend_params):
    """Apply region-specific parameter presets in place.

    No-op for the default 'cortex' region, so existing behavior is unchanged.
    For 'striatum', overwrites the MSN/vasculature/dendrite fields listed in
    the ``_STRIATUM_*`` dicts above. ``getattr`` is used so volume params from
    older pickles (without a ``region`` field) are treated as cortex.
    """
    region = getattr(vol_params, "region", "cortex")
    if region == "cortex":
        return
    if region != "striatum":
        raise ValueError(
            f"Unknown region {region!r}. Supported: 'cortex', 'striatum'."
        )
    for k, v in _STRIATUM_NEURON.items():
        setattr(neur_params, k, v)
    for k, v in _STRIATUM_VASC.items():
        setattr(vasc_params, k, v)
    for k, v in _STRIATUM_DEND.items():
        setattr(dend_params, k, v)

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

        # Calculate number of apical dendrites from density if not specified
        if self.N_den is None:
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

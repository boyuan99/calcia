"""Core volume generation modules."""

from .neurons import (
    generate_neural_body,
    generate_multiple_neurons,
    sample_dense_neurons,
    compute_neuron_statistics,
)

from .neural_volume import (
    generate_neural_volume,
    NeuralVolumeResult,
    NeuronVoxelData,
)

from .dendrites import (
    grow_neuron_dendrites,
    grow_apical_dendrites,
    DendriteResult,
    ApicalDendriteResult,
)

from .fluorescence import (
    set_cell_fluorescence,
    FluorescenceResult,
    CellFluorescenceData,
)

from .background import (
    generate_bg_dendrites,
    generate_axons,
    sort_axons,
    BgDendriteResult,
    AxonResult,
    BgProcessData,
)

from .vasculature import (
    VesselNode,
    VesselConnection,
    VesselNetwork,
    simulate_blood_vessels,
    grow_major_vessels,
    grow_diving_vessels,
    grow_capillaries,
    pseudo_rand_sample_2d,
    pseudo_rand_sample_3d,
)

from .postprocess import fill_nuclei

__all__ = [
    # Neurons
    "generate_neural_body",
    "generate_multiple_neurons",
    "sample_dense_neurons",
    "compute_neuron_statistics",
    # Neural volume
    "generate_neural_volume",
    "NeuralVolumeResult",
    "NeuronVoxelData",
    # Dendrites
    "grow_neuron_dendrites",
    "grow_apical_dendrites",
    "DendriteResult",
    "ApicalDendriteResult",
    # Fluorescence
    "set_cell_fluorescence",
    "FluorescenceResult",
    "CellFluorescenceData",
    # Background / Axons
    "generate_bg_dendrites",
    "generate_axons",
    "sort_axons",
    "BgDendriteResult",
    "AxonResult",
    "BgProcessData",
    # Vasculature
    "VesselNode",
    "VesselConnection",
    "VesselNetwork",
    "simulate_blood_vessels",
    "grow_major_vessels",
    "grow_diving_vessels",
    "grow_capillaries",
    "pseudo_rand_sample_2d",
    "pseudo_rand_sample_3d",
    # Post-generation volume edits
    "fill_nuclei",
]

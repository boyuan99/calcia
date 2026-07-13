"""Post-generation, values-only edits of a neural volume.

These operate on an already-built ``NeuralVolumeResult`` (as returned by
``simulate_neural_volume`` / loaded from a Phase-1 cache), touching only
fluorescence VALUES and mask bookkeeping — never the geometry ground truth in
``locs``. They exist because some indicator/imaging regimes render differently
than NAOMi's raw convention (e.g. dark nuclei) yet must stay physical.
"""
import numpy as np


def fill_nuclei(vol_out, verbose=True):
    """Fill each neuron's nucleus with its soma fluorescence so cells render as
    SOLID bright blobs, not dark-centred rings.

    NAOMi gives the nucleus zero fluorescence (nuc_fluorsc=0), leaving a dark
    centre — the cytoplasmic-GCaMP/tdT "ring". But real washed 1P striatum cells
    are SOLID light blobs (nuclear exclusion is not resolved through scattering
    tissue, and the indicator is not perfectly excluded). This is a PHYSICAL
    correction (match real), NOT a brightness cosmetic. Values-only edit of a
    freshly-loaded volume: merges each nucleus's voxels into its soma footprint
    with the soma's median fluorescence + soma_mask, and zeros gp_nuc.
    """
    if not getattr(vol_out, "gp_nuc", None):
        return 0
    n_solid = 0
    for i in range(min(len(vol_out.gp_vals), len(vol_out.gp_nuc))):
        nuc_idx = np.asarray(vol_out.gp_nuc[i][0])
        cfd = vol_out.gp_vals[i]
        sm = np.asarray(cfd.soma_mask)
        if len(nuc_idx) == 0 or not sm.any():
            continue
        fill = float(np.median(cfd.fluorescence[sm]))
        cfd.indices = np.concatenate([cfd.indices, nuc_idx])
        cfd.fluorescence = np.concatenate(
            [cfd.fluorescence,
             np.full(len(nuc_idx), fill, cfd.fluorescence.dtype)])
        cfd.soma_mask = np.concatenate([sm, np.ones(len(nuc_idx), bool)])
        n_solid += 1
    vol_out.gp_nuc = [(np.array([], dtype=np.int64), 0.0)
                      for _ in vol_out.gp_nuc]
    if verbose:
        print(f"  solid somata: filled {n_solid} nuclei (no dark-centre rings)")
    return n_solid

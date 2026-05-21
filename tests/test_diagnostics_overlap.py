"""Tests for ``calcia.diagnostics.overlap``.

Synthetic `NeuralVolumeOutput` surrogates are used for fine-grained unit
tests. A single slow integration test runs the full Phase 1 pipeline at
small scale to confirm ``summarize`` produces sensible numbers end-to-end.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from calcia.config.params import VolumeParams
from calcia.diagnostics.overlap import (
    COMPONENT_ORDER,
    OverlapReport,
    _crop_vessel_mask,
    component_masks,
    component_vs_vessel,
    owner_count_histogram,
    pairwise_overlap,
    summarize,
)


# =====================================================================
# Helpers
# =====================================================================


def _fake_vol_out(
    shape=(6, 6, 4),
    n_neur=2,
    n_den=1,
    n_den2=1,
    soma_flat=(),
    basal_flat=(),
    apical_flat=(),
    bg_flat=(),
    axon_flat=(),
    vessel_full_shape=None,
    vessel_flat_full=(),
):
    """Build a minimal ``NeuralVolumeOutput`` stand-in.

    `*_flat` are C-order linear indices into the imaging-shape volume.
    """
    vol_params = VolumeParams(
        vol_sz=(shape[0], shape[1], shape[2]), vres=1, N_neur=n_neur, N_den=n_den,
    )
    vol_params.N_den = n_den  # force override (VolumeParams default may differ)

    neur_num = np.zeros(shape, dtype=np.uint16)
    neur_num_ad = np.zeros(shape, dtype=np.uint16)

    # Basal and soma both get IDs in [1..N_neur]; soma is distinguished
    # through gp_soma only. Use the same ID for both in neur_num.
    nn_flat = neur_num.ravel()
    for ix in soma_flat:
        nn_flat[ix] = 1  # neuron #1
    for ix in basal_flat:
        nn_flat[ix] = 1  # neuron #1
    # Apical: neur_num IDs N_neur+1 .. N_neur+N_den, and neur_num_ad>0
    for ix in apical_flat:
        nn_flat[ix] = n_neur + 1
        neur_num_ad.ravel()[ix] = 1
    # bg: IDs > N_neur + N_den
    for ix in bg_flat:
        nn_flat[ix] = n_neur + n_den + 1

    gp_soma = [np.asarray(soma_flat, dtype=np.int32)] + [
        np.asarray([], dtype=np.int32) for _ in range(n_neur - 1)
    ]
    gp_bgvals = [
        (np.asarray(axon_flat, dtype=np.int32), np.ones(len(axon_flat), dtype=np.float32))
    ] if axon_flat else []

    # Vessels: optionally full-depth
    if vessel_full_shape is None:
        neur_ves = None
    else:
        neur_ves = np.zeros(vessel_full_shape, dtype=np.uint8)
        for ix in vessel_flat_full:
            neur_ves.ravel()[ix] = 1

    return SimpleNamespace(
        neur_num=neur_num,
        neur_num_ad=neur_num_ad,
        gp_soma=gp_soma,
        gp_bgvals=gp_bgvals,
        neur_ves=neur_ves,
        params={"vol_params": vol_params},
    )


# =====================================================================
# component_masks
# =====================================================================


def test_component_masks_exclusive_soma_basal_apical():
    """Soma, basal, apical must not share voxels."""
    shape = (4, 4, 4)
    vol_out = _fake_vol_out(
        shape=shape,
        soma_flat=[0, 1, 2],
        basal_flat=[10, 11, 12],
        apical_flat=[20, 21, 22],
    )
    masks = component_masks(vol_out)
    assert not np.logical_and(masks["soma"], masks["basal_dendrite"]).any()
    assert not np.logical_and(masks["soma"], masks["apical_dendrite"]).any()
    assert not np.logical_and(masks["basal_dendrite"], masks["apical_dendrite"]).any()
    # Counts
    assert masks["soma"].sum() == 3
    assert masks["basal_dendrite"].sum() == 3
    assert masks["apical_dendrite"].sum() == 3


def test_component_masks_basal_excludes_apical_when_same_voxel():
    """If a voxel carries apical ID in neur_num but is also hit by neur_num_ad,
    the apical mask takes priority — basal must not double-count it."""
    shape = (3, 3, 3)
    vol_out = _fake_vol_out(shape=shape, basal_flat=[], apical_flat=[5])
    masks = component_masks(vol_out)
    # Voxel 5 is apical; not basal.
    assert masks["apical_dendrite"].ravel()[5]
    assert not masks["basal_dendrite"].ravel()[5]


def test_component_masks_bg_and_axon_distinct_stores():
    """bg_dendrite comes from neur_num; axon comes from gp_bgvals."""
    shape = (3, 3, 3)
    vol_out = _fake_vol_out(shape=shape, bg_flat=[5, 6], axon_flat=[7, 8])
    masks = component_masks(vol_out)
    assert masks["bg_dendrite"].sum() == 2
    assert masks["axon"].sum() == 2
    # They don't overlap here
    assert not np.logical_and(masks["bg_dendrite"], masks["axon"]).any()


def test_component_masks_order_matches_constant():
    vol_out = _fake_vol_out(soma_flat=[0])
    masks = component_masks(vol_out)
    assert list(masks.keys()) == list(COMPONENT_ORDER)


# =====================================================================
# _crop_vessel_mask
# =====================================================================


def test_crop_vessel_mask_full_depth():
    """Full-depth vessel mask should be cropped from the trailing Z."""
    imaging_shape = (4, 4, 3)
    full_shape = (4, 4, 8)  # surface + imaging = 5 + 3
    ves = np.zeros(full_shape, dtype=np.uint8)
    ves[:, :, 5:] = 1  # mark imaging slices as vessel
    out = _crop_vessel_mask(ves, imaging_shape)
    assert out.shape == imaging_shape
    assert out.all()  # every imaging voxel is vessel


def test_crop_vessel_mask_already_imaging_shape():
    imaging_shape = (3, 3, 3)
    ves = np.ones(imaging_shape, dtype=np.uint8)
    out = _crop_vessel_mask(ves, imaging_shape)
    assert out.shape == imaging_shape
    assert out.all()


def test_crop_vessel_mask_none():
    out = _crop_vessel_mask(None, (2, 2, 2))
    assert out.shape == (2, 2, 2)
    assert not out.any()


def test_crop_vessel_mask_shape_error():
    with pytest.raises(ValueError):
        _crop_vessel_mask(np.zeros((4, 4, 3)), (5, 5, 3))


# =====================================================================
# owner_count_histogram
# =====================================================================


def test_owner_count_histogram_sums_to_volume():
    shape = (4, 4, 4)
    vol_out = _fake_vol_out(
        shape=shape,
        soma_flat=[0, 1],
        basal_flat=[2, 3],
        apical_flat=[4],
        bg_flat=[5],
        axon_flat=[6],
    )
    masks = component_masks(vol_out)
    hist = owner_count_histogram(masks)
    assert hist.sum() == np.prod(shape)
    # 7 single-owner voxels (2 soma + 2 basal + 1 apical + 1 bg + 1 axon)
    assert hist[1] == 7
    assert hist[2:].sum() == 0


def test_owner_count_histogram_detects_multi_owner():
    """Overlap the soma and bg dendrite on the same voxel — should land in bin 2."""
    shape = (3, 3, 3)
    vol_out = _fake_vol_out(shape=shape, soma_flat=[0], bg_flat=[0])
    masks = component_masks(vol_out)
    hist = owner_count_histogram(masks)
    # Voxel 0 has soma+bg, so bin 2 should be 1
    assert hist[2] == 1
    assert hist[1] == 0


# =====================================================================
# pairwise_overlap
# =====================================================================


def test_pairwise_overlap_symmetric_and_diag():
    shape = (3, 3, 3)
    vol_out = _fake_vol_out(
        shape=shape,
        soma_flat=[0, 1],
        basal_flat=[2, 3],
        vessel_full_shape=(3, 3, 3),
        vessel_flat_full=[0, 4],
    )
    masks = component_masks(vol_out)
    mat, keys = pairwise_overlap(masks)
    # Symmetric
    assert np.array_equal(mat, mat.T)
    # Diagonal = per-mask count
    for i, k in enumerate(keys):
        assert mat[i, i] == int(masks[k].sum())
    # Off-diagonal: soma ∩ vessel = 1 (voxel 0 is in both)
    si = keys.index("soma")
    vi = keys.index("vessel")
    assert mat[si, vi] == 1


# =====================================================================
# component_vs_vessel
# =====================================================================


def test_component_vs_vessel_fraction():
    shape = (3, 3, 3)
    vol_out = _fake_vol_out(
        shape=shape,
        soma_flat=[0, 1, 2, 3],
        vessel_full_shape=(3, 3, 3),
        vessel_flat_full=[0, 1],
    )
    masks = component_masks(vol_out)
    stats = component_vs_vessel(masks)
    n_inter, frac = stats["soma"]
    assert n_inter == 2
    assert abs(frac - 0.5) < 1e-9


def test_component_vs_vessel_empty_when_no_vessels():
    vol_out = _fake_vol_out(soma_flat=[0])  # no vessel_full_shape
    masks = component_masks(vol_out)
    stats = component_vs_vessel(masks)
    assert stats == {}


# =====================================================================
# summarize
# =====================================================================


def test_summarize_returns_report_with_all_fields():
    shape = (3, 3, 3)
    vol_out = _fake_vol_out(
        shape=shape, soma_flat=[0], basal_flat=[1], apical_flat=[2],
        bg_flat=[3], axon_flat=[4],
        vessel_full_shape=(3, 3, 3), vessel_flat_full=[5],
    )
    report = summarize(vol_out)
    assert isinstance(report, OverlapReport)
    assert report.total_voxels == 27
    # Owner histogram sums to total
    assert int(report.owner_hist.sum()) == 27
    # String rendering doesn't crash
    s = str(report)
    assert "OverlapReport" in s
    assert "soma" in s
    assert "vessel" in s


# =====================================================================
# Integration: run the pipeline at minimum size
# =====================================================================


@pytest.mark.slow
def test_summarize_on_small_pipeline_run():
    """End-to-end: tiny Phase 1 pipeline → summarize produces sane numbers."""
    from calcia.pipeline import simulate_neural_volume

    vol_params = VolumeParams(
        vol_sz=(40, 40, 20), vres=1, N_neur=3, N_den=2, N_bg=2, verbose=0,
    )
    result = simulate_neural_volume(vol_params=vol_params, seed=0, verbose=0)

    report = summarize(result)

    # Owner histogram sums to volume
    imaging_voxels = int(np.prod(result.neur_num.shape))
    assert int(report.owner_hist.sum()) == imaging_voxels
    # At least soma present
    assert report.component_counts["soma"] > 0
    # Pairwise matrix shape
    n = len(report.pair_keys)
    assert report.pair_matrix.shape == (n, n)

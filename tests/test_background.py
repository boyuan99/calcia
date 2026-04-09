"""Tests for Step 7: Background neuropil and axon generation."""

import numpy as np
import pytest

from calcia.algorithms.random_walk import dendrite_random_walk, FLT_MAX
from calcia.config.params import (
    VolumeParams, NeuronParams, DendParams, BgParams, AxonParams,
)
from calcia.volume.background import (
    generate_bg_dendrites,
    generate_axons,
    sort_axons,
    BgDendriteResult,
    AxonResult,
    BgProcessData,
)
from calcia.volume.fluorescence import CellFluorescenceData


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def simple_cost_matrix():
    """Small 20x20x20 cost matrix with random values."""
    np.random.seed(42)
    M = np.random.random((20, 20, 20)).astype(np.float32)
    # Boundary walls
    M[0, :, :] = FLT_MAX
    M[-1, :, :] = FLT_MAX
    M[:, 0, :] = FLT_MAX
    M[:, -1, :] = FLT_MAX
    M[:, :, 0] = FLT_MAX
    M[:, :, -1] = FLT_MAX
    return M


def _make_step6_result(seed=42):
    """Run Steps 2-6 with small parameters for testing."""
    from calcia.volume.neurons import sample_dense_neurons
    from calcia.volume.neural_volume import generate_neural_volume
    from calcia.volume.dendrites import grow_neuron_dendrites, grow_apical_dendrites
    from calcia.volume.fluorescence import set_cell_fluorescence

    np.random.seed(seed)
    vol_params = VolumeParams(vol_sz=(40, 40, 20), vres=2, N_neur=3)
    neur_params = NeuronParams(n_samps=100, nuc_fluorsc=0.5)
    dend_params = DendParams(
        dtParams=(6, 20, 12, 1, 3),
        atParams=(2, 5, 5, 5, 1),
        dims=(10, 10, 10),
        dimsSS=(5, 5, 5),
    )

    neurons, angles, positions = sample_dense_neurons(
        vol_params, neur_params, verbose=0,
    )
    vol_result = generate_neural_volume(
        neurons, positions, vol_params, neur_params, verbose=0,
    )

    np.random.seed(seed)
    dend_result = grow_neuron_dendrites(
        vol_params, dend_params, vol_result,
        positions=positions, rotation_angles=angles, verbose=0,
    )

    np.random.seed(seed)
    apical_result = grow_apical_dendrites(
        vol_params, dend_result.dend_params,
        dend_result, vol_result, verbose=0,
    )

    np.random.seed(seed + 1)
    fluor_result = set_cell_fluorescence(
        vol_params, neur_params, dend_result.dend_params,
        neur_num=apical_result.neur_num,
        neur_soma=vol_result.neur_soma,
        neur_num_ad=apical_result.neur_num_ad,
        positions=positions,
        neur_vol=vol_result.neur_vol,
        verbose=0,
    )

    return {
        "vol_params": vol_params,
        "neur_params": neur_params,
        "dend_params": dend_params,
        "positions": positions,
        "vol_result": vol_result,
        "apical_result": apical_result,
        "fluor_result": fluor_result,
    }


# =====================================================================
# TestDendriteRandomWalk
# =====================================================================

class TestDendriteRandomWalk:
    """Tests for the core random walk algorithm."""

    def test_output_shape(self, simple_cost_matrix):
        """Path should be (N, 3) int32."""
        M = simple_cost_matrix.copy()
        root = np.array([10, 10, 10], dtype=np.int32)
        ends = np.array([15, 15, 15], dtype=np.int32)
        path = dendrite_random_walk(M, root, ends, 0.5, 100, 10.0, 2, 3)
        assert path.ndim == 2
        assert path.shape[1] == 3
        assert path.dtype == np.int32

    def test_path_within_bounds(self, simple_cost_matrix):
        """All path points should be within volume bounds."""
        M = simple_cost_matrix.copy()
        root = np.array([10, 10, 10], dtype=np.int32)
        ends = np.array([15, 15, 5], dtype=np.int32)
        path = dendrite_random_walk(M, root, ends, 0.5, 100, 10.0, 2, 3)
        if len(path) > 0:
            assert np.all(path >= 0)
            assert np.all(path < 20)

    def test_path_connected(self, simple_cost_matrix):
        """Consecutive path points should be 6-connected."""
        M = simple_cost_matrix.copy()
        root = np.array([10, 10, 10], dtype=np.int32)
        ends = np.array([5, 5, 5], dtype=np.int32)
        path = dendrite_random_walk(M, root, ends, 0.5, 100, 10.0, 2, 3)
        if len(path) > 1:
            diffs = np.abs(np.diff(path, axis=0))
            steps = np.sum(diffs, axis=1)
            assert np.all(steps == 1), "Each step must be exactly 1 in one axis"

    def test_max_length(self, simple_cost_matrix):
        """Path length should not exceed maxlength."""
        M = simple_cost_matrix.copy()
        root = np.array([10, 10, 10], dtype=np.int32)
        ends = np.array([1, 1, 1], dtype=np.int32)
        maxlen = 15
        path = dendrite_random_walk(M, root, ends, 0.5, maxlen, 10.0, 5, 1)
        assert len(path) <= maxlen

    def test_min_length_rejection(self):
        """Paths shorter than minlength should return empty."""
        # Tiny volume with most voxels blocked: can only walk ~2 steps
        M = np.full((6, 6, 6), FLT_MAX, dtype=np.float32)
        M[2, 2, 2] = 0.1
        M[2, 3, 2] = 0.1
        M[2, 4, 2] = 0.1
        root = np.array([2, 2, 2], dtype=np.int32)
        ends = np.array([2, 4, 2], dtype=np.int32)
        path = dendrite_random_walk(M, root, ends, 0.5, 100, 10.0, 1, 10)
        assert len(path) == 0

    def test_cost_mutation(self, simple_cost_matrix):
        """M should be mutated for accepted paths."""
        M = simple_cost_matrix.copy()
        M_before = M.copy()
        root = np.array([10, 10, 10], dtype=np.int32)
        ends = np.array([15, 15, 15], dtype=np.int32)
        path = dendrite_random_walk(M, root, ends, 0.5, 100, 10.0, 2, 3)
        if len(path) > 0:
            assert not np.array_equal(M, M_before), "M should be modified"

    def test_blocked_returns_empty(self):
        """All-FLT_MAX volume should produce empty path."""
        M = np.full((10, 10, 10), FLT_MAX, dtype=np.float32)
        root = np.array([5, 5, 5], dtype=np.int32)
        ends = np.array([8, 8, 8], dtype=np.int32)
        path = dendrite_random_walk(M, root, ends, 0.5, 100, 10.0, 1, 3)
        assert len(path) == 0

    def test_direction_bias(self, simple_cost_matrix):
        """High distsc should bias path toward endpoint."""
        M = simple_cost_matrix.copy()
        root = np.array([5, 10, 10], dtype=np.int32)
        ends = np.array([15, 10, 10], dtype=np.int32)
        path = dendrite_random_walk(M, root, ends, 5.0, 50, 10.0, 3, 3)
        if len(path) > 0:
            # Final x should be closer to 15 than start x=5
            assert path[-1, 0] > 5

    def test_reproducibility(self, simple_cost_matrix):
        """Same seed should produce same path."""
        for _ in range(3):
            M1 = simple_cost_matrix.copy()
            M2 = simple_cost_matrix.copy()
            root = np.array([10, 10, 10], dtype=np.int32)
            ends = np.array([15, 15, 15], dtype=np.int32)
            p1 = dendrite_random_walk(M1, root, ends, 0.5, 50, 10.0, 2, 3)
            p2 = dendrite_random_walk(M2, root, ends, 0.5, 50, 10.0, 2, 3)
            # Random walk is deterministic given same M (greedy, no randomness)
            np.testing.assert_array_equal(p1, p2)


# =====================================================================
# TestGenerateBgDendrites
# =====================================================================

class TestGenerateBgDendrites:
    """Tests for background dendrite generation (Part A)."""

    @pytest.fixture(scope="class")
    def step7a_data(self):
        data = _make_step6_result()
        np.random.seed(100)
        bg_params = BgParams(
            maxlength=30, minlength=3, maxel=1, fillweight=50.0,
        )
        result = generate_bg_dendrites(
            data["vol_params"], bg_params, data["dend_params"],
            data["fluor_result"].neur_vol,
            data["apical_result"].neur_num,
            data["fluor_result"].gp_vals,
            data["vol_result"].gp_nuc,
            data["positions"],
            verbose=0,
        )
        return {**data, "bg_result": result, "bg_params": bg_params}

    def test_output_types(self, step7a_data):
        """Result should have correct types."""
        r = step7a_data["bg_result"]
        assert isinstance(r, BgDendriteResult)
        assert r.neur_num.dtype == np.uint16
        assert r.neur_vol.dtype == np.float32
        assert isinstance(r.gp_vals, list)
        assert isinstance(r.N_den2, int)

    def test_neur_num_ids_correct(self, step7a_data):
        """New bg IDs should start at N_neur + N_den + 1."""
        r = step7a_data["bg_result"]
        vp = step7a_data["vol_params"]
        Ncomps = vp.N_neur + vp.N_den
        max_id = int(r.neur_num.max())
        if r.N_den2 > 0:
            assert max_id >= Ncomps + 1
            assert max_id <= Ncomps + r.N_den2

    def test_gp_vals_extended(self, step7a_data):
        """gp_vals should grow by N_den2."""
        r = step7a_data["bg_result"]
        orig = step7a_data["fluor_result"].gp_vals
        assert len(r.gp_vals) == len(orig) + r.N_den2

    def test_fluorescence_positive(self, step7a_data):
        """Background fluorescence should be positive."""
        r = step7a_data["bg_result"]
        orig_len = len(step7a_data["fluor_result"].gp_vals)
        for g in r.gp_vals[orig_len:]:
            if len(g.fluorescence) > 0:
                assert np.all(g.fluorescence > 0)

    def test_no_overlap_existing(self, step7a_data):
        """Bg dendrites should not overwrite existing neuron soma voxels."""
        r = step7a_data["bg_result"]
        orig_neur_num = step7a_data["apical_result"].neur_num
        # Where original had neurons, the new should still have same IDs
        orig_mask = orig_neur_num > 0
        np.testing.assert_array_equal(
            r.neur_num[orig_mask], orig_neur_num[orig_mask],
        )

    def test_neur_locs_extended(self, step7a_data):
        """neur_locs should grow."""
        r = step7a_data["bg_result"]
        orig = step7a_data["positions"]
        assert r.neur_locs.shape[0] > orig.shape[0]
        assert r.neur_locs.shape[1] == 3


# =====================================================================
# TestGenerateAxons
# =====================================================================

class TestGenerateAxons:
    """Tests for axon generation (Part B)."""

    @pytest.fixture(scope="class")
    def step7b_data(self):
        data = _make_step6_result()
        np.random.seed(200)
        axon_params = AxonParams(
            maxlength=30, minlength=3, numbranches=3, varbranches=1,
            maxfill=0.05, padsize=5, maxel=4, maxvoxel=3,
        )
        vp = VolumeParams(
            vol_sz=data["vol_params"].vol_sz,
            vres=data["vol_params"].vres,
            N_neur=data["vol_params"].N_neur,
            N_bg=30,
        )
        result = generate_axons(
            vp, axon_params,
            data["fluor_result"].neur_vol,
            data["apical_result"].neur_num,
            data["fluor_result"].gp_vals,
            data["vol_result"].gp_nuc,
            verbose=0,
        )
        return {**data, "axon_result": result, "axon_params": axon_params, "vp": vp}

    def test_output_types(self, step7b_data):
        """Result should have correct types."""
        r = step7b_data["axon_result"]
        assert isinstance(r, AxonResult)
        assert r.neur_vol.dtype == np.float32
        assert isinstance(r.gp_bgvals, list)
        assert isinstance(r.N_bg_actual, int)

    def test_gp_bgvals_structure(self, step7b_data):
        """Each entry should have (indices, fluorescence) arrays."""
        r = step7b_data["axon_result"]
        for idx, fl in r.gp_bgvals:
            assert idx.ndim == 1
            assert fl.ndim == 1
            assert len(idx) == len(fl)
            assert idx.dtype == np.int32
            assert fl.dtype == np.float32

    def test_fluorescence_additive(self, step7b_data):
        """neur_vol at axon locations should be >= original."""
        r = step7b_data["axon_result"]
        orig = step7b_data["fluor_result"].neur_vol
        # Axon fluorescence is added on top
        diff = r.neur_vol - orig
        # At axon locations diff should be >= 0
        assert np.all(diff >= -1e-6)

    def test_padding_stripped(self, step7b_data):
        """No indices should be outside volume bounds."""
        r = step7b_data["axon_result"]
        vp = step7b_data["vp"]
        grid_shape = tuple(np.array(vp.vol_sz) * vp.vres)
        total = np.prod(grid_shape)
        for idx, fl in r.gp_bgvals:
            assert np.all(idx >= 0)
            assert np.all(idx < total)

    def test_some_axons_generated(self, step7b_data):
        """Should generate at least some axons."""
        r = step7b_data["axon_result"]
        assert r.N_bg_actual > 0

    def test_reproducibility(self):
        """Same seed should produce same result."""
        data = _make_step6_result()
        axon_params = AxonParams(
            maxlength=20, minlength=3, numbranches=2, varbranches=1,
            maxfill=0.02, padsize=5, maxel=4, maxvoxel=3,
        )
        vp = VolumeParams(
            vol_sz=data["vol_params"].vol_sz,
            vres=data["vol_params"].vres,
            N_neur=data["vol_params"].N_neur,
            N_bg=10,
        )

        np.random.seed(300)
        r1 = generate_axons(
            vp, axon_params,
            data["fluor_result"].neur_vol,
            data["apical_result"].neur_num,
            data["fluor_result"].gp_vals,
            data["vol_result"].gp_nuc,
            verbose=0,
        )

        np.random.seed(300)
        r2 = generate_axons(
            vp, axon_params,
            data["fluor_result"].neur_vol,
            data["apical_result"].neur_num,
            data["fluor_result"].gp_vals,
            data["vol_result"].gp_nuc,
            verbose=0,
        )

        assert r1.N_bg_actual == r2.N_bg_actual
        np.testing.assert_array_equal(r1.neur_vol, r2.neur_vol)


# =====================================================================
# TestSortAxons
# =====================================================================

class TestSortAxons:
    """Tests for axon sorting (Part C)."""

    @pytest.fixture
    def sort_data(self):
        """Create simple axon data for sorting."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(20, 20, 10), vres=2, N_neur=3)
        axon_params = AxonParams(N_proc=8)
        grid_shape = tuple(np.array(vol_params.vol_sz) * vol_params.vres)

        # Create 15 fake axons with random indices
        gp_bgvals = []
        for i in range(15):
            n = np.random.randint(10, 50)
            indices = np.random.choice(
                np.prod(grid_shape), n, replace=False,
            ).astype(np.int32)
            fl = np.random.random(n).astype(np.float32)
            gp_bgvals.append((indices, fl))

        # Fake cell positions
        N_total = vol_params.N_neur + vol_params.N_den
        cell_pos = np.random.random((N_total, 3)).astype(np.float32) * np.array(grid_shape)

        return vol_params, axon_params, gp_bgvals, cell_pos

    def test_output_length(self, sort_data):
        """Should return N_proc processes."""
        vp, ap, bgvals, pos = sort_data
        result = sort_axons(vp, ap, bgvals, pos, verbose=0)
        assert len(result) == ap.N_proc

    def test_all_axons_assigned(self, sort_data):
        """All input voxels should appear in output."""
        vp, ap, bgvals, pos = sort_data
        result = sort_axons(vp, ap, bgvals, pos, verbose=0)

        total_input = sum(len(idx) for idx, _ in bgvals)
        total_output = sum(len(bp.indices) for bp in result)
        assert total_output == total_input

    def test_element_types(self, sort_data):
        """Each BgProcessData should have correct types."""
        vp, ap, bgvals, pos = sort_data
        result = sort_axons(vp, ap, bgvals, pos, verbose=0)
        for bp in result:
            assert isinstance(bp, BgProcessData)
            assert bp.indices.dtype == np.int32
            assert bp.fluorescence.dtype == np.float32

    def test_random_mode(self):
        """When N_proc <= N_comps, all axons randomly assigned."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(20, 20, 10), vres=2, N_neur=10)
        axon_params = AxonParams(N_proc=5)  # less than N_neur+N_den
        grid_shape = tuple(np.array(vol_params.vol_sz) * vol_params.vres)

        gp_bgvals = []
        for i in range(10):
            n = 20
            indices = np.random.choice(
                np.prod(grid_shape), n, replace=False,
            ).astype(np.int32)
            fl = np.ones(n, dtype=np.float32) * 0.1
            gp_bgvals.append((indices, fl))

        cell_pos = np.random.random((50, 3)).astype(np.float32) * 40
        result = sort_axons(vol_params, axon_params, gp_bgvals, cell_pos, verbose=0)
        assert len(result) == 5
        total = sum(len(bp.indices) for bp in result)
        assert total == 200  # 10 axons * 20 voxels

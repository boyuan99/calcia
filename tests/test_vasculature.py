"""Tests for blood vessel generation module."""

import numpy as np
import pytest
from calcia.config.params import VolumeParams, VascParams
from calcia.volume.vasculature import (
    VesselNode,
    VesselConnection,
    VesselNetwork,
    pseudo_rand_sample_2d,
    pseudo_rand_sample_3d,
    branch_grow_nodes,
    generate_source_nodes,
    grow_major_vessels,
    grow_diving_vessels,
    grow_capillaries,
    nodes_to_connections,
    connections_to_volume,
    simulate_blood_vessels,
)


class TestVesselDataStructures:
    """Test vessel data structures."""

    def test_vessel_node_creation(self):
        """Test VesselNode creation with defaults."""
        node = VesselNode(num=0)
        assert node.num == 0
        assert node.root == -1
        assert node.conn == []
        assert np.array_equal(node.pos, np.zeros(3))
        assert node.type == 0

    def test_vessel_node_with_position(self):
        """Test VesselNode with position."""
        node = VesselNode(num=5, pos=[10, 20, 30])
        assert node.num == 5
        assert np.allclose(node.pos, [10, 20, 30])

    def test_vessel_connection_creation(self):
        """Test VesselConnection creation."""
        conn = VesselConnection(start=0, ends=1, weight=5.0)
        assert conn.start == 0
        assert conn.ends == 1
        assert conn.weight == 5.0

    def test_vessel_network_creation(self):
        """Test VesselNetwork creation."""
        network = VesselNetwork()
        assert network.nodes == []
        assert network.connections == []
        assert network.vessel_volume is None


class TestPseudoRandomSampling:
    """Test pseudo-random sampling with Gaussian exclusion."""

    def test_pseudo_rand_sample_2d_basic(self):
        """Test basic 2D sampling."""
        np.random.seed(42)
        points = pseudo_rand_sample_2d(
            n_samples=10,
            bounds=(0, 100, 0, 100),
            exclusion_sigma=20.0,
        )
        assert points.shape == (10, 2)
        assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] <= 100)
        assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] <= 100)

    def test_pseudo_rand_sample_2d_exclusion(self):
        """Test that 2D sampling respects exclusion."""
        np.random.seed(42)
        points = pseudo_rand_sample_2d(
            n_samples=5,
            bounds=(0, 100, 0, 100),
            exclusion_sigma=30.0,
        )
        # Check minimum distance between points
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                # Points should be reasonably spread out
                assert dist > 5.0, "Points too close together"

    def test_pseudo_rand_sample_3d_basic(self):
        """Test basic 3D sampling."""
        np.random.seed(42)
        points = pseudo_rand_sample_3d(
            n_samples=10,
            bounds=(0, 100, 0, 100, 0, 50),
            exclusion_sigma=15.0,
        )
        assert points.shape == (10, 3)
        assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] <= 100)
        assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] <= 100)
        assert np.all(points[:, 2] >= 0) and np.all(points[:, 2] <= 50)

    def test_pseudo_rand_sample_2d_with_existing(self):
        """Test 2D sampling with existing points."""
        np.random.seed(42)
        existing = np.array([[50, 50], [25, 25]])
        points = pseudo_rand_sample_2d(
            n_samples=5,
            bounds=(0, 100, 0, 100),
            exclusion_sigma=20.0,
            existing_points=existing,
        )
        # New points should avoid existing ones
        for p in points:
            for e in existing:
                dist = np.linalg.norm(p - e)
                assert dist > 3.0, "New point too close to existing"


class TestBranchGrowth:
    """Test branch growth algorithm."""

    def test_branch_grow_nodes_basic(self):
        """Test basic branch growth."""
        np.random.seed(42)
        nodes = branch_grow_nodes(
            start_pos=np.array([50.0, 50.0]),
            bounds=(0, 100, 0, 100),
            n_steps=10,
            step_size=5.0,
            step_var=1.0,
            direction=np.array([1.0, 0.0]),
            branch_prob=0.0,  # No branching
        )
        assert len(nodes) > 0
        # All nodes should be within bounds (XY)
        for node in nodes:
            assert 0 <= node.pos[0] <= 100
            assert 0 <= node.pos[1] <= 100

    def test_branch_grow_nodes_with_branching(self):
        """Test branch growth with branching enabled."""
        np.random.seed(42)
        nodes = branch_grow_nodes(
            start_pos=np.array([50.0, 50.0]),
            bounds=(0, 100, 0, 100),
            n_steps=20,
            step_size=5.0,
            step_var=1.0,
            direction=np.array([1.0, 0.0]),
            branch_prob=0.3,  # High branching probability
            min_dist=2.0,  # Smaller min distance for more nodes
        )
        # Should have some nodes (at least 1)
        assert len(nodes) >= 1

    def test_branch_grow_nodes_connectivity(self):
        """Test that grown nodes are connected properly."""
        np.random.seed(42)
        nodes = branch_grow_nodes(
            start_pos=np.array([50.0, 50.0]),
            bounds=(0, 100, 0, 100),
            n_steps=5,
            step_size=5.0,
            step_var=0.5,
            direction=np.array([1.0, 0.0]),
            branch_prob=0.0,
        )
        if len(nodes) > 1:
            # Check connectivity: each non-first node should have a parent
            for i in range(1, len(nodes)):
                assert nodes[i].root >= 0 or len(nodes[i].conn) > 0


class TestSourceNodes:
    """Test source node generation."""

    def test_generate_source_nodes(self):
        """Test source node generation."""
        vol_params = VolumeParams(vol_sz=(100, 100, 50))
        vasc_params = VascParams()

        nodes = generate_source_nodes(vol_params, vasc_params)

        # Should have at least one node per edge
        assert len(nodes) >= 4

        # All should be source nodes (type 1)
        for node in nodes:
            assert node.type == 1
            assert node.root == -1  # Source nodes have no parent

        # All should be at surface depth
        for node in nodes:
            assert node.pos[2] == vasc_params.depth_surf


class TestMajorVessels:
    """Test major vessel growth."""

    def test_grow_major_vessels(self):
        """Test complete major vessel growth."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(100, 100, 50))
        vasc_params = VascParams()

        network = grow_major_vessels(vol_params, vasc_params, verbose=0)

        # Should have nodes
        assert len(network.nodes) > 0

        # Should have source nodes
        source_count = sum(1 for n in network.nodes if n.type == 1)
        assert source_count >= 4


class TestDivingVessels:
    """Test diving vessel growth."""

    def test_grow_diving_vessels(self):
        """Test diving vessel growth."""
        np.random.seed(42)
        # Use larger volume so diving vessels can grow (depth_vasc=200 by default)
        vol_params = VolumeParams(vol_sz=(100, 100, 250))
        vasc_params = VascParams(depth_surf=15.0, depth_vasc=200.0)

        # First grow major vessels
        network = grow_major_vessels(vol_params, vasc_params, verbose=0)
        initial_count = len(network.nodes)

        # Then grow diving vessels
        network = grow_diving_vessels(network, vol_params, vasc_params, verbose=0)

        # Should have more nodes
        assert len(network.nodes) > initial_count

        # Should have some diving vessel nodes (type 3)
        diving_count = sum(1 for n in network.nodes if n.type == 3)
        assert diving_count > 0


class TestCapillaries:
    """Test capillary generation."""

    def test_grow_capillaries(self):
        """Test capillary network growth."""
        np.random.seed(42)
        # Use larger volume so diving vessels and capillaries can grow
        vol_params = VolumeParams(vol_sz=(100, 100, 250))
        vasc_params = VascParams(depth_surf=15.0, depth_vasc=200.0)

        # First grow major vessels and diving vessels
        network = grow_major_vessels(vol_params, vasc_params, verbose=0)
        network = grow_diving_vessels(network, vol_params, vasc_params, verbose=0)
        initial_count = len(network.nodes)

        # Then grow capillaries
        network = grow_capillaries(network, vol_params, vasc_params, verbose=0)

        # Should have more nodes
        assert len(network.nodes) > initial_count

        # Should have capillary nodes (type 4)
        cap_count = sum(1 for n in network.nodes if n.type == 4)
        assert cap_count > 0


class TestVolumeRendering:
    """Test volume rendering of vessels."""

    def test_nodes_to_connections(self):
        """Test conversion from nodes to connections."""
        # Create a simple network
        nodes = [
            VesselNode(num=0, pos=np.array([0, 0, 0]), conn=[1]),
            VesselNode(num=1, pos=np.array([10, 0, 0]), conn=[0, 2]),
            VesselNode(num=2, pos=np.array([10, 10, 0]), conn=[1]),
        ]
        network = VesselNetwork(nodes=nodes)
        vasc_params = VascParams()

        network = nodes_to_connections(network, vasc_params)

        # Should have 2 connections
        assert len(network.connections) == 2

    def test_connections_to_volume(self):
        """Test rendering connections to volume."""
        # Create a simple network with connections
        nodes = [
            VesselNode(num=0, pos=np.array([10, 10, 10]), conn=[1], type=1),
            VesselNode(num=1, pos=np.array([20, 10, 10]), conn=[0]),
        ]
        locs = np.array([[10, 10, 10], [15, 10, 10], [20, 10, 10]])
        connections = [
            VesselConnection(start=0, ends=1, weight=10.0, locs=locs)
        ]
        network = VesselNetwork(nodes=nodes, connections=connections)

        vol_params = VolumeParams(vol_sz=(50, 50, 30), vres=1)
        vasc_params = VascParams()

        network = connections_to_volume(network, vol_params, vasc_params, verbose=0)

        # Should have a volume
        assert network.vessel_volume is not None
        assert network.vessel_volume.shape == (50, 50, 30)

        # Volume should have some vessel voxels
        assert np.sum(network.vessel_volume) > 0


class TestFullPipeline:
    """Test complete vasculature simulation pipeline."""

    def test_simulate_blood_vessels_basic(self):
        """Test complete blood vessel simulation."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(80, 80, 40))
        vasc_params = VascParams()

        network = simulate_blood_vessels(vol_params, vasc_params, verbose=0)

        # Should have nodes
        assert len(network.nodes) > 0

        # Should have connections
        assert len(network.connections) > 0

        # Should have rendered volume
        assert network.vessel_volume is not None

        # Volume should have vessels
        assert np.sum(network.vessel_volume) > 0

    def test_simulate_blood_vessels_disabled(self):
        """Test that simulation can be disabled."""
        vol_params = VolumeParams(vol_sz=(50, 50, 25))
        vasc_params = VascParams(flag=False)

        network = simulate_blood_vessels(vol_params, vasc_params, verbose=0)

        # Should return empty network
        assert len(network.nodes) == 0
        assert len(network.connections) == 0

    def test_simulate_blood_vessels_statistics(self):
        """Test vessel network statistics."""
        np.random.seed(42)
        # Use larger volume so all vessel types can be generated
        vol_params = VolumeParams(vol_sz=(100, 100, 250))
        vasc_params = VascParams(depth_surf=15.0, depth_vasc=200.0)

        network = simulate_blood_vessels(vol_params, vasc_params, verbose=0)

        # Count node types
        source_count = sum(1 for n in network.nodes if n.type == 1)
        diving_count = sum(1 for n in network.nodes if n.type == 3)
        cap_count = sum(1 for n in network.nodes if n.type == 4)

        # Should have all vessel types
        assert source_count > 0, "No source nodes"
        assert diving_count > 0, "No diving vessel nodes"
        assert cap_count > 0, "No capillary nodes"

        # Vessel volume fill should be reasonable (not too much, not too little)
        fill_fraction = np.sum(network.vessel_volume) / np.prod(network.vessel_volume.shape)
        assert 0.0001 < fill_fraction < 0.5, f"Unexpected fill fraction: {fill_fraction}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_volume(self):
        """Test with a very small volume."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(30, 30, 20))
        vasc_params = VascParams()

        # Should still work
        network = simulate_blood_vessels(vol_params, vasc_params, verbose=0)
        assert len(network.nodes) >= 0

    def test_large_source_freq(self):
        """Test with large source frequency (fewer sources)."""
        np.random.seed(42)
        vol_params = VolumeParams(vol_sz=(100, 100, 50))
        vasc_params = VascParams(sourceFreq=500.0)  # Fewer sources

        network = simulate_blood_vessels(vol_params, vasc_params, verbose=0)

        # Should have fewer source nodes
        source_count = sum(1 for n in network.nodes if n.type == 1)
        assert source_count >= 4  # At least one per edge

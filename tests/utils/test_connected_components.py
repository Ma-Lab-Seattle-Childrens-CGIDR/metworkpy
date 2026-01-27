"""Tests for connected_components submodule"""

from metworkpy.utils import connected_components


class TestFindConnectedComponents:
    def test_no_edges(self):
        node_list = [f"node{i}" for i in range(9)]
        edge_list = []
        components = connected_components.find_connected_components(
            node_list=node_list, edge_list=edge_list
        )
        # Assert that there is a components for each node
        assert len(components) == len(node_list), (
            "There is not one component per node despite no edges being passed"
        )

    def test_small_graph(self):
        node_list = [f"node{i}" for i in range(1, 10)]
        edge_list = [
            ("node1", "node2"),
            ("node1", "node3"),
            ("node4", "node5"),
            ("node6", "node7"),
            ("node7", "node8"),
            ("node7", "node9"),
        ]
        components = connected_components.find_connected_components(
            node_list=node_list, edge_list=edge_list
        )
        # Assert that there are 3 connected components
        assert len(components) == 3, "There is not the expected 3 components"
        # Now count the size of the components
        assert set([len(it) for it in components]) == {2, 3, 4}, (
            "Components are not of the expected size"
        )


class TestFindRepresentativeNodes:
    def test_no_edges(self):
        node_list = [f"node{idx}" for idx in range(1, 10)]
        edge_list = []
        representative_nodes = connected_components.find_representative_nodes(
            node_list=node_list, edge_list=edge_list
        )
        assert len(representative_nodes) == len(node_list), (
            "Each node should be representative since there are no edges"
        )

    def test_small_graph(self):
        node_list = [f"node{i}" for i in range(1, 10)]
        edge_list = [
            ("node1", "node2"),
            ("node1", "node3"),
            ("node4", "node5"),
            ("node6", "node7"),
            ("node7", "node8"),
            ("node7", "node9"),
        ]
        representative_nodes = connected_components.find_representative_nodes(
            node_list=node_list, edge_list=edge_list
        )
        # There are 3 connected components, and each should have a single
        # representative node
        assert len(representative_nodes) == 3
        # Check that the return value is as expected
        expected_representative_nodes = {
            "node1": {"node1", "node2", "node3"},
            "node4": {"node4", "node5"},
            "node7": {"node6", "node7", "node8", "node9"},
        }
        for rep_node, node_set in representative_nodes.items():
            if rep_node not in {"node4", "node5"}:
                assert rep_node in expected_representative_nodes
                assert expected_representative_nodes[rep_node] == node_set
            else:
                # NOTE: node4 and node5 have the same degree,
                # so which is picked as representative is not garunteed
                assert expected_representative_nodes["node4"] == node_set

# Imports
# Standard Library Imports
import unittest

# External Imports
import networkx as nx

# Local Imports
from metworkpy.network.projection import bipartite_project


class TestBipartiteProject(unittest.TestCase):
    test_graph = None
    test_digraph = None
    test_weighted_graph = None

    @classmethod
    def setUpClass(cls):
        # Create test graph
        g = nx.Graph()
        g.add_nodes_from(["A", "B", "C", "D", "E"])
        g.add_edges_from([("A", "D"), ("A", "E"), ("B", "E"), ("C", "D")])
        cls.test_graph = g

        # Create directed test graph
        dg = nx.DiGraph()
        dg.add_nodes_from(["A", "B", "C", "D", "E"])
        dg.add_edges_from(
            [
                ("A", "D"),
                ("B", "D"),
                ("B", "E"),
                ("D", "C"),
            ]
        )
        cls.test_digraph = dg

        # Create weighted test graph
        wg = nx.DiGraph()
        wg.add_nodes_from(["A", "B", "C", "D", "E"])
        wg.add_edges_from(
            [
                ("A", "D", {"weight": 5}),
                ("B", "D", {"weight": 2}),
                ("B", "E", {"weight": 2}),
                ("D", "C", {"weight": 3}),
            ]
        )
        cls.test_weighted_graph = wg

        cls.node_set = ["A", "B", "C"]

    def test_unweighted_undirected(self):
        projected_graph = bipartite_project(self.test_graph, node_set=self.node_set)
        # Test that only nodes in node set are in final graph
        self.assertListEqual(list(projected_graph.nodes()), self.node_set)
        # Test edges
        self.assertTrue(projected_graph.has_edge("A", "C"))
        self.assertTrue(projected_graph.has_edge("A", "B"))
        self.assertFalse(projected_graph.has_edge("B", "C"))

    def test_unweighted_directed(self):
        directed_projection = bipartite_project(
            self.test_digraph, node_set=self.node_set, directed=True
        )
        # Test node set
        self.assertListEqual(list(directed_projection.nodes()), self.node_set)
        # Test known edges
        self.assertTrue(directed_projection.has_edge("A", "C"))
        self.assertFalse(directed_projection.has_edge("A", "B"))
        self.assertFalse(directed_projection.has_edge("C", "A"))
        # Test if directed=None leaves the graph directed
        directed_projection2 = bipartite_project(
            self.test_digraph, node_set=self.node_set
        )
        self.assertTrue(
            nx.utils.graphs_equal(directed_projection, directed_projection2)
        )
        # Test conversion to undirected
        undirected_projection = bipartite_project(
            self.test_digraph, node_set=self.node_set, directed=False
        )
        self.assertFalse(isinstance(undirected_projection, nx.DiGraph))
        self.assertListEqual(list(undirected_projection.nodes()), self.node_set)
        self.assertTrue(
            nx.utils.graphs_equal(
                undirected_projection, nx.complete_graph(["A", "B", "C"])
            )
        )

    def test_weighted_directed(self):
        weighted_projection = bipartite_project(
            self.test_weighted_graph,
            node_set=self.node_set,
            directed=True,
            weight=min,
            weight_attribute="weight",
        )
        self.assertListEqual(list(weighted_projection.nodes()), self.node_set)
        self.assertTrue(weighted_projection.has_edge("A", "C"))
        self.assertEqual(weighted_projection.get_edge_data("A", "C")["weight"], 3)

    def test_weighted_undirected(self):
        weighted_projection = bipartite_project(
            self.test_weighted_graph,
            node_set=self.node_set,
            directed=False,
            weight=min,
            weight_attribute="weight",
        )
        self.assertListEqual(list(weighted_projection.nodes()), self.node_set)
        self.assertTrue(weighted_projection.has_edge("A", "C"))
        self.assertEqual(weighted_projection.get_edge_data("A", "C")["weight"], 3)
        self.assertEqual(weighted_projection.get_edge_data("A", "B")["weight"], 2)
        self.assertEqual(weighted_projection.get_edge_data("B", "C")["weight"], 2)


if __name__ == "__main__":
    unittest.main()

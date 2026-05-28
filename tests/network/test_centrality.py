# Standard Library Imports
import unittest

# External Imports
import networkx as nx
import numpy as np
import pandas as pd

# Local Imports
from metworkpy.network.centrality import closeness_centrality_subset


class TestClosenessCentralitySubset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.random_graph = nx.connected_watts_strogatz_graph(
            35, 5, 0.2, seed=4724792568309286
        )
        rng = np.random.default_rng(12983019283)
        cls.random_targets = list(rng.choice(cls.random_graph.nodes, 10))
        cls.random_graph_closeness = pd.Series(
            nx.closeness_centrality(cls.random_graph)
        )

        small_graph = nx.Graph()
        small_graph.add_nodes_from(
            [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
            ]
        )
        small_graph.add_edges_from(
            [
                ("E", "A"),
                ("B", "A"),
                ("C", "A"),
                ("G", "A"),
                ("C", "D"),
                ("C", "F"),
                ("G", "H"),
                ("G", "F"),
                ("I", "F"),
                ("I", "H"),
                ("J", "F"),
            ]
        )
        cls.small_graph = small_graph
        cls.small_graph_targets = ["E", "D", "C", "G"]
        cls.small_graph_closeness = pd.Series(
            nx.closeness_centrality(cls.small_graph)
        )

    def test_all_targeted(self):
        calculated_closeness = pd.Series(
            closeness_centrality_subset(
                self.random_graph, targets=self.random_graph.nodes
            )
        )
        pd.testing.assert_series_equal(
            calculated_closeness, self.random_graph_closeness
        )

    def test_random_subset(self):
        sub_close = pd.Series(
            closeness_centrality_subset(self.random_graph, self.random_targets)
        )
        self.assertTrue(len(sub_close) == len(self.random_graph.nodes))

    def test_small_graph(self):
        sub_close = pd.Series(
            closeness_centrality_subset(
                self.small_graph, self.small_graph_targets
            )
        )
        for n in [
            "A",
            "C",
            "D",
            "E",
        ]:
            self.assertGreater(sub_close[n], self.small_graph_closeness[n])


if __name__ == "__main__":
    unittest.main()

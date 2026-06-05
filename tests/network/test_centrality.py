# Standard Library Imports
from metworkpy.network import create_reaction_network
import pathlib
import unittest

# External Imports
import cobra
import networkx as nx
import numpy as np
import pandas as pd

# Local Imports
from metworkpy.utils import read_model
from metworkpy.network.network_construction import (
    create_metabolic_network,
    bipartite_project,
)
from metworkpy.network.centrality import (
    closeness_centrality_subset,
    betweenness_centrality_subset,
    betweenness_centrality_bipartite_subset,
)


class TestCentralitySubset(unittest.TestCase):
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
        cls.random_graph_betweenness = pd.Series(
            nx.betweenness_centrality(cls.random_graph)
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
        cls.small_graph_betweenness = pd.Series(
            nx.betweenness_centrality(cls.small_graph)
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

        calculated_betweenness = pd.Series(
            betweenness_centrality_subset(
                self.random_graph, self.random_graph.nodes
            )
        )
        pd.testing.assert_series_equal(
            calculated_betweenness, self.random_graph_betweenness
        )

    def test_random_subset(self):
        sub_close = pd.Series(
            closeness_centrality_subset(self.random_graph, self.random_targets)
        )
        self.assertTrue(len(sub_close) == len(self.random_graph.nodes))

        sub_between = pd.Series(
            betweenness_centrality_subset(
                self.random_graph, self.random_targets
            )
        )
        self.assertTrue(len(sub_between) == len(self.random_graph.nodes))

    def test_small_graph(self):
        sub_close = pd.Series(
            closeness_centrality_subset(
                self.small_graph, self.small_graph_targets
            )
        )
        sub_btwn = pd.Series(
            betweenness_centrality_subset(
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
        for n in ["A", "C"]:
            self.assertGreater(sub_btwn[n], self.small_graph_betweenness[n])


class TestCentralityBipartiteSubset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cobra.Configuration().solver = "glpk"
        data_path = pathlib.Path(__file__).parent.parent / "data"
        textbook_model = read_model(data_path / "textbook_model.json")
        # Create the bipartite network from the textbook model
        cls.textbook_network = create_metabolic_network(
            model=textbook_model, weighted=False, directed=False
        )
        cls.textbook_reactions = textbook_model.reactions.list_attr("id")
        cls.textbook_reaction_network = create_reaction_network(
            model=textbook_model, weighted=False, directed=False
        )
        # Also, create a line graph for testing
        line_network = nx.Graph()
        line_network.add_edges_from(
            [
                ("A", "a"),
                ("A", "b"),
                ("A", "c"),
                ("a", "B"),
                ("b", "B"),
                ("c", "B"),
            ]
        )
        cls.line_network = line_network
        # And a small graph (but more complicated than a line graph)
        small_network = nx.Graph()
        small_network.add_edges_from(
            [
                ("A", "a"),
                ("A", "b"),
                ("a", "B"),
                ("b", "B"),
                ("A", "c"),
                ("c", "E"),
                ("A", "d"),
                ("A", "e"),
                ("d", "C"),
                ("e", "C"),
                ("C", "f"),
                ("f", "D"),
                ("A", "j"),
                ("j", "F"),
                ("C", "h"),
                ("C", "i"),
                ("h", "F"),
                ("i", "F"),
                ("F", "k"),
                ("k", "G"),
                ("G", "l"),
                ("l", "H"),
                ("G", "m"),
                ("m", "I"),
            ]
        )
        cls.small_network = small_network
        cls.small_network_partition_1 = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
        ]
        cls.small_network_partition_2 = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
        ]

    def test_line_network(self):
        btwn = pd.Series(
            betweenness_centrality_bipartite_subset(
                self.line_network, ["A", "B"], ["A", "B"]
            )
        )
        btwn_expected = pd.Series(
            [0.0, 0.0, 1.0, 1.0, 1.0],
            index=pd.Index(["A", "B", "a", "b", "c"]),
        )
        pd.testing.assert_series_equal(btwn, btwn_expected, check_like=True)

    def test_small_network(self):
        btwn = pd.Series(
            betweenness_centrality_bipartite_subset(
                self.small_network,
                node_partition=self.small_network_partition_1,
            )
        )
        btwn_subset = pd.Series(
            betweenness_centrality_bipartite_subset(
                self.small_network,
                node_partition=self.small_network_partition_1,
                targets=["B", "D", "E", "F"],
            )
        )
        for n in ["A", "C", "a", "b", "d", "e", "f"]:
            assert btwn_subset[n] > btwn[n]

        # Compute the subset betweenness on the projected graph
        proj_graph = bipartite_project(
            self.small_network, self.small_network_partition_1
        )
        proj_btwn_subset = pd.Series(
            betweenness_centrality_subset(proj_graph, ["B", "D", "E", "F"])
        )

        # The betweenness centrality of the nodes should be the same following projection
        pd.testing.assert_series_equal(
            btwn_subset[proj_btwn_subset.index], proj_btwn_subset
        )

    def test_metabolic_network(self):
        rng = np.random.default_rng()
        # First test using all reactions as targets
        btwn = pd.Series(
            betweenness_centrality_subset(self.textbook_reaction_network)
        )
        btwn_bipart = pd.Series(
            betweenness_centrality_bipartite_subset(
                self.textbook_network, self.textbook_reactions
            )
        )
        pd.testing.assert_series_equal(btwn_bipart[btwn.index], btwn)
        # Now test with a random reaction subset
        targets = rng.choice(self.textbook_reactions, 20, replace=False)
        btwn = pd.Series(
            betweenness_centrality_subset(
                self.textbook_reaction_network, targets
            )
        )
        btwn_bipart = pd.Series(
            betweenness_centrality_bipartite_subset(
                self.textbook_network, self.textbook_reactions, targets
            )
        )
        pd.testing.assert_series_equal(btwn_bipart[btwn.index], btwn)


if __name__ == "__main__":
    unittest.main()

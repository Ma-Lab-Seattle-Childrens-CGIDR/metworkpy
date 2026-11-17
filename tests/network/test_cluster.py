"""
Test the cluster submodule
"""

# Standard Library Imports
from __future__ import annotations

import unittest
from collections import defaultdict

# External Imports
import networkx as nx
import numpy as np
import pandas as pd

from metworkpy.examples import get_example_model

# Local Imports
from metworkpy.network import (
    create_gene_network,
    get_network_target_set_clustering,
    get_network_target_set_linkage,
)
from metworkpy.network.cluster import get_mcl_clustering


class TestGroupNetworkCluster(unittest.TestCase):
    model = get_example_model()
    network: nx.Graph | None = None

    @classmethod
    def setUpClass(cls):
        cls.network = create_gene_network(
            model=cls.model,
            directed=False,
            nodes_to_remove=["biomass"],
            essential=False,
        )

    def test_2_groups_mean_linkage(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        cluster_res = get_network_target_set_clustering(
            network=self.network,
            target_sets=[{"g023", "g013"}, {"g004", "g010"}],
            n_clusters=None,
            linkage="mean",
        )
        # Check that the result has the expected form
        # Children should be 1x2
        self.assertTupleEqual(cluster_res.children.shape, (1, 2))
        # Clusters should be of length 1
        self.assertEqual(len(cluster_res.clusters), 1)
        # Clusters should have a single set with 0 and 1
        self.assertSetEqual(cluster_res.clusters[0], {0, 1})
        # Distances should be a 1-D array of length 1
        self.assertTupleEqual(cluster_res.distances.shape, (1,))

    def test_2_groups_min_linkage(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        cluster_res = get_network_target_set_clustering(
            network=self.network,
            target_sets=[{"g023", "g013"}, {"g004", "g010"}],
            n_clusters=None,
            linkage="min",
        )
        # Check that the result has the expected form
        # Children should be 1x2
        self.assertTupleEqual(cluster_res.children.shape, (1, 2))
        # Clusters should be of length 1
        self.assertEqual(len(cluster_res.clusters), 1)
        # Clusters should have a single set with 0 and 1
        self.assertSetEqual(cluster_res.clusters[0], {0, 1})
        # Distances should be a 1-D array of length 1
        self.assertTupleEqual(cluster_res.distances.shape, (1,))

    def test_2_groups_max_linkage(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        cluster_res = get_network_target_set_clustering(
            network=self.network,
            target_sets=[{"g023", "g013"}, {"g004", "g010"}],
            n_clusters=None,
            linkage="max",
        )
        # Check that the result has the expected form
        # Children should be 1x2
        self.assertTupleEqual(cluster_res.children.shape, (1, 2))
        # Clusters should be of length 1
        self.assertEqual(len(cluster_res.clusters), 1)
        # Clusters should have a single set with 0 and 1
        self.assertSetEqual(cluster_res.clusters[0], {0, 1})
        # Distances should be a 1-D array of length 1
        self.assertTupleEqual(cluster_res.distances.shape, (1,))

    def test_3_groups_mean_linkage(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        cluster_res = get_network_target_set_clustering(
            network=self.network,
            target_sets=[{"g023", "g013"}, {"g004", "g010"}, {"g002", "g001"}],
            n_clusters=None,
            linkage="mean",
        )
        # Based on the clusters, the first merge should be 1 and 2,
        # then 0 and 3 should be second
        self.assertTupleEqual(cluster_res.children.shape, (2, 2))
        # Clusters should still be of length 1, and include 0,1,2
        self.assertEqual(len(cluster_res.clusters), 1)
        self.assertSetEqual(cluster_res.clusters[0], {0, 1, 2})
        # Distances should be a 1-D array of length 2
        self.assertTupleEqual(cluster_res.distances.shape, (2,))
        # The first row of children should be 1,2 and the second should be 1,3
        np.testing.assert_array_equal(
            cluster_res.children, np.array([[1, 2], [0, 3]])
        )


class TestGroupNetworkLinkage(unittest.TestCase):
    model = get_example_model()
    network: nx.Graph | None = None

    @classmethod
    def setUpClass(cls):
        cls.network = create_gene_network(
            model=cls.model,
            directed=False,
            nodes_to_remove=["biomass"],
            essential=False,
        )

    def test_2_groups_mean_linkage(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        linkage_mat = get_network_target_set_linkage(
            network=self.network,
            target_sets=[{"g023", "g013"}, {"g001", "g008"}],
            linkage="mean",
        )
        self.assertTupleEqual(linkage_mat.shape, (1, 4))

    def test_3_groups_mean_linkage(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        linkage_mat = get_network_target_set_linkage(
            network=self.network,
            target_sets=[{"g023", "g013"}, {"g001", "g008"}, {"g005", "g006"}],
            linkage="mean",
        )
        self.assertTupleEqual(linkage_mat.shape, (2, 4))


class TestMCL(unittest.TestCase):
    def test_component_clustering(self):
        df = pd.DataFrame(
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 0],
            ],
            index=["A", "B", "C", "D", "E", "F"],
            columns=["A", "B", "C", "D", "E", "F"],
            dtype="float",
        )
        # Should split into A,B,C and D,E,F (since these
        # are the components)
        c1, c2 = get_mcl_clustering(
            df,
            max_iter=100,
        )
        if "A" in c1:
            self.assertSetEqual(c1, {"A", "B", "C"})
            self.assertSetEqual(c2, {"D", "E", "F"})
        else:
            self.assertSetEqual(c2, {"A", "B", "C"})
            self.assertSetEqual(c1, {"D", "E", "F"})

    def test_small_graph(self):
        G = nx.Graph()
        G.add_edges_from(
            [
                ("A", "B"),
                ("A", "C"),
                ("A", "D"),
                ("B", "C"),
                ("C", "D"),
                ("C", "E"),
                ("E", "F"),
                ("E", "G"),
                ("E", "H"),
                ("G", "H"),
                ("H", "I"),
                ("I", "J"),
                ("G", "I"),
                ("J", "K"),
                ("J", "L"),
                ("J", "M"),
                ("L", "O"),
                ("L", "M"),
                ("L", "N"),
                ("K", "M"),
                ("K", "N"),
                ("K", "O"),
                ("M", "N"),
                ("M", "O"),
            ]
        )
        c1 = {"A", "B", "C", "D"}
        c2 = {
            "E",
            "F",
            "G",
            "H",
            "I",
        }
        c3 = {"J", "K", "L", "M", "N", "O"}
        clusters = get_mcl_clustering(G, ascending_size=True)
        print(clusters)
        test_c1, test_c2, test_c3 = clusters
        self.assertSetEqual(c1, test_c1)
        self.assertSetEqual(c2, test_c2)
        self.assertSetEqual(c3, test_c3)

    def test_karate(self):
        G = nx.karate_club_graph()
        clusters = get_mcl_clustering(
            G, expansion=2, inflation=1.6, weight="weight"
        )
        self.assertTrue(
            len(clusters) == 2,
            f"Incorrect number of clusters, expected 2 found {len(clusters)}",
        )
        # Check that the two clusters are each
        # only a single club
        for cluster in clusters:
            clubs = defaultdict(float)
            for m in cluster:
                clubs[G.nodes[m]["club"]] += 1.0
            total = sum(clubs.values())
            prop = max(v / total for v in clubs.values())
            self.assertGreaterEqual(prop, 0.90)


if __name__ == "__main__":
    unittest.main()

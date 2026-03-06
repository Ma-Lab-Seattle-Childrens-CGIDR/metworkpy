"""
Test the cluster submodule
"""

# Standard Library Imports
import unittest
from typing import Optional

# External Imports
import networkx as nx
import numpy as np

# Local Imports
from metworkpy.network import (
    get_network_group_clustering,
    get_network_group_linkage,
)
from metworkpy.examples import get_example_model
from metworkpy.network import create_gene_network


class TestGroupNetworkCluster(unittest.TestCase):
    model = get_example_model()
    network: Optional[nx.Graph] = None

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
        cluster_res = get_network_group_clustering(
            network=self.network,
            groups=[{"g023", "g013"}, {"g004", "g010"}],
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
        cluster_res = get_network_group_clustering(
            network=self.network,
            groups=[{"g023", "g013"}, {"g004", "g010"}],
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
        cluster_res = get_network_group_clustering(
            network=self.network,
            groups=[{"g023", "g013"}, {"g004", "g010"}],
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
        cluster_res = get_network_group_clustering(
            network=self.network,
            groups=[{"g023", "g013"}, {"g004", "g010"}, {"g002", "g001"}],
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
    network: Optional[nx.Graph] = None

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
        linkage_mat = get_network_group_linkage(
            network=self.network,
            groups=[{"g023", "g013"}, {"g001", "g008"}],
            linkage="mean",
        )
        self.assertTupleEqual(linkage_mat.shape, (1, 4))

    def test_3_groups_mean_linkage(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        linkage_mat = get_network_group_linkage(
            network=self.network,
            groups=[{"g023", "g013"}, {"g001", "g008"}, {"g005", "g006"}],
            linkage="mean",
        )
        self.assertTupleEqual(linkage_mat.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()

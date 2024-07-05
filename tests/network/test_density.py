# Standard Library Imports
import unittest

# External Imports
import networkx as nx
import numpy as np
import pandas as pd

# Local Imports
from metworkpy.network.density import label_density, find_dense_clusters, _node_density


class TestLabelDensity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = nx.Graph()
        g.add_edges_from(
            [(0, 1), (0, 2), (2, 3), (3, 4), (3, 5), (2, 6), (5, 7), (0, 8), (1, 5)]
        )
        cls.test_graph = g
        cls.test_labels = {0: 2, 5: 3, 7: 2}

    def test_node_density(self):
        node_density_calc1 = _node_density(
            self.test_graph, labels=pd.Series(self.test_labels), node=4, radius=2
        )
        node_density_expected1 = 0.75
        self.assertTrue(np.isclose(node_density_calc1, node_density_expected1))
        node_density_calc2 = _node_density(
            self.test_graph, labels=pd.Series(self.test_labels), node=6, radius=1
        )
        node_density_expected2 = 0.0
        self.assertTrue(np.isclose(node_density_calc2, node_density_expected2))

    def test_label_density(self):
        label_density_calc = label_density(
            self.test_graph, labels=self.test_labels, radius=1
        )
        label_density_exp = pd.Series(
            {
                0: 0.5,
                1: (5 / 3),
                2: (2 / 4),
                3: (3 / 4),
                4: 0,
                5: (5 / 4),
                6: 0,
                7: (5 / 2),
                8: (2 / 2),
            }
        ).sort_index()
        self.assertTrue(np.isclose(label_density_exp, label_density_calc).all())


class TestFindDenseClusters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = nx.Graph()
        g.add_edges_from(
            [(0, 1), (0, 2), (2, 3), (3, 4), (3, 5), (2, 6), (5, 7), (0, 8), (1, 5)]
        )
        cls.test_graph = g
        cls.test_labels = {0: 2, 5: 3, 7: 2}

    def test_find_dense_clusters(self):
        res_df = find_dense_clusters(
            network=self.test_graph,
            labels=self.test_labels,
            radius=0,
            quantile_cutoff=3 / 9,
        )
        for i in [0, 5, 7]:
            self.assertTrue(i in res_df.index)
        self.assertFalse(2 in res_df.index)
        self.assertAlmostEqual(res_df.loc[0, "density"], 2)
        self.assertAlmostEqual(res_df.loc[5, "density"], 3)
        self.assertAlmostEqual(res_df.loc[7, "density"], 2)
        self.assertNotEqual(res_df.loc[5, "cluster"], res_df.loc[0, "cluster"])


if __name__ == "__main__":
    unittest.main()

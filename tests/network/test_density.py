# Standard Library Imports
import pathlib
import unittest

# External Imports
import cobra
import networkx as nx
import numpy as np
import pandas as pd

# Local Imports
from metworkpy import read_model
from metworkpy.network import create_metabolic_network, bipartite_project
from metworkpy.network.density import (
    label_density,
    find_dense_clusters,
    _node_density,
    gene_target_density,
)


class TestLabelDensity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = nx.Graph()
        g.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (2, 6),
                (5, 7),
                (0, 8),
                (1, 5),
            ]
        )
        cls.test_graph = g
        cls.test_labels = {0: 2, 5: 3, 7: 2}

    def test_node_density(self):
        node_density_calc1 = _node_density(
            self.test_graph,
            labels=pd.Series(self.test_labels),
            node=4,
            radius=2,
        )
        node_density_expected1 = 0.75
        self.assertTrue(np.isclose(node_density_calc1, node_density_expected1))
        node_density_calc2 = _node_density(
            self.test_graph,
            labels=pd.Series(self.test_labels),
            node=6,
            radius=1,
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
        self.assertTrue(
            np.isclose(label_density_exp, label_density_calc).all()
        )


class TestFindDenseClusters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = nx.Graph()
        g.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (2, 6),
                (5, 7),
                (0, 8),
                (1, 5),
            ]
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


class TestGeneTargetDensity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cobra.Configuration().solver = "glpk"
        cls.data_path = (
            pathlib.Path(__file__).parent.parent.absolute() / "data"
        )
        cls.model = read_model(cls.data_path / "test_model.xml")
        metabolic_network = create_metabolic_network(
            model=cls.model,
            weighted=False,
            directed=True,
        )
        cls.reaction_network = bipartite_project(
            metabolic_network,
            node_set=cls.model.reactions.list_attr("id"),
        )

    def test_gene_target_density_r0(self):
        test_net = self.reaction_network
        test_model = self.model
        # Test with single gene label
        gene_targets = ["g_A_B_D_E"]
        # Perform the gene target density
        test_density = gene_target_density(
            metabolic_network=test_net,
            metabolic_model=test_model,
            gene_labels=gene_targets,
            radius=0,
        )
        # Since there is only one targeted gene, with a radius of 0,
        # every reaction but r_A_B_D_E should have a density of 0
        # and r_A_B_D_E should have a density of 1.0
        for rxn, density in test_density.items():
            if rxn == "r_A_B_D_E":
                self.assertAlmostEqual(density, 1.0, delta=1e-7)
            else:
                self.assertAlmostEqual(density, 0.0, delta=1e-7)

    def test_gene_target_density_r1(self):
        test_net = self.reaction_network
        test_model = self.model
        # Test with single gene label
        gene_targets = ["g_A_B_D_E"]
        # Perform the gene target density
        test_density = gene_target_density(
            metabolic_network=test_net,
            metabolic_model=test_model,
            gene_labels=gene_targets,
            radius=1,
        )
        # Since there is only one targeted gene, with a radius of 0,
        # every reaction but r_A_B_D_E should have a density of 0
        # and r_A_B_D_E should have a density of 1.0
        for rxn, density in test_density.items():
            if rxn == "r_A_B_D_E" or rxn == "r_C_E_F":
                self.assertAlmostEqual(density, 0.2, delta=1e-7)
            elif rxn == "R_A_imp" or rxn == "R_B_imp":
                self.assertAlmostEqual(density, 0.5, delta=1e-7)
            elif rxn == "r_D_G":
                self.assertAlmostEqual(density, 1.0 / 3.0, delta=1e-7)
            else:
                self.assertAlmostEqual(density, 0.0, delta=1e-7)


if __name__ == "__main__":
    unittest.main()

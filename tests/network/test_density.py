# Standard Library Imports
import pathlib
import unittest

# External Imports
import cobra
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

# Local Imports
from metworkpy import read_model
from metworkpy.network import create_metabolic_network, bipartite_project
from metworkpy.network.density import (
    label_density,
    find_dense_clusters,
    gene_target_density,
    gene_target_enrichment,
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

    def test_label_density(self):
        label_density_calc = label_density(
            self.test_graph, labels=self.test_labels, radius=1, processes=1
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
        np.testing.assert_allclose(label_density_calc, label_density_exp)


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
            processes=1,
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
            processes=1,
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


class TestGeneTargetEnrichment(unittest.TestCase):
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

    def test_gene_target_enrichment_pvalue_r0(self):
        test_net = self.reaction_network
        test_model = self.model
        # Test with single gene label
        gene_targets = ["g_A_B_D_E"]
        # Perform the gene target enrichment
        test_enrichment = gene_target_enrichment(
            metabolic_network=test_net,
            metabolic_model=test_model,
            gene_targets=gene_targets,
            metric="p-value",
            alternative="greater",
            radius=0,
            processes=1,
        )
        # Since there is only one targeted gene, with a radius of 0,
        # every reaction but r_A_B_D_E should have an enrichment of ~1
        # and r_A_B_D_E should have a density of ~0.0
        for rxn, enrichment in test_enrichment.items():
            if rxn == "r_A_B_D_E":
                # Expected p-value just hand calculated
                self.assertAlmostEqual(enrichment, 0.1, delta=1e-7)
            else:
                self.assertAlmostEqual(enrichment, 1.0, delta=1e-7)

    def test_gene_target_enrichment_pvalue_r1(self):
        test_net = self.reaction_network
        test_model = self.model
        # Test with single gene label
        gene_targets = ["g_A_B_D_E"]
        # Perform the gene target enrichment
        test_enrichment = gene_target_enrichment(
            metabolic_network=test_net,
            metabolic_model=test_model,
            gene_targets=gene_targets,
            metric="p-value",
            alternative="greater",
            radius=1,
            processes=1,
        )
        R_A_imp_cont = np.array([[1, 1], [0, 8]])
        R_A_imp_pval = stats.fisher_exact(
            R_A_imp_cont, alternative="greater"
        ).pvalue
        r_A_B_D_E_cont = np.array([[1, 4], [0, 5]])
        r_A_B_D_E_pval = stats.fisher_exact(
            r_A_B_D_E_cont, alternative="greater"
        ).pvalue

        r_D_G_cont = np.array([[1, 2], [0, 7]])
        r_D_G_pval = stats.fisher_exact(
            r_D_G_cont, alternative="greater"
        ).pvalue

        # Since there is only one targeted gene, with a radius of 0,
        # every reaction but r_A_B_D_E should have an enrichment of ~1
        # and r_A_B_D_E should have a density of ~0.0
        for rxn, enrichment in test_enrichment.items():
            if rxn in {"R_A_imp", "R_B_imp"}:  # Both have same cont table
                # Expected p-value just hand calculated
                self.assertAlmostEqual(enrichment, R_A_imp_pval, delta=1e-7)
            elif rxn in {
                "r_A_B_D_E",
                "r_C_E_F",
            }:  # r_C_E_F has same cont table
                self.assertAlmostEqual(enrichment, r_A_B_D_E_pval, delta=1e-7)
            elif rxn in {"r_D_G"}:
                self.assertAlmostEqual(enrichment, r_D_G_pval, delta=1e-7)
            else:
                self.assertAlmostEqual(enrichment, 1.0, delta=1e-7)

    def test_gene_target_enrichment_odds_r1(self):
        test_net = self.reaction_network
        test_model = self.model
        # Test with single gene label
        gene_targets = ["g_A_B_D_E", "g_C_H"]
        # Perform the gene target enrichment
        test_enrichment = gene_target_enrichment(
            metabolic_network=test_net,
            metabolic_model=test_model,
            gene_targets=gene_targets,
            metric="odds-ratio",
            alternative="greater",
            radius=1,
            processes=1,
        )
        self.assertAlmostEqual(
            test_enrichment["R_A_e_ex"],
            stats.fisher_exact(np.array([[0, 1], [2, 7]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_B_e_ex"],
            stats.fisher_exact(np.array([[0, 1], [2, 7]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_C_e_ex"],
            stats.fisher_exact(np.array([[0, 1], [2, 7]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_F_e_ex"],
            stats.fisher_exact(np.array([[0, 1], [2, 7]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_G_e_ex"],
            stats.fisher_exact(np.array([[0, 1], [2, 7]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_H_e_ex"],
            stats.fisher_exact(np.array([[0, 1], [2, 7]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_A_imp"],
            stats.fisher_exact(np.array([[1, 1], [1, 7]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_B_imp"],
            stats.fisher_exact(np.array([[1, 1], [1, 7]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_C_imp"],
            stats.fisher_exact(np.array([[1, 2], [1, 6]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_F_exp"],
            stats.fisher_exact(np.array([[0, 2], [2, 6]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_G_exp"],
            stats.fisher_exact(np.array([[0, 2], [2, 6]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["R_H_exp"],
            stats.fisher_exact(np.array([[1, 1], [1, 7]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["r_A_B_D_E"],
            stats.fisher_exact(np.array([[1, 4], [1, 4]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["r_C_E_F"],
            stats.fisher_exact(np.array([[2, 3], [0, 5]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["r_C_H"],
            stats.fisher_exact(np.array([[1, 3], [1, 5]])).statistic,
            delta=1e-7,
        )
        self.assertAlmostEqual(
            test_enrichment["r_D_G"],
            stats.fisher_exact(np.array([[1, 2], [1, 6]])).statistic,
            delta=1e-7,
        )


if __name__ == "__main__":
    unittest.main()

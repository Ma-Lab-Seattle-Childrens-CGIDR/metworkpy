# Standard Library Imports
from metworkpy import fuzzy_reaction_set, fuzzy_reaction_intersection
import unittest
from typing import Optional

# External Imports
import networkx as nx
import pandas as pd

# Local Imports
from metworkpy.examples import get_example_model
from metworkpy.network import create_reaction_network


class TestFuzzyReactionSet(unittest.TestCase):
    model = get_example_model()
    network: Optional[nx.Graph] = None

    @classmethod
    def setUpClass(cls):
        cls.model = get_example_model()
        # Convert model into a reaction network
        cls.network = create_reaction_network(
            model=cls.model,
            weighted=False,
            directed=False,
            nodes_to_remove=["biomass"],
        )

    def test_simple_gene_density_r0(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        fuzzy_rxn_set_res = fuzzy_reaction_set(
            metabolic_network=self.network,
            metabolic_model=self.model,
            gene_set={"g010", "g018"},
            membership_fn="simple gene density",
            scale=None,
            essential=False,
            processes=1,
            radius=0,
        )
        # Check that all reactions in the network have a value in the model
        self.assertCountEqual(
            list(fuzzy_rxn_set_res.index), list(self.network.nodes)
        )
        # For the reactions directly associated with the genes,
        # they should have a density of 1, for all others
        # the density should be 0
        expected_density = pd.Series(0.0, index=pd.Index(self.network.nodes))
        expected_density["R_J__Q"] = 1.0
        expected_density["R_exp"] = 1.0
        pd.testing.assert_series_equal(
            fuzzy_rxn_set_res, expected_density, check_like=True
        )

    def test_simple_gene_density_r1(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        fuzzy_rxn_set_res = fuzzy_reaction_set(
            metabolic_network=self.network,
            metabolic_model=self.model,
            gene_set={"g010", "g018"},
            membership_fn="simple gene density",
            scale=None,
            essential=False,
            processes=1,
            radius=1,
        )
        expected_non_zero = {
            "R_exp",
            "R_E_ex",
            "R_O_P__R",
            "R_C_D__J",
            "R_J__Q",
            "R_P_Q__S",
        }
        for rxn, value in fuzzy_rxn_set_res.items():
            if rxn in expected_non_zero:
                self.assertGreater(
                    value, 0.1
                )  # All should be higher than this, R_P_Q__S should be 1/6
            else:
                self.assertEqual(value, 0.0)

    def test_simple_reaction_density(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        fuzzy_rxn_set_res = fuzzy_reaction_set(
            metabolic_network=self.network,
            metabolic_model=self.model,
            gene_set={"g010", "g018"},
            membership_fn="simple reaction density",
            scale=None,
            essential=False,
            processes=1,
            radius=1,
        )
        expected_non_zero = {
            "R_exp",
            "R_E_ex",
            "R_O_P__R",
            "R_C_D__J",
            "R_J__Q",
            "R_P_Q__S",
        }
        for rxn, value in fuzzy_rxn_set_res.items():
            if rxn in expected_non_zero:
                self.assertGreater(
                    value, 0.1
                )  # All should be higher than this, R_P_Q__S should be 1/6
            else:
                self.assertEqual(value, 0.0)

    def test_distance_weighted_gene_density(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        fuzzy_rxn_set_res = fuzzy_reaction_set(
            metabolic_network=self.network,
            metabolic_model=self.model,
            gene_set={"g010", "g018"},
            membership_fn="weighted gene density",
            scale=None,
            essential=False,
            processes=1,
            max_radius=2,
        )

        expected_non_zero = {
            "R_exp",
            "R_E_ex",
            "R_O_P__R",
            "R_C_D__J",
            "R_C_H__I",
            "R_J__Q",
            "R_P_Q__S",
            "R_K__O",
            "R_I__P",
            "C_import",
            "D_import",
            "R_S_T__V_X",
        }
        for rxn, value in fuzzy_rxn_set_res.items():
            if rxn in expected_non_zero:
                self.assertGreater(value, 0.01)
            else:
                self.assertEqual(value, 0.0)

    def test_distance_weighted_reaction_density(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        fuzzy_rxn_set_res = fuzzy_reaction_set(
            metabolic_network=self.network,
            metabolic_model=self.model,
            gene_set={"g010", "g018"},
            membership_fn="weighted reaction density",
            scale=None,
            essential=False,
            processes=1,
            max_radius=2,
        )

        expected_non_zero = {
            "R_exp",
            "R_E_ex",
            "R_O_P__R",
            "R_C_D__J",
            "R_C_H__I",
            "R_J__Q",
            "R_P_Q__S",
            "R_K__O",
            "R_I__P",
            "C_import",
            "D_import",
            "R_S_T__V_X",
        }
        for rxn, value in fuzzy_rxn_set_res.items():
            if rxn in expected_non_zero:
                self.assertGreater(value, 0.01)
            else:
                self.assertEqual(value, 0.0)

    def test_knn_gene_density(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        fuzzy_rxn_set_res = fuzzy_reaction_set(
            metabolic_network=self.network,
            metabolic_model=self.model,
            gene_set={"g010", "g018"},
            membership_fn="knn gene density",
            scale=None,
            essential=False,
            processes=1,
            max_radius=2,
            k_neighbors=1,
        )
        expected_non_zero = {
            "R_E_ex",
            "C_import",
            "D_import",
            "R_exp",
            "R_C_H__I",
            "R_C_D__J",
            "R_I__P",
            "R_J__Q",
            "R_K__O",
            "R_O_P__R",
            "R_P_Q__S",
            "R_S_T__V_X",
        }

        for rxn, value in fuzzy_rxn_set_res.items():
            if rxn in expected_non_zero:
                self.assertGreater(value, 0.01)
            else:
                self.assertEqual(value, 0.0)

    def test_gene_enrichment_r1(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        fuzzy_rxn_set_res = fuzzy_reaction_set(
            metabolic_network=self.network,
            metabolic_model=self.model,
            gene_set={"g010", "g018"},
            membership_fn="gene enrichment",
            scale=None,
            essential=False,
            processes=1,
            radius=1,
        )
        expected_non_zero = {
            "R_exp",
            "R_E_ex",
            "R_O_P__R",
            "R_C_D__J",
            "R_J__Q",
            "R_P_Q__S",
        }
        for rxn, value in fuzzy_rxn_set_res.items():
            print(f"Reaction: {rxn}, value: {value}")
            if rxn in expected_non_zero:
                self.assertGreater(
                    value, 0.1
                )  # All should be higher than this, R_P_Q__S should be 1/6
            else:
                self.assertEqual(value, 0.0)


class TestFuzzyIntersection(unittest.TestCase):
    model = get_example_model()
    network: Optional[nx.Graph] = None

    @classmethod
    def setUpClass(cls):
        cls.model = get_example_model()
        # Convert model into a reaction network
        cls.network = create_reaction_network(
            model=cls.model,
            weighted=False,
            directed=False,
            nodes_to_remove=None,
        )

    def test_fuzzy_intersection(self):
        if self.network is None:
            raise ValueError("Test requires a metabolic network")
        fuzzy_intersection_set = fuzzy_reaction_intersection(
            gene_sets=[{"g018"}, {"g016"}],
            metabolic_network=self.network,
            metabolic_model=self.model,
            intersection_fn="min",
            radius=1,
        )
        # The only reaction which should be non-zero is R_O_P__R
        for rxn, value in fuzzy_intersection_set.items():
            if rxn == "R_O_P__R":
                self.assertGreater(value, 0.1)
            else:
                self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()

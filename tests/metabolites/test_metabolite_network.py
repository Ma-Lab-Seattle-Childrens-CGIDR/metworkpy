# Imports
# Standard Library Imports
import pathlib
import unittest

import cobra
import numpy as np

# External Imports
from cobra.core.configuration import Configuration

# Local Imports
from metworkpy.metabolites.metabolite_network import (
    find_metabolite_synthesis_network_reactions,
    find_metabolite_synthesis_network_genes,
    find_metabolite_consuming_network_reactions,
    find_metabolite_consuming_network_genes,
    add_metabolite_objective_,
    add_metabolite_absorb_reaction_,
    eliminate_maintenance_requirements_,
)
from metworkpy.utils.models import model_eq, read_model


def setup(cls):
    Configuration().solver = "glpk"  # Use GLPK solver for testing
    cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
    cls.model = read_model(cls.data_path / "test_model.xml")


class TestMetaboliteObjective(unittest.TestCase):
    model: cobra.Model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_adding_metabolite_objective(self):
        original_model = self.model.copy()
        with self.model as m:
            add_metabolite_objective_(m, "F_c")
            self.assertEqual(
                str(self.model.objective.expression),
                "1.0*F_c_sink_objective_cd67707b - 1.0*F_c_sink_objective_cd67707b_reverse_e263c",
            )
            self.assertAlmostEqual(m.slim_optimize(), 50)
        self.assertTrue(model_eq(self.model, original_model))


class TestFindMetaboliteNetwork(unittest.TestCase):
    model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_find_metabolite_network_reactions(self):
        # Test Essential Method
        original_model = self.model.copy()
        ess_met_net = find_metabolite_synthesis_network_reactions(
            model=self.model, method="essential", essential_proportion=0.05
        )
        self.assertTrue(model_eq(self.model, original_model))
        ess_f = ess_met_net["F_c"]
        self.assertEqual(ess_f.dtype, "bool")
        for rxn in [
            "R_A_e_ex",
            "R_B_e_ex",
            "R_C_e_ex",
            "R_G_e_ex",
            "R_A_imp",
            "R_B_imp",
            "R_C_imp",
            "R_G_exp",
            "r_A_B_D_E",
            "r_C_E_F",
            "r_D_G",
        ]:
            self.assertTrue(ess_f[rxn])
        for rxn in ["R_F_e_ex", "R_H_e_ex", "R_F_exp", "r_C_H", "R_H_exp"]:
            self.assertFalse(ess_f[rxn])

    def test_find_metabolite_network_reactions_pfba(self):
        original_model = self.model.copy()
        pfba_met_net = find_metabolite_synthesis_network_reactions(
            model=self.model, method="pfba", pfba_proportion=1.0
        )
        self.assertTrue(model_eq(self.model, original_model))
        pfba_f = pfba_met_net["F_c"]
        self.assertEqual(pfba_f.dtype, "float")

        for rxn in ["R_A_e_ex", "R_B_e_ex", "R_C_e_ex"]:
            self.assertAlmostEqual(pfba_f[rxn], -50)
        for rxn in [
            "R_G_e_ex",
            "R_A_imp",
            "R_B_imp",
            "R_C_imp",
            "R_G_exp",
            "r_A_B_D_E",
            "r_C_E_F",
            "r_D_G",
        ]:
            self.assertAlmostEqual(pfba_f[rxn], 50)
        for rxn in ["R_F_e_ex", "R_H_e_ex", "R_F_exp", "r_C_H", "R_H_exp"]:
            self.assertAlmostEqual(pfba_f[rxn], 0.0)

    def test_find_metabolite_network_genes_essential(self):
        # Test essential method
        ess_met_net = find_metabolite_synthesis_network_genes(
            model=self.model, method="essential", essential_proportion=0.05
        )
        ess_f = ess_met_net["F_c"]
        for gene in [
            "g_A_imp",
            "g_B_imp",
            "g_C_imp",
            "g_G_exp",
            "g_A_B_D_E",
            "g_C_E_F",
            "g_D_G",
        ]:
            self.assertTrue(ess_f[gene])
        for rxn in ["g_C_H"]:
            self.assertFalse(ess_f[rxn])

    def test_find_metabolite_network_genes_pfba(self):
        # Test pfba method
        pfba_met_net = find_metabolite_synthesis_network_genes(
            model=self.model, method="pfba", pfba_proportion=1.0
        )
        pfba_f = pfba_met_net["F_c"]
        for gene in [
            "g_A_imp",
            "g_B_imp",
            "g_C_imp",
            "g_G_exp",
            "g_A_B_D_E",
            "g_C_E_F",
            "g_D_G",
        ]:
            self.assertAlmostEqual(pfba_f[gene], 50)
        for gene in ["g_C_H"]:
            self.assertAlmostEqual(pfba_f[gene], 0)


class TestFindMetaboliteConsumingNetwork(unittest.TestCase):
    model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_find_metabolite_consuming_network_reactions(self):
        original_model = self.model.copy()
        consuming_network = find_metabolite_consuming_network_reactions(
            model=self.model, reaction_proportion=0.05, progress_bar=False
        )
        self.assertTrue(model_eq(self.model, original_model))
        b_network = consuming_network["B_c"]
        expected_network_rxns = [
            "r_A_B_D_E",
            "r_C_E_F",
            "r_D_G",
            "R_G_exp",
            "R_F_exp",
            "R_G_e_ex",
            "R_F_e_ex",
            "R_A_e_ex",  # Reverse uses the metabolite
            "R_A_imp",  # Reverse uses the metabolite
        ]
        actual_network_rxns = list(b_network[b_network].index)
        self.assertCountEqual(expected_network_rxns, actual_network_rxns)

    def test_find_metabolite_consuming_network_reactions_ignore_reverse(self):
        original_model = self.model.copy()
        consuming_network = find_metabolite_consuming_network_reactions(
            model=self.model,
            reaction_proportion=0.05,
            check_reverse=False,
            progress_bar=False,
        )
        self.assertTrue(model_eq(self.model, original_model))
        b_network = consuming_network["B_c"]
        expected_network_rxns = [
            "r_A_B_D_E",
            "r_C_E_F",
            "r_D_G",
            "R_G_exp",
            "R_F_exp",
            "R_G_e_ex",
            "R_F_e_ex",
            "R_A_imp",  # Can't run because there is nowhere for the internal A to go...
        ]
        actual_network_rxns = list(b_network[b_network].index)
        self.assertCountEqual(expected_network_rxns, actual_network_rxns)

    def test_find_metabolite_consuming_network_genes(self):
        original_model = self.model.copy()
        consuming_network = find_metabolite_consuming_network_genes(
            model=self.model, reaction_proportion=0.05, progress_bar=False
        )
        self.assertTrue(model_eq(self.model, original_model))
        b_network = consuming_network["B_c"]
        expected_network_rxns = [
            "g_A_imp",
            "g_A_B_D_E",
            "g_C_E_F",
            "g_D_G",
            "g_G_exp",
            "g_F_exp",
        ]
        actual_network_rxns = list(b_network[b_network].index)
        self.assertCountEqual(expected_network_rxns, actual_network_rxns)

    def test_find_metabolite_consuming_network_genes_ignore_reverse(self):
        original_model = self.model.copy()
        consuming_network = find_metabolite_consuming_network_genes(
            model=self.model, reaction_proportion=0.05, progress_bar=False
        )
        self.assertTrue(model_eq(self.model, original_model))
        b_network = consuming_network["B_c"]
        expected_network_rxns = [
            "g_A_imp",  # Still included as it is unable to run if B is absorbed due to equilibrium assumption
            "g_A_B_D_E",
            "g_C_E_F",
            "g_D_G",
            "g_G_exp",
            "g_F_exp",
        ]
        actual_network_rxns = list(b_network[b_network].index)
        self.assertCountEqual(expected_network_rxns, actual_network_rxns)


class TestHelperFunctions(unittest.TestCase):
    model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        Configuration().solver = "glpk"  # Use GLPK solver for testing
        cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
        cls.model = read_model(cls.data_path / "textbook_model.xml")

    def test_add_absorbing_reaction(self):
        test_model = self.model.copy()
        for metabolite in test_model.metabolites:
            with test_model as m:
                max_generation = m.slim_optimize()
                if np.isclose(max_generation, 0.0) or np.isnan(max_generation):
                    pass
                else:
                    add_metabolite_objective_(m, metabolite.id)
                    add_metabolite_absorb_reaction_(m, metabolite.id)
                    new_max_gen = m.slim_optimize()
                    if np.isnan(new_max_gen):
                        # If the metabolite is required to generate ATP,
                        # absorbing it will cause an infeasible problem, in this test just skip these
                        continue
                    self.assertAlmostEqual(m.slim_optimize(), 0.0, delta=m.tolerance)
            # Test that the remove worked as expected
            self.assertAlmostEqual(
                test_model.slim_optimize(), max_generation, delta=m.tolerance
            )

    def test_add_absorbing_reaction_no_maintenance(self):
        test_model = self.model.copy()
        initial_max_objective = test_model.slim_optimize()
        for metabolite in test_model.metabolites:
            with test_model as m:
                max_generation = m.slim_optimize()
                add_metabolite_objective_(m, metabolite.id)
                eliminate_maintenance_requirements_(m)
                add_metabolite_absorb_reaction_(m, metabolite.id)
                new_max_gen = m.slim_optimize()
                if np.isnan(new_max_gen):
                    # If the metabolite is required to generate ATP,
                    # absorbing it will cause an infeasible problem, in this test just skip these
                    continue
                self.assertAlmostEqual(m.slim_optimize(), 0.0, delta=m.tolerance)
            # Test that the remove worked as expected
            self.assertAlmostEqual(
                test_model.slim_optimize(), max_generation, delta=m.tolerance
            )
        self.assertAlmostEqual(
            test_model.slim_optimize(),
            initial_max_objective,
            delta=test_model.tolerance,
        )


if __name__ == "__main__":
    unittest.main()

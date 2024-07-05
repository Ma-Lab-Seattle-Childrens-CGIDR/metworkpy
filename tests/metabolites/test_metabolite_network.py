# Imports
# Standard Library Imports
import pathlib
import unittest

import cobra

# External Imports
from cobra.core.configuration import Configuration

# Local Imports
from metworkpy.metabolites.metabolite_network import (
    find_metabolite_network_reactions,
    find_metabolite_network_genes,
    MetaboliteObjective,
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
        with MetaboliteObjective(model=self.model, metabolite="F_c") as m:
            self.assertEqual(
                str(self.model.objective.expression),
                "1.0*tmp_F_c_sink_rxn - 1.0*tmp_F_c_sink_rxn_reverse_132c6",
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
        ess_met_net = find_metabolite_network_reactions(
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
        pfba_met_net = find_metabolite_network_reactions(
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
        ess_met_net = find_metabolite_network_genes(
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
        pfba_met_net = find_metabolite_network_genes(
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


if __name__ == "__main__":
    unittest.main()

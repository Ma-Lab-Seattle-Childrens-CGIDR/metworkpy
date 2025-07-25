# Imports
# Standard Library Imports
import pathlib
import unittest

import numpy as np
import pandas as pd

# External Imports
from cobra.core.configuration import Configuration

from metworkpy.metabolites.metchange_functions import (
    metchange,
    add_metchange_objective_constraint_,
)

# Local Imports
from metworkpy.utils import read_model, model_eq


def setup(cls):
    Configuration().solver = "glpk"  # Use GLPK solver for testing
    cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
    cls.model = read_model(cls.data_path / "test_model.xml")


class TestMetchangeObjectiveConstraint(unittest.TestCase):
    model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_zero_inconsistency(self):
        test_model = self.model.copy()
        weights = pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            index=[
                "R_A_imp",
                "R_B_imp",
                "R_C_imp",
                "R_F_exp",
                "R_G_exp",
                "R_H_exp",
                "r_A_B_D_E",
                "r_C_E_F",
                "r_C_H",
                "r_D_G",
            ],
        )
        with test_model as m:
            sink_rxn_id = add_metchange_objective_constraint_(
                model=m,
                metabolite="F_c",
                reaction_weights=weights,
                objective_tolerance=0.05,
            )
            # Know that the inconsistency score will be 0
            self.assertAlmostEqual(m.slim_optimize(), 0.0)
            self.assertEqual(m.objective_direction, "min")
            rxn = m.reactions.get_by_id(sink_rxn_id)
            self.assertAlmostEqual(rxn.lower_bound, 0.95 * 50)
        # Make sure model was reverted
        self.assertTrue(model_eq(test_model, self.model))

    def test_forced_inconsistency(self):
        test_model = self.model.copy()
        weights = pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            index=[
                "R_A_imp",
                "R_B_imp",
                "R_C_imp",
                "R_F_exp",
                "R_G_exp",
                "R_H_exp",
                "r_A_B_D_E",
                "r_C_E_F",
                "r_C_H",
                "r_D_G",
            ],
        )
        with test_model as m:
            add_metchange_objective_constraint_(
                model=m,
                metabolite="F_c",
                reaction_weights=weights,
                objective_tolerance=0.0,
            )
            self.assertAlmostEqual(m.slim_optimize(), 50.0)
            self.assertEqual(m.objective_direction, "min")
        self.assertTrue(model_eq(test_model, self.model))


class TestMetchange(unittest.TestCase):
    model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_zero_weights(self):
        test_model = self.model.copy()
        weights = pd.Series()
        with self.assertRaisesRegex(
            ValueError, "Reaction weights is empty, must have at least one weight."
        ):
            _ = metchange(model=test_model, reaction_weights=weights, metabolites=None)
        # Test that it doesn't change the model
        self.assertTrue(model_eq(test_model, self.model))

    def test_subset_metabolites(self):
        test_model = self.model.copy()
        weights = pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            index=[
                "R_A_imp",
                "R_B_imp",
                "R_C_imp",
                "R_F_exp",
                "R_G_exp",
                "R_H_exp",
                "r_A_B_D_E",
                "r_C_E_F",
                "r_C_H",
                "r_D_G",
            ],
        )
        metchange_res = metchange(
            model=test_model,
            reaction_weights=weights,
            metabolites=["F_c"],
            objective_tolerance=0.0,
        )
        self.assertTrue(model_eq(test_model, self.model))
        self.assertTrue(np.isclose(metchange_res["F_c"], 50))

    def test_forced_inconsistency(self):
        test_model = self.model.copy()
        weights = pd.Series(
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            index=[
                "R_A_imp",
                "R_B_imp",
                "R_C_imp",
                "R_F_exp",
                "R_G_exp",
                "R_H_exp",
                "r_A_B_D_E",
                "r_C_E_F",
                "r_C_H",
                "r_D_G",
            ],
        )
        metchange_res = metchange(
            model=test_model,
            reaction_weights=weights,
            metabolites=None,
            objective_tolerance=0.0,
        )
        self.assertTrue(np.isclose(metchange_res["C_c"], 0.0))
        self.assertTrue(np.isclose(metchange_res["B_c"], 0.0))
        self.assertTrue(np.isclose(metchange_res["A_c"], 25.0))
        self.assertTrue(np.isclose(metchange_res["F_c"], 75.0))


if __name__ == "__main__":
    unittest.main()

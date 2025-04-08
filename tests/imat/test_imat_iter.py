# Standard Library Imports
import pathlib
import unittest

# External Imports
import cobra
from cobra.core.configuration import Configuration
import numpy as np
import pandas as pd

# Local Imports
import metworkpy.imat.imat_iter as imat_iter
from metworkpy import read_model, model_bounds_eq, gene_to_rxn_weights
from metworkpy.gpr.gpr_functions import IMAT_FUNC_DICT


# Helper setup class for all tests below
def setup(cls):
    # Taking advantage of highs/osqp combination for testing that is
    # faster than glpk, and still not proprietary
    Configuration().solver = "hybrid"
    # Set a path to the data folder
    cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
    # Read in the textbook model (ecoli core)
    cls.model = read_model(cls.data_path / "textbook_model.xml")
    # Set various default parameters
    cls.epsilon = 1.0
    cls.threshold = 1e-2
    cls.max_iter = 20
    cls.objective_tolerance = 0.3
    # Randomly select genes to assign to different weights
    gene_list = cls.model.genes.list_attr("id")
    gene_weights = pd.Series(0.0, index=pd.Index(gene_list))
    rng = np.random.default_rng(seed=42)
    shuffled_genes = rng.permutation(gene_list)
    num_genes_to_select = len(gene_list) // 10
    low_expr_genes = shuffled_genes[:num_genes_to_select]
    high_expr_genes = shuffled_genes[-num_genes_to_select:]
    gene_weights[low_expr_genes] = -1
    gene_weights[high_expr_genes] = 1
    # Now, convert these to reaction weights
    cls.rxn_weights = gene_to_rxn_weights(
        cls.model, gene_weights=gene_weights, fn_dict=IMAT_FUNC_DICT, fill_val=0.0
    )


class TestImatIterBinaryVariables(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None
    max_iter = None
    objective_tolerance = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_iteration(self):
        # Mainly a catch-fire test, just ensure iteration is actually occurring
        counter = 0
        test_iter = imat_iter.ImatIterBinaryVariables(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Iterate through, should at least iterate once
        for _, _, _ in test_iter:
            counter += 1
        self.assertGreater(counter, 5)

    def test_max_iter(self):
        counter = 0
        test_iter = imat_iter.ImatIterBinaryVariables(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=3,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Iterate through, should at least iterate once
        for _, _, _ in test_iter:
            counter += 1
        self.assertEqual(counter, 3)

    def test_different_solutions(self):
        # Test that the solutions returned are different
        test_iter = imat_iter.ImatIterBinaryVariables(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        rh_y_pos_list = []
        rh_y_neg_list = []
        rl_y_pos_list = []
        counter = 0
        for rh_y_pos, rh_y_neg, rl_y_pos in test_iter:
            # Check that the newest solutions are different to all previous solutions
            for rh_y_pos_test, rh_y_neg_test, rl_y_pos_test in zip(
                rh_y_pos_list, rh_y_neg_list, rl_y_pos_list
            ):
                rh_y_pos_diff = not np.isclose(rh_y_pos, rh_y_pos_test).all()
                rh_y_neg_diff = not np.isclose(rh_y_neg, rh_y_neg_test).all()
                rl_y_pos_diff = not np.isclose(rl_y_pos, rl_y_pos_test).all()
                self.assertTrue(any([rh_y_pos_diff, rh_y_neg_diff, rl_y_pos_diff]))
            # Add the newest different solution to the lists
            rh_y_pos_list.append(rh_y_pos)
            rh_y_neg_list.append(rh_y_neg)
            rl_y_pos_list.append(rl_y_pos)
            counter += 1
        # Make sure that it has actually checked multiple solutions
        self.assertGreater(counter, 3)

    def test_near_optimality(self):
        # Test that the solutions returned are different
        test_iter = imat_iter.ImatIterBinaryVariables(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Variable to hold best solution
        best_solution = None
        for rh_y_pos, rh_y_neg, rl_y_pos in test_iter:
            if best_solution is None:
                best_solution = rh_y_pos.sum() + rh_y_neg.sum() + rl_y_pos.sum()
                continue
            self.assertGreater(
                rh_y_pos.sum() + rh_y_neg.sum() + rl_y_pos.sum(),
                best_solution * (1 - self.objective_tolerance),
            )


class TestImatIterReactionActivity(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None
    max_iter = None
    objective_tolerance = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_iteration(self):
        # Mainly a catch-fire test, just ensure iteration is actually occurring
        counter = 0
        test_iter = imat_iter.ImatIterReactionActivities(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Iterate through, should at least iterate once
        for _ in test_iter:
            counter += 1
        self.assertGreater(counter, 5)

    def test_near_optimality(self):
        counter = 0
        test_iter = imat_iter.ImatIterReactionActivities(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Variable to hold optimal objective
        optimal_objective = None
        # Iterate through, should at least iterate once
        for rxn_activities in test_iter:
            if optimal_objective is None:
                optimal_objective = (
                    (rxn_activities == imat_iter.ReactionActivity.ActiveForward).sum()
                    + (rxn_activities == imat_iter.ReactionActivity.ActiveReverse).sum()
                    + (rxn_activities == imat_iter.ReactionActivity.Inactive).sum()
                )
                continue
            current_objective = (
                (rxn_activities == imat_iter.ReactionActivity.ActiveForward).sum()
                + (rxn_activities == imat_iter.ReactionActivity.ActiveReverse).sum()
                + (rxn_activities == imat_iter.ReactionActivity.Inactive).sum()
            )
            self.assertGreaterEqual(
                current_objective, (1 - self.objective_tolerance) * optimal_objective
            )
            counter += 1
        self.assertGreater(counter, 5)


class TestImatIterModels(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None
    max_iter = None
    objective_tolerance = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_iteration(self):
        # Mainly a catch-fire test, just ensure iteration is actually occurring
        counter = 0
        test_iter = imat_iter.ImatIterModels(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            method="simple",
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Iterate through, should at least iterate once
        for _ in test_iter:
            counter += 1
        self.assertGreater(counter, 5)

    def test_different_method(self):
        # Mainly a catch-fire test, just ensure iteration is actually occurring
        counter = 0
        test_iter = imat_iter.ImatIterModels(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            method="subset",
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Iterate through, should at least iterate once
        for _ in test_iter:
            counter += 1
        self.assertGreater(counter, 5)

    def test_model_changing(self):
        # Check that the returned models are actually different than the initial model
        counter = 0
        test_iter = imat_iter.ImatIterModels(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            method="simple",
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Get a copy of the base model for testing
        base_model = self.model.copy()
        # Iterate through, should at least iterate once
        for updated_model in test_iter:
            # Check that the model has been updated
            self.assertFalse(model_bounds_eq(base_model, updated_model))
            counter += 1  # This could be replaced with enumerated, but I don't like how state is leaked from for loops
        # Ensure that iterations have actually occurred
        self.assertGreater(counter, 5)


class TestImatIterMain(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None
    max_iter = None
    objective_tolerance = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_model_dispatch(self):
        counter = 0
        test_iter = imat_iter.ImatIterModels(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            method="simple",
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Iterate through, should at least iterate once
        for model in test_iter:
            self.assertIsInstance(model, cobra.Model)
            counter += 1
        self.assertGreater(counter, 5)

    def test_binary_variable_dispatch(self):
        counter = 0
        test_iter = imat_iter.ImatIterBinaryVariables(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Iterate through, should be returning named tuple of pandas series
        for ntuple in test_iter:
            self.assertIsInstance(ntuple, imat_iter.ImatBinaryVariables)
            self.assertIsInstance(ntuple.rh_y_pos, pd.Series)
            self.assertIsInstance(ntuple.rh_y_neg, pd.Series)
            self.assertIsInstance(ntuple.rl_y_pos, pd.Series)
            self.assertEqual(ntuple.rh_y_pos.dtypes, "float")
            self.assertEqual(ntuple.rh_y_neg.dtypes, "float")
            self.assertEqual(ntuple.rl_y_pos.dtypes, "float")
            counter += 1
        self.assertGreater(counter, 5)

    def test_reaction_activity_dispatch(self):
        counter = 0
        test_iter = imat_iter.ImatIterReactionActivities(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
        )
        # Iterate through, should be returning named tuple of pandas series
        for act_series in test_iter:
            self.assertIsInstance(act_series, pd.Series)
            self.assertEqual(act_series.dtype, "object")
            self.assertIsInstance(act_series.iloc[0], imat_iter.ReactionActivity)
            counter += 1
        self.assertGreater(counter, 5)


class TestImatIterSampling(unittest.TestCase):
    model = None
    data_path = None
    rxn_weights = None
    epsilon = None
    threshold = None
    max_iter = None
    objective_tolerance = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_imat_sampling(self):
        sample_res = imat_iter.imat_iter_flux_sample(
            model=self.model,
            rxn_weights=self.rxn_weights,
            model_generation_method="simple",
            max_iter=3,  # Reduced to save comp time
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            sampler=None,
            thinning=20,  # Again reducing to save time
            num_samples=100,
            sampler_kwargs={"processes": 1},
        )
        self.assertIsInstance(sample_res, pd.DataFrame)
        self.assertEqual(sample_res.shape[0], 3 * 100)


if __name__ == "__main__":
    unittest.main()

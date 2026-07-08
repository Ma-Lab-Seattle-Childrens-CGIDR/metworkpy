# Standard Library Imports
from metworkpy.imat import ImatIterReactionActivities
import pathlib
from typing import Literal
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
    rng = np.random.default_rng(seed=29384029384029)
    shuffled_genes = rng.permutation(gene_list)
    num_genes_to_select = len(gene_list) // 10
    low_expr_genes = shuffled_genes[:num_genes_to_select]
    high_expr_genes = shuffled_genes[-num_genes_to_select:]
    gene_weights[low_expr_genes] = -1
    gene_weights[high_expr_genes] = 1
    # Now, convert these to reaction weights
    cls.rxn_weights = gene_to_rxn_weights(
        cls.model,
        gene_weights=gene_weights,
        fn_dict=IMAT_FUNC_DICT,
        fill_val=0.0,
    )


class TestImatIterators(unittest.TestCase):
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

    def check_basic_iteration(self, test_iter_class, **kwargs):
        assert self.model is not None
        assert isinstance(self.epsilon, float)
        assert isinstance(self.rxn_weights, pd.Series)
        assert isinstance(self.max_iter, int)
        assert isinstance(self.threshold, float)
        assert isinstance(self.objective_tolerance, float)
        test_iter = test_iter_class(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            **kwargs,
        )
        # Iterate through, should at least iterate once
        counter = 0
        for _ in test_iter:
            counter += 1
        self.assertGreater(counter, 5)

    def check_max_iter(self, test_iter_class, **kwargs):
        assert self.model is not None
        assert isinstance(self.epsilon, float)
        assert isinstance(self.rxn_weights, pd.Series)
        assert isinstance(self.max_iter, int)
        assert isinstance(self.threshold, float)
        assert isinstance(self.objective_tolerance, float)
        test_iter = test_iter_class(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=3,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            **kwargs,
        )
        # Iterate through, should at least iterate once
        counter = 0
        for _ in test_iter:
            counter += 1
        self.assertEqual(counter, 3)

    def test_basic_iteration(self):
        for iter_class in [
            imat_iter.ImatIterBinaryVariables,
            ImatIterReactionActivities,
        ]:
            for iter_method in ["icut", "maxdist", "corner"]:
                self.check_basic_iteration(iter_class, iter_method=iter_method)
        for iter_method in ["icut", "maxdist", "corner"]:
            for model_method in ["simple", "subset"]:
                self.check_basic_iteration(
                    imat_iter.ImatIterModels,
                    iter_method=iter_method,
                    model_method=model_method,
                )

    def test_max_iter(self):
        for iter_class in [
            imat_iter.ImatIterBinaryVariables,
            ImatIterReactionActivities,
        ]:
            for iter_method in ["icut", "maxdist", "corner"]:
                self.check_max_iter(iter_class, iter_method=iter_method)
        for iter_method in ["icut", "maxdist", "corner"]:
            for model_method in ["simple", "subset"]:
                self.check_max_iter(
                    imat_iter.ImatIterModels,
                    iter_method=iter_method,
                    model_method=model_method,
                )

    def check_different_solutions_binary_var_iter(self, **kwargs):
        assert self.model is not None
        assert isinstance(self.epsilon, float)
        assert isinstance(self.rxn_weights, pd.Series)
        assert isinstance(self.max_iter, int)
        assert isinstance(self.threshold, float)
        assert isinstance(self.objective_tolerance, float)
        test_iter = imat_iter.ImatIterBinaryVariables(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            **kwargs,
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
                self.assertTrue(
                    any([rh_y_pos_diff, rh_y_neg_diff, rl_y_pos_diff])
                )
            # Add the newest different solution to the lists
            rh_y_pos_list.append(rh_y_pos)
            rh_y_neg_list.append(rh_y_neg)
            rl_y_pos_list.append(rl_y_pos)
            counter += 1
        # Make sure that it has actually checked multiple solutions
        self.assertGreater(counter, 3)

    def test_different_solutions_binary_var_iter(self):
        for iter_method in ["icut", "maxdist", "corner"]:
            self.check_different_solutions_binary_var_iter(
                iter_method=iter_method
            )

    def check_near_optimum_binary_var_iter(self, **kwargs):
        assert self.model is not None
        assert isinstance(self.epsilon, float)
        assert isinstance(self.rxn_weights, pd.Series)
        assert isinstance(self.max_iter, int)
        assert isinstance(self.threshold, float)
        assert isinstance(self.objective_tolerance, float)
        test_iter = imat_iter.ImatIterBinaryVariables(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            **kwargs,
        )
        # Variable to hold best solution
        best_solution = None
        for rh_y_pos, rh_y_neg, rl_y_pos in test_iter:
            if best_solution is None:
                best_solution = (
                    rh_y_pos.sum() + rh_y_neg.sum() + rl_y_pos.sum()
                )
                continue
            self.assertGreater(
                rh_y_pos.sum() + rh_y_neg.sum() + rl_y_pos.sum(),
                best_solution * (1 - self.objective_tolerance),
            )

    def test_near_optimality_binary_var_iter(self):
        for iter_method in ["icut", "maxdist", "corner"]:
            self.check_near_optimum_binary_var_iter(iter_method=iter_method)

    def check_model_changing_model_iter(self, **kwargs):
        # Check that the returned models are actually different than the initial model
        assert self.model is not None
        assert isinstance(self.epsilon, float)
        assert isinstance(self.rxn_weights, pd.Series)
        assert isinstance(self.max_iter, int)
        assert isinstance(self.threshold, float)
        assert isinstance(self.objective_tolerance, float)
        counter = 0
        test_iter = imat_iter.ImatIterModels(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            **kwargs,
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

    def test_model_changing_model_iter(self):
        for iter_method in ["icut", "maxdist", "corner"]:
            for model_method in ["simple", "subset"]:
                self.check_model_changing_model_iter(
                    iter_method=iter_method, model_method=model_method
                )


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

    def check_dispatch(
        self,
        output: Literal["model", "binary-variables", "reaction-activity"],
        expected_class,
        additional_checks=None,
        **kwargs,
    ):
        assert self.model is not None
        assert isinstance(self.epsilon, float)
        assert isinstance(self.rxn_weights, pd.Series)
        assert isinstance(self.max_iter, int)
        assert isinstance(self.threshold, float)
        assert isinstance(self.objective_tolerance, float)
        counter = 0
        test_iter = imat_iter.ImatIter(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            output=output,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            **kwargs,
        )
        for iter_out in test_iter:
            self.assertIsInstance(iter_out, expected_class)
            if additional_checks is not None:
                additional_checks(iter_out)
            counter += 1
        self.assertGreater(counter, 5)

    def test_dispatch(self):
        # For the binary variables, define additional checks
        # for the structure of the return
        def binary_var_return_validator(iter_return):
            self.assertIsInstance(iter_return, imat_iter.ImatBinaryVariables)
            self.assertIsInstance(iter_return.rh_y_pos, pd.Series)
            self.assertIsInstance(iter_return.rh_y_neg, pd.Series)
            self.assertIsInstance(iter_return.rl_y_pos, pd.Series)
            self.assertEqual(iter_return.rh_y_pos.dtypes, "float")
            self.assertEqual(iter_return.rh_y_neg.dtypes, "float")
            self.assertEqual(iter_return.rl_y_pos.dtypes, "float")

        def rxn_act_return_validator(iter_return):
            self.assertIsInstance(iter_return, pd.Series)
            self.assertEqual(iter_return.dtype, "object")
            self.assertIsInstance(
                iter_return.iloc[0], imat_iter.ReactionActivity
            )

        for iter_method in ["icut", "maxdist", "corner"]:
            self.check_dispatch(
                "binary-variables",
                tuple,
                additional_checks=binary_var_return_validator,
                iter_method=iter_method,
            )
            self.check_dispatch(
                "reaction-activity",
                pd.Series,
                additional_checks=rxn_act_return_validator,
                iter_method=iter_method,
            )
            for model_method in ["simple", "subset"]:
                self.check_dispatch(
                    "model",
                    cobra.Model,
                    iter_method=iter_method,
                    model_method=model_method,
                )


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
        assert self.model is not None
        assert isinstance(self.epsilon, float)
        assert isinstance(self.rxn_weights, pd.Series)
        assert isinstance(self.max_iter, int)
        assert isinstance(self.threshold, float)
        assert isinstance(self.objective_tolerance, float)
        sample_res = imat_iter.imat_iter_flux_sample(
            model=self.model,
            rxn_weights=self.rxn_weights,
            model_method="simple",
            max_iter=3,  # Reduced to save comp time
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            sampler=None,
            n_samples=100,
            sampler_kwargs={"processes": 1, "thinning": 20},
        )
        self.assertIsInstance(sample_res, pd.DataFrame)
        self.assertEqual(sample_res.shape[0], 3 * 100)


class TestImatIterEssentiality(unittest.TestCase):
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

    def check_iter_essentiality(self, **kwargs):
        assert self.model is not None
        assert isinstance(self.epsilon, float)
        assert isinstance(self.rxn_weights, pd.Series)
        assert isinstance(self.max_iter, int)
        assert isinstance(self.threshold, float)
        assert isinstance(self.objective_tolerance, float)
        ess_df = imat_iter.imat_iter_essential(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            processes=1,
            **kwargs,
        )
        self.assertIsInstance(ess_df, pd.DataFrame)
        # Should have a column for every gene
        self.assertEqual(ess_df.shape[1], len(self.model.genes))
        self.assertCountEqual(
            list(ess_df.columns), self.model.genes.list_attr("id")
        )
        # Should be able to iterate at least 5 times
        self.assertGreater(ess_df.shape[0], 5)
        # All the dtypes should be boolean
        # Should have some True, and some False
        for _, row in ess_df.iterrows():
            self.assertFalse(row.all())
            self.assertTrue(row.any())

    def test_iter_essentiality(self):
        for iter_method in ["icut", "maxdist", "corner"]:
            self.check_iter_essentiality(iter_method=iter_method)

    def check_consensus_essentiality(self, **kwargs):
        assert self.model is not None
        assert isinstance(self.epsilon, float)
        assert isinstance(self.rxn_weights, pd.Series)
        assert isinstance(self.max_iter, int)
        assert isinstance(self.threshold, float)
        assert isinstance(self.objective_tolerance, float)
        ess_series = imat_iter.consensus_essentiality(
            model=self.model.copy(),
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            iter_method="icut",
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            processes=1,
            **kwargs,
        )
        self.assertIsInstance(ess_series, pd.Series)
        # Should be indexed by gene
        self.assertCountEqual(
            list(ess_series.index), self.model.genes.list_attr("id")
        )
        # Should have some True and Some False
        self.assertFalse(ess_series.all())
        self.assertTrue(ess_series.any())

    def test_consensus_essentiality(self):
        for consensus_method in ["any", "all", 0.5]:
            self.check_consensus_essentiality(
                consensus_method=consensus_method
            )


if __name__ == "__main__":
    unittest.main()

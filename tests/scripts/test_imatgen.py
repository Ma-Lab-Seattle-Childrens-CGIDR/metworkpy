# Imports
# Standard Library Import
import argparse
import importlib.util
import os
import pathlib
import unittest
from unittest import mock, skipIf

# External Imports
import cobra
import numpy as np
import pandas as pd

# Local Imports
import metworkpy
from metworkpy.scripts import imatgen

BASE_PATH = pathlib.Path(__file__).parent.parent


class TestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure cobra to default to glpk
        cobra.core.configuration.Configuration().solver = "glpk"
        # Get path references, and make a temporary folder
        cls.data_path = BASE_PATH / "data"
        cls.tmp_path = BASE_PATH / "tmp_imatgen"
        os.mkdir(cls.tmp_path)
        # Get the gene expression data
        cls.gene_expression = pd.read_csv(cls.data_path / "test_model_gene_expression.csv", index_col=0, header=0)
        # Get the unchanged model
        cls.model = metworkpy.read_model(cls.data_path / "test_model.json")

    def setUp(self):
        self.test_model = self.model.copy()

    def tearDown(self):
        # Cleanup the tmp directory after each test (to not potentially pollute)
        for filename in os.listdir(self.tmp_path):
            p = self.tmp_path / filename
            os.remove(p)

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.tmp_path)

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(
                    model_file=BASE_PATH / "data" / "test_model.json",
                    gene_expression_file=BASE_PATH / "data" / "test_model_gene_expression.csv",
                    output_file=BASE_PATH / "tmp_imatgen" / "default_test_result_model.json",
                    method="subset",
                    epsilon=1.,
                    threshold=0.001,
                    objective_tolerance=5e-2,
                    model_format=None,
                    output_format="json",
                    samples=None,
                    aggregation_method="median",
                    transpose=False,
                    quantile="0.15",
                    subset=False,
                    verbose=False,
                    solver="glpk",
                    sep=",",
                    loopless=False,
                    processes=None,

                ))
    def test_default_run(self, mock_args):
        imatgen.run()
        # Test that it created the expected file
        self.assertTrue(os.path.exists(self.tmp_path / argparse.ArgumentParser.parse_args().output_file))
        # Test that the output model is the same that would be created by running the IMAT algorithm by hand
        out_model = metworkpy.read_model(self.tmp_path / argparse.ArgumentParser.parse_args().output_file)
        gene_weights = metworkpy.utils.expr_to_gene_weights(expression=self.gene_expression,
                                                            quantile=0.15,
                                                            aggregator=np.median,
                                                            subset=None,
                                                            sample_axis=0)
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(model=self.test_model,
                                                        gene_weights=gene_weights)
        expected_model = metworkpy.imat.generate_model(model=self.test_model,
                                                       rxn_weights=rxn_weights,
                                                       method="subset",
                                                       epsilon=1.,
                                                       threshold=0.001,
                                                       objective_tolerance=5e-2
                                                       )
        self.assertTrue(metworkpy.utils.model_eq(out_model, expected_model))

    @skipIf(
        importlib.util.find_spec("gurobipy") is None, "gurobi is not installed"
    )
    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(
                    model_file=BASE_PATH / "data" / "test_model.json",
                    gene_expression_file=BASE_PATH / "data" / "test_model_gene_expression.csv",
                    output_file=BASE_PATH / "tmp_imatgen" / "default_test_result_model.json",
                    method="subset",
                    epsilon=1.,
                    threshold=0.001,
                    objective_tolerance=5e-2,
                    model_format=None,
                    output_format="json",
                    samples=None,
                    aggregation_method="median",
                    transpose=False,
                    quantile="0.15",
                    subset=False,
                    verbose=False,
                    solver="gurobi",
                    sep=",",
                    loopless=False,
                    processes=None,

                ))
    def test_gurobi_run(self, mock_args):
        imatgen.run()
        # Test that it created the expected file
        self.assertTrue(os.path.exists(self.tmp_path / argparse.ArgumentParser.parse_args().output_file))
        # Test that the output model is the same that would be created by running the IMAT algorithm by hand
        out_model = metworkpy.read_model(self.tmp_path / argparse.ArgumentParser.parse_args().output_file)
        gene_weights = metworkpy.utils.expr_to_gene_weights(expression=self.gene_expression,
                                                            quantile=0.15,
                                                            aggregator=np.median,
                                                            subset=None,
                                                            sample_axis=0)
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(model=self.test_model,
                                                        gene_weights=gene_weights)
        expected_model = metworkpy.imat.generate_model(model=self.test_model,
                                                       rxn_weights=rxn_weights,
                                                       method="subset",
                                                       epsilon=1.,
                                                       threshold=0.001,
                                                       objective_tolerance=5e-2
                                                       )
        self.assertTrue(metworkpy.utils.model_eq(out_model, expected_model))

    @skipIf(
        importlib.util.find_spec("cplex") is None, "cplex is not installed"
    )
    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(
                    model_file=BASE_PATH / "data" / "test_model.json",
                    gene_expression_file=BASE_PATH / "data" / "test_model_gene_expression.csv",
                    output_file=BASE_PATH / "tmp_imatgen" / "default_test_result_model.json",
                    method="subset",
                    epsilon=1.,
                    threshold=0.001,
                    objective_tolerance=5e-2,
                    model_format=None,
                    output_format="json",
                    samples=None,
                    aggregation_method="median",
                    transpose=False,
                    quantile="0.15",
                    subset=False,
                    verbose=False,
                    solver="cplex",
                    sep=",",
                    loopless=False,
                    processes=None,

                ))
    def test_gurobi_run(self, mock_args):
        imatgen.run()
        # Test that it created the expected file
        self.assertTrue(os.path.exists(self.tmp_path / argparse.ArgumentParser.parse_args().output_file))
        # Test that the output model is the same that would be created by running the IMAT algorithm by hand
        out_model = metworkpy.read_model(self.tmp_path / argparse.ArgumentParser.parse_args().output_file)
        gene_weights = metworkpy.utils.expr_to_gene_weights(expression=self.gene_expression,
                                                            quantile=0.15,
                                                            aggregator=np.median,
                                                            subset=None,
                                                            sample_axis=0)
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(model=self.test_model,
                                                        gene_weights=gene_weights)
        expected_model = metworkpy.imat.generate_model(model=self.test_model,
                                                       rxn_weights=rxn_weights,
                                                       method="subset",
                                                       epsilon=1.,
                                                       threshold=0.001,
                                                       objective_tolerance=5e-2
                                                       )
        self.assertTrue(metworkpy.utils.model_eq(out_model, expected_model))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(
                    model_file=BASE_PATH / "data" / "test_model.json",
                    gene_expression_file=BASE_PATH / "data" / "test_model_gene_expression.csv",
                    output_file=BASE_PATH / "tmp_imatgen" / "default_test_result_model.json",
                    method="subset",
                    epsilon=10.,
                    threshold=0.01,
                    objective_tolerance=5e-2,
                    model_format=None,
                    output_format="json",
                    samples=None,
                    aggregation_method="median",
                    transpose=False,
                    quantile="0.15",
                    subset=False,
                    verbose=False,
                    solver="glpk",
                    sep=",",
                    loopless=False,
                    processes=None,

                ))
    def test_run_param_change(self, mock_args):
        imatgen.run()
        # Test that it created the expected file
        self.assertTrue(os.path.exists(self.tmp_path / argparse.ArgumentParser.parse_args().output_file))
        # Test that the output model is the same that would be created by running the IMAT algorithm by hand
        out_model = metworkpy.read_model(self.tmp_path / argparse.ArgumentParser.parse_args().output_file)
        gene_weights = metworkpy.utils.expr_to_gene_weights(expression=self.gene_expression,
                                                            quantile=0.15,
                                                            aggregator=np.median,
                                                            subset=None,
                                                            sample_axis=0)
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(model=self.test_model,
                                                        gene_weights=gene_weights)
        expected_model = metworkpy.imat.generate_model(model=self.test_model,
                                                       rxn_weights=rxn_weights,
                                                       method="subset",
                                                       epsilon=10,
                                                       threshold=0.01,
                                                       objective_tolerance=5e-2
                                                       )
        self.assertTrue(metworkpy.utils.model_eq(out_model, expected_model))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(
                    model_file=BASE_PATH / "data" / "test_model.json",
                    gene_expression_file=BASE_PATH / "data" / "test_model_gene_expression.csv",
                    output_file=BASE_PATH / "tmp_imatgen" / "default_test_result_model.json",
                    method="fva",
                    epsilon=1.,
                    threshold=0.001,
                    objective_tolerance=5e-2,
                    model_format=None,
                    output_format="json",
                    samples=None,
                    aggregation_method="median",
                    transpose=False,
                    quantile="0.10",
                    subset=False,
                    verbose=False,
                    solver="glpk",
                    sep=",",
                    loopless=False,
                    processes=None,

                ))
    def test_run_method_fva(self, mock_args):
        imatgen.run()
        # Test that it created the expected file
        self.assertTrue(os.path.exists(self.tmp_path / argparse.ArgumentParser.parse_args().output_file))
        # Test that the output model is the same that would be created by running the IMAT algorithm by hand
        out_model = metworkpy.read_model(self.tmp_path / argparse.ArgumentParser.parse_args().output_file)
        gene_weights = metworkpy.utils.expr_to_gene_weights(expression=self.gene_expression,
                                                            quantile=0.1,
                                                            aggregator=np.median,
                                                            subset=None,
                                                            sample_axis=0)
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(model=self.test_model,
                                                        gene_weights=gene_weights)
        expected_model = metworkpy.imat.generate_model(model=self.test_model,
                                                       rxn_weights=rxn_weights,
                                                       method="fva",
                                                       epsilon=1.,
                                                       threshold=0.001,
                                                       objective_tolerance=5e-2,
                                                       loopless=False
                                                       )
        for rxn in out_model.reactions:
            expected_rxn = expected_model.reactions.get_by_id(rxn.id)
            self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
            self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
        # Make sure the model is not identical to before running IMAT
        self.assertFalse(metworkpy.model_eq(out_model, self.test_model))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(
                    model_file=BASE_PATH / "data" / "test_model.json",
                    gene_expression_file=BASE_PATH / "data" / "test_model_gene_expression.csv",
                    output_file=BASE_PATH / "tmp_imatgen" / "default_test_result_model.json",
                    method="simple",
                    epsilon=1.,
                    threshold=0.001,
                    objective_tolerance=5e-2,
                    model_format=None,
                    output_format="json",
                    samples=None,
                    aggregation_method="median",
                    transpose=False,
                    quantile="0.10",
                    subset=False,
                    verbose=False,
                    solver="glpk",
                    sep=",",
                    loopless=False,
                    processes=None,

                ))
    def test_run_method_simple(self, mock_args):
        imatgen.run()
        # Test that it created the expected file
        self.assertTrue(os.path.exists(self.tmp_path / argparse.ArgumentParser.parse_args().output_file))
        # Test that the output model is the same that would be created by running the IMAT algorithm by hand
        out_model = metworkpy.read_model(self.tmp_path / argparse.ArgumentParser.parse_args().output_file)
        gene_weights = metworkpy.utils.expr_to_gene_weights(expression=self.gene_expression,
                                                            quantile=0.1,
                                                            aggregator=np.median,
                                                            subset=None,
                                                            sample_axis=0)
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(model=self.test_model,
                                                        gene_weights=gene_weights)
        expected_model = metworkpy.imat.generate_model(model=self.test_model,
                                                       rxn_weights=rxn_weights,
                                                       method="simple",
                                                       epsilon=1.,
                                                       threshold=0.001,
                                                       objective_tolerance=5e-2
                                                       )
        for rxn in out_model.reactions:
            expected_rxn = expected_model.reactions.get_by_id(rxn.id)
            self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
            self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
        # Make sure the model is not identical to before running IMAT
        self.assertFalse(metworkpy.model_eq(out_model, self.test_model))

    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(
                    model_file=BASE_PATH / "data" / "test_model.json",
                    gene_expression_file=BASE_PATH / "data" / "test_model_gene_expression.csv",
                    output_file=BASE_PATH / "tmp_imatgen" / "default_test_result_model.json",
                    method="milp",
                    epsilon=1.,
                    threshold=0.001,
                    objective_tolerance=5e-2,
                    model_format=None,
                    output_format="json",
                    samples=None,
                    aggregation_method="median",
                    transpose=False,
                    quantile="0.10",
                    subset=False,
                    verbose=False,
                    solver="glpk",
                    sep=",",
                    loopless=False,
                    processes=None,

                ))
    def test_run_method_milp(self, mock_args):
        imatgen.run()
        # Test that it created the expected file
        self.assertTrue(os.path.exists(self.tmp_path / argparse.ArgumentParser.parse_args().output_file))
        # Test that the output model is the same that would be created by running the IMAT algorithm by hand
        out_model = metworkpy.read_model(self.tmp_path / argparse.ArgumentParser.parse_args().output_file)
        gene_weights = metworkpy.utils.expr_to_gene_weights(expression=self.gene_expression,
                                                            quantile=0.1,
                                                            aggregator=np.median,
                                                            subset=None,
                                                            sample_axis=0)
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(model=self.test_model,
                                                        gene_weights=gene_weights)
        expected_model = metworkpy.imat.generate_model(model=self.test_model,
                                                       rxn_weights=rxn_weights,
                                                       method="milp",
                                                       epsilon=1.,
                                                       threshold=0.001,
                                                       objective_tolerance=5e-2
                                                       )
        for rxn in out_model.reactions:
            expected_rxn = expected_model.reactions.get_by_id(rxn.id)
            self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
            self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
        # Make sure the model is not identical to before running IMAT
        self.assertFalse(metworkpy.model_eq(out_model, self.test_model))


class TestHelperFunctions(unittest.TestCase):
    def test_parse_samples(self):
        self.assertListEqual(imatgen._parse_samples("1"), [1])
        self.assertListEqual(imatgen._parse_samples("1:5"), [1, 2, 3, 4, 5])
        self.assertListEqual(imatgen._parse_samples("1,3,6:7"), [1, 3, 6, 7])
        self.assertListEqual(imatgen._parse_samples("2:5,1,7"), [2, 3, 4, 5, 1, 7])

    def test_parse_quantile(self):
        self.assertTupleEqual(imatgen._parse_quantile("0.15"), (0.15, 0.85))
        self.assertTupleEqual(imatgen._parse_quantile("0.10,0.90"), (0.10, 0.90))

    def test_parse_aggregation_method(self):
        self.assertEqual(imatgen._parse_aggregation_method("median"), np.median)
        self.assertEqual(imatgen._parse_aggregation_method("max"), np.max)
        self.assertEqual(imatgen._parse_aggregation_method("min"), np.min)
        self.assertEqual(imatgen._parse_aggregation_method("mean"), np.mean)


if __name__ == '__main__':
    unittest.main()

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
import metworkpy.scripts._script_utils
from metworkpy.scripts import imatgen, _script_utils

BASE_PATH = pathlib.Path(__file__).parent.parent


class TestRun(unittest.TestCase):
    default_dict = {
        "model_file": BASE_PATH / "data" / "test_model.json",
        "gene_expression_file": BASE_PATH / "data" / "test_model_gene_expression.csv",
        "output_file": BASE_PATH / "tmp_imatgen" / "test_result_model.json",
        "method": "subset",
        "epsilon": 1.,
        "threshold": 0.001,
        "objective_tolerance": 5e-2,
        "model_format": None,
        "output_format": "json",
        "samples": None,
        "aggregation_method": "median",
        "transpose": False,
        "quantile": "0.15",
        "subset": False,
        "verbose": False,
        "solver": "glpk",
        "sep": ",",
        "loopless": False,
        "processes": None,
    }

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

    def cli_tester(self, **kwargs):
        namespace_dict = self.default_dict | kwargs
        with mock.patch('argparse.ArgumentParser.parse_args',
                        return_value=argparse.Namespace(**namespace_dict)):
            imatgen.run()
            # Test that it created the expected file
            self.assertTrue(os.path.exists(argparse.ArgumentParser.parse_args().output_file))
            # Test that the output model is the same that would be created by running the IMAT algorithm by hand
            out_model = metworkpy.read_model(argparse.ArgumentParser.parse_args().output_file)
            gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
                expression=self.gene_expression,
                quantile=_script_utils._parse_quantile(namespace_dict["quantile"]),
                aggregator=_script_utils._parse_aggregation_method(namespace_dict["aggregation_method"]),
                subset=(None if not namespace_dict["subset"] else self.model.genes.list_attr("id")),
                sample_axis=0)
            rxn_weights = metworkpy.gpr.gene_to_rxn_weights(model=self.test_model,
                                                            gene_weights=gene_weights)
            expected_model = metworkpy.imat.generate_model(model=self.test_model,
                                                           rxn_weights=rxn_weights,
                                                           method=namespace_dict["method"],
                                                           epsilon=namespace_dict["epsilon"],
                                                           threshold=namespace_dict["threshold"],
                                                           objective_tolerance=namespace_dict["objective_tolerance"]
                                                           )
            for rxn in out_model.reactions:
                expected_rxn = expected_model.reactions.get_by_id(rxn.id)
                self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
                self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
            # Make sure the model is not identical to before running IMAT
            self.assertFalse(metworkpy.model_eq(out_model, self.test_model))

    def test_default_run(self):
        self.cli_tester()

    @skipIf(
        importlib.util.find_spec("gurobipy") is None, "gurobi is not installed"
    )
    def test_gurobi_solver(self):
        self.cli_tester(solver="gurobi")

    @skipIf(
        importlib.util.find_spec("cplex") is None, "cplex is not installed"
    )
    def test_cplex_solver(self):
        self.cli_tester(solver="cplex")

    def test_fva(self):
        self.cli_tester(method="fva")

    def test_milp(self):
        self.cli_tester(method="milp")

    def test_simple(self):
        self.cli_tester(method="simple")

    def test_epsilon(self):
        self.cli_tester(epsilon=10.)

    def test_threshold(self):
        self.cli_tester(threshold=0.1)

    def test_objective_tolerance(self):
        self.cli_tester(method="fva", objective_tolerance=0.5)

    def test_model_format_sbml(self):
        self.cli_tester(model_file=BASE_PATH / "data" / "test_model.xml",
                        model_format="sbml")

    def test_model_format_yaml(self):
        self.cli_tester(model_file=BASE_PATH / "data" / "test_model.yaml",
                        model_format="yaml")

    def test_model_format_mat(self):
        self.cli_tester(model_file=BASE_PATH / "data" / "test_model.mat",
                        model_format="mat")


if __name__ == '__main__':
    unittest.main()

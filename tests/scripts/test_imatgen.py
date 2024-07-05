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


# region Helper Functions
def setup(cls, gene_expression_file):
    # Configure cobra to default to glpk
    cobra.core.configuration.Configuration().solver = "glpk"
    # Get path references, and make a temporary folder
    cls.data_path = BASE_PATH / "data"
    cls.tmp_path = BASE_PATH / "tmp_imatgen"
    os.mkdir(cls.tmp_path)
    # Get the gene expression data
    cls.gene_expression = pd.read_csv(
        cls.data_path / gene_expression_file, index_col=0, header=0
    )
    # Get the unchanged model
    cls.model = metworkpy.read_model(cls.data_path / "test_model.json")


def teardown_dir(self):
    # Cleanup the tmp directory after each test (to not potentially pollute)
    for filename in os.listdir(self.tmp_path):
        p = self.tmp_path / filename
        os.remove(p)


def model_bounds_equality(self, model_a, model_b):
    for rxn_a in model_a.reactions:
        rxn_b = model_b.reactions.get_by_id(rxn_a.id)
        self.assertTrue(np.isclose(rxn_a.lower_bound, rxn_b.lower_bound))
        self.assertTrue(np.isclose(rxn_a.upper_bound, rxn_b.upper_bound))


# endregion Helper Functions


class TestRunSingle(unittest.TestCase):
    default_dict = {
        "model_file": BASE_PATH / "data" / "test_model.json",
        "gene_expression_file": BASE_PATH / "data" / "test_model_gene_expression.csv",
        "output_file": BASE_PATH / "tmp_imatgen" / "test_result_model.json",
        "method": "subset",
        "epsilon": 1.0,
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
        "func": imatgen.run_single,
    }

    data_path = None
    tmp_path = None
    gene_expression = None
    model = None

    @classmethod
    def setUpClass(cls):
        setup(cls, "test_model_gene_expression.csv")

    def setUp(self):
        self.test_model = self.model.copy()

    def tearDown(self):
        teardown_dir(self)

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.tmp_path)

    def cli_tester(self, **kwargs):
        namespace_dict = self.default_dict | kwargs
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(**namespace_dict),
        ):
            imatgen.main_run()
            # Test that it created the expected file
            self.assertTrue(
                os.path.exists(argparse.ArgumentParser.parse_args().output_file)
            )
            # Test that the output model is the same that would be created by running the IMAT algorithm by hand
            out_model = metworkpy.read_model(
                argparse.ArgumentParser.parse_args().output_file
            )
            gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
                expression=self.gene_expression,
                quantile=_script_utils._parse_quantile(namespace_dict["quantile"]),
                aggregator=_script_utils._parse_aggregation_method(
                    namespace_dict["aggregation_method"]
                ),
                subset=(
                    None
                    if not namespace_dict["subset"]
                    else self.model.genes.list_attr("id")
                ),
                sample_axis=0,
            )
            rxn_weights = metworkpy.gpr.gene_to_rxn_weights(
                model=self.test_model, gene_weights=gene_weights
            )
            expected_model = metworkpy.imat.generate_model(
                model=self.test_model,
                rxn_weights=rxn_weights,
                method=namespace_dict["method"],
                epsilon=namespace_dict["epsilon"],
                threshold=namespace_dict["threshold"],
                objective_tolerance=namespace_dict["objective_tolerance"],
            )
            for rxn in out_model.reactions:
                expected_rxn = expected_model.reactions.get_by_id(rxn.id)
                self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
                self.assertTrue(np.isclose(rxn.lower_bound, expected_rxn.lower_bound))
            # Make sure the model is not identical to before running IMAT
            self.assertFalse(metworkpy.model_eq(out_model, self.test_model))

    def test_default_run(self):
        self.cli_tester()

    @skipIf(importlib.util.find_spec("gurobipy") is None, "gurobi is not installed")
    def test_gurobi_solver(self):
        self.cli_tester(solver="gurobi")

    @skipIf(importlib.util.find_spec("cplex") is None, "cplex is not installed")
    def test_cplex_solver(self):
        self.cli_tester(solver="cplex")

    def test_fva(self):
        self.cli_tester(method="fva")

    def test_milp(self):
        self.cli_tester(method="milp")

    def test_simple(self):
        self.cli_tester(method="simple")

    def test_epsilon(self):
        self.cli_tester(epsilon=10.0)

    def test_threshold(self):
        self.cli_tester(threshold=0.1)

    def test_objective_tolerance(self):
        self.cli_tester(method="fva", objective_tolerance=0.5)

    def test_model_format_sbml(self):
        self.cli_tester(
            model_file=BASE_PATH / "data" / "test_model.xml", model_format="sbml"
        )

    def test_model_format_yaml(self):
        self.cli_tester(
            model_file=BASE_PATH / "data" / "test_model.yaml", model_format="yaml"
        )

    def test_model_format_mat(self):
        self.cli_tester(
            model_file=BASE_PATH / "data" / "test_model.mat", model_format="mat"
        )


class TestRunMulti(unittest.TestCase):
    data_path = None
    tmp_path = None
    gene_expression = None
    model = None
    default_dict = {
        "model_file": BASE_PATH / "data" / "test_model.json",
        "gene_expression_file": BASE_PATH
        / "data"
        / "test_model_gene_expression_imatgen_multi.csv",
        "output_dir": BASE_PATH / "tmp_imatgen",
        "method": "subset",
        "epsilon": 1.0,
        "threshold": 0.001,
        "objective_tolerance": 5e-2,
        "model_format": None,
        "output_format": "json",
        "aggregation_method": "median",
        "transpose": False,
        "quantile": "0.10",
        "subset": False,
        "verbose": False,
        "solver": "glpk",
        "sep": ",",
        "loopless": False,
        "processes": None,
        "func": imatgen.run_multi,
        "prefix": None,
        "sample_groups": None,
        "fold_change": None,
        "wildtype": "6,7,8",
        "sample_group_names": None,
    }

    @classmethod
    def setUpClass(cls):
        setup(cls, "test_model_gene_expression_imatgen_multi.csv")

    def setUp(self):
        self.test_model = self.model.copy()

    def tearDown(self):
        teardown_dir(self)

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.tmp_path)

    def run_cli(self, **kwargs):
        namespace_dict = self.default_dict | kwargs
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(**namespace_dict),
        ):
            imatgen.main_run()
            args = argparse.ArgumentParser.parse_args()
            # Check that the expected file outputs exist
            if args.wildtype:
                wildtype = _script_utils._parse_samples(args.wildtype)
            else:
                wildtype = []
            if args.prefix:
                prefix = args.prefix + "_"
            else:
                prefix = ""
            if args.output_format:
                extension = _script_utils._parse_format_to_extension(args.output_format)
            else:
                extension = "json"
            if args.sample_groups:
                (
                    sample_groups,
                    sample_names,
                ) = _script_utils._parse_sample_groups_and_names(
                    args.sample_groups, args.sample_group_names
                )
            else:
                sample_groups = [
                    [idx]
                    for idx, _ in enumerate(self.gene_expression.index)
                    if idx not in wildtype
                ]
                sample_names = [f"s{i}" for i in range(1, len(sample_groups) + 1)]
            for sample in sample_names:
                filename = f"{prefix}{sample}.{extension}"
                self.assertTrue(os.path.exists(self.tmp_path / filename))
            # For each of the sample groups, generate rxn weights
            if args.subset:
                subset = self.test_model.genes.list_attr("id")
            else:
                subset = None
            aggregator = _script_utils._parse_aggregation_method(
                args.aggregation_method
            )
            rxn_weights = {}
            wt_expr = self.gene_expression.iloc[wildtype, :].apply(aggregator, axis=0)
            for group, name in zip(sample_groups, sample_names):
                if not args.fold_change:
                    g_weights = metworkpy.utils.expr_to_imat_gene_weights(
                        expression=self.gene_expression.iloc[group, :],
                        quantile=_script_utils._parse_quantile(args.quantile),
                        aggregator=aggregator,
                        subset=subset,
                        sample_axis=0,
                    )
                    rxn_weights[name] = metworkpy.gpr.gene_to_rxn_weights(
                        model=self.test_model,
                        gene_weights=g_weights,
                        fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
                        fill_val=0,
                    )
                else:
                    log2fc = np.log2(
                        self.gene_expression.iloc[group, :]
                        .apply(aggregator, axis=0)
                        .div(wt_expr)
                        + 1e-10
                    )
                    g_weights = log2fc.apply(
                        lambda x: -1
                        if x <= -args.fold_change
                        else (1 if x >= args.fold_change else 0)
                    )
                    rxn_weights[name] = metworkpy.gene_to_rxn_weights(
                        model=self.test_model,
                        gene_weights=g_weights,
                        fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
                        fill_val=0,
                    )
        for name, r_weights in rxn_weights.items():
            filename = f"{prefix}{name}.{extension}"
            generated_model = metworkpy.read_model(
                self.tmp_path / filename, args.output_format
            )
            expected_model = metworkpy.imat.generate_model(
                model=self.test_model,
                rxn_weights=r_weights,
                method=args.method,
                epsilon=args.epsilon,
                threshold=args.threshold,
                objective_tolerance=args.objective_tolerance,
            )
            model_bounds_equality(self, expected_model, generated_model)
            self.assertFalse(metworkpy.model_eq(generated_model, self.model))

    def test_default_run(self):
        self.run_cli()

    @skipIf(importlib.util.find_spec("gurobipy") is None, "gurobi is not installed")
    def test_gurobi(self):
        self.run_cli(solver="gurobi")

    @skipIf(importlib.util.find_spec("cplex") is None, "cplex is not installed")
    def test_cplex(self):
        self.run_cli(solver="cplex")

    def test_fva(self):
        self.run_cli(method="fva")

    def test_milp(self):
        self.run_cli(method="milp")

    def test_simple(self):
        self.run_cli(method="simple")

    def test_epsilon(self):
        self.run_cli(epsilon=10.0)

    def test_threshold(self):
        self.run_cli(threshold=0.1)

    def test_model_format_smbl(self):
        self.run_cli(output_format="sbml")

    def test_model_format_yaml(self):
        self.run_cli(output_format="yaml")

    def test_model_format_mat(self):
        self.run_cli(output_format="mat")

    def test_sample_groups(self):
        self.run_cli(sample_groups="(0,1,2)(3,4,5)")

    def test_sample_names(self):
        self.run_cli(sample_groups="(0,1,2)(3,4,5)", sample_group_names="oe,ko")

    def test_foldchange(self):
        self.run_cli(fold_change=1.0, sample_groups="(0,1,2)(3,4,5)")


if __name__ == "__main__":
    unittest.main()

# Imports
# Standard Library Imports
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import pathlib
import re
import unittest
from unittest import mock, skipIf

# External Imports
import cobra
import numpy as np
import pandas as pd

# Local Imports
import metworkpy
from metworkpy.scripts import _script_utils, metchange

# Setup BASE_PATH
BASE_PATH = pathlib.Path(__file__).parent.parent


class TestRun(unittest.TestCase):
    default_dict = {
        "model_file": BASE_PATH / "data" / "test_model.json",
        "gene_expression_file": BASE_PATH
        / "data"
        / "test_model_gene_expression_metchange.csv",
        "wildtype": "0:5",
        "sample_groups": "",
        "output_file": BASE_PATH / "tmp_metchange" / "metchange_res.csv",
        "model_format": None,
        "metabolites_list": None,
        "metabolites_file": None,
        "sample_group_names": None,
        "quantile_cutoff": 0.2,
        "objective_tolerance": 5e-2,
        "aggregation_method": "median",
        "transpose": False,
        "subset": False,
        "verbose": False,
        "extra_info": False,
        "solver": "glpk",
        "sep": ",",
    }

    @classmethod
    def setUpClass(cls):
        # Configure cobra to default to GLPK
        cobra.core.configuration.Configuration().solver = "glpk"
        # Get path references, make temporary folder
        cls.data_path = BASE_PATH / "data"
        cls.tmp_path = BASE_PATH / "tmp_metchange"
        os.mkdir(cls.tmp_path)
        # Get the gene expression data
        cls.gene_expression = pd.read_csv(
            cls.data_path / "test_model_gene_expression_metchange.csv",
            index_col=0,
            header=0,
        )
        # Get the original model
        cls.model = metworkpy.read_model(cls.data_path / "test_model.json")

    def setUp(self):
        self.test_model = self.model.copy()

    def tearDown(self):
        # Cleanup the tmp directory
        for filename in os.listdir(self.tmp_path):
            p = self.tmp_path / filename
            os.remove(p)

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.tmp_path)

    def run_cli(self, **kwargs) -> pd.DataFrame:
        namespace_dict = self.default_dict | kwargs
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(**namespace_dict),
        ):
            metchange.run()
            # Test that the expected file is created
            self.assertTrue(
                os.path.exists(argparse.ArgumentParser.parse_args().output_file)
            )
            # Read the results in
            metchange_results = pd.read_csv(
                argparse.ArgumentParser.parse_args().output_file, index_col=0, header=0
            )
        return metchange_results

    def match_cli(self, **kwargs):
        namespace_dict = self.default_dict | kwargs
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(**namespace_dict),
        ):
            metchange.run()
            # Test that the expected file is created
            self.assertTrue(
                os.path.exists(argparse.ArgumentParser.parse_args().output_file)
            )
            # Read the results in
            metchange_results_cli = pd.read_csv(
                argparse.ArgumentParser.parse_args().output_file, index_col=0, header=0
            )
            METCHANGE_FUNC_DICT = {"AND": max, "OR": min}
            wt_samples = _script_utils._parse_samples(namespace_dict["wildtype"])
            wt_rxn_weights = []
            for s in wt_samples:
                wt_gene_weights = metworkpy.utils.expr_to_metchange_gene_weights(
                    expression=self.gene_expression.iloc[s],
                    quantile_cutoff=namespace_dict["quantile_cutoff"],
                    subset=(
                        None
                        if not namespace_dict["subset"]
                        else self.test_model.genes.list_attr("id")
                    ),
                    aggregator=_script_utils._parse_aggregation_method(
                        namespace_dict["aggregation_method"]
                    ),
                    sample_axis=0,
                )
                wt_rxn_weights.append(
                    metworkpy.gpr.gene_to_rxn_weights(
                        model=self.test_model,
                        gene_weights=wt_gene_weights,
                        fn_dict=METCHANGE_FUNC_DICT,
                    )
                )
            if namespace_dict["sample_groups"]:
                group_pattern = re.compile(r"\(([\d,:]+)\)")
                sample_groups = [
                    _script_utils._parse_samples(m)
                    for m in group_pattern.findall(namespace_dict["sample_groups"])
                ]
            else:
                sample_groups = [
                    [i]
                    for i in range(len(self.gene_expression.index))
                    if i not in wt_samples
                ]
                sample_names = list(
                    self.gene_expression.index[
                        list(itertools.chain.from_iterable(sample_groups))
                    ]
                )
            sample_rxn_weights_list = []
            for sample_group in sample_groups:
                expr = self.gene_expression.iloc[sample_group]
                gene_weights = metworkpy.utils.expr_to_metchange_gene_weights(
                    expression=expr,
                    quantile_cutoff=namespace_dict["quantile_cutoff"],
                    subset=(
                        None
                        if not namespace_dict["subset"]
                        else self.test_model.genes.list_attr("id")
                    ),
                    aggregator=_script_utils._parse_aggregation_method(
                        namespace_dict["aggregation_method"]
                    ),
                    sample_axis=0,
                )
                sample_rxn_weights_list.append(
                    metworkpy.gene_to_rxn_weights(
                        model=self.test_model,
                        gene_weights=gene_weights,
                        fn_dict=METCHANGE_FUNC_DICT,
                    )
                )
            # Run Metchange
            wt_metchange = pd.DataFrame(
                0.0,
                index=range(len(wt_rxn_weights)),
                columns=self.test_model.metabolites.list_attr("id"),
            )
            for idx, rxn_wrights in enumerate(wt_rxn_weights):
                res = metworkpy.metchange(
                    model=self.test_model,
                    reaction_weights=rxn_wrights,
                    metabolites=None,
                    objective_tolerance=namespace_dict["objective_tolerance"],
                )
                wt_metchange.loc[idx, res.index] = res
            wt_mean = wt_metchange.mean(axis=0)
            wt_std = wt_metchange.std(axis=0)
            metchange_results_expected = pd.DataFrame(
                0.0,
                index=sample_names,
                columns=self.test_model.metabolites.list_attr("id"),
            )
            for name, weights in zip(sample_names, sample_rxn_weights_list):
                res = (
                    metworkpy.metchange(
                        model=self.test_model,
                        reaction_weights=weights,
                        metabolites=None,
                        objective_tolerance=namespace_dict["objective_tolerance"],
                    )
                    - wt_mean
                ) / wt_std
                metchange_results_expected.loc[name, res.index] = res
            metchange_results_expected = metchange_results_expected.transpose()
        self.assertTrue(
            np.isclose(
                metchange_results_cli.fillna(0.0).values,
                metchange_results_expected.fillna(0.0).values,
            ).all()
        )

    def test_default_run(self):
        self.match_cli()

    def test_metabolites_list(self):
        met_list = "A_e,F_c,H_c,H_e"
        metchange_res = self.run_cli(metabolites_list=met_list)
        self.assertListEqual(list(metchange_res.index), met_list.split(","))

    def test_metabolites_file(self):
        with open(BASE_PATH / "data" / "met_list.txt", "r") as f:
            met_file = f.read()
        met_file_metabolites = [
            met.strip() for met in met_file.split(",") if met.strip()
        ]
        metchange_res = self.run_cli(
            metabolites_file=BASE_PATH / "data" / "met_list.txt"
        )
        self.assertListEqual(list(metchange_res.index), met_file_metabolites)

    def test_extra_info(self):
        metchange_res = self.run_cli(extra_info=True)
        info_list = ["name", "formula", "compartment", "charge"]
        for i in info_list:
            self.assertTrue(i in metchange_res.columns)

    @skipIf(importlib.util.find_spec("gurobipy") is None, "gurobi is not installed")
    def test_gurobi_solver(self):
        self.match_cli(solver="gurobi")

    @skipIf(importlib.util.find_spec("cplex") is None, "cplex is not installed")
    def test_cplex_solver(self):
        self.match_cli(solver="cplex")

    def test_tolerance(self):
        self.match_cli(objective_tolerance=0.2)

    def test_sample_groups(self):
        sample_group_names = "marmoset,jackal"
        res = self.run_cli(
            sample_groups="(6,7)(8)", sample_group_names=sample_group_names
        )
        for name in sample_group_names.split(","):
            self.assertTrue(name in res.columns)


if __name__ == "__main__":
    unittest.main()

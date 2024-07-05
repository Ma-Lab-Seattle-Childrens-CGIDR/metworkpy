# Imports
# Standard Library Imports
from __future__ import annotations
import argparse
import importlib.util
import os
import pathlib
import unittest
from unittest import mock, skipIf

# External Imports
import cobra
import pandas as pd

# Local Imports
import metworkpy
from metworkpy.scripts import metdivergence

# Setup BASE_PATH
BASE_PATH = pathlib.Path(__file__).parent.parent


class TestMetdivergenceMain(unittest.TestCase):
    default_dict = {
        "treatment_distribution_file": BASE_PATH
        / "tmp_metdivergence"
        / "treatment_distribution.csv",
        "wildtype_distribution_file": BASE_PATH
        / "tmp_metdivergence"
        / "wildtype_distribution.csv",
        "output_file": BASE_PATH / "tmp_metdivergence" / "metdivergence_results.csv",
        "input_format": "csv",
        "neighbors": 5,
        "divergence_type": "js",
        "metric": "euclidean",
        "model_file": None,
        "model_format": None,
        "sheet_name": None,
        "verbose": False,
    }

    @classmethod
    def setUpClass(cls):
        # Configure cobra to use GLPK
        cobra.core.Configuration.solver = "glpk"
        # Get path references, make temporary folder
        cls.data_path = BASE_PATH / "data"
        cls.tmp_path = BASE_PATH / "tmp_metdivergence"
        os.mkdir(cls.tmp_path)
        cls.model = metworkpy.read_model(cls.data_path / "test_model.json")

    def setUp(self):
        self.test_model = self.model.copy()
        self.imat_gene_expression = pd.read_csv(
            self.data_path / "test_model_gene_expression_imatgen_multi.csv",
            index_col=0,
            header=0,
        )

    def tearDown(self):
        # Cleanup the tmp directory
        for filename in os.listdir(self.tmp_path):
            p = self.tmp_path / filename
            os.remove(p)

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.tmp_path)

    def run_cli(self, **kwargs):
        namespace_dict = self.default_dict | kwargs
        # Create IMAT model for sampling
        gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
            expression=self.imat_gene_expression.loc[["s1", "s2", "s3"], :],
            quantile=0.1,
        )
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(
            model=self.test_model,
            gene_weights=gene_weights,
            fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
            fill_val=0,
        )
        treatment_model = metworkpy.imat.generate_model(
            model=self.test_model,
            rxn_weights=rxn_weights,
            method="fva",
            epsilon=1.0,
            threshold=0.01,
        )
        wt_model = self.test_model
        # Sample the IMAT model and the WT
        wt_sample = cobra.sampling.sample(model=wt_model, n=20)
        treatment_sample = cobra.sampling.sample(model=treatment_model, n=20)
        # Write the sampling results with the appropriate format to the tmp directory
        if namespace_dict["input_format"] == "csv":
            wt_sample.to_csv(
                self.tmp_path / "wildtype_distribution.csv", index=False, header=True
            )
            treatment_sample.to_csv(
                self.tmp_path / "treatment_distribution.csv", index=False, header=True
            )
        elif namespace_dict["input_format"] == "parquet":
            wt_sample.to_parquet(
                self.tmp_path / "wildtype_distribution.parquet", index=False
            )
            treatment_sample.to_parquet(
                self.tmp_path / "treatment_distribution.parquet", index=False
            )
        elif namespace_dict["input_format"] == "feather":
            wt_sample.to_feather(self.tmp_path / "wildtype_distribution.feather")
            treatment_sample.to_feather(
                self.tmp_path / "treatment_distribution.feather"
            )
        elif namespace_dict["input_format"] == "json":
            wt_sample.to_json(self.tmp_path / "wildtype_distribution.json")
            treatment_sample.to_json(self.tmp_path / "treatment_distribution.json")
        elif metdivergence._parse_format(namespace_dict["input_format"]) == "excel":
            wt_sample.to_excel(
                self.tmp_path / "wildtype_distribution.xlsx",
                sheet_name="Flux Samples",
                index=False,
            )
            treatment_sample.to_excel(
                self.tmp_path / "treatment_distribution.xlsx",
                sheet_name="Flux Samples",
                index=False,
            )
        else:
            raise ValueError("Invalid Input Format")
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(**namespace_dict),
        ):
            metdivergence.main_run()
        return pd.read_csv(namespace_dict["output_file"], index_col=0, header=0)

    def test_default(self):
        res = self.run_cli()
        self.assertListEqual(list(res.columns), ["js_divergence"])
        self.assertEqual(len(res), len(self.test_model.reactions))

    def test_kl_givergence(self):
        res = self.run_cli(divergence_type="Kullback-Leibler")
        self.assertListEqual(list(res.columns), ["kl_divergence"])
        self.assertEqual(len(res), len(self.test_model.reactions))

    def format_tester(self, file_format, **kwargs):
        res = self.run_cli(
            input_format=file_format,
            treatment_distribution_file=self.tmp_path
            / f"treatment_distribution.{file_format}",
            wildtype_distribution_file=self.tmp_path
            / f"wildtype_distribution.{file_format}",
            **kwargs,
        )
        self.assertListEqual(list(res.columns), ["js_divergence"])
        self.assertEqual(len(res), len(self.test_model.reactions))

    def test_parquet(self):
        self.format_tester(file_format="parquet")

    def test_feather(self):
        self.format_tester(file_format="feather")

    @skipIf(
        importlib.util.find_spec("openpyxl") is None,
        "openpyxl isn't installed, can't read/write excel",
    )
    def test_excel(self):
        self.format_tester(file_format="xlsx", sheet_name="Flux Samples")

    def test_json(self):
        self.format_tester(file_format="json")

    def test_metric(self):
        res = self.run_cli(metric="cheby")
        self.assertListEqual(list(res.columns), ["js_divergence"])
        self.assertEqual(len(res), len(self.test_model.reactions))

    def test_model_info(self):
        res = self.run_cli(model_file=self.data_path / "test_model.json")
        self.assertListEqual(
            list(res.columns),
            [
                "js_divergence",
                "Reaction Name",
                "Reaction Expression",
                "Subsystem",
                "Genes",
            ],
        )
        self.assertEqual(len(res), len(self.test_model.reactions))


if __name__ == "__main__":
    unittest.main()

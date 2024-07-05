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
from metworkpy.scripts import metsample

# Setup BASE_PATH
BASE_PATH = pathlib.Path(__file__).parent.parent


class TestMetsampleRun(unittest.TestCase):
    default_dict = {
        "model_file": BASE_PATH / "data" / "test_model.json",
        "num_samples": 10,
        "output_file": BASE_PATH / "tmp_metsample" / "sample_res.csv",
        "output_format": "csv",
        "method": "optgp",
        "model_format": "json",
        "processes": 1,
        "thinning": 100,
        "seed": None,
        "validate": False,
        "batches": None,
        "verbose": False,
    }

    @classmethod
    def setUpClass(cls):
        # Configure cobra to use GLPK
        cobra.core.Configuration.solver = "glpk"
        # Get path references, make temporary folder
        cls.data_path = BASE_PATH / "data"
        cls.tmp_path = BASE_PATH / "tmp_metsample"
        os.mkdir(cls.tmp_path)
        # Get the model
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
            metsample.main_run()
            # Test expected file created
            args = argparse.ArgumentParser.parse_args()
            self.assertTrue(os.path.exists(args.output_file))
            format_str = metsample._parse_format(args.output_format)
            if format_str == "csv":
                sample_results = pd.read_csv(args.output_file, index_col=None, header=0)
            elif format_str == "parquet":
                sample_results = pd.read_parquet(args.output_file)
            elif format_str == "feather":
                sample_results = pd.read_feather(args.output_file)
            elif format_str == "excel":
                sample_results = pd.read_excel(
                    args.output_file, index_col=None, sheet_name="Flux Samples"
                )
            elif format_str == "json":
                sample_results = pd.read_json(args.output_file)
        return sample_results

    def test_defaults(self):
        res = self.run_cli()
        self.assertEqual(len(res), 10)
        self.assertEqual(len(res.columns), len(self.test_model.reactions))
        self.assertFalse(res.isna().any(axis=None))

    def test_validate(self):
        res = self.run_cli(validate=True)
        self.assertEqual(len(res), 10)
        self.assertEqual(len(res.columns), len(self.test_model.reactions))
        self.assertFalse(res.isna().any(axis=None))

    def format_tester(self, format):
        res = self.run_cli(
            output_file=BASE_PATH / "tmp_metsample" / f"metsample_res.{format}",
            output_format=format,
        )
        self.assertEqual(len(res), 10)
        self.assertEqual(len(res.columns), len(self.test_model.reactions))
        self.assertFalse(res.isna().any(axis=None))

    def test_parquet(self):
        self.format_tester("parquet")

    def test_feather(self):
        self.format_tester("feather")

    def test_json(self):
        self.format_tester("json")

    @skipIf(
        importlib.util.find_spec("openpyxl") is None,
        "Openpyxl not installed, can't read excel",
    )
    def test_excel(self):
        self.format_tester("xlsx")

    def test_batch_csv(self):
        res = self.run_cli(batches=2)
        self.assertEqual(len(res), 10)
        self.assertEqual(len(res.columns), len(self.test_model.reactions))
        self.assertFalse(res.isna().any(axis=None))

    @skipIf(
        importlib.util.find_spec("fastparquet") is None,
        "fastparquet not installed, can't append to parquet",
    )
    def test_batch_parquet(self):
        res = self.run_cli(
            batches=2,
            output_format="parquet",
            output_file=BASE_PATH / "tmp_metsample" / "metsample_res.parquet",
        )
        self.assertEqual(len(res), 10)
        self.assertEqual(len(res.columns), len(self.test_model.reactions))
        self.assertFalse(res.isna().any(axis=None))

    def test_sample_number(self):
        res = self.run_cli(num_samples=100)
        self.assertEqual(len(res), 100)


class TestHelperFunctions(unittest.TestCase):
    def test_parse_method(self):
        self.assertEqual(metsample._parse_method("achr"), "achr")
        self.assertEqual(
            metsample._parse_method("Automatic-Centering-Hit-and-Run"), "achr"
        )
        self.assertEqual(metsample._parse_method("optgp"), "optgp")
        self.assertEqual(metsample._parse_method("optGP"), "optgp")

    def test_parse_format(self):
        self.assertEqual(metsample._parse_format("csv"), "csv")
        self.assertEqual(metsample._parse_format("comma seperated values"), "csv")
        self.assertEqual(metsample._parse_format("parquet"), "parquet")
        self.assertEqual(metsample._parse_format("Parquet"), "parquet")
        self.assertEqual(metsample._parse_format("feather"), "feather")
        self.assertEqual(metsample._parse_format("Feather"), "feather")
        self.assertEqual(metsample._parse_format("xlsx"), "excel")
        self.assertEqual(metsample._parse_format("Excel"), "excel")
        self.assertEqual(metsample._parse_format("json"), "json")


if __name__ == "__main__":
    unittest.main()

# Imports
# Standard Library Imports
import pathlib
import unittest

import metworkpy.utils.models
# External Imports

# Local Imports
from metworkpy.divergence.ko_divergence_functions import (
    ko_divergence,
    _parse_divergence_method,
    _convert_target_network,
)
from metworkpy.utils.models import read_model


class TestHelperFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"

    def test_convert_target_networks(self):
        data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
        test_model = read_model(data_path / "test_model.json")
        target_list = ["R_C_e_ex", "r_C_E_F", "g_A_B_D_E"]
        expected_list = ["R_C_e_ex", "r_C_E_F", "r_A_B_D_E"]
        converted_list = _convert_target_network(test_model, target_list)
        self.assertListEqual(converted_list, expected_list)
        with self.assertWarnsRegex(
            UserWarning, "Couldn't find fake_gene in model genes or reactions, skipping"
        ):
            _ = _convert_target_network(test_model, ["fake_gene"])

    def test_parse_divergence_method(self):
        self.assertEqual(_parse_divergence_method("JS"), "js")
        self.assertEqual(_parse_divergence_method("jensen-shannon"), "js")
        self.assertEqual(_parse_divergence_method("jensen"), "js")
        self.assertEqual(_parse_divergence_method("Jensen"), "js")

        self.assertEqual(_parse_divergence_method("KL"), "kl")
        self.assertEqual(_parse_divergence_method("kullback–leibler"), "kl")
        self.assertEqual(_parse_divergence_method("kullback_leibler"), "kl")
        self.assertEqual(_parse_divergence_method("kull"), "kl")
        self.assertEqual(_parse_divergence_method("KULL"), "kl")


class TestKoDivergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
        cls.test_model = read_model(cls.data_path / "test_model.json")
        cls.textbook_model = read_model(cls.data_path / "textbook_model.json")

    def test_divergence_ko_js(self):
        div_ko_res = ko_divergence(
            model=self.test_model,
            genes_to_ko=["g_A_imp", "g_B_imp", "g_C_H"],
            target_networks={
                "export": ["R_G_exp", "R_F_exp", "R_H_exp"],
                "H_system": ["r_C_H", "R_H_exp"],
                "upper_route": ["r_A_B_D_E", "r_D_G", "r_C_E_F"],
            },
            divergence_metric="js",
            n_neighbors=3,
            sample_count=100,
            jitter=None,
            distance_metric="euclidean",
            progress_bar=False,
            processes=1,
        )
        # Check that the output dataframe has the expected form
        self.assertCountEqual(list(div_ko_res.index), ["g_A_imp", "g_B_imp", "g_C_H"])
        self.assertCountEqual(
            list(div_ko_res.columns), ["export", "H_system", "upper_route"]
        )

        # Check that there are no nan in the dataframe
        self.assertFalse(div_ko_res.isna().all(axis=None))

        # The export system should be more perturbed than the H_system for both the g_A_imp and g_B_imp KOs
        self.assertGreater(
            div_ko_res.loc["g_A_imp", "export"], div_ko_res.loc["g_A_imp", "H_system"]
        )
        self.assertGreater(
            div_ko_res.loc["g_B_imp", "export"], div_ko_res.loc["g_B_imp", "H_system"]
        )

        # For the g_C_H knockout, the export should be more perturbed than the upper_route
        self.assertGreater(
            div_ko_res.loc["g_C_H", "export"], div_ko_res.loc["g_C_H", "upper_route"]
        )

    def test_divergence_ko_kl(self):
        div_ko_res = ko_divergence(
            model=self.test_model,
            genes_to_ko=["g_A_imp", "g_B_imp", "g_C_H"],
            target_networks={
                "export": ["R_G_exp", "R_F_exp", "R_H_exp"],
                "H_system": ["r_C_H", "R_H_exp"],
                "upper_route": ["r_A_B_D_E", "r_D_G", "r_C_E_F"],
            },
            divergence_metric="kl",
            n_neighbors=3,
            sample_count=100,
            jitter=1e-25,  # Since this will divide by 0 due to the uniformly 0 reaction fluxes
            distance_metric="euclidean",
            progress_bar=False,
            processes=1,
        )
        # Check that the output dataframe has the expected form
        self.assertCountEqual(list(div_ko_res.index), ["g_A_imp", "g_B_imp", "g_C_H"])
        self.assertCountEqual(
            list(div_ko_res.columns), ["export", "H_system", "upper_route"]
        )

        # Check that there are no nan in the dataframe
        self.assertFalse(div_ko_res.isna().all(axis=None))

        # The export system should be more perturbed than the H_system for both the g_A_imp and g_B_imp KOs
        self.assertGreater(
            div_ko_res.loc["g_A_imp", "export"], div_ko_res.loc["g_A_imp", "H_system"]
        )
        self.assertGreater(
            div_ko_res.loc["g_B_imp", "export"], div_ko_res.loc["g_B_imp", "H_system"]
        )

        # For the g_C_H knockout, the export should be more perturbed than the upper_route
        self.assertGreater(
            div_ko_res.loc["g_C_H", "export"], div_ko_res.loc["g_C_H", "upper_route"]
        )

    def test_larger_metabolic_model(self):
        textbook_model = read_model(self.data_path / "textbook_model.json")
        textbook_model_copy = textbook_model.copy()
        div_ko_res = ko_divergence(
            model=textbook_model,
            genes_to_ko=[
                "b0474",
                "b0116",
                "b0726",
                "b0727",
                "b2587",
                "b0356",
                "b1478",
                "b3735",
                "b3733",
            ],
            target_networks={
                "sys1": ["PDH", "PFK", "PFL", "PGI", "PGK", "PGL"],
                "sys2": ["SUCCt2_2", "SUCCt3", "SUCDi", "SUCOAS"],
                "sys3": ["TALA", "THD2", "TKT1", "TKT2"],
            },
            divergence_metric="js",
            n_neighbors=3,
            sample_count=100,
            jitter=None,
            distance_metric="euclidean",
            progress_bar=False,
            processes=1,
        )
        self.assertTrue(
            metworkpy.utils.models.model_eq(
                textbook_model, textbook_model_copy, verbose=False
            )
        )

        self.assertEqual(
            list(div_ko_res.index),
            [
                "b0474",
                "b0116",
                "b0726",
                "b0727",
                "b2587",
                "b0356",
                "b1478",
                "b3735",
                "b3733",
            ],
        )
        self.assertEqual(list(div_ko_res.columns), ["sys1", "sys2", "sys3"])


if __name__ == "__main__":
    unittest.main()

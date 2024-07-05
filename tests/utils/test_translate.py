# Standard Library Imports
import pathlib
import unittest

# External Imports
from cobra.core.configuration import Configuration
import pandas as pd

# Local Imports
from metworkpy.utils.translate import (
    gene_to_reaction_dict,
    gene_to_reaction_df,
    gene_to_reaction_list,
    reaction_to_gene_df,
    reaction_to_gene_dict,
    reaction_to_gene_list,
)
from metworkpy import read_model


def setup(cls):
    data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
    Configuration.solver = "glpk"
    test_model = read_model(str(data_path / "test_model.json"))
    cls.test_model = test_model


class TestListTranslate(unittest.TestCase):
    test_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_reaction_to_gene(self):
        rxn_list = ["r_C_H", "r_C_E_F"]
        gene_list_expected = ["g_C_H", "g_C_E_F"]
        self.assertCountEqual(
            reaction_to_gene_list(self.test_model, rxn_list), gene_list_expected
        )

    def test_gene_to_reaction(self):
        gene_list = ["g_C_H", "g_C_E_F"]
        rxn_list_expected = ["r_C_H", "r_C_E_F"]
        self.assertCountEqual(
            gene_to_reaction_list(self.test_model, gene_list), rxn_list_expected
        )

    def test_reactions_to_gene_essential(self):
        model = read_model(
            pathlib.Path(__file__).parent.parent.absolute()
            / "data"
            / "translate_model.json"
        )
        rxn_list = ["r_A_B"]
        expected_gene_list = []
        self.assertCountEqual(
            reaction_to_gene_list(reaction_list=rxn_list, model=model, essential=True),
            expected_gene_list,
        )
        rxn_list = ["r_B_C"]
        expected_gene_list = ["m"]
        self.assertCountEqual(
            reaction_to_gene_list(reaction_list=rxn_list, model=model, essential=True),
            expected_gene_list,
        )
        rxn_list = ["r_C_D"]
        expected_gene_list = ["m", "n"]
        self.assertCountEqual(
            reaction_to_gene_list(reaction_list=rxn_list, model=model, essential=True),
            expected_gene_list,
        )


class TestDictTranslate(unittest.TestCase):
    test_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_reaction_to_gene(self):
        rxn_list = ["r_C_H", "r_C_E_F"]
        gene_dict_expected = {"r_C_H": ["g_C_H"], "r_C_E_F": ["g_C_E_F"]}
        self.assertDictEqual(
            reaction_to_gene_dict(self.test_model, rxn_list), gene_dict_expected
        )

    def test_gene_to_reaction(self):
        gene_list = ["g_C_H", "g_C_E_F"]
        rxn_dict_expected = {"g_C_H": ["r_C_H"], "g_C_E_F": ["r_C_E_F"]}
        self.assertDictEqual(
            gene_to_reaction_dict(self.test_model, gene_list), rxn_dict_expected
        )


class TestDataFrameTranslate(unittest.TestCase):
    test_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_reaction_to_gene(self):
        rxn_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["r_C_H", "r_C_E_F", "r_D_G"]
        )
        gene_df_expected = pd.DataFrame(
            {
                "reactions": ["r_C_H", "r_C_E_F", "r_D_G"],
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            },
            index=["g_C_H", "g_C_E_F", "g_D_G"],
        )
        gene_df_calc = reaction_to_gene_df(self.test_model, rxn_df)
        self.assertCountEqual(
            gene_df_calc["reactions"].to_list(), rxn_df.index.to_list()
        )
        self.assertTrue((gene_df_calc == gene_df_expected).all().all())

    def test_gene_to_reaction(self):
        gene_df = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["g_C_H", "g_C_E_F", "g_D_G"]
        )
        rxn_df_expected = pd.DataFrame(
            {"genes": ["g_C_H", "g_C_E_F", "g_D_G"], "A": [1, 2, 3], "B": [4, 5, 6]},
            index=["r_C_H", "r_C_E_F", "r_D_G"],
        )
        rxn_df_calc = gene_to_reaction_df(self.test_model, gene_df)
        self.assertCountEqual(rxn_df_calc["genes"].to_list(), gene_df.index.to_list())
        self.assertTrue((rxn_df_calc == rxn_df_expected).all().all())


if __name__ == "__main__":
    unittest.main()

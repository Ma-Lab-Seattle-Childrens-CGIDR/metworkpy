# Standard Library Imports
import unittest

# External Imports
from cobra.core.configuration import Configuration

# Local Imports
from metworkpy.utils.translate import (
    reaction_to_gene_ids,
    gene_to_reaction_ids,
    gene_to_reaction_list,
    reaction_to_gene_list,
)
from metworkpy.examples import get_example_model


class TestIdTranslate(unittest.TestCase):
    test_model = get_example_model()

    @classmethod
    def setUpClass(cls):
        Configuration().solver = "glpk"

    def test_reaction_to_gene_ids(self):
        # Test for reaction with no genes
        self.assertSetEqual(
            reaction_to_gene_ids(
                model=self.test_model, reaction="A_import", essential=False
            ),
            set(),
        )

        # Test for single gene reaction
        self.assertSetEqual(
            reaction_to_gene_ids(
                model=self.test_model, reaction="R_C_D__J", essential=False
            ),
            {"g004"},
        )
        # Test for multiple gene reaction
        self.assertSetEqual(
            reaction_to_gene_ids(
                model=self.test_model, reaction="R_N__U", essential=False
            ),
            {"g011", "g013"},
        )
        self.assertSetEqual(
            reaction_to_gene_ids(
                model=self.test_model, reaction="R_N__U", essential=True
            ),
            {"g011", "g013"},
        )
        self.assertSetEqual(
            reaction_to_gene_ids(
                model=self.test_model, reaction="R_N__T", essential=True
            ),
            set(),
        )
        self.assertSetEqual(
            reaction_to_gene_ids(
                model=self.test_model, reaction="R_N__T", essential=False
            ),
            {"g011", "g012"},
        )

    def test_gene_to_reaction_ids(self):
        # Test for gene associated with single reaction
        self.assertSetEqual(
            gene_to_reaction_ids(
                model=self.test_model, gene="g001", essential=False
            ),
            {"R_A_B__G_H"},
        )
        # Test for gene associated with multiple reactions
        self.assertSetEqual(
            gene_to_reaction_ids(
                model=self.test_model, gene="g011", essential=False
            ),
            {"R_N__U", "R_N__T"},
        )
        self.assertSetEqual(
            gene_to_reaction_ids(
                model=self.test_model, gene="g011", essential=True
            ),
            {"R_N__U"},
        )


class TestListTranslate(unittest.TestCase):
    test_model = get_example_model()

    @classmethod
    def setUpClass(cls):
        Configuration().solver = "glpk"

    def test_gene_to_reaction_list(self):
        self.assertCountEqual(
            gene_to_reaction_list(
                model=self.test_model,
                gene_list=["g001", "g002", "g008"],
                essential=False,
            ),
            ["R_A_B__G_H", "R_C_H__I", "R_G_K__L", "R_I__P"],
        )
        self.assertCountEqual(
            gene_to_reaction_list(
                model=self.test_model,
                gene_list=["g001", "g002", "g008"],
                essential=True,
            ),
            ["R_A_B__G_H", "R_G_K__L", "R_I__P"],
        )

    def test_reaction_to_gene_list(self):
        self.assertCountEqual(
            reaction_to_gene_list(
                model=self.test_model,
                reaction_list=["R_A_B__G_H", "R_C_H__I", "R_C_D__J"],
                essential=False,
            ),
            ["g001", "g002", "g003", "g004"],
        )
        self.assertCountEqual(
            reaction_to_gene_list(
                model=self.test_model,
                reaction_list=["R_A_B__G_H", "R_C_H__I", "R_C_D__J"],
                essential=True,
            ),
            ["g001", "g004"],
        )


if __name__ == "__main__":
    unittest.main()

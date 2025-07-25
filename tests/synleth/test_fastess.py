# Standard Library Imports
import pathlib
import unittest

# External Imports
import cobra

# Local Imports
import metworkpy
from metworkpy.synleth.fastess import fast_ess_genes, fast_ess_rxn

# Environment Setup
BASE_PATH = pathlib.Path(__file__).parent.parent.absolute()
# Set cobra to use GLPK by default since that is bundled with
# cobrapy, and so is always available
cobra.Configuration().solver = "glpk"


class TestFastEss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.textbook_model = metworkpy.read_model(
            BASE_PATH / "data" / "textbook_model.json"
        )

    def test_fast_ess_rxn(self):
        cobra_rxn_ko = [
            rxn.id
            for rxn in cobra.flux_analysis.find_essential_reactions(
                model=self.textbook_model.copy(),
                threshold=0.1 * self.textbook_model.slim_optimize(),
                processes=1,
            )
        ]
        metworkpy_rxn_ko = fast_ess_rxn(
            model=self.textbook_model.copy(),
            essentiality_threshold=0.1,
            pfba_tolerance=1e-7,
            processes=1,
        )
        self.assertCountEqual(metworkpy_rxn_ko, cobra_rxn_ko)

    def test_fast_ess_gene(self):
        cobra_gene_ko = [
            gene.id
            for gene in cobra.flux_analysis.find_essential_genes(
                model=self.textbook_model.copy(),
                threshold=0.1 * self.textbook_model.slim_optimize(),
                processes=1,
            )
        ]
        metworkpy_gene_ko = fast_ess_genes(
            model=self.textbook_model.copy(),
            essentiality_threshold=0.1,
            pfba_tolerance=1e-7,
            processes=1,
        )
        self.assertCountEqual(metworkpy_gene_ko, cobra_gene_ko)

# Imports
# Standard library imports
import os
import pathlib
import unittest

# External Imports
from cobra.core.configuration import Configuration
import numpy as np

# Local Imports
from metworkpy.utils.models import read_model
from metworkpy.network.network_construction import _adj_mat_ud_uw


def setup(cls):
    Configuration().solver = "glpk"
    cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
    cls.test_model = read_model(cls.data_path / "test_model.xml")


class TestAdjMatUdUw(unittest.TestCase):
    test_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        cls.adj_mat = _adj_mat_ud_uw(cls.test_model)

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(self.adj_mat.shape, (num_rxns+num_metabolites,
                                                   num_rxns+num_metabolites))

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertTrue(np.isclose(data, 1.).all())


if __name__ == '__main__':
    unittest.main()

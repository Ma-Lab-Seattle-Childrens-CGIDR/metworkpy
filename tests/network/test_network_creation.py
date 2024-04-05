# Imports
# Standard library imports
import pathlib
import unittest

# External Imports
from cobra.core.configuration import Configuration
from cobra.flux_analysis import flux_variability_analysis
import numpy as np
from scipy.sparse import csc_array, csr_array

# Local Imports
from metworkpy.utils.models import read_model
from metworkpy.network.network_construction import (_adj_mat_ud_uw, _adj_mat_d_uw,
                                                    _adj_mat_d_w, _adj_mat_ud_w)


def setup(cls):
    Configuration().solver = "glpk"
    cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
    cls.test_model = read_model(cls.data_path / "test_model.xml")
    cls.tiny_model = read_model(cls.data_path / "tiny_model.json")


class TestAdjMatUdUw(unittest.TestCase):
    test_model = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        cls.adj_mat = _adj_mat_ud_uw(cls.test_model)
        cls.tiny_adj_mat = _adj_mat_ud_uw(cls.tiny_model)
        cls.tiny_known = csr_array([
            # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
            [0, 0, 0, 1, 1, 0, 0],  # A
            [0, 0, 0, 1, 0, 1, 0],  # B
            [0, 0, 0, 1, 0, 0, 1],  # C
            [1, 1, 1, 0, 0, 0, 0],  # R_A_B_C
            [1, 0, 0, 0, 0, 0, 0],  # R_A_ex
            [0, 1, 0, 0, 0, 0, 0],  # R_B_ex
            [0, 0, 1, 0, 0, 0, 0],  # R_C_ex
        ])

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(self.adj_mat.shape, (num_rxns + num_metabolites,
                                                   num_rxns + num_metabolites))

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertTrue(np.isclose(data, 1.).all())
        # Should all be positive
        self.assertTrue((data >= 0.).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(np.isclose(self.tiny_adj_mat.toarray(),
                                   self.tiny_known.toarray()).all())



class TestAdjMatDUw(unittest.TestCase):
    test_model = None
    tiny_adj_mat = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        cls.adj_mat = _adj_mat_d_uw(cls.test_model)
        cls.tiny_adj_mat = _adj_mat_d_uw(cls.tiny_model)
        cls.tiny_known = csr_array([
            # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
            [0, 0, 0, 1, 1, 0, 0],  # A
            [0, 0, 0, 1, 0, 1, 0],  # B
            [0, 0, 0, 1, 0, 0, 1],  # C
            [1, 1, 1, 0, 0, 0, 0],  # R_A_B_C
            [1, 0, 0, 0, 0, 0, 0],  # R_A_ex
            [0, 1, 0, 0, 0, 0, 0],  # R_B_ex
            [0, 0, 0, 0, 0, 0, 0],  # R_C_ex
        ])

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(self.adj_mat.shape, (num_rxns + num_metabolites,
                                                   num_rxns + num_metabolites))

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertTrue(np.isclose(data, 1.).all())
        # Should all be positive
        self.assertTrue((data >= 0.).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(np.isclose(self.tiny_adj_mat.toarray(),
                                   self.tiny_known.toarray()).all())


class TestAdjMatDW(unittest.TestCase):
    test_model = None
    data_path = None
    tiny_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        fva = flux_variability_analysis(model=cls.test_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))

        cls.adj_mat = _adj_mat_d_w(cls.test_model, rxn_bounds=(rxn_min, rxn_max))

        fva = flux_variability_analysis(model=cls.tiny_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))

        cls.tiny_adj_mat = _adj_mat_d_w(cls.tiny_model, rxn_bounds=(rxn_min, rxn_max))

        cls.tiny_known = csr_array([
            # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
            [0, 0, 0, 50, 0, 0, 0],  # A
            [0, 0, 0, 50, 0, 0, 0],  # B
            [0, 0, 0, 0, 0, 0, 50],  # C
            [0, 0, 50, 0, 0, 0, 0],  # R_A_B_C
            [50, 0, 0, 0, 0, 0, 0],  # R_A_ex
            [0, 50, 0, 0, 0, 0, 0],  # R_B_ex
            [0, 0, 0, 0, 0, 0, 0],  # R_C_ex
        ])

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(self.adj_mat.shape, (num_rxns + num_metabolites,
                                                   num_rxns + num_metabolites))

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertFalse(np.isclose(data, 1.).all())
        # Should all be positive
        self.assertTrue((data >= 0.).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(np.isclose(self.tiny_adj_mat.toarray(),
                                   self.tiny_known.toarray()).all())


class TestAdjMatUdW(unittest.TestCase):
    test_model = None
    data_path = None
    tiny_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        fva = flux_variability_analysis(model=cls.test_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))

        cls.adj_mat = _adj_mat_ud_w(cls.test_model, rxn_bounds=(rxn_min, rxn_max))

        fva = flux_variability_analysis(model=cls.tiny_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))

        cls.tiny_adj_mat = _adj_mat_ud_w(cls.tiny_model, rxn_bounds=(rxn_min, rxn_max))

        cls.tiny_known = csr_array([
            # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
            [0, 0, 0, 50, 50, 0, 0],  # A
            [0, 0, 0, 50, 0, 50, 0],  # B
            [0, 0, 0, 50, 0, 0, 50],  # C
            [50, 50, 50, 0, 0, 0, 0],  # R_A_B_C
            [50, 0, 0, 0, 0, 0, 0],  # R_A_ex
            [0, 50, 0, 0, 0, 0, 0],  # R_B_ex
            [0, 0, 50, 0, 0, 0, 0],  # R_C_ex
        ])

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(self.adj_mat.shape, (num_rxns + num_metabolites,
                                                   num_rxns + num_metabolites))

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertFalse(np.isclose(data, 1.).all())
        # Should all be positive
        self.assertTrue((data >= 0.).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(np.isclose(self.tiny_adj_mat.toarray(),
                                   self.tiny_known.toarray()).all())


if __name__ == '__main__':
    unittest.main()

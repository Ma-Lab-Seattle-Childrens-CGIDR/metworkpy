# Imports
# Standard library imports
import itertools
import pathlib
import unittest

# External Imports
import cobra
from cobra.core.configuration import Configuration
from cobra.flux_analysis import flux_variability_analysis
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csc_array, csr_array

# Local Imports
from metworkpy.utils.models import read_model
from metworkpy.network.network_construction import (
    _adj_mat_ud_uw,
    _adj_mat_d_uw,
    _adj_mat_d_w_flux,
    _adj_mat_d_w_stoichiometry,
    _adj_mat_ud_w_flux,
    _adj_mat_ud_w_stoichiometry,
    create_adjacency_matrix,
    create_metabolic_network,
    create_mutual_information_network,
)
from metworkpy.information import mi_network_adjacency_matrix


# region Metabolic Network
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
        cls.tiny_known = csr_array(
            [
                # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
                [0, 0, 0, 1, 1, 0, 0],  # A
                [0, 0, 0, 1, 0, 1, 0],  # B
                [0, 0, 0, 1, 0, 0, 1],  # C
                [1, 1, 1, 0, 0, 0, 0],  # R_A_B_C
                [1, 0, 0, 0, 0, 0, 0],  # R_A_ex
                [0, 1, 0, 0, 0, 0, 0],  # R_B_ex
                [0, 0, 1, 0, 0, 0, 0],  # R_C_ex
            ]
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape, (num_rxns + num_metabolites, num_rxns + num_metabolites)
        )

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertTrue(np.isclose(data, 1.0).all())
        # Should all be positive
        self.assertTrue((data >= 0.0).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(
            np.isclose(self.tiny_adj_mat.toarray(), self.tiny_known.toarray()).all()
        )


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
        cls.tiny_known = csr_array(
            [
                # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
                [0, 0, 0, 1, 1, 0, 0],  # A
                [0, 0, 0, 1, 0, 1, 0],  # B
                [0, 0, 0, 1, 0, 0, 1],  # C
                [1, 1, 1, 0, 0, 0, 0],  # R_A_B_C
                [1, 0, 0, 0, 0, 0, 0],  # R_A_ex
                [0, 1, 0, 0, 0, 0, 0],  # R_B_ex
                [0, 0, 0, 0, 0, 0, 0],  # R_C_ex
            ]
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape, (num_rxns + num_metabolites, num_rxns + num_metabolites)
        )

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertTrue(np.isclose(data, 1.0).all())
        # Should all be positive
        self.assertTrue((data >= 0.0).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(
            np.isclose(self.tiny_adj_mat.toarray(), self.tiny_known.toarray()).all()
        )


class TestAdjMatDWFlux(unittest.TestCase):
    test_model = None
    data_path = None
    tiny_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        fva = flux_variability_analysis(model=cls.test_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))

        cls.adj_mat = _adj_mat_d_w_flux(cls.test_model, rxn_bounds=(rxn_min, rxn_max))

        fva = flux_variability_analysis(model=cls.tiny_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))

        cls.tiny_adj_mat = _adj_mat_d_w_flux(
            cls.tiny_model, rxn_bounds=(rxn_min, rxn_max)
        )

        cls.tiny_known = csr_array(
            [
                # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
                [0, 0, 0, 50, 0, 0, 0],  # A
                [0, 0, 0, 50, 0, 0, 0],  # B
                [0, 0, 0, 0, 0, 0, 50],  # C
                [0, 0, 50, 0, 0, 0, 0],  # R_A_B_C
                [50, 0, 0, 0, 0, 0, 0],  # R_A_ex
                [0, 50, 0, 0, 0, 0, 0],  # R_B_ex
                [0, 0, 0, 0, 0, 0, 0],  # R_C_ex
            ]
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape, (num_rxns + num_metabolites, num_rxns + num_metabolites)
        )

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertFalse(np.isclose(data, 1.0).all())
        # Should all be positive
        self.assertTrue((data >= 0.0).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(
            np.isclose(self.tiny_adj_mat.toarray(), self.tiny_known.toarray()).all()
        )


class TestAdjMatDWStoichiometry(unittest.TestCase):
    test_model = None
    tiny_adj_mat = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        cls.adj_mat = _adj_mat_d_w_stoichiometry(cls.test_model)
        cls.tiny_adj_mat = _adj_mat_d_w_stoichiometry(cls.tiny_model)
        cls.tiny_known = csr_array(
            [
                # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
                [0, 0, 0, 1, 1, 0, 0],  # A
                [0, 0, 0, 1, 0, 1, 0],  # B
                [0, 0, 0, 1, 0, 0, 1],  # C
                [1, 1, 1, 0, 0, 0, 0],  # R_A_B_C
                [1, 0, 0, 0, 0, 0, 0],  # R_A_ex
                [0, 1, 0, 0, 0, 0, 0],  # R_B_ex
                [0, 0, 0, 0, 0, 0, 0],  # R_C_ex
            ]
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape, (num_rxns + num_metabolites, num_rxns + num_metabolites)
        )

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertTrue(np.isclose(data, 1.0).all())
        # Should all be positive
        self.assertTrue((data >= 0.0).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(
            np.isclose(self.tiny_adj_mat.toarray(), self.tiny_known.toarray()).all()
        )


class TestAdjMatUdWFlux(unittest.TestCase):
    test_model = None
    data_path = None
    tiny_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        fva = flux_variability_analysis(model=cls.test_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))

        cls.adj_mat = _adj_mat_ud_w_flux(cls.test_model, rxn_bounds=(rxn_min, rxn_max))

        fva = flux_variability_analysis(model=cls.tiny_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))

        cls.tiny_adj_mat = _adj_mat_ud_w_flux(
            cls.tiny_model, rxn_bounds=(rxn_min, rxn_max)
        )

        cls.tiny_known = csr_array(
            [
                # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
                [0, 0, 0, 50, 50, 0, 0],  # A
                [0, 0, 0, 50, 0, 50, 0],  # B
                [0, 0, 0, 50, 0, 0, 50],  # C
                [50, 50, 50, 0, 0, 0, 0],  # R_A_B_C
                [50, 0, 0, 0, 0, 0, 0],  # R_A_ex
                [0, 50, 0, 0, 0, 0, 0],  # R_B_ex
                [0, 0, 50, 0, 0, 0, 0],  # R_C_ex
            ]
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape, (num_rxns + num_metabolites, num_rxns + num_metabolites)
        )

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertFalse(np.isclose(data, 1.0).all())
        # Should all be positive
        self.assertTrue((data >= 0.0).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(
            np.isclose(self.tiny_adj_mat.toarray(), self.tiny_known.toarray()).all()
        )


class TestAdjMatUdWStoichiometry(unittest.TestCase):
    test_model = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        cls.adj_mat = _adj_mat_ud_w_stoichiometry(cls.test_model)
        cls.tiny_adj_mat = _adj_mat_ud_w_stoichiometry(cls.tiny_model)
        cls.tiny_known = csr_array(
            [
                # A B C R_A_B_C R_A_ex R_B_ex R_C_ex
                [0, 0, 0, 1, 1, 0, 0],  # A
                [0, 0, 0, 1, 0, 1, 0],  # B
                [0, 0, 0, 1, 0, 0, 1],  # C
                [1, 1, 1, 0, 0, 0, 0],  # R_A_B_C
                [1, 0, 0, 0, 0, 0, 0],  # R_A_ex
                [0, 1, 0, 0, 0, 0, 0],  # R_B_ex
                [0, 0, 1, 0, 0, 0, 0],  # R_C_ex
            ]
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape, (num_rxns + num_metabolites, num_rxns + num_metabolites)
        )

    def test_data(self):
        data = self.adj_mat.data
        # Should all be 1 (or very close)
        self.assertTrue(np.isclose(data, 1.0).all())
        # Should all be positive
        self.assertTrue((data >= 0.0).all())

    def test_type(self):
        self.assertIsInstance(self.adj_mat, csr_array)

    def test_known(self):
        self.assertTrue(
            np.isclose(self.tiny_adj_mat.toarray(), self.tiny_known.toarray()).all()
        )


class TestCreateAdjacencyMatrix(unittest.TestCase):
    test_model = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        fva = flux_variability_analysis(model=cls.test_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))
        cls.test_model_rxn_bounds = (rxn_min, rxn_max)

        fva = flux_variability_analysis(model=cls.tiny_model)
        rxn_max = csc_array(fva["maximum"].values.reshape(-1, 1))
        rxn_min = csc_array(fva["minimum"].values.reshape(-1, 1))
        cls.tiny_model_rxn_bounds = (rxn_min, rxn_max)

    def test_undirected_unweighted(self):
        adj_mat, _, _ = create_adjacency_matrix(
            model=self.test_model, directed=False, weighted=False, out_format="csr"
        )
        adj_mat_known = _adj_mat_ud_uw(model=self.test_model)
        self.assertTrue(np.isclose(adj_mat.toarray(), adj_mat_known.toarray()).all())

    def test_directed_unweighted(self):
        adj_mat, _, _ = create_adjacency_matrix(
            model=self.test_model, directed=True, weighted=False, out_format="csr"
        )
        adj_mat_known = _adj_mat_d_uw(model=self.test_model)
        self.assertTrue(np.isclose(adj_mat.toarray(), adj_mat_known.toarray()).all())

    def test_directed_weighted_flux(self):
        adj_mat, _, _ = create_adjacency_matrix(
            model=self.test_model,
            directed=True,
            weighted=True,
            weight_by="flux",
            out_format="csr",
        )
        adj_mat_known = _adj_mat_d_w_flux(
            model=self.test_model, rxn_bounds=self.test_model_rxn_bounds
        )
        self.assertTrue(np.isclose(adj_mat.toarray(), adj_mat_known.toarray()).all())

    def test_directed_weighted_stoichiometry(self):
        adj_mat, _, _ = create_adjacency_matrix(
            model=self.test_model,
            directed=True,
            weighted=True,
            weight_by="stoichiometry",
            out_format="csr",
        )
        adj_mat_known = _adj_mat_d_w_stoichiometry(model=self.test_model)
        self.assertTrue(np.isclose(adj_mat.toarray(), adj_mat_known.toarray()).all())

    def test_undirected_weighted_flux(self):
        adj_mat, _, _ = create_adjacency_matrix(
            model=self.test_model,
            directed=False,
            weighted=True,
            weight_by="flux",
            out_format="csr",
        )
        adj_mat_known = _adj_mat_ud_w_flux(
            model=self.test_model, rxn_bounds=self.test_model_rxn_bounds
        )
        self.assertTrue(np.isclose(adj_mat.toarray(), adj_mat_known.toarray()).all())

    def test_undirected_weighted_stoichiometry(self):
        adj_mat, _, _ = create_adjacency_matrix(
            model=self.test_model,
            directed=False,
            weighted=True,
            weight_by="stoichiometry",
            out_format="csr",
        )
        adj_mat_known = _adj_mat_ud_w_stoichiometry(model=self.test_model)
        self.assertTrue(np.isclose(adj_mat.toarray(), adj_mat_known.toarray()).all())

    def test_out_format(self):
        for out_form in ["dok", "lil", "csc", "csr"]:
            adj_mat, _, _ = create_adjacency_matrix(
                model=self.test_model,
                directed=False,
                weighted=False,
                out_format=out_form,
            )
            self.assertEqual(out_form, adj_mat.format)
        adj_mat, _, _ = create_adjacency_matrix(
            model=self.test_model, directed=False, weighted=False, out_format="frame"
        )
        self.assertIsInstance(adj_mat, pd.DataFrame)

    def test_loopless(self):
        adj_mat_loopless, _, _ = create_adjacency_matrix(
            model=self.tiny_model,
            directed=True,
            weighted=True,
            weight_by="flux",
            out_format="csr",
            loopless=True,
        )
        # Since there are no loops, should be the same
        adj_mat, _, _ = create_adjacency_matrix(
            model=self.tiny_model,
            directed=True,
            weighted=True,
            weight_by="flux",
            out_format="csr",
            loopless=False,
        )
        self.assertTrue(np.isclose(adj_mat.toarray(), adj_mat_loopless.toarray()).all())


class TestCreateNetwork(unittest.TestCase):
    test_model = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_directed_unweighted(self):
        test_network = create_metabolic_network(
            model=self.test_model, weighted=False, directed=True
        )
        self.assertIsInstance(test_network, nx.DiGraph)
        for start, stop, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 1)
        tiny_network = create_metabolic_network(
            model=self.tiny_model, weighted=False, directed=True
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 1)
        with self.assertRaises(KeyError):
            _ = tiny_network["R_C_ex"]["C"]

    def test_undirected_unweighted(self):
        test_network = create_metabolic_network(
            model=self.test_model, weighted=False, directed=False
        )
        self.assertIsInstance(test_network, nx.Graph)
        for start, stop, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 1)
        tiny_network = create_metabolic_network(
            model=self.tiny_model, weighted=False, directed=False
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 1)
        self.assertEqual(tiny_network["R_C_ex"]["C"]["weight"], 1)

    def test_directed_weighted_stoichiometry(self):
        test_network = create_metabolic_network(
            model=self.test_model,
            weighted=True,
            directed=True,
            weight_by="stoichiometry",
        )
        self.assertIsInstance(test_network, nx.DiGraph)
        for start, stop, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 1)
        tiny_network = create_metabolic_network(
            model=self.tiny_model,
            weighted=True,
            directed=True,
            weight_by="stoichiometry",
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 1)
        with self.assertRaises(KeyError):
            _ = tiny_network["R_C_ex"]["C"]

    def test_undirected_weighted_stoichiometry(self):
        test_network = create_metabolic_network(
            model=self.test_model,
            weighted=True,
            directed=False,
            weight_by="stoichiometry",
        )
        self.assertIsInstance(test_network, nx.Graph)
        for start, stop, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 1)
        tiny_network = create_metabolic_network(
            model=self.tiny_model,
            weighted=True,
            directed=False,
            weight_by="stoichiometry",
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 1)
        self.assertEqual(tiny_network["R_C_ex"]["C"]["weight"], 1)

    def test_directed_weighted_flux(self):
        test_network = create_metabolic_network(
            model=self.test_model, weighted=True, directed=True, weight_by="flux"
        )
        self.assertIsInstance(test_network, nx.DiGraph)
        for start, stop, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 50)
        tiny_network = create_metabolic_network(
            model=self.tiny_model, weighted=True, directed=True, weight_by="flux"
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 50)
        with self.assertRaises(KeyError):
            _ = tiny_network["R_C_ex"]["C"]

    def test_undirected_weighted_flux(self):
        test_network = create_metabolic_network(
            model=self.test_model, weighted=True, directed=False, weight_by="flux"
        )
        self.assertIsInstance(test_network, nx.Graph)
        for start, stop, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 50)
        tiny_network = create_metabolic_network(
            model=self.tiny_model, weighted=True, directed=False, weight_by="flux"
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 50)
        self.assertEqual(tiny_network["R_C_ex"]["C"]["weight"], 50)


# endregion Metabolic Network

# region Mutual Information Network


class TestMutualInformationNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Configuration().solver = "glpk"
        cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
        cls.test_model = read_model(cls.data_path / "test_model.xml")

    def test_create_mutual_information_network(self):
        test_network = create_mutual_information_network(
            model=self.test_model, n_samples=1000, n_neighbors=3
        )
        # More proximate reactions should have greater mutual information
        self.assertGreater(
            test_network.get_edge_data("r_A_B_D_E", "r_D_G")["weight"],
            test_network.get_edge_data("r_A_B_D_E", "R_H_e_ex")["weight"],
        )
        for rxn in self.test_model.reactions:
            test_network.has_node(rxn.id)
        test_samples = cobra.sampling.sample(self.test_model, n=1000)
        mi_adj_mat = mi_network_adjacency_matrix(test_samples, n_neighbors=3)
        test_network = create_mutual_information_network(
            flux_samples=test_samples, n_neighbors=3
        )
        rxn_ids = self.test_model.reactions.list_attr("id")
        for i, j in itertools.combinations(range(mi_adj_mat.shape[1]), 2):
            self.assertAlmostEqual(
                mi_adj_mat[i, j],
                test_network.get_edge_data(rxn_ids[i], rxn_ids[j])["weight"],
            )


# endregion Mutual Information Network

if __name__ == "__main__":
    unittest.main()

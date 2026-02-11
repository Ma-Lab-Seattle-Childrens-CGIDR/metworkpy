# Imports
# Standard library imports
import itertools
import pathlib
import unittest

# External Imports
import cobra
from cobra.core.configuration import Configuration
import networkx as nx
import pandas as pd

# Local Imports
from metworkpy.utils.models import read_model
from metworkpy.network.network_construction import (
    _create_adj_matrix_d_uw,
    _create_adj_matrix_ud_uw,
    _create_adj_matrix_d_w_flux,
    _create_adj_matrix_ud_w_flux,
    _create_adj_matrix_d_w_stoich,
    _create_adj_matrix_ud_w_stoich,
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
        cls.adj_mat = _create_adj_matrix_ud_uw(cls.test_model, threshold=0.0)
        cls.tiny_adj_mat = _create_adj_matrix_ud_uw(
            cls.tiny_model, threshold=0.0
        )
        cls.tiny_known = pd.DataFrame(
            [
                #  R_A_B_C R_A_ex R_B_ex R_C_ex A B C
                [0, 0, 0, 0, 1, 1, 1],  # R_A_B_C
                [0, 0, 0, 0, 1, 0, 0],  # R_A_ex
                [0, 0, 0, 0, 0, 1, 0],  # R_B_ex
                [0, 0, 0, 0, 0, 0, 1],  # R_C_ex
                [1, 1, 0, 0, 0, 0, 0],  # A
                [1, 0, 1, 0, 0, 0, 0],  # B
                [1, 0, 0, 1, 0, 0, 0],  # C
            ],
            index=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            columns=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            dtype=bool,
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)


class TestAdjMatDUw(unittest.TestCase):
    test_model = None
    tiny_adj_mat = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        cls.adj_mat = _create_adj_matrix_d_uw(cls.test_model, threshold=0)
        cls.tiny_adj_mat = _create_adj_matrix_d_uw(cls.tiny_model, threshold=0)
        cls.tiny_known = pd.DataFrame(
            [
                #  R_A_B_C R_A_ex R_B_ex R_C_ex A B C
                [0, 0, 0, 0, 1, 1, 1],  # R_A_B_C
                [0, 0, 0, 0, 1, 0, 0],  # R_A_ex
                [0, 0, 0, 0, 0, 1, 0],  # R_B_ex
                [0, 0, 0, 0, 0, 0, 0],  # R_C_ex
                [1, 1, 0, 0, 0, 0, 0],  # A
                [1, 0, 1, 0, 0, 0, 0],  # B
                [1, 0, 0, 1, 0, 0, 0],  # C
            ],
            index=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            columns=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            dtype=bool,
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)


class TestAdjMatDWFlux(unittest.TestCase):
    test_model = None
    data_path = None
    tiny_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

        cls.adj_mat = _create_adj_matrix_d_w_flux(
            cls.test_model, threshold=0.0
        )

        cls.tiny_adj_mat = _create_adj_matrix_d_w_flux(
            cls.tiny_model, threshold=0.0
        )

        cls.tiny_known = pd.DataFrame(
            [
                #  R_A_B_C R_A_ex R_B_ex R_C_ex A B C
                [0, 0, 0, 0, 0, 0, 50],  # R_A_B_C
                [0, 0, 0, 0, 50, 0, 0],  # R_A_ex
                [0, 0, 0, 0, 0, 50, 0],  # R_B_ex
                [0, 0, 0, 0, 0, 0, 0],  # R_C_ex
                [50, 0, 0, 0, 0, 0, 0],  # A
                [50, 0, 0, 0, 0, 0, 0],  # B
                [0, 0, 0, 50, 0, 0, 0],  # C
            ],
            index=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            columns=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            dtype=float,
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)


class TestAdjMatDWStoichiometry(unittest.TestCase):
    test_model = None
    tiny_adj_mat = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        cls.adj_mat = _create_adj_matrix_d_w_stoich(
            cls.test_model, threshold=0.0
        )
        cls.tiny_adj_mat = _create_adj_matrix_d_w_stoich(
            cls.tiny_model, threshold=0.0
        )
        cls.tiny_known = pd.DataFrame(
            [
                #  R_A_B_C R_A_ex R_B_ex R_C_ex A B C
                [0, 0, 0, 0, 1, 1, 1],  # R_A_B_C
                [0, 0, 0, 0, 1, 0, 0],  # R_A_ex
                [0, 0, 0, 0, 0, 1, 0],  # R_B_ex
                [0, 0, 0, 0, 0, 0, 0],  # R_C_ex
                [1, 1, 0, 0, 0, 0, 0],  # A
                [1, 0, 1, 0, 0, 0, 0],  # B
                [1, 0, 0, 1, 0, 0, 0],  # C
            ],
            index=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            columns=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            dtype=float,
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)


class TestAdjMatUdWFlux(unittest.TestCase):
    test_model = None
    data_path = None
    tiny_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

        cls.adj_mat = _create_adj_matrix_ud_w_flux(
            cls.test_model, threshold=0.0
        )

        cls.tiny_adj_mat = _create_adj_matrix_ud_w_flux(
            cls.tiny_model, threshold=0.0
        )

        cls.tiny_known = pd.DataFrame(
            [
                #  R_A_B_C R_A_ex R_B_ex R_C_ex A B C
                [0, 0, 0, 0, 50, 50, 50],  # R_A_B_C
                [0, 0, 0, 0, 50, 0, 0],  # R_A_ex
                [0, 0, 0, 0, 0, 50, 0],  # R_B_ex
                [0, 0, 0, 0, 0, 0, 50],  # R_C_ex
                [50, 50, 0, 0, 0, 0, 0],  # A
                [50, 0, 50, 0, 0, 0, 0],  # B
                [50, 0, 0, 50, 0, 0, 0],  # C
            ],
            index=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            columns=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            dtype=float,
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)


class TestAdjMatUdWStoichiometry(unittest.TestCase):
    test_model = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        cls.adj_mat = _create_adj_matrix_ud_w_stoich(
            cls.test_model, threshold=0.0
        )
        cls.tiny_adj_mat = _create_adj_matrix_ud_w_stoich(
            cls.tiny_model, threshold=0.0
        )
        cls.tiny_known = pd.DataFrame(
            [
                #  R_A_B_C R_A_ex R_B_ex R_C_ex A B C
                [0, 0, 0, 0, 1, 1, 1],  # R_A_B_C
                [0, 0, 0, 0, 1, 0, 0],  # R_A_ex
                [0, 0, 0, 0, 0, 1, 0],  # R_B_ex
                [0, 0, 0, 0, 0, 0, 1],  # R_C_ex
                [1, 1, 0, 0, 0, 0, 0],  # A
                [1, 0, 1, 0, 0, 0, 0],  # B
                [1, 0, 0, 1, 0, 0, 0],  # C
            ],
            index=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            columns=pd.Index(
                [
                    "R_A_B_C",
                    "R_A_ex",
                    "R_B_ex",
                    "R_C_ex",
                    "A",
                    "B",
                    "C",
                ]
            ),
            dtype=float,
        )

    def test_shape(self):
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)


class TestCreateAdjacencyMatrix(unittest.TestCase):
    test_model = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_undirected_unweighted(self):
        adj_mat = create_adjacency_matrix(
            model=self.test_model,
            directed=False,
            weighted=False,
        )
        adj_mat_known = _create_adj_matrix_ud_uw(
            model=self.test_model, threshold=0.0
        )
        pd.testing.assert_frame_equal(adj_mat_known, adj_mat)

    def test_directed_unweighted(self):
        adj_mat = create_adjacency_matrix(
            model=self.test_model,
            directed=True,
            weighted=False,
        )
        adj_mat_known = _create_adj_matrix_d_uw(
            model=self.test_model, threshold=0.0
        )
        pd.testing.assert_frame_equal(adj_mat_known, adj_mat)

    def test_directed_weighted_flux(self):
        adj_mat = create_adjacency_matrix(
            model=self.test_model,
            directed=True,
            weighted=True,
            weight_by="flux",
            threshold=0.0,
        )
        adj_mat_known = _create_adj_matrix_d_w_flux(
            model=self.test_model, threshold=0.0
        )
        pd.testing.assert_frame_equal(adj_mat_known, adj_mat)

    def test_directed_weighted_stoichiometry(self):
        adj_mat = create_adjacency_matrix(
            model=self.test_model,
            directed=True,
            weighted=True,
            weight_by="stoichiometry",
            threshold=0.0,
        )
        adj_mat_known = _create_adj_matrix_d_w_stoich(
            model=self.test_model, threshold=0.0
        )
        pd.testing.assert_frame_equal(adj_mat_known, adj_mat)

    def test_undirected_weighted_flux(self):
        adj_mat = create_adjacency_matrix(
            model=self.test_model,
            directed=False,
            weighted=True,
            weight_by="flux",
            threshold=0.0,
        )
        adj_mat_known = _create_adj_matrix_ud_w_flux(
            model=self.test_model, threshold=0.0
        )
        pd.testing.assert_frame_equal(adj_mat_known, adj_mat)

    def test_undirected_weighted_stoichiometry(self):
        adj_mat = create_adjacency_matrix(
            model=self.test_model,
            directed=False,
            weighted=True,
            weight_by="stoichiometry",
            threshold=0.0,
        )
        adj_mat_known = _create_adj_matrix_ud_w_stoich(
            model=self.test_model, threshold=0.0
        )
        pd.testing.assert_frame_equal(adj_mat_known, adj_mat)


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
            model=self.test_model,
            weighted=True,
            directed=True,
            weight_by="flux",
        )
        self.assertIsInstance(test_network, nx.DiGraph)
        for start, stop, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 50)
        tiny_network = create_metabolic_network(
            model=self.tiny_model,
            weighted=True,
            directed=True,
            weight_by="flux",
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 50)
        with self.assertRaises(KeyError):
            _ = tiny_network["R_C_ex"]["C"]

    def test_undirected_weighted_flux(self):
        test_network = create_metabolic_network(
            model=self.test_model,
            weighted=True,
            directed=False,
            weight_by="flux",
        )
        self.assertIsInstance(test_network, nx.Graph)
        for start, stop, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 50)
        tiny_network = create_metabolic_network(
            model=self.tiny_model,
            weighted=True,
            directed=False,
            weight_by="flux",
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 50)
        self.assertEqual(tiny_network["R_C_ex"]["C"]["weight"], 50)

    def test_bipartite(self):
        textbook_model = cobra.io.load_model(
            "textbook"
        )  # ecoli core metabolism
        # Test for not directed, not weighted
        textbook_network = create_metabolic_network(
            model=textbook_model, weighted=False, directed=False
        )
        self.assertTrue(nx.is_bipartite(textbook_network))
        self.assertTrue(
            nx.algorithms.bipartite.is_bipartite_node_set(
                textbook_network, textbook_model.reactions.list_attr("id")
            )
        )
        self.assertTrue(
            nx.algorithms.bipartite.is_bipartite_node_set(
                textbook_network, textbook_model.metabolites.list_attr("id")
            )
        )
        # Test for directed, not weighted
        textbook_network = create_metabolic_network(
            model=textbook_model, weighted=False, directed=True
        )
        self.assertTrue(nx.is_bipartite(textbook_network))
        # Can't check bipartite nodes for directed graphs
        # Test for not directed, weighted by stoichiometry
        textbook_network = create_metabolic_network(
            model=textbook_model,
            weighted=True,
            directed=False,
            weight_by="stoichiometry",
        )
        self.assertTrue(nx.is_bipartite(textbook_network))
        self.assertTrue(
            nx.algorithms.bipartite.is_bipartite_node_set(
                textbook_network, textbook_model.reactions.list_attr("id")
            )
        )
        self.assertTrue(
            nx.algorithms.bipartite.is_bipartite_node_set(
                textbook_network, textbook_model.metabolites.list_attr("id")
            )
        )
        # Test for not directed, weighted by flux
        textbook_network = create_metabolic_network(
            model=textbook_model,
            weighted=True,
            directed=False,
            weight_by="flux",
        )
        self.assertTrue(nx.is_bipartite(textbook_network))
        self.assertTrue(
            nx.algorithms.bipartite.is_bipartite_node_set(
                textbook_network, textbook_model.reactions.list_attr("id")
            )
        )
        self.assertTrue(
            nx.algorithms.bipartite.is_bipartite_node_set(
                textbook_network, textbook_model.metabolites.list_attr("id")
            )
        )
        # Test for directed, weighted by stoichiometry
        textbook_network = create_metabolic_network(
            model=textbook_model,
            weighted=True,
            directed=True,
            weight_by="stoichiometry",
        )
        self.assertTrue(nx.is_bipartite(textbook_network))
        # Test for directed, weighted by stoichiometry
        textbook_network = create_metabolic_network(
            model=textbook_model,
            weighted=True,
            directed=True,
            weight_by="flux",
        )
        self.assertTrue(nx.is_bipartite(textbook_network))


# endregion Metabolic Network

# region Mutual Information Network


class TestMutualInformationNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Configuration().solver = "glpk"
        cls.data_path = (
            pathlib.Path(__file__).parent.parent.absolute() / "data"
        )
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
                mi_adj_mat.iloc[i, j],
                test_network.get_edge_data(rxn_ids[i], rxn_ids[j])["weight"],
            )


# endregion Mutual Information Network

if __name__ == "__main__":
    unittest.main()

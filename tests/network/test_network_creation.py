# Imports
from __future__ import annotations

# Standard library imports
import itertools
import pathlib
import string
import unittest

# External Imports
import cobra
import networkx as nx
import numpy as np
import pandas as pd
from cobra.core.configuration import Configuration
from scipy import sparse

from metworkpy.information import mi_network_adjacency_matrix
from metworkpy.network.network_construction import (
    _create_sparse_adjacency_matrix,
    _create_stoichiometric_matrix,
    create_adjacency_matrix,
    create_group_neighborhood_network,
    create_metabolic_network,
    create_mutual_information_network,
)

# Local Imports
from metworkpy.utils.models import read_model


# region Metabolic Network
def setup(cls):
    Configuration().solver = "glpk"
    cls.data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
    cls.test_model = read_model(cls.data_path / "test_model.xml")
    cls.tiny_model = read_model(cls.data_path / "tiny_model.json")


class TestAdjMatUdUw(unittest.TestCase):
    test_model: cobra.Model | None = None
    tiny_model: cobra.Model | None = None
    data_path: pathlib.Path | None = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        assert isinstance(cls.test_model, cobra.Model)
        assert isinstance(cls.tiny_model, cobra.Model)
        cls.adj_mat = _create_sparse_adjacency_matrix(
            cls.test_model,
            forward=sparse.coo_array(
                np.ones((len(cls.test_model.reactions),))
            ),
            reverse=sparse.coo_array(
                np.zeros((len(cls.test_model.reactions),))
            ),
            weighted=False,
            directed=False,
        )

        cls.tiny_adj_mat = _create_sparse_adjacency_matrix(
            cls.tiny_model,
            forward=sparse.coo_array(
                np.ones((len(cls.tiny_model.reactions),))
            ),
            reverse=sparse.coo_array([1, 1, 1, 0]),
            weighted=False,
            directed=False,
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
        assert self.test_model is not None
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, sparse.coo_array)

    def test_known(self):
        np.testing.assert_allclose(
            self.tiny_adj_mat.todense(), self.tiny_known.to_numpy()
        )


class TestAdjMatDUw(unittest.TestCase):
    test_model = None
    tiny_adj_mat = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        assert cls.test_model is not None
        assert cls.tiny_model is not None
        cls.adj_mat = _create_sparse_adjacency_matrix(
            cls.test_model,
            forward=sparse.coo_array(
                np.ones((len(cls.test_model.reactions),))
            ),
            reverse=sparse.coo_array(
                np.zeros((len(cls.test_model.reactions),))
            ),
            weighted=False,
            directed=True,
        )

        cls.tiny_adj_mat = _create_sparse_adjacency_matrix(
            cls.tiny_model,
            forward=sparse.coo_array(
                np.ones((len(cls.tiny_model.reactions),))
            ),
            reverse=sparse.coo_array([1, 1, 1, 0]),
            weighted=False,
            directed=True,
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
        assert self.test_model is not None
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, sparse.coo_array)

    def test_known(self):
        assert self.tiny_adj_mat is not None
        np.testing.assert_allclose(
            self.tiny_adj_mat.todense(), self.tiny_known.to_numpy()
        )


class TestAdjMatDWFVA(unittest.TestCase):
    test_model = None
    data_path = None
    tiny_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        assert cls.test_model is not None
        assert cls.tiny_model is not None

        cls.adj_mat = create_adjacency_matrix(
            cls.test_model, weight="fva", directed=True, array_type="frame"
        )

        cls.tiny_adj_mat = create_adjacency_matrix(
            cls.tiny_model, weight="fva", directed=True, array_type="frame"
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
        assert self.test_model is not None
        assert isinstance(self.adj_mat, pd.DataFrame)
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)  # ty: ignore[invalid-argument-type]


class TestAdjMatDWpFBA(unittest.TestCase):
    test_model = None
    data_path = None
    tiny_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        assert cls.test_model is not None
        assert cls.tiny_model is not None

        cls.adj_mat = create_adjacency_matrix(
            cls.test_model, weight="pfba", directed=True, array_type="frame"
        )
        cls.tiny_adj_mat = create_adjacency_matrix(
            cls.tiny_model, weight="pfba", directed=True, array_type="frame"
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
        assert self.test_model is not None
        assert isinstance(self.adj_mat, pd.DataFrame)
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)  # ty: ignore[invalid-argument-type]


class TestAdjMatDWStoichiometry(unittest.TestCase):
    test_model = None
    tiny_adj_mat = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        assert cls.test_model is not None
        assert cls.tiny_model is not None
        cls.adj_mat = create_adjacency_matrix(
            cls.test_model,
            weight="stoichiometry",
            directed=True,
            array_type="frame",
        )
        cls.tiny_adj_mat = create_adjacency_matrix(
            cls.tiny_model,
            weight="stoichiometry",
            directed=True,
            array_type="frame",
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
        assert self.test_model is not None
        assert isinstance(self.adj_mat, pd.DataFrame)
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        assert self.tiny_adj_mat is not None
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)


class TestAdjMatUdWStoichiometry(unittest.TestCase):
    test_model = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        assert cls.test_model is not None
        assert cls.tiny_model is not None
        cls.adj_mat = create_adjacency_matrix(
            cls.test_model,
            weight="stoichiometry",
            directed=False,
            array_type="frame",
        )
        cls.tiny_adj_mat = create_adjacency_matrix(
            cls.tiny_model,
            weight="stoichiometry",
            directed=False,
            array_type="frame",
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
        assert self.test_model is not None
        assert isinstance(self.adj_mat, pd.DataFrame)
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        assert isinstance(self.tiny_adj_mat, pd.DataFrame)
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)


class TestAdjMatUdWpFBA(unittest.TestCase):
    test_model = None
    data_path = None
    tiny_model = None

    @classmethod
    def setUpClass(cls):
        setup(cls)
        assert cls.test_model is not None
        assert cls.tiny_model is not None

        cls.adj_mat = create_adjacency_matrix(
            cls.test_model, weight="pfba", directed=False, array_type="frame"
        )
        cls.tiny_adj_mat = create_adjacency_matrix(
            cls.tiny_model, weight="pfba", directed=False, array_type="frame"
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
        assert self.test_model is not None
        assert isinstance(self.adj_mat, pd.DataFrame)
        num_metabolites = len(self.test_model.metabolites)
        num_rxns = len(self.test_model.reactions)
        self.assertTupleEqual(
            self.adj_mat.shape,
            (num_rxns + num_metabolites, num_rxns + num_metabolites),
        )

    def test_type(self):
        self.assertIsInstance(self.adj_mat, pd.DataFrame)

    def test_known(self):
        assert isinstance(self.tiny_adj_mat, pd.DataFrame)
        pd.testing.assert_frame_equal(self.tiny_known, self.tiny_adj_mat)


class TestCreateAdjacencyMatrix(unittest.TestCase):
    test_model = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_array_type(self):
        assert self.test_model is not None
        for array_type, type_ in zip(
            [
                "dense",
                "frame",
                "bsr",
                "coo",
                "csc",
                "csr",
                "dia",
                "dok",
                "lil",
            ],
            [
                np.ndarray,
                pd.DataFrame,
                sparse.bsr_array,
                sparse.coo_array,
                sparse.csc_array,
                sparse.csr_array,
                sparse.dia_array,
                sparse.dok_array,
                sparse.lil_array,
            ],
        ):
            self.assertIsInstance(
                create_adjacency_matrix(
                    model=self.test_model,
                    weight="stoichiometry",
                    directed=True,
                    array_type=array_type,  # type: ignore
                ),
                type_,
            )


class TestCreateNetwork(unittest.TestCase):
    test_model = None
    tiny_model = None
    data_path = None

    @classmethod
    def setUpClass(cls):
        setup(cls)

    def test_directed_unweighted(self):
        assert self.test_model is not None
        assert self.tiny_model is not None
        test_network = create_metabolic_network(
            model=self.test_model, weight=None, directed=True
        )
        self.assertIsInstance(test_network, nx.DiGraph)
        for _, _, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 1)
        tiny_network = create_metabolic_network(
            model=self.tiny_model, weighted=False, directed=True
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 1)
        with self.assertRaises(KeyError):
            _ = tiny_network["R_C_ex"]["C"]

    def test_undirected_unweighted(self):
        assert self.test_model is not None
        assert self.tiny_model is not None
        test_network = create_metabolic_network(
            model=self.test_model, weight=None, directed=False
        )
        self.assertIsInstance(test_network, nx.Graph)
        for _, _, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 1)
        tiny_network = create_metabolic_network(
            model=self.tiny_model, weighted=False, directed=False
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 1)
        self.assertEqual(tiny_network["R_C_ex"]["C"]["weight"], 1)

    def test_directed_weighted_stoichiometry(self):
        assert self.test_model is not None
        assert self.tiny_model is not None
        test_network = create_metabolic_network(
            model=self.test_model,
            weight="stoichiometry",
            directed=True,
        )
        self.assertIsInstance(test_network, nx.DiGraph)
        for _, _, data in test_network.edges(data=True):
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
        assert self.test_model is not None
        assert self.tiny_model is not None
        test_network = create_metabolic_network(
            model=self.test_model,
            weight="stoichiometry",
            directed=False,
        )
        self.assertIsInstance(test_network, nx.Graph)
        for _, _, data in test_network.edges(data=True):
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
        assert self.test_model is not None
        assert self.tiny_model is not None
        test_network = create_metabolic_network(
            model=self.test_model,
            weight="fva",
            directed=True,
        )
        self.assertIsInstance(test_network, nx.DiGraph)
        for _, _, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 50)
        tiny_network = create_metabolic_network(
            model=self.tiny_model,
            weight="fva",
            directed=True,
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 50)
        with self.assertRaises(KeyError):
            _ = tiny_network["R_C_ex"]["C"]

    def test_undirected_weighted_flux(self):
        assert self.test_model is not None
        assert self.tiny_model is not None
        test_network = create_metabolic_network(
            model=self.test_model,
            weight="fva",
            directed=False,
        )
        self.assertIsInstance(test_network, nx.Graph)
        for _, _, data in test_network.edges(data=True):
            self.assertEqual(data["weight"], 50)
        tiny_network = create_metabolic_network(
            model=self.tiny_model,
            weight="fva",
            directed=False,
        )
        self.assertEqual(tiny_network["C"]["R_C_ex"]["weight"], 50)
        self.assertEqual(tiny_network["R_C_ex"]["C"]["weight"], 50)

    def test_bipartite(self):
        textbook_model = cobra.io.load_model(
            "textbook"
        )  # ecoli core metabolism
        # Test for not directed, not weighted
        textbook_network = create_metabolic_network(
            model=textbook_model, weight=None, directed=False
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
            model=textbook_model, weight=None, directed=True
        )
        self.assertTrue(nx.is_bipartite(textbook_network))
        # Can't check bipartite nodes for directed graphs
        # Test for not directed, weighted by stoichiometry
        textbook_network = create_metabolic_network(
            model=textbook_model,
            weight="stoichiometry",
            directed=False,
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
        # Test for not directed, weighted by fva
        textbook_network = create_metabolic_network(
            model=textbook_model,
            weight="fva",
            directed=False,
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
        # Test for not directed, weighted by pfba
        textbook_network = create_metabolic_network(
            model=textbook_model,
            weight="pfba",
            directed=False,
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
            weight="stoichiometry",
            directed=True,
        )
        self.assertTrue(nx.is_bipartite(textbook_network))
        # Test for directed, weighted by stoichiometry
        textbook_network = create_metabolic_network(
            model=textbook_model,
            weight="fva",
            directed=True,
        )
        self.assertTrue(nx.is_bipartite(textbook_network))


class TestCreateGroupConnectivityNetwork(unittest.TestCase):
    def test_small_group_network(self):
        # Construct a test network
        g = nx.Graph()
        g.add_nodes_from(["a", "b", "c", "d", "e", "f", "g"])
        g.add_edges_from([("a", "b"), ("c", "d"), ("e", "f"), ("a", "g")])
        groups = {
            "group1": {"a", "c"},
            "group2": {"d", "e"},
            "group3": {"b", "f"},
            "group4": {"g"},
        }
        connectivity_graph = create_group_neighborhood_network(
            network=g,
            groups=groups,  # type: ignore
            max_distance=1,
        )
        # Create the expected graph manually
        expected_connectivity_graph = nx.Graph()
        expected_connectivity_graph.add_nodes_from(
            ["group1", "group2", "group3", "group4"]
        )
        expected_connectivity_graph.add_edges_from(
            [
                ("group1", "group2"),
                ("group1", "group3"),
                ("group1", "group4"),
                ("group2", "group3"),
            ]
        )
        # Check the graph is as expected
        self.assertTrue(
            nx.is_isomorphic(connectivity_graph, expected_connectivity_graph)
        )


# endregion Metabolic Network


class TestAdjMat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Configuration().solver = "glpk"
        cls.data_path = (
            pathlib.Path(__file__).parent.parent.absolute() / "data"
        )
        cls.test_model = read_model(cls.data_path / "test_model.xml")
        cls.textbook_model = read_model(cls.data_path / "textbook_model.xml")

        # Create a tiny model for use in testing here,
        #
        # RXN1: A+B -> C
        # RXN2: C->D+E
        cls.tiny_model = cobra.Model("tiny_model")
        met_a = cobra.Metabolite(
            "A",
            "ch4",
            "Metabolite A",
        )
        met_b = cobra.Metabolite(
            "B",
            "ch4",
            "Metabolite B",
        )
        met_c = cobra.Metabolite(
            "C",
            "ch4",
            "Metabolite C",
        )
        met_d = cobra.Metabolite(
            "D",
            "ch4",
            "Metabolite D",
        )
        met_e = cobra.Metabolite(
            "E",
            "ch4",
            "Metabolite E",
        )

        rxn1 = cobra.Reaction(
            "rxn1", "Reaction 1", lower_bound=-1, upper_bound=10
        )
        rxn2 = cobra.Reaction(
            "rxn2", "Reaction 2", lower_bound=0, upper_bound=10
        )

        rxn1.add_metabolites({met_a: -2.0, met_b: -1.0, met_c: 2})
        rxn2.add_metabolites({met_c: -1.0, met_d: 3.0, met_e: 1.0})

        cls.tiny_model.add_reactions([rxn1, rxn2])

    def test_sparse_stoich_matrix(self):
        # Catch fire with the test model
        sparse_stoich = _create_stoichiometric_matrix(self.test_model)
        self.assertIsInstance(sparse_stoich, sparse.coo_array)
        self.assertEqual(
            sparse_stoich.shape[0], len(self.test_model.metabolites)
        )
        self.assertEqual(
            sparse_stoich.shape[1], len(self.test_model.reactions)
        )

        # Specific test with the tiny model
        expected_tiny_model_stoich = sparse.coo_array(
            (
                [-2.0, -1.0, 2.0, -1.0, 3.0, 1.0],  # coefficients
                (
                    [0, 1, 2, 2, 3, 4],  # Rows
                    [0, 0, 0, 1, 1, 1],  # Columns
                ),
            )
        )

        actual_tiny_model_stoich = _create_stoichiometric_matrix(
            self.tiny_model
        )

        np.testing.assert_allclose(
            actual_tiny_model_stoich.todense(),
            expected_tiny_model_stoich.todense(),
        )

    def test_sparse_adj_wd(self):
        # Test _create_sparse_adjacency_matrix for weighted and directed matrix

        # All Irreversible
        forward = sparse.coo_array(np.ones(len(self.tiny_model.reactions)))
        reverse = sparse.coo_array(np.zeros(len(self.tiny_model.reactions)))

        test_adj_stoich_all_irreversible = _create_sparse_adjacency_matrix(
            self.tiny_model,
            forward=forward,
            reverse=reverse,
            directed=True,
            weighted=True,
        )
        self.assertIsInstance(
            test_adj_stoich_all_irreversible, sparse.coo_array
        )

        expected_adj_stoich_all_irreversible = np.array(
            [
                # 1  2  A  B  C  D  E
                [0, 0, 0, 0, 2, 0, 0],  # rxn1
                [0, 0, 0, 0, 0, 3, 1],  # rxn2
                [2, 0, 0, 0, 0, 0, 0],  # A
                [1, 0, 0, 0, 0, 0, 0],  # B
                [0, 1, 0, 0, 0, 0, 0],  # C
                [0, 0, 0, 0, 0, 0, 0],  # D
                [0, 0, 0, 0, 0, 0, 0],  # E
            ]
        )
        np.testing.assert_allclose(
            test_adj_stoich_all_irreversible.todense(),
            expected_adj_stoich_all_irreversible,
        )

        # All Reversible
        forward = sparse.coo_array(np.ones(len(self.tiny_model.reactions)))
        reverse = sparse.coo_array(np.ones(len(self.tiny_model.reactions)))

        test_adj_stoich_all_reversible = _create_sparse_adjacency_matrix(
            self.tiny_model,
            forward=forward,
            reverse=reverse,
            directed=True,
            weighted=True,
        )
        self.assertIsInstance(test_adj_stoich_all_reversible, sparse.coo_array)

        expected_adj_stoich_all_reversible = np.array(
            [
                # 1  2  A  B  C  D  E
                [0, 0, 2, 1, 2, 0, 0],  # rxn1
                [0, 0, 0, 0, 1, 3, 1],  # rxn2
                [2, 0, 0, 0, 0, 0, 0],  # A
                [1, 0, 0, 0, 0, 0, 0],  # B
                [2, 1, 0, 0, 0, 0, 0],  # C
                [0, 3, 0, 0, 0, 0, 0],  # D
                [0, 1, 0, 0, 0, 0, 0],  # E
            ]
        )
        # Expected should be symmetric
        assert np.allclose(
            expected_adj_stoich_all_reversible
            - expected_adj_stoich_all_reversible.T,
            0.0,
        )
        np.testing.assert_allclose(
            test_adj_stoich_all_reversible.todense(),
            expected_adj_stoich_all_reversible,
        )

    def test_sparse_adj_wud(self):
        forward = sparse.coo_array(np.ones(len(self.tiny_model.reactions)))
        reverse = sparse.coo_array(np.zeros(len(self.tiny_model.reactions)))

        test_adj_stoich_all_irreversible = _create_sparse_adjacency_matrix(
            self.tiny_model,
            forward=forward,
            reverse=reverse,
            directed=False,
            weighted=True,
        )
        self.assertIsInstance(
            test_adj_stoich_all_irreversible, sparse.coo_array
        )

        expected_adj_stoich_all_irreversible = np.array(
            [
                # 1  2  A  B  C  D  E
                [0, 0, 2, 1, 2, 0, 0],  # rxn1
                [0, 0, 0, 0, 1, 3, 1],  # rxn2
                [2, 0, 0, 0, 0, 0, 0],  # A
                [1, 0, 0, 0, 0, 0, 0],  # B
                [2, 1, 0, 0, 0, 0, 0],  # C
                [0, 3, 0, 0, 0, 0, 0],  # D
                [0, 1, 0, 0, 0, 0, 0],  # E
            ]
        )
        np.testing.assert_allclose(
            test_adj_stoich_all_irreversible.todense(),
            expected_adj_stoich_all_irreversible,
        )

    def test_sparse_adj_uwd(self):
        # All Irreversible
        forward = sparse.coo_array(np.ones(len(self.tiny_model.reactions)))
        reverse = sparse.coo_array(np.zeros(len(self.tiny_model.reactions)))

        test_adj_stoich_all_irreversible = _create_sparse_adjacency_matrix(
            self.tiny_model,
            forward=forward,
            reverse=reverse,
            directed=True,
            weighted=False,
        )
        self.assertIsInstance(
            test_adj_stoich_all_irreversible, sparse.coo_array
        )

        expected_adj_stoich_all_irreversible = np.array(
            [
                # 1  2  A  B  C  D  E
                [0, 0, 0, 0, 1, 0, 0],  # rxn1
                [0, 0, 0, 0, 0, 1, 1],  # rxn2
                [1, 0, 0, 0, 0, 0, 0],  # A
                [1, 0, 0, 0, 0, 0, 0],  # B
                [0, 1, 0, 0, 0, 0, 0],  # C
                [0, 0, 0, 0, 0, 0, 0],  # D
                [0, 0, 0, 0, 0, 0, 0],  # E
            ]
        )
        np.testing.assert_allclose(
            test_adj_stoich_all_irreversible.todense(),
            expected_adj_stoich_all_irreversible,
        )

        # Test with larger model
        sparse_adj = _create_sparse_adjacency_matrix(
            model=self.textbook_model,
            forward=sparse.coo_array(
                np.ones((len(self.textbook_model.reactions),))
            ),
            reverse=sparse.coo_array(
                np.zeros(len(self.textbook_model.reactions))
            ),
            weighted=False,
            directed=True,
        )
        np.testing.assert_allclose(sparse_adj.data, 1.0)

    def test_sparse_adj_uwud(self):
        forward = sparse.coo_array(np.ones(len(self.tiny_model.reactions)))
        reverse = sparse.coo_array(np.ones(len(self.tiny_model.reactions)))

        test_adj_stoich_all_reversible = _create_sparse_adjacency_matrix(
            self.tiny_model,
            forward=forward,
            reverse=reverse,
            directed=False,
            weighted=False,
        )
        self.assertIsInstance(test_adj_stoich_all_reversible, sparse.coo_array)

        expected_adj_stoich_all_reversible = np.array(
            [
                # 1  2  A  B  C  D  E
                [0, 0, 1, 1, 1, 0, 0],  # rxn1
                [0, 0, 0, 0, 1, 1, 1],  # rxn2
                [1, 0, 0, 0, 0, 0, 0],  # A
                [1, 0, 0, 0, 0, 0, 0],  # B
                [1, 1, 0, 0, 0, 0, 0],  # C
                [0, 1, 0, 0, 0, 0, 0],  # D
                [0, 1, 0, 0, 0, 0, 0],  # E
            ]
        )
        # Expected should be symmetric
        assert np.allclose(
            expected_adj_stoich_all_reversible
            - expected_adj_stoich_all_reversible.T,
            0.0,
        )
        np.testing.assert_allclose(
            test_adj_stoich_all_reversible.todense(),
            expected_adj_stoich_all_reversible,
        )


class TestCurrencyMetabolites(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small model with currency metabolites
        metabolite_dict = {}
        for char in string.ascii_lowercase[1:11]:
            metabolite_dict[char] = cobra.Metabolite(
                char.upper(), "", f"Metabolite {char}"
            )
        rxn_list = [
            cobra.Reaction(
                f"rxn{idx}", f"Reaction {idx}", lower_bound=0, upper_bound=10
            )
            for idx in range(1, 8)
        ]
        # B->C
        rxn_list[0].add_metabolites(
            {metabolite_dict["b"]: -1, metabolite_dict["c"]: 1}
        )
        # C->H
        rxn_list[1].add_metabolites(
            {metabolite_dict["c"]: -1, metabolite_dict["h"]: 1}
        )
        # E+C->F+D
        rxn_list[2].add_metabolites(
            {
                metabolite_dict["e"]: -1,
                metabolite_dict["c"]: -1,
                metabolite_dict["f"]: 1,
                metabolite_dict["d"]: 1,
            }
        )
        # F+I->G+J+K
        rxn_list[3].add_metabolites(
            {
                metabolite_dict["f"]: -1,
                metabolite_dict["i"]: -1,
                metabolite_dict["g"]: 1,
                metabolite_dict["j"]: 1,
                metabolite_dict["k"]: 1,
            }
        )
        # C->I
        rxn_list[4].add_metabolites(
            {metabolite_dict["c"]: -1, metabolite_dict["i"]: 1}
        )
        # C->D
        rxn_list[5].add_metabolites(
            {metabolite_dict["c"]: -1, metabolite_dict["d"]: 1}
        )
        # I->J+K
        rxn_list[6].add_metabolites(
            {
                metabolite_dict["i"]: -1,
                metabolite_dict["j"]: 1,
                metabolite_dict["k"]: 1,
            }
        )

        cls.curr_met_model = cobra.Model()
        cls.curr_met_model.add_reactions(rxn_list)
        # Create a graph with no currency metabolites removed
        cls.test_graph = create_metabolic_network(
            model=cls.curr_met_model,
            weight=None,
            directed=False,
        )

    def check_has_edges(self, graph, u, vs):
        for v in vs:
            self.assertTrue(graph.has_edge(u, v))

    def check_has_no_edges(self, graph, u, vs):
        for v in vs:
            print(f"Checking for edge ({u}, {v})")
            self.assertFalse(graph.has_edge(u, v))

    def test_single_pair(self):
        print(
            cobra.util.create_stoichiometric_matrix(
                self.curr_met_model, "DataFrame"
            )
        )
        test_graph_curr_removed = create_metabolic_network(
            model=self.curr_met_model,
            weight=None,
            directed=False,
            currency_metabolites=[("C", "D")],
        )
        assert isinstance(self.test_graph, nx.Graph)
        assert isinstance(test_graph_curr_removed, nx.Graph)
        # There should be an edges between rxn3 and C and D
        # in test_graph, but not in test_graph_curr_removed
        self.check_has_edges(self.test_graph, "rxn3", ["C", "D"])
        self.check_has_no_edges(test_graph_curr_removed, "rxn3", ["C", "D"])
        # Both should have rxn6 connected to C and D
        self.check_has_edges(self.test_graph, "rxn6", ["C", "D"])
        self.check_has_edges(test_graph_curr_removed, "rxn6", ["C", "D"])
        # Both should have rxn1,2,5 connected to C
        for rxn in ["rxn1", "rxn2", "rxn5"]:
            self.check_has_edges(self.test_graph, rxn, ["C"])

    def test_multiple_on_one_side(self):
        test_graph_curr_removed = create_metabolic_network(
            model=self.curr_met_model,
            weight=None,
            directed=False,
            currency_metabolites=[("I", ("J", "K"))],
        )
        # Test graph should have rxn4 connected to f,g,i,j,k
        # curr_removed should have rxn4 connected to f,g only
        self.check_has_edges(
            self.test_graph, "rxn4", ["F", "G", "I", "J", "K"]
        )
        self.check_has_edges(test_graph_curr_removed, "rxn4", ["F", "G"])
        self.check_has_no_edges(
            test_graph_curr_removed, "rxn4", ["I", "J", "K"]
        )
        # Both should have rxn7 connected to I, J, and K
        self.check_has_edges(self.test_graph, "rxn7", ["I", "J", "K"])
        self.check_has_edges(test_graph_curr_removed, "rxn7", ["I", "J", "K"])
        # Both should have rxn5 connected to I
        self.check_has_edges(self.test_graph, "rxn5", ["I"])
        self.check_has_edges(test_graph_curr_removed, "rxn5", ["I"])

    def test_multiple_curr(self):
        test_graph_curr_removed = create_metabolic_network(
            model=self.curr_met_model,
            weight=None,
            directed=False,
            currency_metabolites=[("C", "D"), ("I", ("J", "K"))],
        )
        # There should be an edges between rxn3 and C and D
        # in test_graph, but not in test_graph_curr_removed
        self.check_has_edges(self.test_graph, "rxn3", ["C", "D"])
        self.check_has_no_edges(test_graph_curr_removed, "rxn3", ["C", "D"])
        # Both should have rxn6 connected to C and D
        self.check_has_edges(self.test_graph, "rxn6", ["C", "D"])
        self.check_has_edges(test_graph_curr_removed, "rxn6", ["C", "D"])
        # Both should have rxn1,2,5 connected to C
        for rxn in ["rxn1", "rxn2", "rxn5"]:
            self.check_has_edges(self.test_graph, rxn, ["C"])
        # Test graph should have rxn4 connected to f,g,i,j,k
        # curr_removed should have rxn4 connected to f,g only
        self.check_has_edges(
            self.test_graph, "rxn4", ["F", "G", "I", "J", "K"]
        )
        self.check_has_edges(test_graph_curr_removed, "rxn4", ["F", "G"])
        self.check_has_no_edges(
            test_graph_curr_removed, "rxn4", ["I", "J", "K"]
        )
        # Both should have rxn7 connected to I, J, and K
        self.check_has_edges(self.test_graph, "rxn7", ["I", "J", "K"])
        self.check_has_edges(test_graph_curr_removed, "rxn7", ["I", "J", "K"])
        # Both should have rxn5 connected to I
        self.check_has_edges(self.test_graph, "rxn5", ["I"])
        self.check_has_edges(test_graph_curr_removed, "rxn5", ["I"])


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
        assert isinstance(mi_adj_mat, pd.DataFrame)
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

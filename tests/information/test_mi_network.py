# Standard Library Imports
import itertools
import unittest

# External Imports
import networkx as nx
import numpy as np
import pandas as pd
import scipy

# Local Imports
from metworkpy.information.mutual_information_network import (
    mi_network_adjacency_matrix,
    mi_pairwise_grouped,
    create_grouped_mi_network,
)
import metworkpy.information.mutual_information_functions as mi


class TestMiNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create multivariate Gaussian's for testing the mutual information network
        norm_2d_0 = scipy.stats.multivariate_normal(
            mean=[0, 0], cov=[[1, 0], [0, 1]], seed=314
        )
        norm_2d_0_3 = scipy.stats.multivariate_normal(
            mean=[0, 0], cov=[[1, 0.3], [0.3, 1]], seed=314
        )
        norm_2d_0_6 = scipy.stats.multivariate_normal(
            mean=[0, 0], cov=[[1, 0.6], [0.6, 1]], seed=314
        )

        # Sample the distributions
        cls.norm_2d_0_sample_1000 = norm_2d_0.rvs(size=1000)
        cls.norm_2d_0_3_sample_1000 = norm_2d_0_3.rvs(size=1000)
        cls.norm_2d_0_6_sample_1000 = norm_2d_0_6.rvs(size=1000)

        # Known Mutual information for multivariate gaussian with different covariances
        # See the Kraskov et al. paper for details
        cls.norm_2d_0_mi = -(1 / 2) * np.log(1 - 0**2)
        cls.norm_2d_0_3_mi = -(1 / 2) * np.log(1 - 0.3**2)
        cls.norm_2d_0_6_mi = -(1 / 2) * np.log(1 - 0.6**2)

        # Stack together the samples to get a samples array
        cls.samples = np.hstack(
            (
                cls.norm_2d_0_sample_1000,
                cls.norm_2d_0_3_sample_1000,
                cls.norm_2d_0_6_sample_1000,
            )
        )

    def test_mi_network_serial(self):
        mi_network = mi_network_adjacency_matrix(
            self.samples, n_neighbors=3, processes=1
        )
        assert isinstance(mi_network, np.ndarray)
        # Note, in the samples matrix the columns are arranged such that:
        # - 0,1 Have no covariance
        # - 2,3 Have a covariance of 0.3
        # - 4,5 Have a covariance of 0.6

        # Make sure the network is symmetrical
        for i, j in itertools.combinations(range(6), 2):
            self.assertAlmostEqual(mi_network[i, j], mi_network[j, i])

        # Check that known values match
        self.assertTrue(
            np.isclose(mi_network[0, 1], self.norm_2d_0_mi, atol=0.08)
        )
        self.assertTrue(
            np.isclose(mi_network[1, 0], self.norm_2d_0_mi, atol=0.08)
        )
        self.assertTrue(
            np.isclose(mi_network[2, 3], self.norm_2d_0_3_mi, atol=0.05)
        )
        self.assertTrue(
            np.isclose(mi_network[3, 2], self.norm_2d_0_3_mi, atol=0.05)
        )
        self.assertTrue(
            np.isclose(mi_network[4, 5], self.norm_2d_0_6_mi, atol=0.05)
        )
        self.assertTrue(
            np.isclose(mi_network[5, 4], self.norm_2d_0_6_mi, atol=0.05)
        )

        # Calculate the MI using the methods from mutual_information_functions
        self.assertTrue(
            np.isclose(
                mi_network[0, 1],
                mi.mutual_information(
                    self.samples[:, (0,)],
                    self.samples[:, (1,)],
                    discrete_x=False,
                    discrete_y=False,
                    n_neighbors=3,
                ),
            )
        )
        self.assertTrue(
            np.isclose(
                mi_network[2, 3],
                mi.mutual_information(
                    self.samples[:, (2,)],
                    self.samples[:, (3,)],
                    discrete_x=False,
                    discrete_y=False,
                    n_neighbors=3,
                ),
            )
        )
        self.assertTrue(
            np.isclose(
                mi_network[4, 5],
                mi.mutual_information(
                    self.samples[:, (4,)],
                    self.samples[:, (5,)],
                    discrete_x=False,
                    discrete_y=False,
                    n_neighbors=3,
                ),
            )
        )

    def test_mi_network_parallel(self):
        mi_network_serial = mi_network_adjacency_matrix(
            self.samples, n_neighbors=3, processes=1
        )
        mi_network_parallel = mi_network_adjacency_matrix(
            self.samples, n_neighbors=3, processes=2
        )
        self.assertTrue(
            (np.isclose(mi_network_parallel, mi_network_serial)).all()
        )


class TestMINetworkGrouped(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a multivariate gaussian sample for testing
        rng = np.random.default_rng(910248102498)
        means = np.array([20, 15, 20, 10, 5, 20, 20])
        cov = np.array(
            [
                [5.0, 2.0, 3.0, 1.0, 0.5, 10.0, 1.0],
                [0.0, 5.0, 4.0, 1.0, 2.0, 5.0, 8.0],
                [0.0, 0.0, 5.0, 1.0, 9.0, 8.0, 2.0],
                [0.0, 0.0, 0.0, 5.0, 3.0, 7.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 9.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
            ]
        )
        cls.cov = cov @ np.transpose(cov)

        cls.dataset_arr = rng.multivariate_normal(means, cls.cov, 1_000)
        cls.dataset_df = pd.DataFrame(cls.dataset_arr)

    def test_numpy_input(self):
        groups = [[0, 3, 5], [1, 6], [2, 4]]
        result = mi_pairwise_grouped(
            self.dataset_arr,
            groups=groups,
            calculate_pvalue=False,
            cutoff=0.0,
            processes=1,
        )
        assert isinstance(result, pd.DataFrame)
        self.assertTupleEqual(result.shape, (3, 3))
        self.assertTrue(result.ge(0.0).all().all())

    def test_df_input(self):
        groups = [[0, 3, 5], [1, 6], [2, 4]]
        result = mi_pairwise_grouped(
            self.dataset_df,
            groups=groups,
            calculate_pvalue=False,
            cutoff=0.0,
            processes=1,
        )
        assert isinstance(result, pd.DataFrame)
        self.assertTupleEqual(result.shape, (3, 3))
        self.assertTrue(result.ge(0.0).all().all())

    def test_dict_groups(self):
        groups = {
            "A": [0, 3, 5],
            "B": [1, 6],
            "C": [2, 4],
        }
        expected_index = pd.Index(groups.keys())
        result = mi_pairwise_grouped(
            self.dataset_df,
            groups=groups,
            calculate_pvalue=False,
            cutoff=0.0,
            processes=1,
        )
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_index_equal(expected_index, result.index)
        pd.testing.assert_index_equal(expected_index, result.columns)
        self.assertTupleEqual(result.shape, (3, 3))
        self.assertTrue(result.ge(0.0).all().all())

    def test_pvalues(self):
        groups = {
            "A": [0, 3, 5],
            "B": [1, 6],
            "C": [2, 4],
        }
        expected_index = pd.Index(groups.keys())
        result = mi_pairwise_grouped(
            self.dataset_df,
            groups=groups,
            calculate_pvalue=True,
            permutations=100,
            cutoff=0.0,
            processes=1,
        )
        assert isinstance(result, tuple)
        mi_result, pval_result = result
        assert isinstance(mi_result, pd.DataFrame)
        assert isinstance(pval_result, pd.DataFrame)
        for df in [mi_result, pval_result]:
            pd.testing.assert_index_equal(expected_index, df.index)
            pd.testing.assert_index_equal(expected_index, df.columns)
        # All the values in the p-value matrix should be between 0 and 1
        self.assertLessEqual(pval_result.max().max(), 1.0)
        self.assertGreaterEqual(pval_result.min().min(), 0.0)

    def test_pvalue_edge_attributes(self):
        groups = {
            "A": [0, 3, 5],
            "B": [1, 6],
            "C": [2, 4],
        }
        result_network = create_grouped_mi_network(
            self.dataset_df,
            groups=groups,
            calculate_pvalue=True,
            permutations=100,
            cutoff=0.0,
            processes=1,
        )
        assert isinstance(result_network, nx.Graph)
        for g in groups.keys():
            self.assertTrue(g in result_network.nodes)
        # For each edge, check that a p-value attribute is present
        for i in groups.keys():
            for j in groups.keys():
                if i == j:
                    continue
                self.assertTrue("weight" in result_network[i][j])
                self.assertTrue("p-value" in result_network[i][j])
                self.assertLessEqual(result_network[i][j]["p-value"], 1.0)
                self.assertGreaterEqual(result_network[i][j]["p-value"], 0.0)


if __name__ == "__main__":
    unittest.main()

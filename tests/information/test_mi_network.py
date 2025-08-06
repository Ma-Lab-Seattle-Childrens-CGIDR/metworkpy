# Standard Library Imports
import itertools
import unittest

# External Imports
import numpy as np
import scipy

# Local Imports
from metworkpy.information.mutual_information_network import mi_network_adjacency_matrix
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
        # Note, in the samples matrix the columns are arranged such that:
        # - 0,1 Have no covariance
        # - 2,3 Have a covariance of 0.3
        # - 4,5 Have a covariance of 0.6

        # Make sure the network is symmetrical
        for i, j in itertools.combinations(range(6), 2):
            self.assertAlmostEqual(mi_network[i, j], mi_network[j, i])

        # Check that known values match
        self.assertTrue(np.isclose(mi_network[0, 1], self.norm_2d_0_mi, atol=0.08))
        self.assertTrue(np.isclose(mi_network[1, 0], self.norm_2d_0_mi, atol=0.08))
        self.assertTrue(np.isclose(mi_network[2, 3], self.norm_2d_0_3_mi, atol=0.05))
        self.assertTrue(np.isclose(mi_network[3, 2], self.norm_2d_0_3_mi, atol=0.05))
        self.assertTrue(np.isclose(mi_network[4, 5], self.norm_2d_0_6_mi, atol=0.05))
        self.assertTrue(np.isclose(mi_network[5, 4], self.norm_2d_0_6_mi, atol=0.05))

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

    def test_my_network_parallel(self):
        mi_network_serial = mi_network_adjacency_matrix(
            self.samples, n_neighbors=3, processes=1
        )
        mi_network_parallel = mi_network_adjacency_matrix(
            self.samples, n_neighbors=3, processes=2
        )
        self.assertTrue((np.isclose(mi_network_parallel, mi_network_serial)).all())


if __name__ == "__main__":
    unittest.main()

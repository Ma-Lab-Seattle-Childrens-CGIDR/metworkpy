# Standard Library Imports
import unittest

# External Imports
import numpy as np
import pandas as pd
import scipy

# Local Imports
from metworkpy.information import mutual_information_functions as mi


class TestHelperFunctions(unittest.TestCase):
    def test_parse_metric(self):
        with self.assertRaises(ValueError):
            mi._parse_metric(0.5)
        self.assertEqual(5., mi._parse_metric(5.))
        self.assertEqual(2., mi._parse_metric("Euclidean"))
        self.assertEqual(1., mi._parse_metric("Manhattan"))
        self.assertEqual(np.inf, mi._parse_metric("Chebyshev"))
        self.assertEqual(10., mi._parse_metric(10))

    def test_validate_sample(self):
        x = np.array([1, 2, 3, 4, 5])
        y = x.reshape(-1, 1)
        df = pd.Series(x)
        self.assertTrue((mi._validate_sample(df) == y).all())
        self.assertTrue((mi._validate_sample(x) == y).all())
        self.assertTrue((mi._validate_sample(y) == y).all())
        self.assertEqual(mi._validate_sample(x).shape, (5, 1))

    def test_validate_samples(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        z = np.array([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            _ = mi._validate_samples(x, z)
        a, b = mi._validate_samples(x, y)
        self.assertEqual(a.shape, (5, 1))
        self.assertEqual(b.shape, (5, 1))
        self.assertTrue((a == x.reshape(-1, 1)).all())
        self.assertTrue((b == y.reshape(-1, 1)).all())

    def test_check_discrete(self):
        x = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        y = np.array(['a', 'b', 'c', 'a', 'c'])
        with self.assertRaises(ValueError):
            _ = mi._check_discrete(sample=x, is_discrete=True)
        self.assertTrue((mi._check_discrete(sample=y, is_discrete=True) == y.reshape(-1, 1)).all())


class TestContCont(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create the distributions for the multivariate Gaussian with various covariances
        norm_2d_0 = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        norm_2d_0_3 = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0.3], [0.3, 1]])
        norm_2d_0_6 = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1, 0.6], [0.6, 1]])

        # Sample the distributions
        cls.norm_2d_0_sample_1000 = norm_2d_0.rvs(size=1000)
        cls.norm_2d_0_3_sample_1000 = norm_2d_0_3.rvs(size=1000)
        cls.norm_2d_0_6_sample_1000 = norm_2d_0_6.rvs(size=1000)

        # Known Mutual information for multivariate gaussian with different covariances
        # See the Kraskov et al. paper for details
        cls.norm_2d_0_mi = -(1 / 2) * np.log(1 - 0 ** 2)
        cls.norm_2d_0_3_mi = -(1 / 2) * np.log(1 - 0.3 ** 2)
        cls.norm_2d_0_6_mi = -(1 / 2) * np.log(1 - 0.6 ** 2)

    def test_cont_cont_gen(self):
        mi_0 = mi._mi_cont_cont_gen(x=self.norm_2d_0_sample_1000[:, 0].reshape(-1, 1),
                                    y=self.norm_2d_0_sample_1000[:, 1].reshape(-1, 1),
                                    n_neighbors=1,
                                    metric_x=np.inf,
                                    metric_y=np.inf)
        mi_0_3 = mi._mi_cont_cont_gen(x=self.norm_2d_0_3_sample_1000[:, 0].reshape(-1, 1),
                                      y=self.norm_2d_0_3_sample_1000[:, 1].reshape(-1, 1),
                                      n_neighbors=1,
                                      metric_x=np.inf,
                                      metric_y=np.inf)
        mi_0_6 = mi._mi_cont_cont_gen(x=self.norm_2d_0_6_sample_1000[:, 0].reshape(-1, 1),
                                      y=self.norm_2d_0_6_sample_1000[:, 1].reshape(-1, 1),
                                      n_neighbors=1,
                                      metric_x=np.inf,
                                      metric_y=np.inf)
        # Estimates so using a high tolerance
        self.assertTrue(np.isclose(mi_0, self.norm_2d_0_mi, atol=0.1))
        self.assertTrue(np.isclose(mi_0_3, self.norm_2d_0_3_mi, atol=0.1))
        self.assertTrue(np.isclose(mi_0_6, self.norm_2d_0_6_mi, atol=0.1))

    def test_cont_cont_cheb_only(self):
        mi_0 = mi._mi_cont_cont_cheb_only(x=self.norm_2d_0_sample_1000[:, 0].reshape(-1, 1),
                                          y=self.norm_2d_0_sample_1000[:, 1].reshape(-1, 1),
                                          n_neighbors=1, )
        mi_0_3 = mi._mi_cont_cont_cheb_only(x=self.norm_2d_0_3_sample_1000[:, 0].reshape(-1, 1),
                                            y=self.norm_2d_0_3_sample_1000[:, 1].reshape(-1, 1),
                                            n_neighbors=1)
        mi_0_6 = mi._mi_cont_cont_cheb_only(x=self.norm_2d_0_6_sample_1000[:, 0].reshape(-1, 1),
                                            y=self.norm_2d_0_6_sample_1000[:, 1].reshape(-1, 1),
                                            n_neighbors=1)
        self.assertTrue(np.isclose(mi_0, self.norm_2d_0_mi, atol=0.1))
        self.assertTrue(np.isclose(mi_0_3, self.norm_2d_0_3_mi, atol=0.1))
        self.assertTrue(np.isclose(mi_0_6, self.norm_2d_0_6_mi, atol=0.1))


if __name__ == '__main__':
    unittest.main()

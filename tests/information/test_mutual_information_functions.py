# Standard Library Imports
import unittest

# External Imports
import numpy as np
import pandas as pd
import scipy

import metworkpy.utils._arguments
import metworkpy.utils._jitter

# Local Imports
from metworkpy.information import mutual_information_functions as mi


class TestMainMI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create the distributions for the multivariate Gaussian with various
        # covariances
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

        # Additional Multivariate Gaussian for testing higher dimensional MI
        norm_2d_2d_0_3_0_6 = scipy.stats.multivariate_normal(
            mean=[0, 0, 0, 0],
            cov=[
                [1, 0.3, 0.6, 0.6],
                [0.3, 1, 0.6, 0.6],
                [0.6, 0.6, 1, 0.3],
                [0.6, 0.6, 0.3, 1],
            ],
            seed=314,
        )
        cls.norm_2d_2d_0_3_0_6_sample_1000 = norm_2d_2d_0_3_0_6.rvs(size=1000)

    def test_disc_disc(self):
        x = np.array([0, 1, 1, 0, 0])
        y = np.array([1, 0, 0, 0, 1])
        x_col = np.array([0, 1, 1, 0, 0]).reshape(-1, 1)
        y_col = np.array([1, 0, 0, 0, 1]).reshape(-1, 1)
        H_x = H_y = -(3 / 5) * np.log(3 / 5) - (2 / 5) * np.log(2 / 5)
        H_xy = (
            -1 / 5 * np.log(1 / 5) - (2 / 5) * np.log(2 / 5) - (2 / 5) * np.log(2 / 5)
        )
        I_xy = H_x + H_y - H_xy

        # Test with column vectors
        self.assertAlmostEqual(
            mi.mutual_information(x=x_col, y=y_col, discrete_x=True, discrete_y=True),
            I_xy,
        )
        # Test with row vectors
        self.assertAlmostEqual(
            mi.mutual_information(x=x, y=y, discrete_x=True, discrete_y=True), I_xy
        )

        # Test equality with dispatched method
        self.assertAlmostEqual(
            mi.mutual_information(x=x, y=y, discrete_x=True, discrete_y=True),
            mi._mi_disc_disc(x=x_col, y=y_col),
        )

    def test_disc_cont(self):
        # MODIFIED FROM SKLEARN (sklearn/feature_selection/tests/test_mutual_info.py):

        # To test define a joint distribution as follows:
        # p(x, y) = p(x) p(y | x)
        # X ~ Bernoulli(p)
        # (Y | x = 0) ~ Uniform(-1, 1)
        # (Y | x = 1) ~ Uniform(0, 2)

        # Use the following formula for mutual information:
        # I(X; Y) = H(Y) - H(Y | X)
        # Two entropies can be computed by hand:
        # H(Y) = -(1-p)/2 * ln((1-p)/2) - p/2*log(p/2) - 1/2*log(1/2)
        # H(Y | X) = ln(2)

        # Now we need to implement sampling from out distribution, which is
        # done easily using conditional distribution logic.

        n_samples = 1000
        generator = np.random.default_rng(1618)

        for p in [0.3, 0.5, 0.7]:
            x = generator.uniform(low=0.0, high=1.0, size=n_samples) > p

            y = np.empty(n_samples, dtype=float)

            y_dist1_index = x
            y_dist2_index = ~x

            y[y_dist1_index] = generator.uniform(-1.0, 1.0, size=np.sum(x))
            y[y_dist2_index] = generator.uniform(0, 2, size=np.sum(~x))

            mi_theory = -0.5 * (
                (1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)
            ) - np.log(2)

            for n_neighbors in [3, 5, 7]:
                mi_computed = mi.mutual_information(
                    x=x, y=y, discrete_x=True, discrete_y=False, n_neighbors=n_neighbors
                )
                self.assertTrue(np.isclose(mi_computed, mi_theory, rtol=1e-1))

    def test_cont_cont_1d(self):
        mi_0 = mi.mutual_information(
            x=self.norm_2d_0_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
            metric_x="cheb",
            metric_y="cheb",
        )
        mi_0_3 = mi.mutual_information(
            x=self.norm_2d_0_3_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_3_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
            metric_x="cheb",
            metric_y="cheb",
        )
        mi_0_6 = mi.mutual_information(
            x=self.norm_2d_0_6_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_6_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
            metric_x="cheb",
            metric_y="cheb",
        )
        self.assertTrue(np.isclose(mi_0, self.norm_2d_0_mi, atol=0.08))
        self.assertTrue(np.isclose(mi_0_3, self.norm_2d_0_3_mi, atol=0.05))
        self.assertTrue(np.isclose(mi_0_6, self.norm_2d_0_6_mi, atol=0.05))

    def test_cont_cont_2d(self):
        # Test the cheb method
        mi_2d_2d_0_3_0_6 = mi.mutual_information(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            discrete_x=False,
            discrete_y=False,
            n_neighbors=3,
            metric_x=np.inf,
            metric_y=np.inf,
        )
        mi_2d_2d_0_3_0_6_cheb = mi._mi_cont_cont_cheb_only(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            n_neighbors=3,
        )
        self.assertTrue(np.isclose(mi_2d_2d_0_3_0_6, mi_2d_2d_0_3_0_6_cheb))

        # Test the general method
        mi_2d_2d_0_3_0_6 = mi.mutual_information(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            discrete_x=False,
            discrete_y=False,
            n_neighbors=3,
            metric_x="euclidean",
            metric_y="cheby",
        )
        mi_2d_2d_0_3_0_6_gen = mi._mi_cont_cont_gen(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            metric_x=2.0,
            metric_y=np.inf,
            n_neighbors=3,
        )
        self.assertTrue(np.isclose(mi_2d_2d_0_3_0_6, mi_2d_2d_0_3_0_6_gen))

    def test_cont_cont_jitter(self):
        # Test that jitter doesn't significantly alter the results
        mi_2d_2d_0_3_0_6_jitter = mi.mutual_information(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            discrete_x=False,
            discrete_y=False,
            n_neighbors=3,
            jitter=1e-10,
            jitter_seed=314,
            metric_x="euclidean",
            metric_y="cheby",
        )
        mi_2d_2d_0_3_0_6 = mi.mutual_information(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            discrete_x=False,
            discrete_y=False,
            n_neighbors=3,
            jitter=None,
            jitter_seed=314,
            metric_x="euclidean",
            metric_y="cheby",
        )
        self.assertTrue(
            np.isclose(mi_2d_2d_0_3_0_6_jitter, mi_2d_2d_0_3_0_6, rtol=1e-7)
        )


class TestContCont(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create the distributions for the multivariate Gaussian with various covariances
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

        # Additional Multivariate Gaussian for testing higher dimensional MI
        norm_2d_2d_0_3_0_6 = scipy.stats.multivariate_normal(
            mean=[0, 0, 0, 0],
            cov=[
                [1, 0.3, 0.6, 0.6],
                [0.3, 1, 0.6, 0.6],
                [0.6, 0.6, 1, 0.3],
                [0.6, 0.6, 0.3, 1],
            ],
            seed=314,
        )
        cls.norm_2d_2d_0_3_0_6_sample_1000 = norm_2d_2d_0_3_0_6.rvs(size=1000)

    def test_cont_cont_gen(self):
        mi_0 = mi._mi_cont_cont_gen(
            x=self.norm_2d_0_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
            metric_x=np.inf,
            metric_y=np.inf,
        )
        mi_0_3 = mi._mi_cont_cont_gen(
            x=self.norm_2d_0_3_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_3_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
            metric_x=np.inf,
            metric_y=np.inf,
        )
        mi_0_6 = mi._mi_cont_cont_gen(
            x=self.norm_2d_0_6_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_6_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
            metric_x=np.inf,
            metric_y=np.inf,
        )
        # Estimates so using a high tolerance
        # Especially for the case where the true value is 0

        # Some concerns over how high tolerance here has to be,
        # it is partially a reflection of the random sampling (sometimes it is much closer),
        # perhaps sampling sub distributions and then raking the mean of those could improve the accuracy?
        #   However, this method can have issues with ties, so that probably won't work (at least not without jitter,
        #   which adds its own noise)
        self.assertTrue(np.isclose(mi_0, self.norm_2d_0_mi, atol=0.08))
        self.assertTrue(np.isclose(mi_0_3, self.norm_2d_0_3_mi, atol=0.05))
        self.assertTrue(np.isclose(mi_0_6, self.norm_2d_0_6_mi, atol=0.05))

    def test_cont_cont_gen_non_scalar(self):
        # Test that it is able to calculate with more than 1D scalars
        _ = mi._mi_cont_cont_gen(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            n_neighbors=3,
            metric_x=2.0,
            metric_y=2.0,
        )

    def test_cont_cont_cheb_only(self):
        mi_0 = mi._mi_cont_cont_cheb_only(
            x=self.norm_2d_0_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=1,
        )
        mi_0_3 = mi._mi_cont_cont_cheb_only(
            x=self.norm_2d_0_3_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_3_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=1,
        )
        mi_0_6 = mi._mi_cont_cont_cheb_only(
            x=self.norm_2d_0_6_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_6_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=1,
        )
        self.assertTrue(np.isclose(mi_0, self.norm_2d_0_mi, atol=0.08))
        self.assertTrue(np.isclose(mi_0_3, self.norm_2d_0_3_mi, atol=0.05))
        self.assertTrue(np.isclose(mi_0_6, self.norm_2d_0_6_mi, atol=0.05))

    def test_method_equivalence(self):
        # The Chebyshev only method should be equivalent to the general method for scalars (and for
        mi_0_6_gen = mi._mi_cont_cont_gen(
            x=self.norm_2d_0_6_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_6_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
            metric_x=2.0,  # Metric doesn't matter for scalars
            metric_y=2.0,
        )
        mi_0_6_cheb = mi._mi_cont_cont_cheb_only(
            x=self.norm_2d_0_6_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_6_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
        )
        self.assertTrue(np.isclose(mi_0_6_cheb, mi_0_6_gen))

        # Test with multidimensional

        mi_2d_2d_0_3_0_6_gen = mi._mi_cont_cont_gen(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            n_neighbors=3,
            metric_x=np.inf,
            metric_y=np.inf,
        )
        mi_2d_2d_0_3_0_6_cheb = mi._mi_cont_cont_cheb_only(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            n_neighbors=3,
        )
        self.assertTrue(np.isclose(mi_2d_2d_0_3_0_6_cheb, mi_2d_2d_0_3_0_6_gen))

    def test_dispatch(self):
        # Test scalar dispatch (always to cheb_only
        mi_0_3_cheb = mi._mi_cont_cont_cheb_only(
            x=self.norm_2d_0_6_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_6_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
        )
        mi_0_3_disp = mi._mi_cont_cont(
            x=self.norm_2d_0_6_sample_1000[:, 0].reshape(-1, 1),
            y=self.norm_2d_0_6_sample_1000[:, 1].reshape(-1, 1),
            n_neighbors=3,
            metric_x=np.inf,
            metric_y=np.inf,
        )

        self.assertTrue(np.isclose(mi_0_3_disp, mi_0_3_cheb))

        # Test 2D dispatch to the general method
        mi_2d_2d_0_3_0_6_gen = mi._mi_cont_cont_gen(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            n_neighbors=3,
            metric_x=2.0,
            metric_y=np.inf,
        )
        mi_2d_2d_0_3_0_6_disp = mi._mi_cont_cont(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            n_neighbors=3,
            metric_x=2.0,
            metric_y=np.inf,
        )

        self.assertTrue(np.isclose(mi_2d_2d_0_3_0_6_disp, mi_2d_2d_0_3_0_6_gen))

        # Test 2D dispatch to cheb_only
        mi_2d_2d_0_3_0_6_cheb = mi._mi_cont_cont_cheb_only(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            n_neighbors=3,
        )
        mi_2d_2d_0_3_0_6_disp = mi._mi_cont_cont(
            x=self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            y=self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
            n_neighbors=3,
            metric_x=np.inf,
            metric_y=np.inf,
        )
        self.assertTrue(np.isclose(mi_2d_2d_0_3_0_6_disp, mi_2d_2d_0_3_0_6_cheb))


class TestDiscCont(unittest.TestCase):
    def test_mi_disc_cont(self):
        # MODIFIED FROM SKLEARN (sklearn/feature_selection/tests/test_mutual_info.py):

        # To test define a joint distribution as follows:
        # p(x, y) = p(x) p(y | x)
        # X ~ Bernoulli(p)
        # (Y | x = 0) ~ Uniform(-1, 1)
        # (Y | x = 1) ~ Uniform(0, 2)

        # Use the following formula for mutual information:
        # I(X; Y) = H(Y) - H(Y | X)
        # Two entropies can be computed by hand:
        # H(Y) = -(1-p)/2 * ln((1-p)/2) - p/2*log(p/2) - 1/2*log(1/2)
        # H(Y | X) = ln(2)

        # Now we need to implement sampling from out distribution, which is
        # done easily using conditional distribution logic.

        n_samples = 1000
        generator = np.random.default_rng(1618)

        for p in [0.3, 0.5, 0.7]:
            x = generator.uniform(low=0.0, high=1.0, size=n_samples) > p

            y = np.empty(n_samples, dtype=float)

            y_dist1_index = x
            y_dist2_index = ~x

            y[y_dist1_index] = generator.uniform(-1.0, 1.0, size=np.sum(x))
            y[y_dist2_index] = generator.uniform(0, 2, size=np.sum(~x))

            mi_theory = -0.5 * (
                (1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)
            ) - np.log(2)

            for n_neighbors in [3, 5, 7]:
                mi_computed = mi._mi_disc_cont(
                    discrete=x.reshape(-1, 1),
                    continuous=y.reshape(-1, 1),
                    n_neighbors=n_neighbors,
                )
                self.assertTrue(np.isclose(mi_computed, mi_theory, rtol=1e-1))


class TestDiscDisc(unittest.TestCase):
    # Adapted from sklearn mi tests (see license file for full license)
    def test_mi_disc_disc(self):
        x = np.array([0, 1, 1, 0, 0]).reshape(-1, 1)
        y = np.array([1, 0, 0, 0, 1]).reshape(-1, 1)
        H_x = H_y = -(3 / 5) * np.log(3 / 5) - (2 / 5) * np.log(2 / 5)
        H_xy = (
            -1 / 5 * np.log(1 / 5) - (2 / 5) * np.log(2 / 5) - (2 / 5) * np.log(2 / 5)
        )
        I_xy = H_x + H_y - H_xy

        self.assertAlmostEqual(
            mi._mi_disc_disc(x=x, y=y), I_xy
        )  # TODO: Add more edge cases


class TestHelperFunctions(unittest.TestCase):
    def test_parse_metric(self):
        with self.assertRaises(ValueError):
            metworkpy.utils._arguments._parse_metric(0.5)
        self.assertEqual(5.0, metworkpy.utils._arguments._parse_metric(5.0))
        self.assertEqual(2.0, metworkpy.utils._arguments._parse_metric("Euclidean"))
        self.assertEqual(1.0, metworkpy.utils._arguments._parse_metric("Manhattan"))
        self.assertEqual(np.inf, metworkpy.utils._arguments._parse_metric("Chebyshev"))
        self.assertEqual(10.0, metworkpy.utils._arguments._parse_metric(10))

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
        y = np.array(["a", "b", "c", "a", "c"])
        with self.assertRaises(ValueError):
            _ = mi._check_discrete(sample=x, is_discrete=True)
        self.assertTrue(
            (mi._check_discrete(sample=y, is_discrete=True) == y.reshape(-1, 1)).all()
        )

    def test_jitter_single(self):
        generator = np.random.default_rng(seed=314)
        arr_test = generator.normal(loc=1.0, scale=5, size=(10, 3))
        arr_jittered = metworkpy.utils._jitter._jitter_single(
            arr_test, jitter=1e-10, generator=generator
        )
        # Test that they are not the same
        self.assertTrue(~np.all(arr_test == arr_jittered))
        # But that they are very close
        self.assertTrue(np.isclose(arr_jittered, arr_test).all())

    def test_jitter(self):
        generator = np.random.default_rng(314)
        arr1 = generator.normal(loc=1.0, scale=10, size=(10, 5))
        arr2 = generator.normal(loc=2.0, scale=5.0, size=(10, 3))

        arr1_jit, arr2_jit = metworkpy.utils._jitter._jitter(
            arr1,
            arr2,
            jitter=(1e-9, 1e-10),
            jitter_seed=42,
            discrete_x=False,
            discrete_y=False,
        )

        self.assertTrue(~np.all(arr1_jit == arr1))
        self.assertTrue(np.all(np.isclose(arr1, arr1_jit)))

        self.assertTrue(~np.all(arr2_jit == arr2))
        self.assertTrue(np.all(np.isclose(arr2, arr2_jit)))


if __name__ == "__main__":
    unittest.main()

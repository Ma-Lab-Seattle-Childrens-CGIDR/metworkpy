# Standard Library Imports
import unittest

# External Imports
import numpy as np
import scipy.stats
from scipy.spatial.distance import jensenshannon

# Local imports
from metworkpy.divergence.js_divergence_functions import (
    _js_cont,
    _js_disc,
    js_divergence,
    js_divergence_array,
)


class TestMainKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        # Setup for the continuous cases
        cls.norm_0_3 = generator.normal(loc=0, scale=3, size=500).reshape(-1, 1)
        cls.norm_2_3 = generator.normal(loc=2, scale=3, size=500).reshape(-1, 1)
        cls.norm_5_3 = generator.normal(loc=5, scale=3, size=500).reshape(-1, 1)
        cls.norm_2_10 = generator.normal(loc=2, scale=10, size=500).reshape(-1, 1)
        cls.norm_2_15 = generator.normal(loc=2, scale=15, size=500).reshape(-1, 1)
        cls.norm_2_10_rep = generator.normal(loc=2, scale=10, size=500).reshape(-1, 1)

        # Setup for the discrete cases
        cls.uniform_choice = generator.choice([0, 1, 2, 3, 4, 5], size=500)
        cls.uniform_choice_rep = generator.choice([0, 1, 2, 3, 4, 5], size=500)

        # Use known distributions to calculate a theoretical value
        p_weights = np.array([5 / 30, 10 / 30, 7 / 30, 8 / 30])
        q_weights = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

        cls.theory_js = jensenshannon(p_weights, q_weights, base=np.e) ** 2

        cls.p_sample = generator.choice(
            [0, 1, 2, 3], size=5000, replace=True, p=p_weights
        )
        cls.q_sample = generator.choice(
            [0, 1, 2, 3], size=5000, replace=True, p=q_weights
        )

    def test_symetry(self):
        # Continuous
        self.assertTrue(
            np.isclose(
                js_divergence(self.norm_2_10, self.norm_0_3),
                js_divergence(self.norm_0_3, self.norm_2_10),
            )
        )
        # Discrete
        self.assertTrue(
            np.isclose(
                js_divergence(self.p_sample, self.q_sample, discrete=True),
                js_divergence(self.q_sample, self.p_sample, discrete=True),
            )
        )

    def test_identical_distributions(self):
        # Continuous
        self.assertTrue(
            np.isclose(
                0.0, js_divergence(self.norm_2_10, self.norm_2_10_rep), atol=1e-1
            )
        )
        self.assertTrue(
            np.isclose(
                js_divergence(
                    self.uniform_choice, self.uniform_choice_rep, discrete=True
                ),
                0.0,
                atol=5e-2,
            )
        )

    def test_greater_sd(self):
        # JS should increase when the difference in Standard deviations between the
        # distributions increases
        small_sd_diff = js_divergence(self.norm_2_3, self.norm_2_10)
        large_sd_diff = js_divergence(self.norm_2_3, self.norm_2_15)
        self.assertLess(small_sd_diff, large_sd_diff)

    def test_greater_mean(self):
        # JS Divergence should increase when the difference in means in greater
        small_mean_diff = js_divergence(self.norm_0_3, self.norm_2_3)
        large_mean_diff = js_divergence(self.norm_0_3, self.norm_5_3)
        self.assertLess(small_mean_diff, large_mean_diff)

    def test_known_js(self):
        comp_js = js_divergence(self.p_sample, self.q_sample, discrete=True)
        self.assertTrue(np.isclose(comp_js, self.theory_js, atol=5e-3))

    def test_input_handling(self):
        self.assertTrue(
            np.isclose(
                js_divergence(self.norm_0_3, self.norm_2_10),
                js_divergence(self.norm_0_3.ravel(), self.norm_2_10.ravel()),
            )
        )
        self.assertTrue(
            np.isclose(
                js_divergence(self.norm_0_3, self.norm_2_10),
                js_divergence(list(self.norm_0_3), list(self.norm_2_10)),
            )
        )

        self.assertTrue(
            np.isclose(
                js_divergence(self.p_sample, self.q_sample, discrete=True),
                js_divergence(list(self.p_sample), list(self.q_sample), discrete=True),
            )
        )

        self.assertTrue(
            np.isclose(
                js_divergence(self.p_sample, self.q_sample, discrete=True),
                js_divergence(
                    self.p_sample.reshape(-1, 1),
                    self.q_sample.reshape(-1, 1),
                    discrete=True,
                ),
            )
        )

    def test_jitter(self):
        js_no_jitter = js_divergence(self.norm_0_3, self.norm_2_10)
        js_jitter = js_divergence(
            self.norm_0_3, self.norm_2_10, jitter_seed=42, jitter=1e-10
        )
        self.assertTrue(np.isclose(js_jitter, js_no_jitter))


class TestContinuousJS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        cls.norm_0_3 = generator.normal(loc=0, scale=3, size=500).reshape(-1, 1)
        cls.norm_2_3 = generator.normal(loc=2, scale=3, size=500).reshape(-1, 1)
        cls.norm_5_3 = generator.normal(loc=5, scale=3, size=500).reshape(-1, 1)
        cls.norm_2_10 = generator.normal(loc=2, scale=10, size=500).reshape(-1, 1)
        cls.norm_2_15 = generator.normal(loc=2, scale=15, size=500).reshape(-1, 1)
        cls.norm_2_10_rep = generator.normal(loc=2, scale=10, size=500).reshape(-1, 1)

        # Multidimensional case
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

    def test_symetry(self):
        # JS should be symmetric, so the estimates should be symmetric
        self.assertTrue(
            np.isclose(
                _js_cont(self.norm_2_10, self.norm_0_3),
                _js_cont(self.norm_0_3, self.norm_2_10),
            )
        )

    def test_identical_distributions(self):
        # JS should be close to 0 for identical distributions
        self.assertTrue(
            np.isclose(0.0, _js_cont(self.norm_2_10, self.norm_2_10_rep), atol=1e-1)
        )

    def test_greater_sd(self):
        # JS should increase when the difference in Standard deviations between the
        # distributions increases
        small_sd_diff = _js_cont(self.norm_2_3, self.norm_2_10)
        large_sd_diff = _js_cont(self.norm_2_3, self.norm_2_15)
        self.assertLess(small_sd_diff, large_sd_diff)

    def test_greater_mean(self):
        # JS Divergence should increase when the difference in means in greater
        small_mean_diff = _js_cont(self.norm_0_3, self.norm_2_3)
        large_mean_diff = _js_cont(self.norm_0_3, self.norm_5_3)
        self.assertLess(small_mean_diff, large_mean_diff)

    def test_multidimensional(self):
        _ = _js_cont(
            self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
        )


class TestDiscreteJS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        cls.uniform_choice = generator.choice([0, 1, 2, 3, 4, 5], size=500)
        cls.uniform_choice_rep = generator.choice([0, 1, 2, 3, 4, 5], size=500)

        # Use known distributions to calculate a theoretical value
        p_weights = np.array([5 / 30, 10 / 30, 7 / 30, 8 / 30])
        q_weights = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

        cls.theory_js = jensenshannon(p_weights, q_weights, base=np.e) ** 2

        cls.p_sample = generator.choice(
            [0, 1, 2, 3], size=5000, replace=True, p=p_weights
        )
        cls.q_sample = generator.choice(
            [0, 1, 2, 3], size=5000, replace=True, p=q_weights
        )

    def test_identical_distributions(self):
        # JS estimate for identical distributions should be close to 0
        self.assertTrue(
            np.isclose(
                _js_disc(self.uniform_choice, self.uniform_choice_rep), 0.0, atol=5e-2
            )
        )

    def test_symetry(self):
        self.assertTrue(
            np.isclose(
                _js_disc(self.p_sample, self.q_sample),
                _js_disc(self.q_sample, self.p_sample),
            )
        )

    def test_known_js(self):
        comp_js = _js_disc(self.p_sample, self.q_sample)
        self.assertTrue(np.isclose(comp_js, self.theory_js, atol=5e-3))


class TestDivergenceArrayJS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        # Setup for the continuous cases
        cls.NROW = 500
        cls.NCOL = 20
        cls.norm_0_3 = generator.normal(
            loc=0, scale=3, size=cls.NROW * cls.NCOL
        ).reshape(cls.NROW, cls.NCOL)
        cls.norm_2_3 = generator.normal(
            loc=2, scale=3, size=cls.NROW * cls.NCOL
        ).reshape(cls.NROW, cls.NCOL)
        cls.norm_5_3 = generator.normal(
            loc=5, scale=3, size=cls.NROW * cls.NCOL
        ).reshape(cls.NROW, cls.NCOL)
        cls.norm_2_10 = generator.normal(
            loc=2, scale=10, size=cls.NROW * cls.NCOL
        ).reshape(cls.NROW, cls.NCOL)
        cls.norm_2_15 = generator.normal(
            loc=2, scale=15, size=cls.NROW * cls.NCOL
        ).reshape(cls.NROW, cls.NCOL)
        cls.norm_2_10_rep = generator.normal(
            loc=2, scale=10, size=cls.NROW * cls.NCOL
        ).reshape(cls.NROW, cls.NCOL)

    def test_symetry(self):
        self.assertTrue(
            np.all(
                np.isclose(
                    js_divergence_array(
                        self.norm_0_3, self.norm_2_10, n_neighbors=3, processes=1
                    ),
                    js_divergence_array(
                        self.norm_2_10, self.norm_0_3, n_neighbors=3, processes=1
                    ),
                )
            )
        )

    def test_identical_dist(self):
        self.assertTrue(
            np.all(
                np.isclose(
                    js_divergence_array(
                        self.norm_2_10, self.norm_2_10_rep, n_neighbors=5, processes=1
                    ),
                    0.0,
                    atol=1e-1,
                )
            )
        )

    def test_greater_sd(self):
        small_sd = js_divergence_array(
            self.norm_2_3, self.norm_2_10, n_neighbors=3, processes=1
        )
        large_sd = js_divergence_array(
            self.norm_2_3, self.norm_2_15, n_neighbors=3, processes=1
        )
        self.assertTrue(np.all(np.less(small_sd, large_sd)))

    def test_greater_mean(self):
        small_mean = js_divergence_array(
            self.norm_0_3, self.norm_2_3, n_neighbors=3, processes=1
        )
        large_mean = js_divergence_array(
            self.norm_0_3, self.norm_5_3, n_neighbors=3, processes=1
        )
        self.assertTrue(np.all(np.less(small_mean, large_mean)))

    def test_parallel(self):
        for p, q in zip([self.norm_0_3, self.norm_0_3], [self.norm_2_3, self.norm_5_3]):
            serial_js_div = js_divergence_array(p, q, n_neighbors=3, processes=1)
            parallel_js_div = js_divergence_array(p, q, n_neighbors=3, processes=2)
            self.assertTrue(np.all(np.isclose(parallel_js_div, serial_js_div)))


if __name__ == "__main__":
    unittest.main()

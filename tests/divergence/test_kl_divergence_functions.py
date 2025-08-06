# Standard Library Imports
import unittest

# External Imports
import numpy as np
from scipy.stats import multivariate_normal

# Local Imports
import metworkpy.divergence.kl_divergence_functions
from metworkpy.divergence.kl_divergence_functions import kl_divergence_array


class TestMainKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        # Setup for Continuous Case
        cls.norm_0_3 = generator.normal(loc=0, scale=3, size=500).reshape(-1, 1)
        cls.norm_2_10 = generator.normal(loc=2, scale=10, size=500).reshape(-1, 1)
        cls.norm_2_10_rep = generator.normal(loc=2, scale=10, size=500).reshape(-1, 1)
        cls.theory_kl_div = np.log(10 / 3) + (3**2 + (0 - 2) ** 2) / (2 * 10**2) - 0.5
        # Setup for the discrete case
        cls.known_p = generator.choice(
            [0, 1, 2, 3], size=50000, replace=True, p=[7 / 25, 12 / 25, 4 / 25, 2 / 25]
        )
        cls.known_p_rep = generator.choice(
            [0, 1, 2, 3], size=10000, replace=True, p=[7 / 25, 12 / 25, 4 / 25, 2 / 25]
        )
        cls.known_q = generator.choice(
            [0, 1, 2, 3], size=50000, replace=True, p=[1 / 4, 1 / 4, 1 / 4, 1 / 4]
        )
        cls.theory_kl_p_q = (
            (7 / 25) * np.log((7 / 25) / (1 / 4))
            + (12 / 25) * np.log((12 / 25) / (1 / 4))
            + (4 / 25) * np.log((4 / 25) / (1 / 4))
            + (2 / 25) * np.log((2 / 25) / (1 / 4))
        )
        cls.theory_kl_q_p = (
            (1 / 4) * np.log((1 / 4) / (7 / 25))
            + (1 / 4) * np.log((1 / 4) / (12 / 25))
            + (1 / 4) * np.log((1 / 4) / (4 / 25))
            + (1 / 4) * np.log((1 / 4) / (2 / 25))
        )

    def test_kl_cont(self):
        calc_kl_div = metworkpy.divergence.kl_divergence_functions.kl_divergence(
            p=self.norm_0_3, q=self.norm_2_10, n_neighbors=3
        )
        self.assertTrue(np.isclose(calc_kl_div, self.theory_kl_div, rtol=2e-1))

        calc_kl_div_0 = metworkpy.divergence.kl_divergence_functions.kl_divergence(
            p=self.norm_2_10, q=self.norm_2_10_rep, n_neighbors=4
        )
        self.assertTrue(np.isclose(calc_kl_div_0, 0.0, rtol=1e-1, atol=0.05))

    def test_identical_distributions(self):
        # Identical distributions should have KL divergence estimates near 0
        self.assertTrue(
            np.isclose(
                metworkpy.divergence.kl_divergence_functions._kl_disc(
                    self.known_p, self.known_p_rep
                ),
                0.0,
                atol=5e-2,
            )
        )

    def test_known_kl(self):
        # Test against theory KL values
        calc_kl_p_q = metworkpy.divergence.kl_divergence_functions.kl_divergence(
            self.known_p, self.known_q, discrete=True
        )
        calc_kl_q_p = metworkpy.divergence.kl_divergence_functions.kl_divergence(
            self.known_q, self.known_p, discrete=True
        )
        self.assertTrue(np.isclose(calc_kl_p_q, self.theory_kl_p_q, rtol=1e-1))
        self.assertTrue(np.isclose(calc_kl_q_p, self.theory_kl_q_p, rtol=1e-1))

    def test_input_handling(self):
        self.assertTrue(
            np.isclose(
                metworkpy.divergence.kl_divergence_functions.kl_divergence(
                    self.norm_0_3, self.norm_2_10
                ),
                metworkpy.divergence.kl_divergence_functions.kl_divergence(
                    self.norm_0_3.ravel(), self.norm_2_10.ravel()
                ),
            )
        )
        self.assertTrue(
            np.isclose(
                metworkpy.divergence.kl_divergence_functions.kl_divergence(
                    self.norm_0_3, self.norm_2_10
                ),
                metworkpy.divergence.kl_divergence_functions.kl_divergence(
                    list(self.norm_0_3), list(self.norm_2_10)
                ),
            )
        )

        self.assertTrue(
            np.isclose(
                metworkpy.divergence.kl_divergence_functions.kl_divergence(
                    self.known_p, self.known_q, discrete=True
                ),
                metworkpy.divergence.kl_divergence_functions.kl_divergence(
                    list(self.known_p), list(self.known_q), discrete=True
                ),
            )
        )

        self.assertTrue(
            np.isclose(
                metworkpy.divergence.kl_divergence_functions.kl_divergence(
                    self.known_p, self.known_q, discrete=True
                ),
                metworkpy.divergence.kl_divergence_functions.kl_divergence(
                    self.known_p.reshape(-1, 1),
                    self.known_q.reshape(-1, 1),
                    discrete=True,
                ),
            )
        )


class TestContinuousKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        cls.norm_0_3 = generator.normal(loc=0, scale=3, size=500).reshape(-1, 1)
        cls.norm_2_10 = generator.normal(loc=2, scale=10, size=500).reshape(-1, 1)
        cls.norm_2_10_rep = generator.normal(loc=2, scale=10, size=500).reshape(-1, 1)
        cls.theory_kl_div = np.log(10 / 3) + (3**2 + (0 - 2) ** 2) / (2 * 10**2) - 0.5

        # Multidimensional case
        norm_2d_2d_0_3_0_6 = multivariate_normal(
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

    def test_kl_cont(self):
        calc_kl_div = metworkpy.divergence.kl_divergence_functions._kl_cont(
            p=self.norm_0_3, q=self.norm_2_10, n_neighbors=3
        )
        self.assertTrue(np.isclose(calc_kl_div, self.theory_kl_div, rtol=2e-1))

        calc_kl_div_0 = metworkpy.divergence.kl_divergence_functions._kl_cont(
            p=self.norm_2_10, q=self.norm_2_10_rep, n_neighbors=4
        )
        self.assertTrue(np.isclose(calc_kl_div_0, 0.0, rtol=1e-1, atol=0.05))

    def test_multidimensional(self):
        _ = metworkpy.divergence.kl_divergence_functions._kl_cont(
            self.norm_2d_2d_0_3_0_6_sample_1000[:, [0, 1]],
            self.norm_2d_2d_0_3_0_6_sample_1000[:, [2, 3]],
        )

    def test_jitter(self):
        kl_no_jitter = metworkpy.divergence.kl_divergence_functions.kl_divergence(
            self.norm_0_3, self.norm_2_10
        )
        kl_jitter = metworkpy.divergence.kl_divergence_functions.kl_divergence(
            self.norm_0_3, self.norm_2_10, jitter_seed=42, jitter=1e-10
        )
        self.assertTrue(np.isclose(kl_jitter, kl_no_jitter))


class TestDiscreteKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(42)
        cls.known_p = generator.choice(
            [0, 1, 2, 3], size=5000, replace=True, p=[7 / 25, 12 / 25, 4 / 25, 2 / 25]
        )
        cls.known_p_rep = generator.choice(
            [0, 1, 2, 3], size=5000, replace=True, p=[7 / 25, 12 / 25, 4 / 25, 2 / 25]
        )
        cls.known_q = generator.choice(
            [0, 1, 2, 3], size=5000, replace=True, p=[1 / 4, 1 / 4, 1 / 4, 1 / 4]
        )
        cls.theory_kl_p_q = (
            (7 / 25) * np.log((7 / 25) / (1 / 4))
            + (12 / 25) * np.log((12 / 25) / (1 / 4))
            + (4 / 25) * np.log((4 / 25) / (1 / 4))
            + (2 / 25) * np.log((2 / 25) / (1 / 4))
        )
        cls.theory_kl_q_p = (
            (1 / 4) * np.log((1 / 4) / (7 / 25))
            + (1 / 4) * np.log((1 / 4) / (12 / 25))
            + (1 / 4) * np.log((1 / 4) / (4 / 25))
            + (1 / 4) * np.log((1 / 4) / (2 / 25))
        )

    def test_identical_distributions(self):
        # Identical distributions should have KL divergence estimates near 0
        self.assertTrue(
            np.isclose(
                metworkpy.divergence.kl_divergence_functions._kl_disc(
                    self.known_p, self.known_p_rep
                ),
                0.0,
                atol=5e-2,
            )
        )

    def test_known_kl(self):
        # Test against theory KL values
        calc_kl_p_q = metworkpy.divergence.kl_divergence_functions._kl_disc(
            self.known_p, self.known_q
        )
        calc_kl_q_p = metworkpy.divergence.kl_divergence_functions._kl_disc(
            self.known_q, self.known_p
        )

        self.assertTrue(np.isclose(calc_kl_p_q, self.theory_kl_p_q, rtol=1e-1))
        self.assertTrue(np.isclose(calc_kl_q_p, self.theory_kl_q_p, rtol=1e-1))


class TestDivergenceArrayKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        NCOL = 20
        NROW = 10000
        cls.NCOL = NCOL
        cls.NROW = NROW
        cls.norm_0_3 = generator.normal(loc=0, scale=3, size=NROW * NCOL).reshape(
            NROW, NCOL
        )
        cls.norm_2_10 = generator.normal(loc=2, scale=10, size=NROW * NCOL).reshape(
            NROW, NCOL
        )
        cls.norm_2_10_rep = generator.normal(loc=2, scale=10, size=NROW * NCOL).reshape(
            NROW, NCOL
        )
        cls.theory_kl_div = np.log(10 / 3) + (3**2 + (0 - 2) ** 2) / (2 * 10**2) - 0.5

    def test_serial(self):
        # Test Known Divergence
        test_kl_divergence = kl_divergence_array(
            self.norm_0_3, self.norm_2_10, n_neighbors=5, processes=1
        )
        self.assertEqual(len(test_kl_divergence), self.NCOL)
        self.assertTrue(
            np.all(np.isclose(test_kl_divergence, self.theory_kl_div, rtol=2e-1))
        )
        # Test 0 Divergence
        test_kl_divergence = kl_divergence_array(
            self.norm_2_10, self.norm_2_10_rep, n_neighbors=5, processes=1
        )
        self.assertTrue(
            np.all(np.isclose(test_kl_divergence, 0.0, rtol=1e-1, atol=0.05))
        )

    def test_parallel(self):
        kl_div_serial = kl_divergence_array(
            self.norm_0_3, self.norm_2_10, n_neighbors=3, processes=1
        )
        kl_div_parallel = kl_divergence_array(
            self.norm_0_3, self.norm_2_10, n_neighbors=3, processes=2
        )
        self.assertTrue(np.all(np.isclose(kl_div_parallel, kl_div_serial)))


if __name__ == "__main__":
    unittest.main()

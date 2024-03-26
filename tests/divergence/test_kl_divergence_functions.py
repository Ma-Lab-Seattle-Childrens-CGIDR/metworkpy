# Standard Library Imports
import unittest

# External Imports
import numpy as np

# Local Imports
from metworkpy.divergence.kl_divergence_functions import kl_divergence, _kl_cont, \
    _kl_disc


class TestContinuousKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        cls.norm_0_3 = generator.normal(loc=0, scale=3, size=500).reshape(-1, 1)
        cls.norm_2_10 = generator.normal(loc=2, scale=10, size=500).reshape(-1, 1)
        cls.norm_2_10_rep = generator.normal(loc=2, scale=10, size=500).reshape(-1,1)
        cls.theory_kl_div = np.log(10 / 3) + (3 ** 2 + (0 - 2) ** 2) / (
                    2 * 10 ** 2) - 0.5

    def test_kl_cont(self):
        calc_kl_div = _kl_cont(p=self.norm_0_3, q=self.norm_2_10,
                               n_neighbors=3)
        self.assertTrue(np.isclose(calc_kl_div, self.theory_kl_div, rtol=2e-1))

        calc_kl_div_0 = _kl_cont(p=self.norm_2_10, q=self.norm_2_10_rep,
                                 n_neighbors=4)
        self.assertTrue(np.isclose(calc_kl_div_0, 0., rtol=1e-1, atol=0.05))


if __name__ == '__main__':
    unittest.main()

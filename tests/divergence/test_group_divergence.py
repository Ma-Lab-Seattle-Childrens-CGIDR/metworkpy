# Standard Library Imports
import unittest

# External Imports
import numpy as np

# Local Imports


def kl_div_theory(mean1, mean2, std1, std2):
    return (
        np.log(std2 / std1) + (std1**2 + (mean1 - mean2) ** 2) / (2 * (std2**2)) - 0.5
    )


class TestGroupDivergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        arr1 = np.hstack(
            (
                generator.normal(loc=0, scale=3, size=1_000).reshape(-1, 1),
                generator.normal(loc=10, scale=3, size=1_000).reshape(-1, 1),
                generator.normal(loc=0, scale=15, size=1_000).reshape(-1, 1),
                generator.normal(loc=10, scale=15, size=1_000).reshape(-1, 1),
                generator.normal(loc=0, scale=3, size=1_000).reshape(-1, 1),
            )
        )
        arr2 = np.hstack(
            (
                generator.normal(loc=10, scale=3, size=1_000).reshape(-1, 1),
                generator.normal(loc=10, scale=15, size=1_000).reshape(-1, 1),
                generator.normal(loc=0, scale=3, size=1_000).reshape(-1, 1),
                generator.normal(loc=0, scale=15, size=1_000).reshape(-1, 1),
                generator.normal(loc=10, scale=15, size=1_000).reshape(-1, 1),
            )
        )
        cls.arr1 = arr1
        cls.arr2 = arr2

    def test_calculate_divergence_grouped(self):
        pass


if __name__ == "__main__":
    unittest.main()

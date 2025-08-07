# Standard Library Imports
import unittest

# External Imports
import numpy as np
import pandas as pd

# Local Imports
from metworkpy.divergence.group_divergence import calculate_divergence_grouped
from metworkpy.divergence.kl_divergence_functions import kl_divergence


class TestGroupDivergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(314)
        df1 = pd.DataFrame(
            np.hstack(
                (
                    generator.normal(loc=0, scale=3, size=1_000).reshape(-1, 1),
                    generator.normal(loc=10, scale=3, size=1_000).reshape(-1, 1),
                    generator.normal(loc=0, scale=15, size=1_000).reshape(-1, 1),
                    generator.normal(loc=10, scale=15, size=1_000).reshape(-1, 1),
                    generator.normal(loc=0, scale=3, size=1_000).reshape(-1, 1),
                )
            ),
            columns=[0, 1, 2, 3, 4],
        )
        df2 = pd.DataFrame(
            np.hstack(
                (
                    generator.normal(loc=10, scale=3, size=1_000).reshape(-1, 1),
                    generator.normal(loc=10, scale=15, size=1_000).reshape(-1, 1),
                    generator.normal(loc=0, scale=3, size=1_000).reshape(-1, 1),
                    generator.normal(loc=0, scale=15, size=1_000).reshape(-1, 1),
                    generator.normal(loc=10, scale=15, size=1_000).reshape(-1, 1),
                )
            ),
            columns=[0, 1, 2, 3, 4],
        )
        cls.df1 = df1
        cls.df2 = df2

    def test_calculate_divergence_grouped(self):
        divergence_groups = {
            "A": [0],
            "B": [1],
            "C": [2],
            "D": [3],
            "E": [4],
            "F": [0, 2, 4],
            "G": [1, 4, 2],
        }
        divergence_results = calculate_divergence_grouped(
            self.df1,
            self.df2,
            divergence_groups=divergence_groups,
            divergence_type="kl",
            processes=1,
        )
        self.assertIsInstance(divergence_results, pd.Series)
        self.assertCountEqual(list(divergence_results.index), divergence_groups.keys())
        self.assertAlmostEqual(
            divergence_results["A"], kl_divergence(self.df1[0], self.df2[0]), delta=1e-7
        )
        self.assertAlmostEqual(
            divergence_results["B"], kl_divergence(self.df1[1], self.df2[1]), delta=1e-7
        )
        self.assertAlmostEqual(
            divergence_results["C"], kl_divergence(self.df1[2], self.df2[2]), delta=1e-7
        )
        self.assertAlmostEqual(
            divergence_results["D"], kl_divergence(self.df1[3], self.df2[3]), delta=1e-7
        )
        self.assertAlmostEqual(
            divergence_results["E"], kl_divergence(self.df1[4], self.df2[4]), delta=1e-7
        )
        self.assertAlmostEqual(
            divergence_results["F"],
            kl_divergence(self.df1[[0, 2, 4]], self.df2[[0, 2, 4]]),
            delta=1e-7,
        )
        self.assertAlmostEqual(
            divergence_results["G"],
            kl_divergence(self.df1[[1, 4, 2]], self.df2[[1, 4, 2]]),
            delta=1e-7,
        )


if __name__ == "__main__":
    unittest.main()

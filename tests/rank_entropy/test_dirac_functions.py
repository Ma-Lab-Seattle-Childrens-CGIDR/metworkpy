# Standard Library Imports
import math
import unittest

# External Imports
import numpy as np

# Local Imports
from metworkpy.rank_entropy import dirac_functions


class TestRankFunctions(unittest.TestCase):
    def test_rank_vector(self):
        # Test ordered vector
        self.assertTrue(
            np.all(
                np.equal(
                    dirac_functions._rank_vector(np.arange(10, dtype=int)),
                    np.ones(math.comb(10, 2), dtype=int),
                )
            )
        )
        # Test reversed vector
        self.assertTrue(
            np.all(
                np.equal(
                    dirac_functions._rank_vector(np.arange(10, dtype=int)[::-1]),
                    np.zeros(math.comb(10, 2), dtype=int),
                )
            )
        )
        # Known Test Vector
        test_vec = np.array([2, 1, 3, 5, 4, 6])
        known_res = np.ones(math.comb(6, 2), dtype=int)
        known_res[0] = 0
        known_res[12] = 0
        self.assertTrue(
            np.all(np.equal(dirac_functions._rank_vector(test_vec), known_res))
        )
        # Test Repeated Values
        test_vec = np.array([1, 1, 2])
        known_res = np.array([0, 1, 1])
        self.assertTrue(
            np.all(np.equal(dirac_functions._rank_vector(test_vec), known_res))
        )


if __name__ == "__main__":
    unittest.main()

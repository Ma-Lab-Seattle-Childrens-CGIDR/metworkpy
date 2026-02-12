import unittest

import numpy as np

from metworkpy.utils.permutation import permutation_test


class TestPermutationTest(unittest.TestCase):
    def test_pairings(self):
        test_x = np.arange(8).reshape((4, 2))
        test_y = test_x * 10

        # Create a function for statistic that will do some
        # basic checks
        # Abusing the behavior of default lists for remembering
        # previously passed arrays
        def test_stat_func(
            x: np.ndarray, y: np.ndarray, prev_x_list=[], prev_y_list=[]
        ):
            np.testing.assert_array_equal(x.shape, np.array([4, 2]))
            np.testing.assert_array_equal(y.shape, np.array([4, 2]))
            # If there is a previous x,y check that x is the same but y is reordered
            if (
                len(prev_x_list) > 0
            ):  # The first should just be the dataset directly
                prev_x = prev_x_list[-1]
                np.testing.assert_allclose(x, prev_x)
            if len(prev_y_list) > 0:
                prev_y = prev_y_list[-1]
                # Ensure y is different
                assert np.sum((y != prev_y)) > 0
            prev_x_list.append(x.copy())
            prev_y_list.append(y.copy())
            return 0.0

        permutation_test(
            test_x,
            test_y,
            statistic=test_stat_func,
            axis=0,
            permutation_type="pairings",
            n_resamples=3,
            estimation_method="empirical",
            rng=1618,
        )

    def test_independent(self):
        test_x = np.arange(8).reshape((4, 2))
        test_y = np.arange(10).reshape((5, 2)) * 10

        def test_stat_func(
            x: np.ndarray, y: np.ndarray, prev_x_list=[], prev_y_list=[]
        ):
            np.testing.assert_array_equal(x.shape, np.array([4, 2]))
            np.testing.assert_array_equal(y.shape, np.array([5, 2]))
            # If there is a previous x,y check that are both different
            if len(prev_x_list) > 0:
                # Ensure x is different
                assert np.sum((x != prev_x_list[-1])) > 0
            if len(prev_y_list) > 0:
                # Ensure y is different
                assert np.sum((y != prev_y_list[-1])) > 0
            prev_x_list.append(x.copy())
            prev_y_list.append(y.copy())
            return 0.0

        permutation_test(
            test_x,
            test_y,
            statistic=test_stat_func,
            axis=0,
            permutation_type="independent",
            n_resamples=3,
            estimation_method="empirical",
            rng=1618,
        )


if __name__ == "__main__":
    unittest.main()

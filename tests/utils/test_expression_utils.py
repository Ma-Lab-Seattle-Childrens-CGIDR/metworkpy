# Standard Library Imports
import unittest

# External imports
import numpy as np
import pandas as pd

# Local imports
from metworkpy.utils.expression_utils import (
    count_to_rpkm,
    count_to_fpkm,
    count_to_tpm,
    rpkm_to_tpm,
    fpkm_to_tpm,
    count_to_cpm,
    expr_to_imat_gene_weights,
)


class TestConversionFunctions(unittest.TestCase):
    feature_length = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up values for use in the tests, mainly fake expression data
        """
        gene_list = [f"g_{i:03}" for i in range(1, 101)]
        sample_list = [f"s_{i:03}" for i in range(1, 6)]
        # create rng generator
        rng = np.random.default_rng(42)
        # Generate random count values of the right size
        count_values = rng.integers(low=0, high=500_000, size=(5, 100))
        cls.count_data = pd.DataFrame(
            count_values, index=sample_list, columns=gene_list
        )
        # create feature length data
        feature_length_data = rng.integers(low=20, high=500, size=100)
        cls.feature_length = pd.Series(feature_length_data, index=gene_list)
        # Create feature_length with missing genes
        cls.feature_length_missing = cls.feature_length.iloc[0:90]
        # Also create small examples with known values
        cls.small_counts = pd.DataFrame(
            {
                "A": [83, 50, 85, 17, 91],
                "B": [20, 6, 23, 45, 75],
                "C": [54, 65, 53, 97, 38],
                "D": [58, 79, 77, 32, 64],
            }
        )
        cls.small_feature_length = pd.Series(
            [10, 50, 25, 60], index=["A", "B", "C", "D"]
        )

    def test_count_to_rpkm(self):
        """
        Test the count_to_rpkm function
        """
        # Calculate the rpkm from count data
        rpkm = count_to_rpkm(self.count_data, self.feature_length)
        # Should return a dataframe
        self.assertIsInstance(rpkm, pd.DataFrame)
        # Should be the same size as before
        self.assertEqual(rpkm.shape, self.count_data.shape)
        # Should have the same column names as before
        self.assertTrue(rpkm.columns.equals(self.count_data.columns))
        # Should have the same index as count data
        self.assertTrue(rpkm.index.equals(self.count_data.index))
        # Use known values for the small examples
        small_rpkm = count_to_rpkm(self.small_counts, self.small_feature_length)
        known_rpkm = pd.DataFrame(
            {
                "A": [
                    38604651.1627907,
                    25000000,
                    35714285.7142857,
                    8900523.56020942,
                    33955223.880597,
                ],
                "B": [
                    1860465.11627907,
                    600000,
                    1932773.1092437,
                    4712041.88481675,
                    5597014.92537313,
                ],
                "C": [
                    10046511.627907,
                    13000000,
                    8907563.02521008,
                    20314136.1256544,
                    5671641.79104478,
                ],
                "D": [
                    4496124.03100775,
                    6583333.33333333,
                    5392156.8627451,
                    2792321.11692845,
                    3980099.50248756,
                ],
            }
        )
        np.testing.assert_array_almost_equal(small_rpkm, known_rpkm)
        # Check that it throws a warning when given incomplete gene data
        with self.assertWarns(Warning):
            count_to_rpkm(self.count_data, self.feature_length_missing)

    def test_count_to_fpkm(self):
        """
        Test the count_to_fpkm function
        """
        # Since this is just a wrapper, just test that it returns the same
        # as count_to_rpkm
        fpkm = count_to_fpkm(self.count_data, self.feature_length)
        rpkm = count_to_rpkm(self.count_data, self.feature_length)
        self.assertTrue(np.all(np.isclose(fpkm, rpkm)))
        # Check that it throws a warning when given incomplete gene data
        with self.assertWarns(Warning):
            count_to_fpkm(self.count_data, self.feature_length_missing)

    def test_count_to_tpm(self):
        """
        Test the count_to_tpm function
        """
        tpm = count_to_tpm(self.count_data, self.feature_length)
        self.assertTrue(tpm.index.equals(self.count_data.index))
        self.assertTrue(tpm.columns.equals(self.count_data.columns))
        self.assertIsInstance(tpm, pd.DataFrame)
        self.assertEqual(tpm.shape, self.count_data.shape)
        # Check that it throws a warning when given incomplete gene data
        with self.assertWarns(Warning):
            count_to_tpm(self.count_data, self.feature_length_missing)
        # Test that the sum of each row is 1e6
        self.assertTrue(np.all(np.isclose(tpm.sum(axis=1), 1e6)))
        known_tpm = pd.DataFrame(
            {
                "A": [
                    701803.833145434,
                    553301.364810033,
                    687516.850903208,
                    242395.437262357,
                    690091.001011122,
                ],
                "B": [
                    33821.8714768884,
                    13279.2327554408,
                    37206.7942841736,
                    128326.996197719,
                    113751.263902932,
                ],
                "C": [
                    182638.105975197,
                    287716.709701217,
                    171474.7910488,
                    553231.939163498,
                    115267.947421638,
                ],
                "D": [
                    81736.1894024803,
                    145702.692733309,
                    103801.563763818,
                    76045.6273764259,
                    80889.7876643074,
                ],
            }
        )
        np.testing.assert_array_almost_equal(
            count_to_tpm(self.small_counts, self.small_feature_length), known_tpm
        )

    def test_count_to_cpm(self):
        """
        Test the count_to_cpm function
        """
        cpm = count_to_cpm(self.count_data)
        self.assertIsInstance(cpm, pd.DataFrame)
        self.assertTrue(cpm.index.equals(self.count_data.index))
        self.assertTrue(cpm.columns.equals(self.count_data.columns))
        self.assertTrue(np.all(np.isclose(cpm.sum(axis=1), 1e6)))

    def test_rpkm_to_tpm(self):
        """
        Test the rpkm_to_tpm function
        """
        tpm_known = count_to_tpm(self.count_data, self.feature_length)
        rpkm = count_to_rpkm(self.count_data, self.feature_length)
        tpm_test = rpkm_to_tpm(rpkm)
        self.assertIsInstance(tpm_test, pd.DataFrame)
        self.assertTrue(tpm_test.index.equals(rpkm.index))
        self.assertTrue(tpm_test.columns.equals(rpkm.columns))
        self.assertTrue(np.all(np.isclose(tpm_test, tpm_known)))

    def test_fpkm_to_tpm(self):
        """
        Test the fpkm_to_tpm function
        """
        rpkm = count_to_rpkm(self.count_data, self.feature_length)
        fpkm = count_to_fpkm(self.count_data, self.feature_length)
        tpm_rpkm = rpkm_to_tpm(rpkm)
        tpm_fpkm = fpkm_to_tpm(fpkm)
        self.assertTrue(np.all(np.isclose(tpm_rpkm, tpm_fpkm)))


class TestConversionToWeights(unittest.TestCase):
    def test_series_conversion(self):
        test_series = pd.Series([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        quantile = 0.1, 0.9
        expected = pd.Series([-1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        actual = expr_to_imat_gene_weights(test_series, quantile)
        self.assertTrue(np.all(actual == expected))

    def test_dataframe_conversion(self):
        test_frame = pd.DataFrame(
            {
                "A": [-4, -4, -3, -2, -1, 0, 2, 2, 3, 5, 5],
                "B": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                "C": [-6, -4, -3, -2, -1, 0, 0, 2, 3, 3, 5],
            }
        )
        expected = pd.Series([-1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        actual = expr_to_imat_gene_weights(
            test_frame, quantile=(0.1, 0.9), sample_axis=1
        )
        self.assertTrue(np.all(actual == expected))

    def test_subset_genes(self):
        test_series = pd.Series(
            [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            index=[
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
                "Q",
                "R",
                "S",
            ],
        )
        subset = ["A", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "S", "Z"]
        result_series = expr_to_imat_gene_weights(
            test_series, quantile=(0.1, 0.9), subset=subset
        )
        self.assertEqual(result_series["Z"], 0)
        self.assertEqual(result_series["A"], -1)
        self.assertEqual(result_series["S"], 1)


if __name__ == "__main__":
    unittest.main()

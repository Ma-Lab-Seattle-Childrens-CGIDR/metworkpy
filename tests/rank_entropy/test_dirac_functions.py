# Standard Library Imports
import math
import unittest

# External Imports
import numpy as np
from scipy.stats import norm

# Local Imports
from metworkpy.rank_entropy import dirac_functions, _datagen


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

    def test_rank_array(self):
        test_array = np.array(
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 1, 3, 4, 5], [1, 3, 2, 5, 4]]
        )
        expected_array = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            ]
        )
        actual_array = dirac_functions._rank_array(test_array)
        self.assertTupleEqual(actual_array.shape, (4, 10))
        self.assertTrue(np.all(np.equal(actual_array, expected_array)))

    def test_rank_matching_scores(self):
        test_array = np.array(
            [
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [1, 3, 2, 5, 4],
                [4, 5, 2, 1, 3],
            ]
        )
        expected_array = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.2])
        actual_array = dirac_functions._rank_matching_scores(test_array)
        self.assertTupleEqual(actual_array.shape, (8,))
        self.assertListEqual(list(actual_array), list(expected_array))

    def test_rank_conservation_index(self):
        test_array = np.array(
            [
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [3, 1, 2, 5, 4],
                [1, 3, 2, 5, 4],
                [4, 5, 2, 1, 3],
            ]
        )
        expected_rank_conservation_index = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.2]
        ).mean()
        actual_rank_conservation_index = dirac_functions._rank_conservation_index(
            test_array
        )
        self.assertAlmostEqual(
            actual_rank_conservation_index, expected_rank_conservation_index
        )
        low_entropy_test_array = np.array(
            [
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
            ]
        )
        high_entropy_test_array = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
                [4, 3, 2, 6, 5, 1],
                [2, 1, 3, 4, 6, 5],
                [6, 1, 2, 5, 4, 3],
                [6, 1, 2, 3, 4, 5],
                [2, 1, 6, 5, 4, 3],
            ]
        )
        self.assertLess(
            dirac_functions._rank_conservation_index(high_entropy_test_array),
            dirac_functions._rank_conservation_index(low_entropy_test_array),
        )

    def test_differential_entropy(self):
        low_entropy_test_array = np.array(
            [
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
                [1, 3, 2, 4, 6, 5],
            ]
        )
        high_entropy_test_array = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [6, 5, 4, 3, 2, 1],
                [4, 3, 2, 6, 5, 1],
                [2, 1, 3, 4, 6, 5],
                [6, 1, 2, 5, 4, 3],
                [6, 1, 2, 3, 4, 5],
                [2, 1, 6, 5, 4, 3],
            ]
        )
        # Check that identical arrays have no difference
        self.assertAlmostEqual(
            dirac_functions._dirac_differential_entropy(
                low_entropy_test_array, low_entropy_test_array
            ),
            0.0,
        )
        self.assertAlmostEqual(
            dirac_functions._dirac_differential_entropy(
                high_entropy_test_array, high_entropy_test_array
            ),
            0.0,
        )
        # Check that there is a difference between the rank conservation index of the two arrays
        self.assertGreater(
            dirac_functions._dirac_differential_entropy(
                low_entropy_test_array, high_entropy_test_array
            ),
            0.0,
        )


class TestDiracGeneSetEntropy(unittest.TestCase):
    def test_dirac_gene_set_entropy(self):
        (
            test_expression_data,
            ordered_samples,
            unordered_samples,
            ordered_genes,
            unordered_genes,
        ) = _datagen._generate_rank_entropy_data(
            n_ordered_samples=20,
            n_unordered_samples=15,
            n_genes_ordered=20,
            n_genes_unordered=25,
            dist=norm(loc=100, scale=25),
            shuffle_genes=True,
            shuffle_samples=True,
            seed=None,
        )
        rank_conservation_diff, pval = dirac_functions.dirac_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=True,
        )
        self.assertGreater(rank_conservation_diff, 0.0)
        self.assertLessEqual(pval, 0.05)

    def test_parallel(self):
        (
            test_expression_data,
            ordered_samples,
            unordered_samples,
            ordered_genes,
            unordered_genes,
        ) = _datagen._generate_rank_entropy_data(
            n_ordered_samples=20,
            n_unordered_samples=15,
            n_genes_ordered=20,
            n_genes_unordered=25,
            dist=norm(loc=100, scale=25),
            shuffle_genes=True,
            shuffle_samples=True,
            seed=None,
        )
        (
            rank_conservation_diff_serial,
            pval_serial,
        ) = dirac_functions.dirac_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=True,
            processes=1,
        )
        (
            rank_conservation_diff_parallel,
            pval_parallel,
        ) = dirac_functions.dirac_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=True,
            processes=2,
        )
        self.assertAlmostEqual(
            rank_conservation_diff_parallel, rank_conservation_diff_serial
        )
        self.assertAlmostEqual(pval_serial, pval_parallel)

    def test_empirical_cdf(self):
        (
            test_expression_data,
            ordered_samples,
            unordered_samples,
            ordered_genes,
            unordered_genes,
        ) = _datagen._generate_rank_entropy_data(
            n_ordered_samples=20,
            n_unordered_samples=15,
            n_genes_ordered=20,
            n_genes_unordered=25,
            dist=norm(loc=100, scale=25),
            shuffle_genes=True,
            shuffle_samples=True,
            seed=None,
        )
        rank_conservation_diff, pval = dirac_functions.dirac_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=False,
        )
        self.assertGreater(rank_conservation_diff, 0.0)
        self.assertLessEqual(pval, 0.05)


if __name__ == "__main__":
    unittest.main()

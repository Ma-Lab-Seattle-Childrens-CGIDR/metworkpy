# Imports
# Standard Library Imports
import unittest

# External Imports
import numpy as np
from scipy.stats import norm

# Local Imports
from metworkpy.rank_entropy.crane_functions import (
    crane_gene_set_entropy,
    _crane_differential_entropy,
    _rank_array,
    _rank_grouping_score,
    _rank_centroid,
)
from metworkpy.rank_entropy import _datagen


class TestCraneHelperFunctions(unittest.TestCase):
    def test_rank_array(self):
        test_array = np.arange(20).reshape(4, 5)
        ranked_array = _rank_array(test_array)
        for col in range(5):
            self.assertTrue(
                np.all(np.equal(ranked_array[:, col], ranked_array[:, col][0]))
            )
        # +1 since rank array starts from 1
        np.testing.assert_array_equal(np.arange(5) + 1, ranked_array[0])

    def test_rank_centroid(self):
        test_array = np.array(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
            ]
        )
        test_centroid = _rank_centroid(test_array)
        self.assertTupleEqual(test_centroid.shape, (5,))
        np.testing.assert_array_equal(test_centroid, np.array([1, 2, 3, 4, 5]))

        test_array = np.array(
            [
                [2, 1, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
            ]
        )
        test_centroid = _rank_centroid(test_array)
        np.testing.assert_array_equal(test_centroid, np.array([5 / 4, 7 / 4, 3, 4, 5]))

        # Test random array
        test_array = np.random.rand(10, 20)
        test_centroid = _rank_centroid(test_array)
        self.assertTupleEqual(test_centroid.shape, (20,))
        self.assertAlmostEqual(np.mean(test_centroid), 10.5)
        # All values should be close together based on distribution of random ranks
        self.assertLess(np.std(test_centroid), 2.5)

    def test_rank_grouping_score(self):
        test_array = np.arange(20).reshape(4, 5)
        test_grouping_score = _rank_grouping_score(test_array)
        self.assertAlmostEqual(test_grouping_score, 0.0)

        test_array = np.random.rand(4, 5)
        test_grouping_score = _rank_grouping_score(test_array)
        self.assertGreater(test_grouping_score, 0.2)

        rand_array = np.random.rand(4, 5)
        ord_array = np.arange(20).reshape(4, 5)
        self.assertGreater(
            _rank_grouping_score(rand_array), _rank_grouping_score(ord_array)
        )

    def test_crane_differential_entropy(self):
        test_a = np.arange(20).reshape(4, 5)
        test_b = np.random.rand(4, 5)
        self.assertGreater(_crane_differential_entropy(test_a, test_b), 0.0)
        self.assertAlmostEqual(_crane_differential_entropy(test_a, test_a), 0.0)
        self.assertAlmostEqual(_crane_differential_entropy(test_b, test_b), 0.0)


class TestCraneGeneSetEntropy(unittest.TestCase):
    def test_crane_gene_set_entropy(self):
        # Test with the ordered genes to check that they are different
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
            seed=314,
        )
        rank_conservation_diff, pval = crane_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=True,
        )
        self.assertGreater(rank_conservation_diff, 0.0)
        self.assertLessEqual(pval, 0.05)
        # Check with the disorded genes to ensure that they are not different
        rank_conservation_diff, pval = crane_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=unordered_genes,
            kernel_density_estimate=True,
        )
        self.assertLess(rank_conservation_diff, 2.0)
        self.assertGreaterEqual(pval, 0.1)

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
            seed=271,
        )
        (
            rank_conservation_diff_serial,
            pval_serial,
        ) = crane_gene_set_entropy(
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
        ) = crane_gene_set_entropy(
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
            seed=314,
        )
        rank_conservation_diff, pval = crane_gene_set_entropy(
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

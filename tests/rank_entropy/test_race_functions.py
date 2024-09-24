# Standard Library Imports
import unittest

# External Imports
import numpy as np
from scipy.stats import norm

# Local Imports
from metworkpy.rank_entropy.race_functions import (
    _rank_correlation_mean,
    _race_differential_entropy,
    race_gene_set_entropy,
)
from metworkpy.rank_entropy import _datagen


class TestRaceHelperFunctions(unittest.TestCase):
    def test_rank_correlation_mean(self):
        test_array = np.arange(20).reshape(4, 5)
        self.assertAlmostEqual(_rank_correlation_mean(test_array), 0.0)

        # Test that two opposite vectors have a value of 1
        test_vec = np.arange(8)
        test_array = np.vstack((test_vec, test_vec[::-1]))
        self.assertAlmostEqual(_rank_correlation_mean(test_array), 1.0)

        # Test that two identical vectors have a value of 0
        test_vec = np.arange(8)
        test_array = np.vstack((test_vec, test_vec))
        self.assertAlmostEqual(_rank_correlation_mean(test_array), 0.0)

    def test_race_differential_entropy(self):
        test_a = np.arange(20).reshape(4, 5)
        test_b = np.random.rand(4, 5)
        self.assertGreater(_race_differential_entropy(test_a, test_b), 0.0)
        self.assertAlmostEqual(_race_differential_entropy(test_a, test_a), 0.0)
        self.assertAlmostEqual(_race_differential_entropy(test_b, test_b), 0.0)


class TestRaceGeneSetEntropy(unittest.TestCase):
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
        rank_conservation_diff, pval = race_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=True,
            iterations=100,
        )
        self.assertGreater(rank_conservation_diff, 0.0)
        self.assertLessEqual(pval, 0.05)
        # Check with the disorded genes to ensure that they are not different
        rank_conservation_diff, pval = race_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=unordered_genes,
            kernel_density_estimate=True,
            iterations=100,
        )
        self.assertLess(rank_conservation_diff, 2.0)
        self.assertGreaterEqual(pval, 0.05)

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
        ) = race_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=True,
            processes=1,
            iterations=100,
        )
        (
            rank_conservation_diff_parallel,
            pval_parallel,
        ) = race_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=True,
            processes=2,
            iterations=100,
        )
        self.assertAlmostEqual(
            rank_conservation_diff_parallel,
            rank_conservation_diff_serial,
        )
        self.assertAlmostEqual(
            pval_serial,
            pval_parallel,
            places=4,
        )

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
        rank_conservation_diff, pval = race_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=False,
            iterations=100,
        )
        self.assertGreater(rank_conservation_diff, 0.0)
        self.assertLessEqual(pval, 0.05)


if __name__ == "__main__":
    unittest.main()

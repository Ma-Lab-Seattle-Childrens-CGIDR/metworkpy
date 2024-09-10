# Imports
# Standard Library Imports
import unittest

# External Imports
import numpy as np
from scipy.stats import norm, expon, nbinom, geom, beta, rankdata

# Local Imports
from metworkpy.rank_entropy._datagen import (
    _unordered_array,
    _ordered_array,
    _ordered_vector,
    _unordered_vector,
    _generate_rank_entropy_data,
)


class TestDataGenHelpers(unittest.TestCase):
    def test_unordered_vector(self):
        test_dist = norm(loc=10, scale=5)
        test_unordered_vec = _unordered_vector(20, dist=test_dist)
        sorted_vec = np.sort(test_unordered_vec)
        # Check dimension
        self.assertEqual(len(test_unordered_vec), 20)
        # Ensure it doesn't generate an ordered vector (random chance of this happening is essentially 0)
        self.assertFalse(np.all(np.equal(test_unordered_vec, sorted_vec)))

    def test_ordered_vector(self):
        test_dist = norm(loc=10, scale=5)
        test_unordered_vec = _ordered_vector(20, dist=test_dist)
        sorted_vec = np.sort(test_unordered_vec)
        # Check dimension
        self.assertEqual(len(test_unordered_vec), 20)
        # Ensure it doesn't generate an ordered vector (random chance of this happening is essentially 0)
        self.assertTrue(np.all(np.equal(test_unordered_vec, sorted_vec)))

    def test_unordered_array(self):
        test_dist = norm(loc=10, scale=5)
        unordered_array = _unordered_array(nrow=10, ncol=20, dist=test_dist)
        self.assertTupleEqual(unordered_array.shape, (10, 20))
        for row in range(10):
            sorted_row = np.sort(unordered_array[row, :])
            self.assertFalse(np.all(np.equal(unordered_array[row, :], sorted_row)))

    def test_ordered_array(self):
        test_dist = norm(loc=10, scale=5)
        ordered_array = _ordered_array(
            nrow=10, ncol=20, dist=test_dist, col_shuffle=False
        )
        self.assertTupleEqual(ordered_array.shape, (10, 20))
        for row in range(10):
            sorted_row = np.sort(ordered_array[row, :])
            self.assertTrue(np.all(np.equal(ordered_array[row, :], sorted_row)))

    def test_ordered_array_col_shuffle(self):
        test_dist = norm(loc=10, scale=5)
        ordered_array = _ordered_array(
            nrow=10, ncol=20, dist=test_dist, col_shuffle=True
        )
        self.assertTupleEqual(ordered_array.shape, (10, 20))
        for row in range(10):
            sorted_row = np.sort(ordered_array[row, :])
            self.assertFalse(np.all(np.equal(ordered_array[row, :], sorted_row)))

    def test_different_distributions(self):
        # This should probably be some form of test array, but this is easy for now
        test_exponential = expon(loc=0, scale=1)
        self.diff_dist_helper(test_exponential)
        test_nbinom = nbinom(n=20, p=0.3)
        self.diff_dist_helper(test_nbinom)
        test_geom = geom(p=0.3, loc=0)
        self.diff_dist_helper(test_geom)
        test_beta = beta(0.5, 0.5)
        self.diff_dist_helper(test_beta)

    def diff_dist_helper(self, dist):
        # Unordered Vector
        test_unordered_vec = _unordered_vector(20, dist=dist)
        sorted_vec = np.sort(test_unordered_vec)
        # Check dimension
        self.assertEqual(len(test_unordered_vec), 20)
        # Ensure it doesn't generate an ordered vector (random chance of this happening is essentially 0)
        self.assertFalse(np.all(np.equal(test_unordered_vec, sorted_vec)))

        # Ordered Vector
        test_unordered_vec = _ordered_vector(20, dist=dist)
        sorted_vec = np.sort(test_unordered_vec)
        # Check dimension
        self.assertEqual(len(test_unordered_vec), 20)
        # Ensure it doesn't generate an ordered vector (random chance of this happening is essentially 0)
        self.assertTrue(np.all(np.equal(test_unordered_vec, sorted_vec)))

        # Unordered Array
        unordered_array = _unordered_array(nrow=10, ncol=20, dist=dist)
        self.assertTupleEqual(unordered_array.shape, (10, 20))
        for row in range(10):
            sorted_row = np.sort(unordered_array[row, :])
            self.assertFalse(np.all(np.equal(unordered_array[row, :], sorted_row)))

        # Ordered Array
        ordered_array = _ordered_array(nrow=10, ncol=20, dist=dist, col_shuffle=False)
        self.assertTupleEqual(ordered_array.shape, (10, 20))
        for row in range(10):
            sorted_row = np.sort(ordered_array[row, :])
            self.assertTrue(np.all(np.equal(ordered_array[row, :], sorted_row)))

        # Ordered Array Col Shuffle
        ordered_array = _ordered_array(nrow=10, ncol=20, dist=dist, col_shuffle=True)
        self.assertTupleEqual(ordered_array.shape, (10, 20))
        for row in range(10):
            sorted_row = np.sort(ordered_array[row, :])
            self.assertFalse(np.all(np.equal(ordered_array[row, :], sorted_row)))


class TestGenerateRankEntropyData(unittest.TestCase):
    def default_tests(
        self,
        data,
        ordered_samples,
        unordered_samples,
        ordered_genes,
        unordered_genes,
        n_ordered_samples=10,
        n_unordered_samples=10,
        n_genes_ordered=20,
        n_genes_unordered=20,
    ):
        self.assertTupleEqual(
            data.shape,
            (
                n_ordered_samples + n_unordered_samples,
                n_genes_ordered + n_genes_unordered,
            ),
        )
        ranked_mat = rankdata(data[:, ordered_genes], method="ordinal", axis=1)
        for col in range(ranked_mat.shape[1]):
            column = ranked_mat[ordered_samples, col]
            self.assertTrue(np.all(np.equal(column, column[0])))
        for row in unordered_samples:
            sorted_row = np.sort(data[row, ordered_genes])
            self.assertFalse(np.all(np.equal(data[row, ordered_genes], sorted_row)))
        for row in range(n_ordered_samples + n_unordered_samples):
            sorted_row = np.sort(data[row, unordered_genes])
            self.assertFalse(np.all(np.equal(data[row, unordered_genes], sorted_row)))

    def test_no_shuffle(self):
        n_ordered_samples = 10
        n_unordered_samples = 10
        n_genes_ordered = 20
        n_genes_unordered = 20
        dist = norm(10, 5)
        (
            data,
            ordered_samples,
            unordered_samples,
            ordered_genes,
            unordered_genes,
        ) = _generate_rank_entropy_data(
            n_ordered_samples=n_ordered_samples,
            n_unordered_samples=n_unordered_samples,
            n_genes_ordered=n_genes_ordered,
            n_genes_unordered=n_genes_unordered,
            dist=dist,
            shuffle_genes=False,
            shuffle_samples=False,
        )
        self.default_tests(
            data=data,
            ordered_samples=ordered_samples,
            unordered_samples=unordered_samples,
            ordered_genes=ordered_genes,
            unordered_genes=unordered_genes,
            n_ordered_samples=n_ordered_samples,
            n_unordered_samples=n_unordered_samples,
            n_genes_ordered=n_genes_ordered,
            n_genes_unordered=n_genes_unordered,
        )
        for row in ordered_samples:
            sorted_row = np.sort(data[row, ordered_genes])
            self.assertTrue(np.all(np.equal(data[row, ordered_genes], sorted_row)))

    def test_gene_shuffle(self):
        n_ordered_samples = 10
        n_unordered_samples = 10
        n_genes_ordered = 20
        n_genes_unordered = 20
        dist = norm(10, 5)
        (
            data,
            ordered_samples,
            unordered_samples,
            ordered_genes,
            unordered_genes,
        ) = _generate_rank_entropy_data(
            n_ordered_samples=n_ordered_samples,
            n_unordered_samples=n_unordered_samples,
            n_genes_ordered=n_genes_ordered,
            n_genes_unordered=n_genes_unordered,
            dist=dist,
            shuffle_genes=True,
            shuffle_samples=False,
        )
        self.default_tests(
            data=data,
            ordered_samples=ordered_samples,
            unordered_samples=unordered_samples,
            ordered_genes=ordered_genes,
            unordered_genes=unordered_genes,
            n_ordered_samples=n_ordered_samples,
            n_unordered_samples=n_unordered_samples,
            n_genes_ordered=n_genes_ordered,
            n_genes_unordered=n_genes_unordered,
        )
        # Make sure the genes are actually scrambled
        for row in ordered_samples:
            sorted_row = np.sort(data[row, ordered_genes])
            self.assertFalse(np.all(np.equal(data[row, ordered_genes], sorted_row)))

    def test_shuffle_samples(self):
        n_ordered_samples = 10
        n_unordered_samples = 10
        n_genes_ordered = 20
        n_genes_unordered = 20
        dist = norm(10, 5)
        (
            data,
            ordered_samples,
            unordered_samples,
            ordered_genes,
            unordered_genes,
        ) = _generate_rank_entropy_data(
            n_ordered_samples=n_ordered_samples,
            n_unordered_samples=n_unordered_samples,
            n_genes_ordered=n_genes_ordered,
            n_genes_unordered=n_genes_unordered,
            dist=dist,
            shuffle_genes=False,
            shuffle_samples=True,
        )
        self.default_tests(
            data=data,
            ordered_samples=ordered_samples,
            unordered_samples=unordered_samples,
            ordered_genes=ordered_genes,
            unordered_genes=unordered_genes,
            n_ordered_samples=n_ordered_samples,
            n_unordered_samples=n_unordered_samples,
            n_genes_ordered=n_genes_ordered,
            n_genes_unordered=n_genes_unordered,
        )

    def test_shuffle_both(self):
        n_ordered_samples = 10
        n_unordered_samples = 10
        n_genes_ordered = 20
        n_genes_unordered = 20
        dist = norm(10, 5)
        (
            data,
            ordered_samples,
            unordered_samples,
            ordered_genes,
            unordered_genes,
        ) = _generate_rank_entropy_data(
            n_ordered_samples=n_ordered_samples,
            n_unordered_samples=n_unordered_samples,
            n_genes_ordered=n_genes_ordered,
            n_genes_unordered=n_genes_unordered,
            dist=dist,
            shuffle_genes=True,
            shuffle_samples=True,
        )
        self.default_tests(
            data=data,
            ordered_samples=ordered_samples,
            unordered_samples=unordered_samples,
            ordered_genes=ordered_genes,
            unordered_genes=unordered_genes,
            n_ordered_samples=n_ordered_samples,
            n_unordered_samples=n_unordered_samples,
            n_genes_ordered=n_genes_ordered,
            n_genes_unordered=n_genes_unordered,
        )


if __name__ == "__main__":
    unittest.main()

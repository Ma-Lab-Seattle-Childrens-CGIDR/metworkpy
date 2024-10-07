# Imports
# Standard Library Imports
import unittest

# External Imports
import numpy as np
from scipy.stats import norm

# Local Imports
from metworkpy.rank_entropy.crane_functions import (
    crane_gene_set_entropy,
    crane_gene_set_classification,
    CraneClassifier,
    _crane_differential_entropy,
    _rank_array,
    _rank_grouping_score,
    _rank_centroid,
)
from metworkpy.rank_entropy import _datagen, crane_functions


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
        self.assertTupleEqual(test_centroid.shape, (1, 5))
        np.testing.assert_array_equal(
            test_centroid, np.array([1, 2, 3, 4, 5]).reshape(1, -1)
        )

        test_array = np.array(
            [
                [2, 1, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
            ]
        )
        test_centroid = _rank_centroid(test_array)
        np.testing.assert_array_equal(
            test_centroid, np.array([5 / 4, 7 / 4, 3, 4, 5]).reshape(1, -1)
        )

        # Test random array
        rng = np.random.default_rng(51928)
        test_array = rng.uniform(low=0, high=1, size=10 * 20).reshape(10, 20)
        test_centroid = _rank_centroid(test_array)
        self.assertTupleEqual(test_centroid.shape, (1, 20))
        self.assertAlmostEqual(np.mean(test_centroid), 10.5)
        # All values should be close together based on distribution of random ranks
        self.assertLess(np.std(test_centroid), 2.5)

    def test_rank_grouping_score(self):
        test_array = np.arange(20).reshape(4, 5)
        test_grouping_score = _rank_grouping_score(test_array)
        self.assertAlmostEqual(test_grouping_score, 0.0)

        rng = np.random.default_rng(123456)

        test_array = rng.uniform(0, 1, size=4 * 5).reshape(4, 5)
        test_grouping_score = _rank_grouping_score(test_array)
        self.assertGreater(test_grouping_score, 0.2)

        rand_array = rng.uniform(0, 1, size=4 * 5).reshape(4, 5)
        ord_array = np.arange(20).reshape(4, 5)
        self.assertGreater(
            _rank_grouping_score(rand_array), _rank_grouping_score(ord_array)
        )

    def test_crane_differential_entropy(self):
        rng = np.random.default_rng(12312941024)
        test_a = np.arange(20).reshape(4, 5)
        test_b = rng.uniform(0, 1, size=4 * 5).reshape(4, 5)
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
        rank_conservation_diff, pval = crane_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=False,
        )
        self.assertGreater(rank_conservation_diff, 0.0)
        self.assertLessEqual(pval, 0.05)


class TestCraneClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_genes = 10
        cls.num_samples_g1 = 20
        cls.num_samples_g2 = 15
        # Generate test data with good ability to classify
        (test_expression_data1, _, _, _, _) = _datagen._generate_rank_entropy_data(
            n_ordered_samples=cls.num_samples_g1,
            n_unordered_samples=0,
            n_genes_ordered=cls.num_genes,
            n_genes_unordered=0,
            dist=norm(loc=100, scale=25),
            shuffle_genes=True,
            shuffle_samples=True,
            seed=314,
        )
        (test_expression_data2, _, _, _, _) = _datagen._generate_rank_entropy_data(
            n_ordered_samples=cls.num_samples_g2,
            n_unordered_samples=0,
            n_genes_ordered=cls.num_genes,
            n_genes_unordered=0,
            dist=norm(loc=100, scale=25),
            shuffle_genes=True,
            shuffle_samples=True,
            seed=1618,
        )
        cls.good_class_data_X = np.vstack(
            [test_expression_data1, test_expression_data2]
        )
        cls.good_class_data_y = np.array([0] * 20 + [1] * 15)

        # Generate test data with a bad ability to classify
        (test_expression_data1, _, _, _, _) = _datagen._generate_rank_entropy_data(
            n_ordered_samples=0,
            n_unordered_samples=cls.num_samples_g1,
            n_genes_ordered=0,
            n_genes_unordered=cls.num_genes,
            dist=norm(loc=100, scale=25),
            shuffle_genes=True,
            shuffle_samples=True,
            seed=3512,
        )
        (test_expression_data2, _, _, _, _) = _datagen._generate_rank_entropy_data(
            n_ordered_samples=0,
            n_unordered_samples=cls.num_samples_g2,
            n_genes_ordered=0,
            n_genes_unordered=cls.num_genes,
            dist=norm(loc=100, scale=25),
            shuffle_genes=True,
            shuffle_samples=True,
            seed=168,
        )
        cls.bad_class_data_X = np.vstack([test_expression_data1, test_expression_data2])
        cls.bad_class_data_y = np.array(
            [0] * cls.num_samples_g1 + [1] * cls.num_samples_g2
        )

    def test_crane_classification_rate(self):
        class_rate_ordered = crane_functions._crane_classification_rate(
            self.good_class_data_X[: self.num_samples_g1, :],
            self.good_class_data_X[self.num_samples_g1 :, :],
        )
        self.assertAlmostEqual(class_rate_ordered, 1.0)

        class_rate_disordered = crane_functions._crane_classification_rate(
            self.bad_class_data_X[: self.num_samples_g1, :],
            self.bad_class_data_X[self.num_samples_g1 :, :],
        )
        self.assertLess(class_rate_disordered, 1.0)

    def test_crane_gene_set_classification(self):
        class_rate, pvalue = crane_gene_set_classification(
            expression_data=self.good_class_data_X,
            sample_group1=np.array(range(self.num_samples_g1)),
            sample_group2=np.array(
                range(self.num_samples_g1, self.num_samples_g1 + self.num_samples_g2)
            ),
            gene_network=np.array(range(self.num_genes)),
            kernel_density_estimate=True,
            bw_method=None,
            iterations=1_000,
            replace=True,
            seed=516,
            processes=1,
        )
        # Classification rate should be between 0 and 1
        self.assertGreaterEqual(class_rate, 0.0)
        self.assertLessEqual(class_rate, 1.0)
        # P value should be significant
        self.assertLessEqual(pvalue, 0.05)

        # Now with the bad classification data
        class_rate, pvalue = crane_gene_set_classification(
            expression_data=self.bad_class_data_X,
            sample_group1=np.array(range(self.num_samples_g1)),
            sample_group2=np.array(
                range(self.num_samples_g1, self.num_samples_g1 + self.num_samples_g2)
            ),
            gene_network=np.array(range(self.num_genes)),
            kernel_density_estimate=True,
            bw_method=None,
            iterations=1_000,
            replace=True,
            seed=516,
            processes=1,
        )
        # Classification rate should be between 0 and 1
        self.assertGreaterEqual(class_rate, 0.0)
        self.assertLessEqual(class_rate, 1.0)
        # P value should be significant
        self.assertGreaterEqual(pvalue, 0.05)

    def test_crane_gene_set_classification_parallel(self):
        class_rate_serial, pvalue_serial = crane_gene_set_classification(
            expression_data=self.good_class_data_X,
            sample_group1=np.array(range(self.num_samples_g1)),
            sample_group2=np.array(
                range(self.num_samples_g1, self.num_samples_g1 + self.num_samples_g2)
            ),
            gene_network=np.array(range(self.num_genes)),
            kernel_density_estimate=True,
            bw_method=None,
            iterations=1_000,
            replace=True,
            seed=516,
            processes=1,
        )

        class_rate_parallel, pvalue_parallel = crane_gene_set_classification(
            expression_data=self.good_class_data_X,
            sample_group1=np.array(range(self.num_samples_g1)),
            sample_group2=np.array(
                range(self.num_samples_g1, self.num_samples_g1 + self.num_samples_g2)
            ),
            gene_network=np.array(range(self.num_genes)),
            kernel_density_estimate=True,
            bw_method=None,
            iterations=1_000,
            replace=True,
            seed=516,
            processes=2,
        )

        self.assertAlmostEqual(pvalue_serial, pvalue_parallel)
        self.assertAlmostEqual(class_rate_serial, class_rate_parallel)

    def test_crane_classifier(self):
        rng = np.random.default_rng(154)

        good_classifier = CraneClassifier()
        train_rows = rng.choice(
            list(range(self.num_samples_g1 + self.num_samples_g2)),
            size=int(0.8 * (self.num_samples_g1 + self.num_samples_g2)),
            replace=False,
        )
        test_rows = np.ones(self.num_samples_g1 + self.num_samples_g2, dtype=bool)
        test_rows[train_rows] = False

        X_train = self.good_class_data_X[train_rows, :]
        y_train = self.good_class_data_y[train_rows]

        X_test = self.good_class_data_X[test_rows, :]
        y_test = self.good_class_data_y[test_rows]

        # Fit the classifier
        good_classifier = good_classifier.fit(X_train, y_train)

        # Classify the test samples
        y_pred = good_classifier.classify(X_test)

        # This classifier should be perfect
        self.assertAlmostEqual(np.equal(y_pred, y_test).mean(), 1.0)

        ## Repeat with the bad class data
        bad_classifier = CraneClassifier()
        train_rows = rng.choice(
            list(range(self.num_samples_g1 + self.num_samples_g2)),
            size=int(0.8 * (self.num_samples_g1 + self.num_samples_g2)),
            replace=False,
        )
        test_rows = np.ones(self.num_samples_g1 + self.num_samples_g2, dtype=bool)
        test_rows[train_rows] = False

        X_train = self.bad_class_data_X[train_rows, :]
        y_train = self.bad_class_data_y[train_rows]

        X_test = self.bad_class_data_X[test_rows, :]
        y_test = self.bad_class_data_y[test_rows]

        # Fit the classifier
        bad_classifier = bad_classifier.fit(X_train, y_train)

        # Classify the test samples
        y_pred = bad_classifier.classify(X_test)

        # This classifier should be perfect
        self.assertLess(np.equal(y_pred, y_test).mean(), 0.7)


if __name__ == "__main__":
    unittest.main()

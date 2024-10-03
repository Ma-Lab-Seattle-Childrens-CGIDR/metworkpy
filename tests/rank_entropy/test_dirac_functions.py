# Standard Library Imports
import math
import unittest

# External Imports
import numpy as np
from scipy.stats import norm

from metworkpy import dirac_gene_set_classification, DiracClassifier

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
        rank_conservation_diff, pval = dirac_functions.dirac_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=True,
        )
        self.assertGreater(rank_conservation_diff, 0.0)
        self.assertLessEqual(pval, 0.05)
        # Check with the disordered genes to ensure that they are not different
        rank_conservation_diff, pval = dirac_functions.dirac_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=unordered_genes,
            kernel_density_estimate=True,
        )
        self.assertLess(rank_conservation_diff, 0.1)
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
        rank_conservation_diff, pval = dirac_functions.dirac_gene_set_entropy(
            test_expression_data,
            sample_group1=ordered_samples,
            sample_group2=unordered_samples,
            gene_network=ordered_genes,
            kernel_density_estimate=False,
        )
        self.assertGreater(rank_conservation_diff, 0.0)
        self.assertLessEqual(pval, 0.05)


class TestDiracClassification(unittest.TestCase):
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

    def test_dirac_classification_rate(self):
        class_rate_ordered = dirac_functions._dirac_classification_rate(
            self.good_class_data_X[: self.num_samples_g1, :],
            self.good_class_data_X[self.num_samples_g1 :, :],
        )
        self.assertAlmostEqual(class_rate_ordered, 1.0)

        class_rate_disordered = dirac_functions._dirac_classification_rate(
            self.bad_class_data_X[: self.num_samples_g1, :],
            self.bad_class_data_X[self.num_samples_g1 :, :],
        )
        self.assertLess(class_rate_disordered, 1.0)

    def test_dirac_gene_set_classification(self):
        class_rate, pvalue = dirac_gene_set_classification(
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
        class_rate, pvalue = dirac_gene_set_classification(
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

    def test_dirac_gene_set_classification_parallel(self):
        class_rate_serial, pvalue_serial = dirac_gene_set_classification(
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

        class_rate_parallel, pvalue_parallel = dirac_gene_set_classification(
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

    def test_dirac_classifier(self):
        rng = np.random.default_rng(154)

        good_classifier = DiracClassifier()
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
        bad_classifier = DiracClassifier()
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

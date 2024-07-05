# Import Statements
# Standard Library Imports
from __future__ import annotations
import unittest

# External Imports
import numpy as np
from scipy.sparse import csr_array, csc_array

# Local Imports
import metworkpy.network._array_utils
from metworkpy.network._array_utils import _broadcast_mult_arr_vec


class TestSplitArrayColumns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create array used for the splitting
        cls.test_arr_np = np.array(
            [
                [1, 2, 0, 0, 3, 0, 5, 0, 6, 1],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 2],
                [0, 1, 0, 5, 0, 0, 1, 0, 0, 0],
                [0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
            ]
        )
        cls.split1 = np.array(
            [
                [1, 0, 3, 5, 6],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        cls.split2 = np.array(
            [
                [2, 0, 0, 0, 1],
                [0, 0, 0, 1, 2],
                [1, 5, 0, 0, 0],
                [2, 1, 0, 0, 0],
            ]
        )

    def test_numpy(self):
        # Split into 2 sub-arrays
        arr1, arr2 = metworkpy.network._array_utils._split_arr_col(
            self.test_arr_np, into=2
        )
        self.assertTupleEqual(arr1.shape, (4, 5))
        self.assertTupleEqual(arr2.shape, (4, 5))
        self.assertTrue((arr1 == self.split1).all())
        self.assertTrue((arr2 == self.split2).all())

    def test_csc(self):
        _split_arr_col_helper(self, csc_array)

    def test_csr(self):
        _split_arr_col_helper(self, csr_array)


def _split_arr_col_helper(test_obj: TestSplitArrayColumns, array_format):
    arr1, arr2 = metworkpy.network._array_utils._split_arr_col(
        array_format(test_obj.test_arr_np), into=2
    )
    test_obj.assertIsInstance(arr1, array_format)
    test_obj.assertIsInstance(arr2, array_format)

    test_obj.assertTupleEqual(arr1.shape, (4, 5))
    test_obj.assertTupleEqual(arr2.shape, (4, 5))
    test_obj.assertTrue((arr1.toarray() == test_obj.split1).all())
    test_obj.assertTrue((arr2.toarray() == test_obj.split2).all())


class TestSplitArrayRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create array used for the splitting
        cls.test_arr_np = np.array(
            [
                [1, 2, 0, 0, 3, 0, 5, 0, 6, 1],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 2],
                [0, 1, 0, 5, 0, 0, 1, 0, 0, 0],
                [0, 2, 0, 1, 0, 0, 0, 0, 0, 0],
            ]
        ).T
        cls.split1 = np.array(
            [
                [1, 0, 3, 5, 6],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        ).T
        cls.split2 = np.array(
            [
                [2, 0, 0, 0, 1],
                [0, 0, 0, 1, 2],
                [1, 5, 0, 0, 0],
                [2, 1, 0, 0, 0],
            ]
        ).T

    def test_numpy(self):
        # Split into 2 sub-arrays
        arr1, arr2 = metworkpy.network._array_utils._split_arr_row(
            self.test_arr_np, into=2
        )
        self.assertTupleEqual(arr1.shape, (5, 4))
        self.assertTupleEqual(arr2.shape, (5, 4))
        self.assertTrue((arr1 == self.split1).all())
        self.assertTrue((arr2 == self.split2).all())

    def test_csc(self):
        _split_arr_row_helper(self, csc_array)

    def test_csr(self):
        _split_arr_row_helper(self, csr_array)


def _split_arr_row_helper(test_obj: TestSplitArrayColumns, array_format):
    arr1, arr2 = metworkpy.network._array_utils._split_arr_row(
        array_format(test_obj.test_arr_np), into=2
    )
    test_obj.assertIsInstance(arr1, array_format)
    test_obj.assertIsInstance(arr2, array_format)

    test_obj.assertTupleEqual(arr1.shape, (5, 4))
    test_obj.assertTupleEqual(arr2.shape, (5, 4))
    test_obj.assertTrue((arr1.toarray() == test_obj.split1).all())
    test_obj.assertTrue((arr2.toarray() == test_obj.split2).all())


class TestSplitArrSign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_arr_np = np.array(
            [
                [0, 1, 0, -1, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, -1, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, -1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        cls.pos_arr_np = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        cls.neg_arr_np = np.array(
            [
                [0, 0, 0, -1, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

    def test_numpy(self):
        pos_arr, neg_arr = metworkpy.network._array_utils._split_arr_sign(
            self.test_arr_np
        )
        self.assertTupleEqual(pos_arr.shape, self.test_arr_np.shape)
        self.assertTupleEqual(neg_arr.shape, self.test_arr_np.shape)

        self.assertTrue((pos_arr == self.pos_arr_np).all())
        self.assertTrue((neg_arr == self.neg_arr_np).all())

    def test_csc(self):
        _split_sign_sparse_helper(self, csc_array)

    def test_csr(self):
        _split_sign_sparse_helper(self, csr_array)


def _split_sign_sparse_helper(test_obj, array_format):
    pos_arr, neg_arr = metworkpy.network._array_utils._split_arr_sign(
        array_format(test_obj.test_arr_np)
    )
    test_obj.assertIsInstance(pos_arr, array_format)
    test_obj.assertTupleEqual(pos_arr.shape, test_obj.test_arr_np.shape)
    test_obj.assertTupleEqual(neg_arr.shape, test_obj.test_arr_np.shape)

    test_obj.assertTrue((pos_arr.toarray() == test_obj.pos_arr_np).all())
    test_obj.assertTrue((neg_arr.toarray() == test_obj.neg_arr_np).all())


class TestSparseMax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arr1 = csr_array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ]
        )
        cls.arr2 = csr_array(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]
        )
        cls.arr3 = csr_array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )
        cls.max_arr = csr_array(
            [
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
            ]
        )

    def test_sparse_max(self):
        self.assertFalse(
            (
                metworkpy.network._array_utils._sparse_max(
                    self.arr1, self.arr2, self.arr3
                )
                != self.max_arr
            )
            .toarray()
            .any()
        )


class TestSparseMean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arr1 = csr_array(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
            ]
        )
        cls.arr2 = csr_array(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]
        )
        cls.arr3 = csr_array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        )
        cls.mean_arr = csr_array(
            [
                [1 / 3, 2 / 3, 0],
                [1 / 3, 0, 1 / 3],
                [1 / 3, 1 / 3, 0],
            ]
        )

    def test_sparse_max(self):
        sparse_mean = metworkpy.network._array_utils._sparse_mean(
            self.arr1, self.arr2, self.arr3
        )
        self.assertIsInstance(sparse_mean, csr_array)
        self.assertTrue(
            np.isclose(sparse_mean.toarray(), self.mean_arr.toarray()).all()
        )


class TestBroadcastMultArrVec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_arr = csr_array(
            [
                [1, 0, 0, 5, 4, 0, 0],
                [0, 2, 0, 0, 3, 0, 0],
                [1, 2, 0, 0, 0, 6, 0],
                [0, 0, 7, 0, 0, 0, 1],
            ]
        )

        cls.test_vec = csr_array([[2, 1, 1, 0, 0, 2, 10]]).T

        cls.test_res = csr_array(
            [
                [2, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0],
                [2, 2, 0, 0, 0, 12, 0],
                [0, 0, 7, 0, 0, 0, 10],
            ]
        )

    def test_broadcast_mult(self):
        self.assertFalse(
            (_broadcast_mult_arr_vec(self.test_arr, self.test_vec) != self.test_res)
            .toarray()
            .any()
        )


if __name__ == "__main__":
    unittest.main()

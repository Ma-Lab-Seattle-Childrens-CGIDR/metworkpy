# Standard Library Imports
import unittest

# External Imports
import numpy as np

# Local Imports
from metworkpy.divergence._data_validation import (
    _validate_discrete,
    _validate_sample,
    _validate_samples,
)


class TestDataValidation(unittest.TestCase):
    def test_validate_discrete(self):
        disc_arr = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]).reshape(-1, 4)
        with self.assertRaises(ValueError):
            _validate_discrete(disc_arr)
        _ = _validate_sample(disc_arr.reshape(-1, 1))

    def test_validate_sample(self):
        list_arr = [1, 2, 3, 4, 5]
        validated_list_arr = _validate_sample(list_arr)
        self.assertIsInstance(validated_list_arr, np.ndarray)
        self.assertEqual(validated_list_arr.shape[1], 1)
        arr_3d = np.zeros(shape=(4, 3, 5))
        with self.assertRaisesRegex(
            ValueError, r"Sample must have a maximum of 2 axes"
        ):
            _validate_sample(arr_3d)
        arr_1d = np.ones(shape=10)
        validated_arr_1d = _validate_sample(arr_1d)
        self.assertEqual(validated_arr_1d.shape[1], 1)

    def test_validate_samples(self):
        # Test coercion, and 1D reshaping
        list_arr1 = [1, 2, 3, 4, 4, 5, 6]
        list_arr2 = [5, 4, 3, 4, 5, 6, 6]
        v_list_arr1, v_list_arr2 = _validate_samples(list_arr1, list_arr2)
        self.assertIsInstance(v_list_arr1, np.ndarray)
        self.assertIsInstance(v_list_arr2, np.ndarray)
        self.assertEqual(v_list_arr1.shape[1], 1)
        self.assertEqual(v_list_arr2.shape[1], 1)
        # Test mismatched length arrays
        arr1_5 = np.ones(shape=(10, 5))
        arr2_6 = np.ones(shape=(10, 4))
        with self.assertRaisesRegex(
            ValueError, r"Both p and q distributions must have the same dimension.+"
        ):
            _ = _validate_samples(arr1_5, arr2_6)
        # Test incorrect number of dimensions
        arr1_3d = np.ones(shape=(4, 3, 5))
        arr2_3d = np.ones(shape=(4, 3, 5))
        with self.assertRaisesRegex(
            ValueError, r"p and q must have a maximum of two axes.+"
        ):
            _ = _validate_samples(arr1_3d, arr2_3d)


if __name__ == "__main__":
    unittest.main()

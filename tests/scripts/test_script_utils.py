# Imports
# Standard Library Imports
import unittest

# External Imports
import numpy as np

# Local Imports
import metworkpy.scripts._script_utils


class TestHelperFunctions(unittest.TestCase):
    def test_parse_samples(self):
        self.assertListEqual(metworkpy.scripts._script_utils._parse_samples("1"), [1])
        self.assertListEqual(
            metworkpy.scripts._script_utils._parse_samples("1:5"), [1, 2, 3, 4, 5]
        )
        self.assertListEqual(
            metworkpy.scripts._script_utils._parse_samples("1,3,6:7"), [1, 3, 6, 7]
        )
        self.assertListEqual(
            metworkpy.scripts._script_utils._parse_samples("2:5,1,7"),
            [2, 3, 4, 5, 1, 7],
        )

    def test_parse_quantile(self):
        self.assertTupleEqual(
            metworkpy.scripts._script_utils._parse_quantile("0.15"), (0.15, 0.85)
        )
        self.assertTupleEqual(
            metworkpy.scripts._script_utils._parse_quantile("0.10,0.90"), (0.10, 0.90)
        )

    def test_parse_aggregation_method(self):
        self.assertEqual(
            metworkpy.scripts._script_utils._parse_aggregation_method("median"),
            np.median,
        )
        self.assertEqual(
            metworkpy.scripts._script_utils._parse_aggregation_method("max"), np.max
        )
        self.assertEqual(
            metworkpy.scripts._script_utils._parse_aggregation_method("min"), np.min
        )
        self.assertEqual(
            metworkpy.scripts._script_utils._parse_aggregation_method("mean"), np.mean
        )


if __name__ == "__main__":
    unittest.main()

# Standard Library Imports
import unittest

# External Imports
import cobra  # type: ignore

# Local Imports
from metworkpy.examples import get_example_model


class TestExampleModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cobra.Configuration().solver = "glpk"

    def test_get_example_model(self):
        test_model = get_example_model()
        self.assertIsInstance(test_model, cobra.Model)
        # Check that the biomass is still the same
        self.assertAlmostEqual(test_model.slim_optimize(), 25.0)


if __name__ == "__main__":
    unittest.main()

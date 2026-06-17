# Imports
# Standard Library Imports
import pathlib
import unittest
import random

# External Imports
import cobra
import pandas as pd

# Local Imports
from metworkpy.sampling.corners import corner_sampling
from metworkpy.utils import read_model

random.seed(314159265)


class TestCornerSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
        cobra.Configuration().solver = "hybrid"
        cls.test_model = read_model(data_path / "textbook_model.json")

    def test_corner_sampling_serial(self):
        # Sample from the model
        samples = corner_sampling(
            model=self.test_model,
            n_samples=10,
            reaction_list=None,
            processes=1,
        )
        self.assertIsInstance(samples, pd.DataFrame)
        # There should be a column for each reaction in the model
        reaction_list = self.test_model.reactions.list_attr("id")
        self.assertCountEqual(list(samples.columns), reaction_list)
        # There should be 10 rows
        self.assertEqual(samples.shape[0], 10)
        # There should be no NaN in the dataframe
        self.assertFalse(samples.isna().any().any())

    def test_corner_sampling_seeded(self):
        # Sample from the model
        samples1 = corner_sampling(
            model=self.test_model,
            n_samples=10,
            reaction_list=None,
            processes=1,
            seed=423434,
        )
        # Repeat
        samples2 = corner_sampling(
            model=self.test_model,
            n_samples=10,
            reaction_list=None,
            processes=1,
            seed=423434,
        )
        # Check that they are (approximately) equal
        # This may break...there's not really a reason the solutions
        # themselves should be the same
        pd.testing.assert_frame_equal(samples1, samples2)

    def test_corner_sampling_parallel(self):
        # Sample from the model
        samples = corner_sampling(
            model=self.test_model,
            n_samples=10,
            reaction_list=None,
            processes=2,
        )
        self.assertIsInstance(samples, pd.DataFrame)
        # There should be a column for each reaction in the model
        reaction_list = self.test_model.reactions.list_attr("id")
        self.assertCountEqual(list(samples.columns), reaction_list)
        # There should be 10 rows
        self.assertEqual(samples.shape[0], 10)
        # There should be no NaN in the dataframe
        self.assertFalse(samples.isna().any().any())

    def test_corner_sampling_parallel_seeded(self):
        # Sample from the model
        samples1 = corner_sampling(
            model=self.test_model,
            n_samples=10,
            reaction_list=None,
            processes=2,
            seed=423434,
        )
        # Repeat
        samples2 = corner_sampling(
            model=self.test_model,
            n_samples=10,
            reaction_list=None,
            processes=2,
            seed=423434,
        )
        # Check that they are (approximately) equal
        # This may break...there's not really a reason the solutions
        # themselves should be the same
        pd.testing.assert_frame_equal(samples1, samples2)


if __name__ == "__main__":
    unittest.main()

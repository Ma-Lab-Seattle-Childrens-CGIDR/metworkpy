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
from metworkpy.sampling.upsampling import upsample
from metworkpy.utils import read_model

random.seed(314159265)


class TestUpsample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = pathlib.Path(__file__).parent.parent.absolute() / "data"
        cobra.Configuration().solver = "hybrid"
        cls.test_model = read_model(data_path / "textbook_model.json")
        # Create a sampler from the test_model
        cls.test_optgp_sampler = cobra.sampling.OptGPSampler(
            model=cls.test_model, thinning=100, processes=1
        )
        samples = cls.test_optgp_sampler.sample(100)
        cls.test_optgp_samples: pd.DataFrame = samples[
            cls.test_optgp_sampler.validate(samples) == "v"  # type: ignore
        ]
        # Also get some corner samples for evaluation
        cls.test_corner_samples = corner_sampling(
            model=cls.test_model, n_samples=25, processes=1
        )

    def test_upsampling_optgp(self):
        num_test_samples = 1_000
        upsampled = upsample(
            self.test_optgp_samples,
            n_samples=num_test_samples,
            processes=1,
            seed=1238098,
        )
        # The shape of the upsampled points should be 1_100, n_columns
        self.assertEqual(
            upsampled.shape[0],
            num_test_samples + self.test_optgp_samples.shape[0],
        )
        self.assertEqual(
            upsampled.shape[1], len(self.test_optgp_samples.columns)
        )
        # All the newly generated samples (and the old ones) should be valid
        self.assertEqual(
            (self.test_optgp_sampler.validate(upsampled) == "v").sum(),  # type: ignore
            num_test_samples + self.test_optgp_samples.shape[0],
        )

    def test_upsampling_corners(self):
        num_test_samples = 1_000
        upsampled = upsample(
            self.test_corner_samples,
            n_samples=num_test_samples,
            processes=1,
            seed=72983759287,
        )
        # The shape of the upsampled points should be 1_100, n_columns
        self.assertEqual(
            upsampled.shape[0],
            num_test_samples + self.test_corner_samples.shape[0],
        )
        self.assertEqual(
            upsampled.shape[1], len(self.test_corner_samples.columns)
        )
        # All the newly generated samples (and the old ones) should be valid
        self.assertEqual(
            (self.test_optgp_sampler.validate(upsampled) == "v").sum(),  # type: ignore
            self.test_corner_samples.shape[0] + num_test_samples,
        )

    def test_upsampling_seeded(self):
        num_test_samples = 1_000
        upsampled1 = upsample(
            self.test_optgp_samples,
            n_samples=num_test_samples,
            processes=1,
            seed=109283,
        )
        upsampled2 = upsample(
            self.test_optgp_samples,
            n_samples=num_test_samples,
            processes=1,
            seed=109283,
        )
        pd.testing.assert_frame_equal(upsampled1, upsampled2)

    def test_upsampling_seeded_parallel(self):
        num_test_samples = 1_000
        upsampled1 = upsample(
            self.test_optgp_samples,
            n_samples=num_test_samples,
            processes=2,
            seed=56293761,
        )
        upsampled2 = upsample(
            self.test_optgp_samples,
            n_samples=num_test_samples,
            processes=2,
            seed=56293761,
        )
        pd.testing.assert_frame_equal(upsampled1, upsampled2)


if __name__ == "__main__":
    unittest.main()

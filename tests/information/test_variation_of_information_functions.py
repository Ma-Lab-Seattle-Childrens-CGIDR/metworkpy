# Standard Library Imports
import unittest

# External Imports
import numpy as np
import pandas as pd


# Local Imports
from metworkpy.information import variation_of_information_functions as voi


class TestMainVOI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(61924711)
        cls.rng = rng
        # Setup Known vectors
        x = np.array([0, 1, 1, 0, 0])
        y = np.array([1, 0, 0, 0, 1])
        z = np.array([0, 0, 1, 0, 1])
        x_col = x.reshape(-1, 1)
        y_col = y.reshape(-1, 1)
        z_col = z.reshape(-1, 1)
        cls.x = x
        cls.y = y
        cls.x_col = x_col
        cls.y_col = y_col
        cls.z_col = z_col
        H_x = H_y = -(3 / 5) * np.log(3 / 5) - (2 / 5) * np.log(2 / 5)
        H_xy = (
            -1 / 5 * np.log(1 / 5)
            - (2 / 5) * np.log(2 / 5)
            - (2 / 5) * np.log(2 / 5)
        )
        cls.VOI_xy = 2 * H_xy - H_x - H_y
        # Setup Random x,y, and z vectors
        x_rand = rng.integers(2, size=50)
        y_rand = np.zeros(50, dtype=np.int_)
        z_rand = np.zeros(50, dtype=np.int_)
        for idx, x_val in enumerate(x_rand):
            y_rand[idx] = rng.choice(
                [0, 1], p=[0.2, 0.8] if x_val == 0 else [0.4, 0.6]
            )
        for idx, y_val in enumerate(y_rand):
            z_rand[idx] = rng.choice(
                [0, 1], p=[0.3, 0.7] if y_val == 0 else [0.8, 0.2]
            )
        # Create column vectors from these
        cls.x_rand_col = x_rand.reshape(-1, 1)
        cls.y_rand_col = y_rand.reshape(-1, 1)
        cls.z_rand_col = z_rand.reshape(-1, 1)
        # Also create a pandas DataFrame from these
        cls.rand_df = pd.DataFrame({"x": x_rand, "y": y_rand, "z": z_rand})

    def test_voi(self):
        test_voi = voi._voi(self.x_col, self.y_col)
        expected_voi = self.VOI_xy
        self.assertAlmostEqual(test_voi, expected_voi)

    def test_symetry(self):
        # First with the Known
        test_voi_xy = voi._voi(self.x_col, self.y_col)
        test_voi_yx = voi._voi(self.y_col, self.x_col)
        self.assertAlmostEqual(test_voi_xy, test_voi_yx)
        # Then with the random
        test_voi_xy = voi._voi(self.x_rand_col, self.y_rand_col)
        test_voi_yx = voi._voi(self.y_rand_col, self.x_rand_col)
        self.assertAlmostEqual(test_voi_xy, test_voi_yx)

    def test_triangle(self):
        # First with the known
        voi_xy = voi._voi(self.x_col, self.y_col)
        voi_yz = voi._voi(self.y_col, self.z_col)
        voi_xz = voi._voi(self.x_col, self.z_col)
        self.assertLessEqual(voi_xz, voi_xy + voi_yz)
        # Then with the random
        voi_xy = voi._voi(self.x_rand_col, self.y_rand_col)
        voi_yz = voi._voi(self.y_rand_col, self.z_rand_col)
        voi_xz = voi._voi(self.x_rand_col, self.z_rand_col)
        self.assertLessEqual(voi_xz, voi_xy + voi_yz)

    def test_pandas_input(self):
        # Test with pandas series
        voi_xy_np = voi.variation_of_information(
            self.x_rand_col, self.y_rand_col
        )
        voi_xy_pd = voi.variation_of_information(
            self.rand_df["x"], self.rand_df["y"]
        )
        self.assertAlmostEqual(voi_xy_np, voi_xy_pd)
        # Then with DataFrames
        # Test with pandas series
        voi_xy_np = voi.variation_of_information(
            self.x_rand_col, self.y_rand_col
        )
        voi_xy_pd = voi.variation_of_information(
            self.rand_df["x"].to_frame(), self.rand_df["y"].to_frame()
        )
        self.assertAlmostEqual(voi_xy_np, voi_xy_pd)

    def test_string_input(self):
        x_string = self.x_rand_col.astype("S10")
        y_string = self.y_rand_col.astype("S10")
        voi_xy_int = voi._voi(self.x_rand_col, self.y_rand_col)
        voi_xy_string = voi._voi(x_string, y_string)
        self.assertAlmostEqual(voi_xy_int, voi_xy_string)

    def test_same_dist(self):
        voi_xx = voi._voi(self.x_rand_col, self.x_rand_col)
        self.assertAlmostEqual(voi_xx, 0.0)


if __name__ == "__main__":
    unittest.main()

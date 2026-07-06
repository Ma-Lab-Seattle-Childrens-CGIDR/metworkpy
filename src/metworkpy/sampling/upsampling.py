"""
Submodule for upsampling from a set of points in a convex polytope
"""

# Standard Library Imports
from typing import Optional, Union

# External Imports
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# Local
from metworkpy.metworkpy_types import Array1D, Array2D


def upsample(
    samples: pd.DataFrame,
    n_samples: int = 1_000,
    processes: Optional[int] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> pd.DataFrame:
    """
    Perform upsampling of samples drawn from a convex polytope

    Parameters
    ----------
    samples : pd.DataFrame
        Samples taken from a convex polytope, for example using
        the `metworkpy.sampling.corner_sampling` function.
    n_samples : int, default=1000
        The number of additional samples to generate
    processes : int, optional
        Number of processes for performing the
        calculations
    seed : int or np.random.Generator, optional
        Seed for the random number generator performing
        the sampling

    Returns
    -------
    samples : pd.DataFrame
        A DataFrame of the original samples, with shape
        (n+n_samples, m) where the original samples DataFrame had
        shape (n,m). That is, it is the orignal DataFrame
        with the additional samples appended to the end.
        The column named remain unchanged.

    Notes
    -----
    This method iteratively selects a random proportion of
    the samples in the original sample array, and treats these
    samples as corners and samples from the interior of the
    polytope they describe.
    """
    # Seed needs to not be none, so that it can be
    # passed with the id to generate reproducible results
    if seed is None:
        rng = np.random.default_rng()
        seed = rng.integers(0, np.iinfo(np.int_).max)
    if isinstance(seed, np.random.Generator):
        seed = seed.integers(0, np.iinfo(np.int_).max)
    # Store the sample columns, and convert it to a np.array
    sample_cols = samples.columns
    sample_array = samples.to_numpy()
    # Create a results array to store the additional samples
    upsample_array = np.zeros((n_samples, len(sample_cols)))
    for idx, sample in enumerate(
        Parallel(n_jobs=processes, return_as="generator")(
            delayed(_upsample_worker)(samples=sample_array, seed=[i, seed])
            for i in range(n_samples)
        )
    ):
        upsample_array[idx] = sample
    # Concatenate on the new samples
    combined_samples = pd.DataFrame(
        np.vstack([sample_array, upsample_array]), columns=sample_cols
    )
    return combined_samples


def _upsample_worker(
    samples: Array2D, seed: Optional[Union[int, list[int]]]
) -> Array1D:
    rng = np.random.default_rng(seed)
    # Select the proportion of samples which will be used as corners in the
    # polytope to sample inside
    n_corners = rng.integers(2, samples.shape[0] + 1)
    # Actually select the corners
    corners = rng.choice(samples, size=n_corners, replace=False, axis=0)
    # Get a weights array, and normalize it
    weights_array: Array1D = rng.random(n_corners)
    weights_array /= weights_array.sum()
    # Take the weighted sum of the corners
    return (corners.T * weights_array).sum(axis=1)

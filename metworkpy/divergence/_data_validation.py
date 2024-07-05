"""
Functions for validating arguments provided to the divergence functions in this module
"""

import numpy as np
from numpy._typing import ArrayLike


def _validate_sample(arr: ArrayLike) -> np.ndarray:
    # Coerce to ndarray if needed
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    # Check that the sample only has two axes
    if len(arr.shape) > 2:
        raise ValueError("Sample must have a maximum of 2 axes")
    # If 1D, change to a column vector
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _validate_samples(p: ArrayLike, q: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    # Coerce
    try:
        p = _validate_sample(p)
        q = _validate_sample(q)
    except ValueError as err:
        raise ValueError(
            f"p and q must have a maximum of two axes, but p has {len(p.shape)} axes, "
            f"and q has"
            f"{len(q.shape)} axes."
        ) from err

    if p.shape[1] != q.shape[1]:
        raise ValueError(
            f"Both p and q distributions must have the same dimension, but p has a "
            f"dimension {p.shape[1]}"
            f"and q has a dimension {q.shape[1]}"
        )

    return p, q


# This is just a stub, but can be used to add any additional validation logic to the
# handling of discrete inputs
def _validate_discrete(sample):
    if sample.shape[1] != 1:
        raise ValueError(
            "For samples from discrete distributions, only a single dimension for the "
            "samples is supported"
            ", sample should have shape (n_samples, 1)."
        )
    return sample

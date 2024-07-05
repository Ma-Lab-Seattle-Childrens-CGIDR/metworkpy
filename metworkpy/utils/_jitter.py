"""
Functions to add noise to arrays to avoid ties
"""

# Standard Library Imports
from typing import Union

# External Imports
import numpy as np


# Local Imports


def _jitter_single(arr: np.ndarray, jitter: float, generator: np.random.Generator):
    """
    Add jitter to single array

    :param arr: Array to add noise to
    :type arr: np.ndarray
    :param jitter: Standard deviation of noise to add
    :type jitter: float
    :param generator: Numpy random number generator used to generate the noise
    :type generator: np.random.generator
    :return: Array with noise added (same shape as input array)
    :rtype: np.ndarray
    """
    return arr + generator.normal(loc=0.0, scale=jitter, size=arr.shape)


def _jitter(
    x: np.ndarray,
    y: np.ndarray,
    jitter: Union[float, tuple[float, float]],
    jitter_seed: int,
    discrete_x: bool,
    discrete_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add noise to two arrays based on whether they are discrete

    :param x: First array to add noise to
    :type x: np.ndarray
    :param y: Second array to add noise to
    :type y: np.ndarray
    :param jitter: Standard deviation of noise to add. Can be a single float
        which is used as the standard deviation
        for the noise added to both arrays, or a tuple of floats where the
        first value is used for the noise applied to
        x, and the second for y.
    :type jitter: Union[float, tuple[float,float]]
    :param jitter_seed: Seed for random number generator used to generate the
        noise
    :type jitter_seed: int
    :param discrete_x: Whether x is a discrete distribution
    :type discrete_x: bool
    :param discrete_y: Whether y is a discrete distribution
    :type discrete_y: bool
    :return: x and y arrays with noise added as a tuple
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    generator = np.random.default_rng(jitter_seed)
    if isinstance(jitter, tuple):
        if len(jitter) != 2:
            raise ValueError(
                f"If jitter is a tuple, must have length 2 not {len(jitter)}"
            )
        jitter_x, jitter_y = jitter
    elif isinstance(jitter, float):
        jitter_x = jitter
        jitter_y = jitter
    else:
        raise ValueError(
            "Unexpected type for jitter, should be float or tuple of floats"
        )
    if not discrete_x:
        x = _jitter_single(x, jitter=jitter_x, generator=generator)
    if not discrete_y:
        y = _jitter_single(y, jitter=jitter_y, generator=generator)
    return x, y

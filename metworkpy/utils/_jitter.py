"""Functions to add noise to arrays to avoid ties"""

# Standard Library Imports
from typing import Union

# External Imports
import numpy as np


# Local Imports


def _jitter_single(arr: np.ndarray, jitter: float, generator: np.random.Generator):
    """Add jitter to single array

    Parameters
    ----------
    arr : np.ndarray
        Array to add noise to
    jitter : float
        Standard deviation of noise to add
    generator : np.random.generator
        Numpy random number generator used to generate the noise

    Returns
    -------
    np.ndarray
        Array with noise added (same shape as input array)
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
    """Add noise to two arrays based on whether they are discrete

    Parameters
    ----------
    x : np.ndarray
        First array to add noise to
    y : np.ndarray
        Second array to add noise to
    jitter : Union[float, tuple[float,float]]
        Standard deviation of noise to add. Can be a single float which
        is used as the standard deviation for the noise added to both
        arrays, or a tuple of floats where the first value is used for
        the noise applied to x, and the second for y.
    jitter_seed : int
        Seed for random number generator used to generate the noise
    discrete_x : bool
        Whether x is a discrete distribution
    discrete_y : bool
        Whether y is a discrete distribution

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x and y arrays with noise added as a tuple
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

"""
A wrapper function around methods which can calculate divergences for continuous and discrete distributions
"""

# Imports
# Standard Library Imports
from typing import Optional, Union, Callable, Any

# External Imports
import numpy as np
from numpy.typing import ArrayLike, NDArray

# Local Imports
from metworkpy.utils._arguments import _parse_metric
from metworkpy.divergence._data_validation import (
    _validate_samples,
    _validate_discrete,
)
from metworkpy.utils._jitter import _jitter_single


def _wrap_divergence_functions(
    p: ArrayLike,
    q: ArrayLike,
    discrete_method: Callable[[NDArray[Any], NDArray[Any]], float],
    continuous_method: Callable[[NDArray[float], NDArray[float], int, float], float],
    n_neighbors: int = 5,
    discrete: bool = False,
    jitter: Optional[float] = None,
    jitter_seed: Optional[int] = None,
    distance_metric: Union[float, str] = "euclidean",
) -> float:
    """
    Calculate the divergence between two distributions represented by samples p and q
    :param p: Array representing sample from a distribution, should have shape (n_samples, n_dimensions). If `p` is
        one dimensional, it will be reshaped to (n_samples,1). If it is not a np.ndarray, this function will attempt to
        coerce it into one.
    :type p: ArrayLike
    :param q: Array representing sample from a distribution, should have shape (n_samples, n_dimensions). If `q` is
        one dimensional, it will be reshaped to (n_samples,1). If it is not a np.ndarray, this function will attempt to
        coerce it into one.
    :type q: ArrayLike
    :param discrete_method: Method to use to calculate the divergence between two discrete distributions,
        should take two positional arguments for p and q.
    :param continuous_method: Method to use to calculate the divergence between two continuous distributions,
        should take two positional arguments for p and q, as well as keyword arguments for n_neighbors, and
        metric (which will be a float representing a Minkowski p-norm).
    :param n_neighbors: Number of neighbors to use for computing mutual information. Will attempt to coerce into an
        integer. Must be at least 1. Default 5.
    :type n_neighbors: int
    :param discrete: Whether the samples are from discrete distributions
    :type discrete: bool
    :param jitter: Amount of noise to add to avoid ties. If None no noise is added. If a float, that is the standard
        deviation of the random noise added to the continuous samples. If a tuple, the first element is the standard
        deviation of the noise added to the x array, the second element is the standard deviation added to the y array.
    :type jitter: Union[None, float, tuple[float,float]]
    :param jitter_seed:Seed for the random number generator used for adding noise
    :type jitter_seed:Union[None, int]
    :param distance_metric: Metric to use for computing distance between points in p and q, can be "Euclidean",
        "Manhattan", or "Chebyshev". Can also be a float representing the Minkowski p-norm.
    :type distance_metric: Union[str, float]
    :return: The divergence between p and q
    :rtype: float
    """
    try:
        n_neighbors = int(n_neighbors)
    except ValueError as err:
        raise ValueError(
            f"n_neighbors must be able to be converted to an integer, but a {type(n_neighbors)} was"
            f"given instead."
        ) from err
    distance_metric = _parse_metric(distance_metric)
    p, q = _validate_samples(p, q)
    if jitter and not discrete:
        generator = np.random.default_rng(jitter_seed)
        p = _jitter_single(p, jitter=jitter, generator=generator)
        q = _jitter_single(q, jitter=jitter, generator=generator)
    if discrete:
        try:
            p = _validate_discrete(p)
            q = _validate_discrete(q)
        except ValueError as err:
            raise ValueError(
                f"p and q must represent single dimensional samples, and so have shape (n_samples, 1)"
                f"but p has dimension {p.shape[1]}, and q has dimension {q.shape[1]}."
            ) from err
        return discrete_method(p, q)

    return continuous_method(p, q, n_neighbors=n_neighbors, metric=distance_metric)

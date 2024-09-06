"""
Functions for calculating the Jenson-Shannon divergence between two sampled distributions
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Optional, Union

# External Imports
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from scipy.spatial.distance import jensenshannon
from scipy.special import digamma

# Local Imports
from metworkpy.divergence._main_wrapper import _wrap_divergence_functions
from metworkpy.divergence._pairwise_divergence import _divergence_array


# region Main Function
def js_divergence(
    p: ArrayLike,
    q: ArrayLike,
    n_neighbors: int = 5,
    discrete: bool = False,
    jitter: Optional[float] = None,
    jitter_seed: Optional[int] = None,
    distance_metric: Union[float, str] = "euclidean",
) -> float:
    """
    Calculate the Jensen-Shannon divergence between two distributions represented by samples p and q

    :param p: Array representing sample from a distribution, should have shape (n_samples, n_dimensions). If `p` is
        one dimensional, it will be reshaped to (n_samples,1). If it is not a np.ndarray, this function will attempt to
        coerce it into one.
    :type p: ArrayLike
    :param q: Array representing sample from a distribution, should have shape (n_samples, n_dimensions). If `q` is
        one dimensional, it will be reshaped to (n_samples,1). If it is not a np.ndarray, this function will attempt to
        coerce it into one.
    :type q: ArrayLike
    :param n_neighbors: Number of neighbors to use for computing divergence. Will attempt to coerce into an
        integer. Must be at least 1. Default 5.
    :type n_neighbors: int
    :param discrete: Whether the samples are from discrete distributions
    :type discrete: bool
    :param jitter: Amount of noise to add to avoid ties. If None no noise is added. If a float, that is the standard
        deviation of the random noise added to the continuous samples. If a tuple, the first element is the standard
        deviation of the noise added to the x array, the second element is the standard deviation added to the y array.
    :type jitter: Union[None, float, tuple[float,float]]
    :param jitter_seed: Seed for the random number generator used for adding noise
    :type jitter_seed: Union[None, int]
    :param distance_metric: Metric to use for computing distance between points in p and q, can be \"Euclidean\",
        \"Manhattan\", or \"Chebyshev\". Can also be a float representing the Minkowski p-norm.
    :type distance_metric: Union[str, float]
    :return: The Jensen-Shannon divergence between p and q
    :rtype: float

    .. seealso::

       Ross, B. C. (2014). Mutual Information between Discrete and Continuous Data Sets. PLoS ONE, 9(2), e87357.
    """
    return _wrap_divergence_functions(
        p=p,
        q=q,
        discrete_method=_js_disc,
        continuous_method=_js_cont,
        n_neighbors=n_neighbors,
        discrete=discrete,
        jitter=jitter,
        jitter_seed=jitter_seed,
        distance_metric=distance_metric,
    )


# endregion Main Function


# region Array Divergence


def js_divergence_array(
    p: pd.DataFrame | np.ndarray,
    q: pd.DataFrame | np.ndarray,
    n_neighbors: int = 5,
    metric: float | str = 2.0,
    processes: int = 1,
) -> np.ndarray | pd.Series:
    """
    Calculate the Jensen-Shannon divergence between the columns in two arrays using the
    nearest neighbors method.

    :param p: Flux sample array, with columns representing different reactions and rows representing
        different samples. Should have same number of columns as q.
    :type p: pd.DataFrame | np.ndarray
    :param q: Flux sample array, with columns representing different reactions and rows representing
        different samples. Should have same number of columns as p.
    :type q: pd.DataFrame | np.ndarray
    :param n_neighbors: Number of neighbors to use when estimating divergence
    :type n_neighbors: int
    :param metric: Metric to use for computing distance between points in p and q, can be \"Euclidean\",
        \"Manhattan\", or \"Chebyshev\". Can also be a float representing the Minkowski p-norm.
    :type metric: float | str
    :param processes: Number of processes to use when calculating the divergence (default 1)
    :type processes: int
    :return: Array with length equal to the number of columns in p and q, the ith value representing
        the divergence between the ith column of p and the ith column of q. If both p and q are
        numpy ndarrays, this returns a ndarray with shape (ncols,). If either p or q are pandas
        DataFrames then returns a pandas Series with index the same as the columns in the DataFrame
        (p takes priority if the column names differ).
    :rtype: np.ndarray | pd.DataFrame
    """
    return _divergence_array(
        p=p,
        q=q,
        divergence_function=_js_cont,
        n_neighbors=n_neighbors,
        metric=metric,
        processes=processes,
    )


# endregion Array Divergence


# region Continuous Case
# Because this method is similar to mutual information between a continuous and discrete distribution
# This method is also inspired by sklearn
def _js_cont(
    p: np.ndarray,
    q: np.ndarray,
    n_neighbors: int = 5,
    metric: float = 2.0,
    **kwargs,
) -> float:
    """
    Calculate the Jensen-Shannon divergence between samples from two continuous distributions using the
    nearest neighbor method.

    :param p: Array representing samples from the first continuous distribution, should have shape
        (n_samples, n_dimensions), where n_dimensions>=1
    :type p: np.ndarray
    :param q: Array representing samples from the second continuous distribution, should have shape
        (n_samples, n_dimensions), where n_dimensions>=1
    :type q: np.ndarray
    :param n_neighbors:The number of neighbors to use for computing mutual information
    :type n_neighbors: int
    :param metric: Metric to use for computing distance between points in y (must be `float>=1` representing
        Minkowski p-norm)
    :type metric: float
    :param kwargs: Arguments passed to KDTree constructor
    :return: Jensen-Shannon divergence between p and q
    :rtype: float


    .. seealso::

       Ross, B. C. (2014). Mutual Information between Discrete and Continuous Data Sets. PLoS ONE, 9(2), e87357.
            Paper from which this method was obtained
       :func: `_js_disc`
            Function for calculating Jensen-Shannon divergence when the distributions are discrete

    """
    combined = np.vstack([p, q])
    n_data_points = combined.shape[0]
    classes = np.vstack([np.zeros((len(p), 1)), np.ones((len(q), 1))])
    # This is overkill, but it matches the MI formulation better, and allows for easier
    # refactoring into the divergence
    # between more than 2 distributions
    discrete_classes, counts = np.unique(classes, return_counts=True)

    full_tree = KDTree(combined, **kwargs)
    radius_array = np.empty(shape=n_data_points, dtype=float)
    count_array = np.empty(shape=n_data_points, dtype=float)

    for d_class, count in zip(discrete_classes, counts):
        same_class_index = (classes == d_class).squeeze()
        count_array[same_class_index] = count
        type_tree = KDTree(combined[same_class_index, :], **kwargs)
        # Get the neighbors (1st neighbor will just be the point itself)
        dist, _ = type_tree.query(
            combined[same_class_index, :], k=[n_neighbors + 1], p=metric
        )
        dist = dist.squeeze()
        radius_array[same_class_index] = np.nextafter(dist, np.inf)

    neighbors_within_radius = (
        full_tree.query_ball_point(combined, radius_array, p=metric, return_length=True)
        - 1
    )

    # Use formula 9 from Ross, 2014
    return (
        digamma(n_data_points)
        + digamma(n_neighbors)
        - np.sum(
            np.divide(
                digamma(count_array) + digamma(neighbors_within_radius),
                count_array,
            )
        )
        / len(discrete_classes)
    )


# endregion Continuous Case


# region Discrete Case
def _js_disc(p: np.ndarray, q: np.ndarray):
    """
    Calculate the Jensen-Shannon divergence between samples from two discrete distributions

    :param p: Array representing samples from the first discrete distribution, should have shape
        (n_samples, 1)
    :type p: np.ndarray
    :param q: Array representing samples from the second discrete distribution, should have shape
        (n_samples, 1)
    :type q: np.ndarray
    :return: Jensen-Shannon divergence between p and q
    :rtype: float

    """
    p_elements, p_counts = np.unique(p, return_counts=True)
    q_elements, q_counts = np.unique(q, return_counts=True)

    p_total = p.shape[0]
    q_total = q.shape[0]

    comb_elements = np.union1d(p_elements, q_elements)

    p_freq = (
        np.array(
            [
                p_counts[p_elements == elem].item() if elem in p_elements else 0.0
                for elem in comb_elements
            ]
        )
        / p_total
    )

    q_freq = (
        np.array(
            [
                q_counts[q_elements == elem].item() if elem in q_elements else 0.0
                for elem in comb_elements
            ]
        )
        / q_total
    )

    return jensenshannon(p_freq, q_freq, base=np.e) ** 2


# endregion Discrete Case

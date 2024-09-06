"""
Function for calculating the Kullback-Leibler divergence between two probability distributions based on samples from
those distributions.
"""

# Standard Library Imports
from __future__ import annotations

from typing import Optional, Union

# External Imports
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

# Local Imports
from metworkpy.divergence._main_wrapper import _wrap_divergence_functions
from metworkpy.divergence._pairwise_divergence import _divergence_array


# region Main Function
def kl_divergence(
    p: ArrayLike,
    q: ArrayLike,
    n_neighbors: int = 5,
    discrete: bool = False,
    jitter: Optional[float] = None,
    jitter_seed: Optional[int] = None,
    distance_metric: Union[float, str] = "euclidean",
) -> float:
    """
    Calculate the Kulback-Leibler divergence between two distributions represented by samples p and q

    :param p: Array representing sample from a distribution, should have shape (n_samples, n_dimensions). If `p` is
        one dimensional, it will be reshaped to (n_samples,1). If it is not a np.ndarray, this function will attempt to
        coerce it into one.
    :type p: ArrayLike
    :param q: Array representing sample from a distribution, should have shape (n_samples, n_dimensions). If `q` is
        one dimensional, it will be reshaped to (n_samples,1). If it is not a np.ndarray, this function will attempt to
        coerce it into one.
    :type q: ArrayLike
    :param n_neighbors: Number of neighbors to use for computing mutual information. Will attempt to coerce into an
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
    :return: The Kulback-Leibler divergence between p and q
    :rtype: float

    .. note::

       - This function is not symetrical, and q is treated as representing the reference condition. If you want a
         symetric metric try the Jenson-Shannon divergence.

    .. seealso::

       1. Q. Wang, S. R. Kulkarni and S. Verdu, \"Divergence Estimation for Multidimensional Densities Via
       k-Nearest-Neighbor Distances\" in IEEE Transactions on Information Theory, vol. 55, no. 5, pp. 2392-2405,
       May 2009, doi: 10.1109/TIT.2009.2016060.

            Method for estimating the mutual information between samples from two continuous distributions based
            on nearest-neighbor distances.
    """
    return _wrap_divergence_functions(
        p=p,
        q=q,
        discrete_method=_kl_disc,
        continuous_method=_kl_cont,
        n_neighbors=n_neighbors,
        discrete=discrete,
        jitter=jitter,
        jitter_seed=jitter_seed,
        distance_metric=distance_metric,
    )


# endregion Main Function


# region Discrete Divergence
def _kl_disc(p: np.ndarray, q: np.ndarray):
    """
    Compute the Kullback-Leibler divergence for two samples from two finite discrete distributions

    :param p: Sample from the p distribution, with shape (n_samples, 1)
    :type p: np.ndarray
    :param q: Sample from the q distribution, with shape (n_samples, 1)
    :type q: np.ndarray
    :return: The Kullback-Leibler divergence between the two distributions represented by the p and q samples
    :rtype: float

    """
    p_elements, p_counts = np.unique(p, return_counts=True)
    q_elements, q_counts = np.unique(q, return_counts=True)
    p_freq = p_counts / p_counts.sum()
    q_freq = q_counts / q_counts.sum()

    kl = 0.0
    for val in np.union1d(p_elements, q_elements):
        pf = p_freq[p_elements == val]
        qf = q_freq[q_elements == val]
        # If the length of the pf vector is 0, add a 0. element
        if len(pf) == 0:
            pf = np.zeros(shape=(1,))
        # If the length of qf is 0 (so the estimate of the probability is 0), the divergence defined as +inf
        if len(qf) == 0:
            return np.inf
        kl += (pf * np.log(pf / qf)).item()
    return kl


# endregion Discrete Divergence


def kl_divergence_array(
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
        divergence_function=_kl_cont,
        n_neighbors=n_neighbors,
        metric=metric,
        processes=processes,
    )


# region Continuous Divergence
def _kl_cont(p: np.ndarray, q: np.ndarray, n_neighbors: int = 5, metric: float = 2.0):
    """
    Calculate the Kullback-Leibler divergence for two samples from two continuous distributions

    :param p: Sample from the p distribution, with shape (n_samples, n_dimensions)
    :type p: np.ndarray
    :param q: Sample from the q distribution, with shape (n_samples, n_dimensions
    :type q: np.ndarray
    :param n_neighbors: Number of neighbors to use for the estimator
    :type n_neighbors: int
    :param metric: Minkowski p-norm to use for calculating distances, must be at least 1
    :type metric: float
    :return: The Kullback-Leibler divergence between the distributions represented by the p and q samples
    :rtype: float

    """
    # Construct the KDTrees for finding neighbors, and neighbor distances
    p_tree = KDTree(p)
    q_tree = KDTree(q)

    # Find the distance to the kth nearest neighbor of each p point in both p and q samples
    # Note: The distance arrays are column vectors
    p_dist, _ = p_tree.query(p, k=[n_neighbors + 1], p=metric)
    q_dist, _ = q_tree.query(p, k=[n_neighbors], p=metric)

    # Reshape p and q_dist into 1D arrays
    p_dist = p_dist.squeeze()
    q_dist = q_dist.squeeze()

    # Find the KL-divergence estimate using equation (5) from Wang and Kulkarni, 2009
    return (
        (p.shape[1] / p.shape[0]) * np.sum(np.log(np.divide(q_dist, p_dist)))
        + np.log(q.shape[0] / (p.shape[0] - 1))
    ).item()


# endregion Continuous Divergence

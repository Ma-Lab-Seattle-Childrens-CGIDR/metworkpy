"""
Function for calculating the Kullback-Leibler divergence between two probability distributions based on samples from
those distributions.
"""

# Standard Library Imports
from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

# External Imports
import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma

# Local Imports
from metworkpy.divergence._main_wrapper import (
    _wrap_divergence_functions,
    DivergenceResult,
)
from metworkpy.divergence._pairwise_divergence import (
    _divergence_array,
    ArrayInput,
    Array1D,
)
from metworkpy.metworkpy_types import Array2D


# region Main Function
def kl_divergence(
    p: np.typing.ArrayLike,
    q: np.typing.ArrayLike,
    calculate_pvalue: bool = False,
    alternative: Literal["less", "greater", "two-sided"] = "greater",
    permutations: int = 500,
    permutation_rng: Optional[Union[np.random.Generator, int]] = None,
    permutation_estimation_method: Literal[
        "kernel", "empirical"
    ] = "empirical",
    n_neighbors: int = 5,
    discrete: bool = False,
    jitter: Optional[float] = None,
    jitter_seed: Optional[int] = None,
    distance_metric: Union[float, str] = "euclidean",
    clip: bool = False,
) -> Union[float, DivergenceResult]:
    """Calculate the Kulback-Leibler divergence between two distributions represented by samples p and q

    Parameters
    ----------
    p : ArrayLike
        Array representing sample from a distribution, should have shape
        (n_samples, n_dimensions). If `p` is one dimensional, it will be
        reshaped to (n_samples,1). If it is not a np.ndarray, this
        function will attempt to coerce it into one.
    q : ArrayLike
        Array representing sample from a distribution, should have shape
        (n_samples, n_dimensions). If `q` is one dimensional, it will be
        reshaped to (n_samples,1). If it is not a np.ndarray, this
        function will attempt to coerce it into one.
    calculate_pvalue : bool, default=False
        Whether the p-value should be calculated using a permutation test
    alternative : 'less', 'greater', or 'two-sided', default='greater'
        The alternative to use, see
        `metworkpy.utils.permutation.permutation_test`
    permutations : int, default=9999
         The number of permuatations to use when calculating the p-value
    permutation_rng : np.random.Generator or int, Optional
        A numpy random generator to use for sampling, or an int
        to seed the default generator.
    permutation_estimation_method : {"kernel", "empirical"}, default="empirical"
        Method to use for estimating p-value, either an empirical cdf,
        or a gaussian_kde
    n_neighbors : int
        Number of neighbors to use for computing mutual information.
        Will attempt to coerce into an integer. Must be at least 1.
        Default 5.
    discrete : bool
        Whether the samples are from discrete distributions
    jitter : Union[None, float, tuple[float,float]]
        Amount of noise to add to avoid ties. If None no noise is added.
        If a float, that is the standard deviation of the random noise
        added to the continuous samples. If a tuple, the first element
        is the standard deviation of the noise added to the x array, the
        second element is the standard deviation added to the y array.
    jitter_seed : Union[None, int]
        Seed for the random number generator used for adding noise
    distance_metric : Union[str, float]
        Metric to use for computing distance between points in p and q,
        can be \"Euclidean\", \"Manhattan\", or \"Chebyshev\". Can also
        be a float representing the Minkowski p-norm.
    clip : bool, default=False
        Whether or not to clip the divergence values at 0.0

    Returns
    -------
    float
        The Kulback-Leibler divergence between p and q

    Notes
    -----

    - This function is not symmetrical, p is treated as a 'true' distribution, and q as an
      approximating distribution. If you want a symmetric metric try the Jenson-Shannon
      divergence.

    See Also
    --------

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
        calculate_pvalue=calculate_pvalue,
        alternative=alternative,
        permutations=permutations,
        permutation_rng=permutation_rng,
        permutation_estimation_method=permutation_estimation_method,
        n_neighbors=n_neighbors,
        discrete=discrete,
        jitter=jitter,
        jitter_seed=jitter_seed,
        distance_metric=distance_metric,
        clip=clip,
    )


def kl_divergence_array(
    p: ArrayInput,
    q: ArrayInput,
    axis: int = 1,
    processes: int = 1,
    **kwargs,
) -> Union[Array1D, Tuple[Array1D, Array1D]]:
    """Calculate the Kullback-Leibler divergence between two arrays along the
    specified axis.

    Parameters
    ----------
    p : ArrayInput
        Sample array, where slices along the specified axis represent
        the distributions to calculate the divergence between.
    q : ArrayInput
        Sample array, where slices along the specified axis represent
        the distributions to calculate the divergence between.
    axis : int, default=1
        Axis to slice along to get the arrays representing samples from
        the distributions to calculate the divergence between. For example,
        axis=1 specified that 2-dimensional the p and q arrays will be sliced
        along the columns. The size of p and q along this axis must match.
    processes : int
        Number of processes to use when calculating the divergence
        (default 1)
    kwargs
        Keyword arguments are passed to the `kl_divergence` function

    Returns
    -------
    Array1D or Tuple of Array1D, Array1D
        Array with length equal to the shape along the axis in p and q, the
        ith value representing the divergence between the ith slice along
        specified axis of p and q. f both p and q are numpy ndarrays,
        this returns a ndarray with shape (ncols,). If either p or q are
        pandas DataFrames then returns a pandas Series with index the
        same as the columns in the DataFrame (p takes priority if the
        column names differ).

    Notes
    -----
        If either p or q are pandas DataFrames, they both must be and their
        indices along the specified axis must be the same
    """
    return _divergence_array(
        p=p,
        q=q,
        divergence_function=kl_divergence,  # type: ignore
        axis=axis,
        processes=processes,
        **kwargs,
    )


# endregion Main Function


# region Discrete Divergence
def _kl_disc(p: np.ndarray, q: np.ndarray):
    """Compute the Kullback-Leibler divergence for two samples from two finite discrete distributions

    Parameters
    ----------
    p : np.ndarray
        Sample from the p distribution, with shape (n_samples, 1)
    q : np.ndarray
        Sample from the q distribution, with shape (n_samples, 1)

    Returns
    -------
    float
        The Kullback-Leibler divergence between the two distributions
        represented by the p and q samples
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
        # If the length of qf is 0 (so the estimate of the probability is 0),
        # the divergence defined as +inf
        if len(qf) == 0:
            return np.inf
        kl += (pf * np.log(pf / qf)).item()
    return kl


# endregion Discrete Divergence


# region Continuous Divergence
def _kl_cont(
    p: Array2D,
    q: Array2D,
    n_neighbors: int = 5,
    distance_metric: float = 2.0,
    clip=False,
):
    """
    Calculate the Kullback-Leibler divergence for two samples from two continuous distributions

    Parameters
    ----------
    p : np.ndarray
        Sample from the p distribution, with shape (n_samples,
        n_dimensions)
    q : np.ndarray
        Sample from the q distribution, with shape (n_samples,
        n_dimensions)
    n_neighbors : int
        Number of neighbors to use for the estimator
    distance_metric : float
        Minkowski p-norm to use for calculating distances, must be at
        least 1
    clip : bool, default=False
        Whether or not to clip the divergence values at 0.0


    Returns
    -------
    float
        The Kullback-Leibler divergence between the distributions
        represented by the p and q samples
    """
    # Construct the KDTrees for finding neighbors, and neighbor distances
    p_tree = KDTree(p)
    q_tree = KDTree(q)

    # Find the distance to the kth nearest neighbor of each p point in both p and q samples
    # Note: The distance arrays are column vectors
    p_dist, _ = p_tree.query(
        p, k=[n_neighbors + 1], p=distance_metric
    )  # rho in wang et al. eq 5
    q_dist, _ = q_tree.query(
        p, k=[n_neighbors], p=distance_metric
    )  # nu in wang et al. eq 5

    # Reshape p and q_dist into 1D arrays
    p_dist = p_dist.squeeze()
    q_dist = q_dist.squeeze()

    # Find the KL-divergence estimate using equation (5) from Wang and Kulkarni, 2009

    div = (
        (p.shape[1] / p.shape[0]) * np.sum(np.log(np.divide(q_dist, p_dist)))
        + np.log(q.shape[0] / (p.shape[0] - 1))
    ).item()
    if not clip:
        return div
    return max(div, 0.0)


def _kl_cont_adaptive(
    p: Array2D,
    q: Array2D,
    epsilon_mult: float = 1.0,
    distance_metric: float = 2.0,
    clip: bool = False,
):
    r"""
    Calculate the Kullback-Leibler divergence for two samples from
    two continuous distributions using an adaptive k-NN
    estimator

    Parameters
    ----------
    p : np.ndarray
        Sample from the p distribution, with shape (n_samples,
        n_dimensions)
    q : np.ndarray
        Sample from the q distribution, with shape (n_samples,
        n_dimensions)
    epsilon_mult : float, default=1.0
        Number to multiply epsilon (the radius used to find density)
        by, see notes for more details
    distance_metric : float
        Minkowski p-norm to use for calculating distances, must be at
        least 1
    clip : bool, default=False
        Whether or not to clip the divergence values at 0.0


    Returns
    -------
    float
        The Kullback-Leibler divergence between the distributions
        represented by the p and q samples

    Notes
    -----
    Uses the second version of an adaptive-k estimator from [1]_,
    found in equation (25).

    This method uses a constant radius around each point in p to
    calculate density based on the number of points in p and q which
    lie within the ball of radius $\epsilon(i)$ centered at
    each point i in p. The constant radius for each point in p
    is determined by the maximum distance to its nearest neighbor
    in either p or q. This radius will then be multiplied by `epsilon_mult`.
    A value of `epsilon_mult` greater than 1.0 can reduce the
    bias of the estimator.

    Reference
    ---------
    ..[1] Wang, Q.; Kulkarni, S. R.; Verdu, S. Divergence Estimation for
          Multidimensional Densities Via K-Nearest-Neighbor Distances.
          IEEE Trans. Inform. Theory 2009, 55 (5), 2392–2405.
          https://doi.org/10.1109/TIT.2009.2016060.
    """
    # Construct the KDTrees for finding neighbors, and neighbor distances
    p_tree = KDTree(p)
    q_tree = KDTree(q)

    # First find the epsilon using equations (26), (27), and (28)

    # Find the distances from X to its nearest neighbors in p, and
    # its nearest neighbors in q (this is for calculating epsilon(i))
    # Since the points are in p, need to find the second neighbor
    # (first is the point itself)
    p_nn_dist, _ = np.nextafter(
        p_tree.query(p, [2], p=distance_metric), np.inf
    )
    q_nn_dist, _ = np.nextafter(
        q_tree.query(p, [1], p=distance_metric), np.inf
    )

    # Find the epsilon (radius around each point),
    # this will be the maximum of the p and q nn distances
    epsilon = np.maximum(p_nn_dist, q_nn_dist).ravel() * epsilon_mult

    # Find the number of neighbors in p and q
    # which are within epsilon of each point in p
    # This corresponds to finding the l and k values
    # for equation 25
    # NOTE: The nextafter increases the epsilon in order to
    # keep the nearest neighbor insideh
    # NOTE: This is a 1-D array of number of neighbors
    l_i = (
        p_tree.query_ball_point(
            x=p,
            r=epsilon.ravel(),
            p=distance_metric,
            return_length=True,
        )
        - 1  # Subtract 1 to remove the points themselves
    )
    k_i = q_tree.query_ball_point(
        x=p,
        r=epsilon.ravel(),
        p=distance_metric,
        return_length=True,
    )

    # Now, calculate the actual divergence estimate
    return np.sum(digamma(l_i) - digamma(k_i)) / p.shape[0] + np.log(
        q.shape[0] / (p.shape[0] - 1)
    )


# endregion Continuous Divergence

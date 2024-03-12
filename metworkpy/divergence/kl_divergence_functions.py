"""
Function for calculating the Kullback-Leibler divergence between two probability distributions based on samples from
those distributions.
"""
# Standard Library Imports
from typing import Union

# External Imports
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

# Local Imports
from metworkpy.utils._jitter import _jitter_single
from metworkpy.utils._arguments import _parse_metric


# region Main Function
def kl_divergence(p: ArrayLike, q: ArrayLike, n_neighbors: int = 5, discrete: bool = False, jitter: float = None,
                  jitter_seed: int = None, distance_metric: Union[float, str] = "euclidean") -> float:
    """
    Calculate the Kullback-Leibler Divergence between two distributions represented by samples p and q.
    :param p:
    :type p:
    :param q:
    :type q:
    :param n_neighbors:
    :type n_neighbors:
    :param discrete:
    :type discrete:
    :param jitter:
    :type jitter:
    :param jitter_seed:
    :type jitter_seed:
    :param distance_metric:
    :type distance_metric:
    :return:
    :rtype:

    .. note::
       - This function is no symetrical, and q is treated as representing the reference condition. If you want a
         symetric metric try the Jenson-Shannon divergence

    .. seealso::
       1. 'Q. Wang, S. R. Kulkarni and S. Verdu, "Divergence Estimation for Multidimensional Densities Via
          k-Nearest-Neighbor Distances," in IEEE Transactions on Information Theory, vol. 55, no. 5, pp. 2392-2405,
          May 2009, doi: 10.1109/TIT.2009.2016060.'<https://ieeexplore.ieee.org/document/4839047>_
    """
    distance_metric = _parse_metric(distance_metric)
    p, q = _validate_samples(p=p, q=q)
    if jitter and not discrete:
        generator = np.random.default_rng(jitter_seed)
        p = _jitter_single(p, jitter=jitter, generator=generator)
        q = _jitter_single(q, jitter=jitter, generator=generator)
    if discrete:
        return _kl_disc(p,q)
    return _kl_cont(p,q,n_neighbors=n_neighbors, metric=distance_metric)


# endregion Main Function

# region Discrete Divergence
def _kl_disc(p: np.ndarray, q:np.ndarray):
    """
    Compute the Kullback-Leibler divergence for two samples from two finite discrete distributions
    :param p: Sample from the p distribution, with shape (n_samples, 1)
    :type p: np.ndarray
    :param q: Sample from the q distribution, with shape (n_samples, 1)
    :type q: np.ndarray
    :return: The Kullback-Leibler divergence between the two distributions represented by the p and q samples
    :rtype: float
    """
    try:
        p,q = _validate_discrete(p), _validate_discrete(q)
    except ValueError as err:
        raise ValueError(f"p and q must represent single dimensional samples, and so have shape (n_samples, 1)"
                         f"but p has dimension {p.shape[1]}, and q has dimension {q.shape[1]}.") from err
    p_elements, p_counts = np.unique(p, return_counts=True)
    q_elements, q_counts = np.unique(q, return_counts=True)
    p_freq = p_counts/p_counts.sum()
    q_freq = q_counts/q_counts.sum()

    kl = 0.
    for val in np.union1d(p_elements, q_elements):
        pf = p_freq[p_elements==val]
        qf = q_freq[q_elements==val]
        # If the length of the pf vector is 0, add a 0. element
        if len(pf)==0:
            pf = np.zeros(shape=(1,))
        # If the length of qf is 0 (so the estimate of the probability is 0), the divergence defined as +inf
        if len(qf) == 0:
            return np.inf
        kl+= (pf * log(pf/qf)).item()
    return kl

# endregion Discrete Divergence


# region Continuous Divergence
def _kl_cont(p: np.ndarray, q: np.ndarray, n_neighbors: int = 5, metric: float = 2.):
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
    return ((p.shape[1] / p.shape[0]) * np.sum(np.log(np.divide(q_dist, p_dist))) + np.log(
        q.shape[0] / (p.shape[0 - 1]))).item()


# endregion Continuous Divergence

# region Helper Functions

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
        raise ValueError(f"p and q must have a maximum of two axes, but p has {len(p.shape)} axes, and q has "
                         f"{len(q.shape)} axes.") from err

    if p.shape[1] != q.shape[1]:
        raise ValueError(f"Both p and q distributions must have the same dimension, but p has a dimension {p.shape[1]} "
                         f"and q has a dimension {q.shape[1]}")

    return p, q

def _validate_discrete(sample):
    if sample.shape[1]!=1:
        raise ValueError("For samples from discrete distributions, only a single dimension for the samples is supported"
                         ", sample should have shape (n_samples, 1).")
    return sample

# endregion Helper Functions

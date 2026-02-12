"""Functions for calculating the Jenson-Shannon divergence between two sampled distributions"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Literal, Optional, Tuple, Union

# External Imports
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from scipy.spatial.distance import jensenshannon
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


# region Main Function
def js_divergence(
    p: ArrayLike,
    q: ArrayLike,
    calculate_pvalue: bool = False,
    alternative: Literal["less", "greater", "two-sided"] = "greater",
    permutations: int = 500,
    permutation_rng: Optional[Union[np.random.Generator, int]] = None,
    permutation_estimation_method: Literal["kernel", "empirical"] = "kernel",
    n_neighbors: int = 5,
    discrete: bool = False,
    jitter: Optional[float] = None,
    jitter_seed: Optional[int] = None,
    distance_metric: Union[float, str] = "euclidean",
) -> Union[float, DivergenceResult]:
    """Calculate the Jensen-Shannon divergence between two distributions represented by samples p and q

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
        The alternative to use, passed to SciPy's `permutation_test<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html>`_
    permutations : int, default=9999
         The number of permuatations to use when calculating the p-value
    permutation_rng : np.random.Generator or int, Optional
        A numpy random generator to use for sampling, or an int
        to seed the default generator.
    permutation_estimation_method : {"kernel", "empirical"}, default="kernel"
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

    Returns
    -------
    float
        The Jensen-Shannon divergence between p and q

    See Also
    --------

    Ross, B. C. (2014). Mutual Information between Discrete and Continuous Data Sets. PLoS ONE, 9(2), e87357.
    """
    return _wrap_divergence_functions(
        p=p,
        q=q,
        discrete_method=_js_disc,
        continuous_method=_js_cont,
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
    )


# endregion Main Function


# region Array Divergence


def js_divergence_array(
    p: ArrayInput,
    q: ArrayInput,
    axis: int = 1,
    processes: int = 1,
    **kwargs,
) -> Union[Array1D, Tuple[Array1D, Array1D]]:
    """Calculate the Jensen-Shannon divergence between arrays along the
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
        Keyword arguments are passed to the `js_divergence` function

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
        divergence_function=js_divergence,  # type: ignore
        axis=axis,
        processes=processes,
        **kwargs,
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
    """Calculate the Jensen-Shannon divergence between samples from two continuous distributions using the
    nearest neighbor method.

    Parameters
    ----------
    p : np.ndarray
        Array representing samples from the first continuous
        distribution, should have shape (n_samples, n_dimensions), where
        n_dimensions>=1
    q : np.ndarray
        Array representing samples from the second continuous
        distribution, should have shape (n_samples, n_dimensions), where
        n_dimensions>=1
    n_neighbors : int
        The number of neighbors to use for computing mutual information
    metric : float
        Metric to use for computing distance between points in y (must
        be `float>=1` representing Minkowski p-norm)
    **kwargs
        Arguments passed to KDTree constructor

    Returns
    -------
    float
        Jensen-Shannon divergence between p and q

    See Also
    --------

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
        full_tree.query_ball_point(
            combined, radius_array, p=metric, return_length=True
        )
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
    """Calculate the Jensen-Shannon divergence between samples from two discrete distributions

    Parameters
    ----------
    p : np.ndarray
        Array representing samples from the first discrete distribution,
        should have shape (n_samples, 1)
    q : np.ndarray
        Array representing samples from the second discrete
        distribution, should have shape (n_samples, 1)

    Returns
    -------
    float
        Jensen-Shannon divergence between p and q
    """
    p_elements, p_counts = np.unique(p, return_counts=True)
    q_elements, q_counts = np.unique(q, return_counts=True)

    p_total = p.shape[0]
    q_total = q.shape[0]

    comb_elements = np.union1d(p_elements, q_elements)

    p_freq = (
        np.array(
            [
                p_counts[p_elements == elem].item()
                if elem in p_elements
                else 0.0
                for elem in comb_elements
            ]
        )
        / p_total
    )

    q_freq = (
        np.array(
            [
                q_counts[q_elements == elem].item()
                if elem in q_elements
                else 0.0
                for elem in comb_elements
            ]
        )
        / q_total
    )

    return jensenshannon(p_freq, q_freq, base=np.e) ** 2


# endregion Discrete Case

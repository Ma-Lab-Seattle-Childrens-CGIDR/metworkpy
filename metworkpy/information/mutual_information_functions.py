"""
Module for finding mutual information between two sampled distributions using nearest neighbor methods. Includes
methods to compute mutual information between two continuous distributions, between two discrete distributions, and
between a continuous and discrete distribution.
"""

# Inspired by the implementation in scikit-learn, although the implementations differ so any errors are
# Braden Griebel's.

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Union

# External Imports
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree, distance_matrix
from scipy.special import digamma

# local imports
from metworkpy.utils._arguments import _parse_metric
from metworkpy.utils._jitter import _jitter


# region Main Mutual Information Function
def mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    discrete_x: bool = False,
    discrete_y: bool = False,
    n_neighbors: int = 5,
    jitter: Union[None, float] = None,
    jitter_seed: Union[None, int] = None,
    metric_x: Union[str, float] = "euclidean",
    metric_y: Union[str, float] = "euclidean",
    truncate: bool = False,
) -> float:
    """
    :param x: Array representing sample from a distribution, should have shape (n_samples, n_dimensions). If ``x`` is
        one dimensional, it will be reshaped to (n_samples, 1). If it is not a np.ndarray, this function will
        attempt to coerce it into one.
    :type x: ArrayLike
    :param y: Array representing sample from a distribution, should have shape (n_samples, n_dimensions). If ``y`` is
        one dimensional, it will be reshaped to  (n_samples, 1). If it is not a np.ndarray, this function will
        attempt to coerce it into one.
    :type y: ArrayLike
    :param discrete_x: Whether x is discrete or continuous
    :type discrete_x: bool
    :param discrete_y: Whether y is discrete or continuous
    :type discrete_y: bool
    :param n_neighbors: Number of neighbors to use for computing mutual information. Will attempt to coerce into
        an integer. Must be at least 1. Default 5.
    :type n_neighbors: int
    :param jitter: Amount of noise to add to avoid ties. If None no noise is added. If a float, that is the standard
        deviation of the random noise added to the continuous samples. If a tuple, the first element is the standard
        deviation of the noise added to the x array, the second element is the standard deviation added to the y array.
    :type jitter: Union[None, float, tuple[float,float]]
    :param jitter_seed: Seed for the random number generator used for adding noise
    :type jitter_seed: Union[None, int]
    :param metric_x: Metric to use for computing distance between points in x, can be "Euclidean",
        "Manhattan", or "Chebyshev". Can also be a float representing the Minkowski p-norm.
    :type metric_x: Union[str, int]
    :param metric_y: Metric to use for computing distance between points in y, can be "Euclidean",
        "Manhattan", or "Chebyshev". Can also be a float representing the Minkowski p-norm.
    :type metric_y: Union[str, int]
    :param truncate: Whether to ensure the mutual information is positive
    :type truncate: bool
    :return: The mutual information between x and y
    :rtype: float

    .. note::

       - The metrics can either be provided as a float greater than 1 representing the Minkowski p-norm, or a string
         representing the name of a metric such as 'Manhattan', 'Chebyshev', or 'Euclidean'.
       - For scalar samples (samples from a 1-D distribution), all the metrics are the same.
       - In the case of two continuous distributions, the distance in the z space (i.e. the joint (X,Y) space), is
         determined by the maximum norm (||z-z\\`|| = max{||x-x\\`||, ||y-y\\`||}), see [1] for more details.
       - Always returns value in nats (i.e. mutual information is calculated using the natural logarithm.

    .. seealso::

       1. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E, 69(6), 066138.
            Method for estimating mutual information between samples from two continuous distributions.
       2. Ross, B. C. (2014). Mutual Information between Discrete and Continuous Data Sets. PLoS ONE, 9(2), e87357.
           Method for estimating mutual information between a sample from a discrete distribution and a sample
           from a continuous distribution.

    """
    try:
        n_neighbors = int(n_neighbors)
    except ValueError as err:
        raise ValueError(
            f"n_neighbors must be able to be converted to an integer, but a {type(n_neighbors)} was"
            f"given instead."
        ) from err

    # Check that n_neighbors is greater than or equal to 1
    if not (n_neighbors >= 1):
        raise ValueError(
            f"n_neighbors must be at least 1, but argument was {n_neighbors}"
        )

    # Validate the x and y samples
    x, y = _validate_samples(x, y)

    # Check that if either x or y are discrete, they are either 1 dimensional or a column vector
    x = _check_discrete(sample=x, is_discrete=discrete_x)
    y = _check_discrete(sample=y, is_discrete=discrete_y)

    # Parse the metrics to floats
    metric_x, metric_y = _parse_metric(metric_x), _parse_metric(metric_y)
    if jitter:
        x, y = _jitter(
            x,
            y,
            jitter=jitter,
            jitter_seed=jitter_seed,
            discrete_x=discrete_x,
            discrete_y=discrete_y,
        )
    mi = None
    if discrete_x ^ discrete_y:  # if one of x or y is discrete
        if discrete_x:
            mi = _mi_disc_cont(
                continuous=y,
                discrete=x,
                n_neighbors=n_neighbors,
                metric_cont=metric_y,
            )
        if discrete_y:
            mi = _mi_disc_cont(
                continuous=x,
                discrete=y,
                n_neighbors=n_neighbors,
                metric_cont=metric_x,
            )
    elif not (discrete_x or discrete_y):  # if both are continuous
        mi = _mi_cont_cont(
            x=x,
            y=y,
            n_neighbors=n_neighbors,
            metric_x=metric_x,
            metric_y=metric_y,
        )
    elif discrete_x and discrete_y:
        mi = _mi_disc_disc(x=x, y=y)
    else:
        raise ValueError(
            "Error with discrete_x and/or discrete_y parameters, both must be boolean."
        )
    if truncate:
        return max(mi, 0)
    return mi


# endregion Main Mutual Information Function

# region Continuous-Continuous MI


def _mi_cont_cont(
    x: np.ndarray,
    y: np.ndarray,
    n_neighbors: int,
    metric_x: float,
    metric_y: float,
):
    """
    Calculate the mutual information between two continuous distributions using the nearest neighbor method.
    Dispatches to either _mi_cont_cont_cheb_only, or _mi_cont_cont_gen depending on which is applicable (
    cheb_only is more efficient, but only applicable when both metrics are np.inf or both samples are scalar).
    :param x: Array representing samples from a continuous distribution, shape (n_samples, n_dims_x)
    :type x: np.ndarray
    :param y: Array representing samples from a continuous distribution using the nearest neighbor method.
    :type y: np.ndarray
    :param n_neighbors: Number of neighbors to use for estimating the mutual information
    :type n_neighbors: int
    :param metric_x: Metric to use for computing distance between points in x (must be `float>=1` representing
        the Minkowski p-norm)
    :type metric_x: float
    :param metric_y: Metric to use for computing distance between points in y (must be `float>=1` representing
        the Minkowski p-norm)
    :type metric_y: float
    :return: The mutual information score between the two distributions represented by the x and y samples
    :rtype: float
    """
    if ((metric_x == np.inf) and (metric_y == np.inf)) or (
        x.shape[1] == 1 and y.shape[1] == 1
    ):
        return _mi_cont_cont_cheb_only(x=x, y=y, n_neighbors=n_neighbors)
    return _mi_cont_cont_gen(
        x=x, y=y, n_neighbors=n_neighbors, metric_x=metric_x, metric_y=metric_y
    )


def _mi_cont_cont_cheb_only(x: np.ndarray, y: np.ndarray, n_neighbors: int):
    """
    Calculate the mutual information between two continuous distributions using the nearest neighbor method.
    This version only allows for Chebyshev distances as the x and y metrics.
    :param x: Array representing samples from a continuous distribution, shape (n_samples, n_dims_x)
    :type x: np.ndarray
    :param y: Array representing samples from a continuous distribution using the nearest neighbor method.
    :type y: np.ndarray
    :param n_neighbors: Number of neighbors to use for estimating the mutual information
    :type n_neighbors: int
    :return: The mutual information score between the two distributions represented by the x and y samples
    :rtype: float
    """
    # Stack x and y to form the z space
    z = np.hstack((x, y))

    # Create the KDTrees needed for querying
    z_tree = KDTree(z)
    x_tree = KDTree(x)
    y_tree = KDTree(y)

    # Find the distances from z to n_neighbor point
    r, _ = z_tree.query(z, k=[n_neighbors + 1], p=np.inf)
    r = r.squeeze()

    # Find the number of neighbors within radius r
    r = np.nextafter(
        r, 0
    )  # Shrink r barely, to ensure it doesn't include the kth neighbor
    x_neighbors = x_tree.query_ball_point(x=x, r=r, p=np.inf, return_length=True) - 1
    y_neighbors = y_tree.query_ball_point(x=y, r=r, p=np.inf, return_length=True) - 1

    # Now use equation (8) from Kraskov, Stogbauer, and Grassberger 2004
    return (
        digamma(n_neighbors)
        - np.mean(digamma(x_neighbors + 1) + digamma(y_neighbors + 1))
        + digamma(z.shape[0])
    )


def _mi_cont_cont_gen(
    x: np.ndarray,
    y: np.ndarray,
    n_neighbors: int,
    metric_x: float,
    metric_y: float,
):
    """
    Calculate the mutual information between two continuous distributions using the nearest neighbor method.
    This method allows for any x and y metrics, but is much slower than the _mi_cont_cont_cheb_only.
    :param x: Array representing samples from a continuous distribution, shape (n_samples, n_dims_x)
    :type x: np.ndarray
    :param y: Array representing samples from a continuous distribution using the nearest neighbor method.
    :type y: np.ndarray
    :param n_neighbors: Number of neighbors to use for estimating the mutual information
    :type n_neighbors: int
    :param metric_x: Metric to use for computing distance between points in x (must be `float>=1` representing
        the Minkowski p-norm)
    :type metric_x: float
    :param metric_y: Metric to use for computing distance between points in y (must be `float>=1` representing
        the Minkowski p-norm)
    :type metric_y: float
    :return: The mutual information score between the two distributions represented by the x and y samples
    :rtype: float
    """
    x_dist = distance_matrix(x, x, p=metric_x)  # Distance in x space
    y_dist = distance_matrix(y, y, p=metric_y)  # Distance in y space
    z_dist = np.maximum(x_dist, y_dist)  # Equivalent to p=np.inf Minkosky p-norm
    # For finding the kth neighbor, things to note:
    # - We are using the numpy partition function, which sorts the kth element into the sorted position
    #   with kth=0 sorting the first element, kth=1 sorting the second, etc.
    # - We want to find the n_neighbor, the first sorted element will be the point itself
    # - This together means that the n_neighbor is the correct index to partition on,
    #   since the +1 of excluding the point itself (always 0) and the -1 of the index cancel.
    # TO SAVE ON SPACE THIS IS DONE IN PLACE, the z_dist matrix has an arbitrary ordering across axis=1
    # neighbor_distances takes the n_neighbors column, which corresponds to the distance to the nth neighbor
    z_dist.partition(kth=n_neighbors, axis=1)
    neighbor_distances = z_dist[:, n_neighbors].reshape(-1, 1)
    # Now, the number of points in x and y which are within neighbor_distances (or epsilon(i)/2)
    x_neighbors = (x_dist < neighbor_distances).sum(axis=1) - 1
    y_neighbors = (y_dist < neighbor_distances).sum(axis=1) - 1
    # Now use equation (8) from Kraskov, Stogbauer, and Grassberger 2004
    return (
        digamma(n_neighbors)
        - np.mean(digamma(x_neighbors + 1) + digamma(y_neighbors + 1))
        + digamma(x.shape[0])
    )


# endregion Continuous-Continuous MI

# region Continuous-Discrete MI


def _mi_disc_cont(
    discrete: np.ndarray,
    continuous: np.ndarray,
    n_neighbors: int,
    metric_cont: float = 2.0,
    **kwargs,
) -> float:
    """
    Calculate the mutual information between a discrete and continuous distribution using the nearest neighbor method.
    :param discrete: Array representing the samples from the discrete distribution, should have shape (n_samples, 1)
    :type discrete: np.ndarray
    :param continuous: Array representing the continuous distribution, should have shape (n_samples, n_dimensions), where
        n_dimensions>=1
    :type continuous: np.ndarray
    :param n_neighbors: The number of neighbors to use for computing mutual information
    :type n_neighbors: int
    :param metric_cont: Metric to use for computing distance between points in y (must be `float>=1` representing
        Minkowski p-norm)
    :type metric_cont: float
    :param kwargs: Arguments passed to KDTree
    :return: Mutual information between x and y
    :rtype: float

    .. note::
       Mutual information should always be positive, but this method can produce a negative value since it is an
       estimate. Also, in order for this method to work each class must have at least n_neighbors+1 data points.

    .. seealso::
       `Ross, B. C. (2014). Mutual Information between Discrete and Continuous Data Sets. PLoS ONE, 9(2), e87357.
       `<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357>_
            Paper from which this method was obtained
       :func: `_mi_disc_cont_scalar`
            Function for calculating the mutual information when the continuous distribution is scalar
    """
    discrete_classes, counts = np.unique(discrete, return_counts=True)
    if (counts < n_neighbors + 1).any():
        raise ValueError(
            f"Some classes contain insufficient points to have n_neighbors neighbors: "
            f"{discrete_classes[counts < (n_neighbors + 1)]} contain {counts[counts < (n_neighbors + 1)]}."
            f" You can try using a smaller number of neighbors, such as "
            f"{counts[counts < (n_neighbors + 1)].min() - 1}, or filter out the classes with insufficient "
            f"numbers of points."
        )

    n_data_points = discrete.shape[0]

    # Create the KDTree which includes every point
    full_tree = KDTree(continuous, **kwargs)

    radius_array = np.empty(n_data_points, dtype=float)
    count_array = np.empty(n_data_points, dtype=float)
    for d_class, count in zip(discrete_classes, counts):
        same_class_index = (discrete == d_class).squeeze()
        count_array[same_class_index] = count
        type_tree = KDTree(continuous[same_class_index, :], **kwargs)
        # Get the neighbors (the 1st neighbor will just be the point itself)
        dist, _ = type_tree.query(
            continuous[same_class_index, :], k=[n_neighbors + 1], p=metric_cont
        )
        dist = dist.squeeze()
        # Add the values to the radius array
        # Increase the distance slightly to make sure that the kth neighbor will be included
        radius_array[same_class_index] = np.nextafter(dist, np.inf)

    # Find the number of neighbors within the radius, subtract 1 since it will include the point itself
    neighbors_within_radius = (
        full_tree.query_ball_point(
            continuous, radius_array, p=metric_cont, return_length=True
        )
        - 1
    )
    return (
        digamma(n_data_points)
        - digamma(count_array).mean()
        + digamma(n_neighbors)
        - digamma(neighbors_within_radius).mean()
    )  # See equation 2 of Ross, 2014


# endregion Continuous-Discrete MI

# region Discrete-Discrete MI


def _mi_disc_disc(x: np.ndarray, y: np.ndarray):
    """
    Calculate the mutual information between samples from two discrete distributions
    :param x: Array representing the samples from a discrete distribution, should have shape (n_samples, 1)
    :type x: np.ndarray
    :param y: Array representing the samples from a discrete distribution, should have shape (n_samples, 1)
    :type y: np.ndarray
    :return: The mutual information between x and y
    :rtype: float
    """
    z = np.hstack((x, y))
    # Get the unique elements and the counts
    x_element, x_count = np.unique(x, return_counts=True)
    y_element, y_count = np.unique(y, return_counts=True)
    z_element, z_count = np.unique(z, axis=0, return_counts=True)
    # Find the paired, and marginal frequencies
    x_freq = x_count / x_count.sum()
    y_freq = y_count / y_count.sum()
    z_freq = z_count / z_count.sum()
    # now calculate the MI
    mi = 0.0  # Start at 0, and accumulate it using a sum formula
    for y_i, y_f in zip(y_element, y_freq):
        for x_i, x_f in zip(x_element, x_freq):
            # Find the joint frequency
            joint = z_freq[(z_element[:, 0] == x_i) & (z_element[:, 1] == y_i)]
            if not joint.size > 0:
                continue
            mi += (
                joint * np.log(joint / (x_f * y_f))
            ).item()  # NOTE: Log is base e (i.e. natural)
    return mi


# endregion Discrete-Discrete MI


# region Helper Functions
# noinspection PyProtectedMember
def _validate_sample(sample: ArrayLike) -> np.ndarray:
    # Coerce to np array
    if not isinstance(sample, np.ndarray):
        sample = np.array(sample)
    if len(sample.shape) > 2:
        raise ValueError("Sample must have a maximum of 2 axes")
    # If 1D, change to (n_samples,1)
    if len(sample.shape) == 1:
        sample = sample.reshape(-1, 1)
    return sample


def _validate_samples(x: ArrayLike, y: ArrayLike):
    try:
        x = _validate_sample(x)
        y = _validate_sample(y)
    except ValueError as err:
        raise ValueError(
            f"Both samples arrays must have a maximum of 2 axes, but x has {len(x.shape)}"
            f"axes and y has {len(x.shape)} axes"
        ) from err
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"The first dimension of x and y should match, but x has first dimension {x.shape[0]}, "
            f"and y has a first dimension of {y.shape[0]}"
        )
    return x, y


def _check_discrete(sample, is_discrete):
    if not isinstance(is_discrete, bool):
        raise ValueError("discrete_* arguments must be boolean")
    if is_discrete:
        if (
            len(sample.shape) == 1
        ):  # if x is 1 dimensional, reshape it into a column vector
            sample = sample.reshape(-1, 1)
        if sample.shape[1] != 1:
            raise ValueError(
                "For samples from discrete distributions, currently only a single dimension is "
                "supported. You can try with x as a sample from a continuous distribution, or encode the "
                "multiple dimensions into one."
            )
    return sample


# endregion Helper Functions

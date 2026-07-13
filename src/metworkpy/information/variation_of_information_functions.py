"""
Functions to compute the variation of information
between two discrete distributions
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from .mutual_information_functions import _validate_samples, _check_discrete


def variation_of_information(
    x: ArrayLike,
    y: ArrayLike,
) -> float:
    """
    Calculate the Variation of Information between two samples from
    two discrete distributions

    Parameters
    ----------
    x : ArrayLike
        Array representing sample from a distribution, should have shape
        (n_samples, n_dimensions). If ``x`` is one dimensional, it will
        be reshaped to (n_samples, 1). If it is not a np.ndarray, this
        function will attempt to coerce it into one.
    y : ArrayLike
        Array representing sample from a distribution, should have shape
        (n_samples, n_dimensions). If ``y`` is one dimensional, it will
        be reshaped to  (n_samples, 1). If it is not a np.ndarray, this
        function will attempt to coerce it into one.

    Returns
    -------
    float
        The variation of information between x and y
    """
    # Validate the x and y samples
    x, y = _validate_samples(x, y)

    # Check that if either x or y are discrete, they are either 1 dimensional or a column vector
    x = _check_discrete(sample=x, is_discrete=True)
    y = _check_discrete(sample=y, is_discrete=True)
    return _voi(x, y)


def _voi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Variation of Information (VOI) between samples from
    two discrete distributions

    Parameters
    ----------
    x : np.ndarray
        Array representing the samples from a discrete distribution,
        should have shape (n_samples, 1)
    y : np.ndarray
        Array representing the samples from a discrete distribution,
        should have shape (n_samples, 1)
    Returns
    -------
    float
        The Variation of Information between x and y
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
    # now calculate the MI, one of the terms in the VOI
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

    # Now calculate the marginal entropies
    x_entropy = stats.entropy(x_freq)
    y_entropy = stats.entropy(y_freq)
    # Return the VOI
    # VOI(X;Y) = H(X) + H(Y) - 2I(X,Y)
    return x_entropy + y_entropy - 2 * mi

"""
Functions for performing permutation tests
"""

# imports
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
from scipy import stats


# TODO: Generalize to n datasets
def permutation_test(
    dataset1: np.ndarray,
    dataset2: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    axis: int = 0,
    permutation_type: Literal["independent", "pairings"] = "independent",
    n_resamples=500,
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    estimation_method: Literal["kernel", "empirical"] = "empirical",
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> Tuple[float, float]:
    """
    Perform a permutation test for a sample statistic

    Parameters
    ----------
    dataset1,dataset2 : np.ndarray
        The two datasets to perform the permutation testing on, must
        have broadcastable shapes except along axis
    statistic : Callable
        Function which takes two numpy arrays (which have the same shape
        except along axis), and returns a float
    axis : int, default=0
        The sample axis for the two datsets
    permutation_type : {'independent', 'pairings'}, default='independent'
        The type of permutation to perform,
        - pairings: Shuffles which observations are paired,
          but the assignment of observation to sample isn't changed
        - independent: Shuffles which samples observations are assigned to
    n_resamples : int, default=500
        The number of permutations to perform
    alternative : {"less", "greater", "two-sided"}, default='two-sided'
        Alternative hypothesis
    estimation_method : {"kernel", "empirical"}, default="empirical"
        Method to use for estimating p-value, either an empirical estimate,
        or a gaussian_kde. The empirical method returns an upper bound on the
        p-value that is somewhat conservative, and is based on [1]_
        and the implementation in SciPy.
    rng : np.random.Generator or int, Optional
        A numpy random generator to use for sampling, or an int
        to seed the default generator.

    Returns
    -------
    tuple of float,float
        Tuple of the sample statistic and the calculated p-value

    Notes
    -----

    .. [1] Phipson, B., & Smyth, G. K. (2010). Permutation p-values
       should never be zero: Calculating exact p-values when
       permutations are randomly drawn. Statistical Applications
       in Genetics and Molecular Biology, 9(1).
       https://doi.org/10.2202/1544-6115.1585
    """
    # Calculate the sample statistic
    sample_stat = statistic(dataset1, dataset2)
    # Get a numpy RNG
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    # Array to hold the null distribution
    null_distribution = np.empty((n_resamples,))
    # Find the null distribution
    if permutation_type == "independent":
        combined = np.concatenate([dataset1, dataset2], axis=axis)
        # Stack the arrays for sampling
        for idx in range(n_resamples):
            # Shuffle combined
            rng.shuffle(combined, axis=axis)
            # Split into x and y to randomly assign to samples
            x = combined.take(range(dataset1.shape[axis]), axis=axis)
            y = combined.take(
                range(dataset1.shape[axis], combined.shape[axis]), axis=axis
            )
            null_distribution[idx] = statistic(x, y)
    elif permutation_type == "pairings":
        # To avoid modifying the input arrays, copy then
        x = dataset1
        y = dataset2.copy()
        for idx in range(n_resamples):
            # Shuffle pairings
            rng.shuffle(y, axis=axis)
            null_distribution[idx] = statistic(x, y)
    else:
        raise ValueError(
            f"permutation_type must be either one of 'independent', or 'pairings' but received {permutation_type}"
        )
    if estimation_method == "kernel":
        distribution = stats.gaussian_kde(null_distribution)
        prob_less: float = distribution.integrate_box_1d(
            -np.inf,
            sample_stat,
        )
        prob_greater: float = distribution.integrate_box_1d(
            sample_stat, np.inf
        )
    elif estimation_method == "empirical":
        # Apply an adjustment based on 'Permutation p-values should never be zero:
        # calculating exact p-values when permutations are randomly drawn'
        # First find eps, which is floating point tolerance
        # Also based on Scipy's permutation test implementation
        eps = (
            0
            if not np.isdtype(null_distribution.dtype, "real floating")
            else np.finfo(null_distribution.dtype).eps * 100
        )
        gamma = np.abs(eps * sample_stat)
        prob_less = float(
            np.count_nonzero(null_distribution <= sample_stat + gamma) + 1
        ) / float(n_resamples + 1)
        prob_greater = float(
            np.count_nonzero(null_distribution >= sample_stat - gamma) + 1
        ) / float(n_resamples + 1)
    else:
        raise ValueError(
            f"estimation_method must be 'kernel' or 'empirical' but received {estimation_method}"
        )
    if alternative == "less":
        return sample_stat, prob_less
    elif alternative == "greater":
        return sample_stat, prob_greater
    elif alternative == "two-sided":
        return sample_stat, 2 * min(prob_less, prob_greater)
    else:
        raise ValueError(
            f"alternative must be either 'less', 'greater', or 'two-sided', but received {alternative}"
        )

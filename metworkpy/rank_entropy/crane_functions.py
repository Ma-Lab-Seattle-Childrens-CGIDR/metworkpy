"""
Functions for computing Centroid Rank Entropy (CRANE)
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Optional, Literal, Callable, Union, Tuple

# External Imports
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import rankdata, gaussian_kde

# Local imports
from metworkpy.rank_entropy._bootstrap_pvalue import _bootstrap_rank_entropy_p_value


# region Main Fuctions


def crane_gene_set_classification(
    expression_data: NDArray[float | int] | pd.DataFrame,
    sample_group1,
    sample_group2,
    gene_network,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str | float | Callable[[gaussian_kde], float]]] = None,
    iterations: int = 10_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> Tuple[float, float]:
    """
    Calculate the classification rate using CRANE rank centroid distances for a given network and its significance

    :param expression_data: Gene expression data, either a numpy array or a pandas DataFrame, with rows representing
        different samples, and columns representing different genes
    :type expression_data: np.ndarray | pd.DataFrame
    :param sample_group1: Which samples belong to group1. If expression_data is a numpy array, this should be
        a something able to index the rows of the array. If expression_data is a pandas dataframe, this should be
        something that can index rows of a dataframe inside a .loc (see pandas documentation for details)
    :param sample_group2: Which samples belong to group2, see sample_group1 information for more details.
    :param gene_network: Which genes belong to the gene network. If expression_data is a numpy array, this
        should be something able to index the columns of the array. If expression_data is a pandas dataframe, this
        should be something be anything that can index columns of a dataframe inside a .loc (see pandas documentation
        for details)
    :param kernel_density_estimate: Whether to use a kernel density estimate for calculating the p-value. If True,
        will use a Gaussian Kernel Density Estimate, if False will use an empirical CDF
    :type kernel_density_estimate: bool
    :param bw_method: Bandwidth method, see [scipy.stats.gaussian_kde](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) for details
    :type bw_method: Optional[Union[str|float|Callable[[gaussian_kde], float]]]
    :param iterations: Number of iterations to perform during bootstrapping the null distribution
    :type iterations: int
    :param replace: Whether to sample with replacement when randomly sampling from the sample groups
        during bootstrapping
    :type replace: bool
    :param seed: Seed to use for the random number generation during bootstrapping
    :type seed: int
    :param processes: Number of processes to use during the bootstrapping, default 1
    :type processes: int
    :return: Tuple of the classification rate, and the significance level found via bootstrapping
    :rtype: Tuple[float,float]
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_group1=sample_group1,
        sample_group2=sample_group2,
        gene_network=gene_network,
        rank_entropy_fun=_crane_classification_rate,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


def crane_gene_set_entropy(
    expression_data: NDArray[float | int] | pd.DataFrame,
    sample_group1,
    sample_group2,
    gene_network,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str | float | Callable[[gaussian_kde], float]]] = None,
    iterations: int = 1_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> Tuple[float, float]:
    """
    Calculate the difference in centroid rank entropy, and it's significance

    :param expression_data: Gene expression data, either a numpy array or a pandas DataFrame, with rows representing
        different samples, and columns representing different genes
    :type expression_data: np.ndarray | pd.DataFrame
    :param sample_group1: Which samples belong to group1. If expression_data is a numpy array, this should be
        a something able to index the rows of the array. If expression_data is a pandas dataframe, this should be
        something that can index rows of a dataframe inside a .loc (see pandas documentation for details)
    :param sample_group2: Which samples belong to group2, see sample_group1 information for more details.
    :param gene_network: Which genes belong to the gene network. If expression_data is a numpy array, this
        should be something able to index the columns of the array. If expression_data is a pandas dataframe, this
        should be something be anything that can index columns of a dataframe inside a .loc (see pandas documentation
        for details)
    :param kernel_density_estimate: Whether to use a kernel density estimate for calculating the p-value. If True,
        will use a Gaussian Kernel Density Estimate, if False will use an empirical CDF
    :type kernel_density_estimate: bool
    :param bw_method: Bandwidth method, see [scipy.stats.gaussian_kde](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) for details
    :type bw_method: Optional[Union[str|float|Callable[[gaussian_kde], float]]]
    :param iterations: Number of iterations to perform during bootstrapping the null distribution
    :type iterations: int
    :param replace: Whether to sample with replacement when randomly sampling from the sample groups
        during bootstrapping
    :type replace: bool
    :param seed: Seed to use for the random number generation during bootstrapping
    :type seed: int
    :param processes: Number of processes to use during the bootstrapping, default 1
    :type processes: int
    :return: Tuple of the difference in centroid rank entropy, and the significance level found via bootstrapping
    :rtype: Tuple[float,float]
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_group1=sample_group1,
        sample_group2=sample_group2,
        gene_network=gene_network,
        rank_entropy_fun=_crane_differential_entropy,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


# endregion Main Functions

# region Rank Centroid Functions


def _rank_array(
    in_array: NDArray[int | float],
    method: Literal[
        "average",
        "min",
        "max",
        "dense",
        "ordinal",
    ] = "average",
) -> NDArray[float]:
    return rankdata(in_array, method=method, axis=1, nan_policy="omit")


def _rank_centroid(in_array: [int | float]) -> NDArray[int]:
    return _rank_array(in_array=in_array).mean(axis=0)


def _rank_grouping_score(in_array: [int | float]) -> NDArray[int]:
    ranked_array = _rank_array(in_array)
    centroid = ranked_array.mean(axis=0)
    return np.sqrt(np.square(np.subtract(ranked_array, centroid)).sum(axis=1)).mean()


def _crane_differential_entropy(
    a: NDArray[int | float],
    b: NDArray[int | float],
) -> float:
    return np.abs(_rank_grouping_score(a) - _rank_grouping_score(b))


# endregion Rank Centroid Functions

# region Classification rate functions


def _crane_classification_rate(
    a: NDArray[float | int], b: NDArray[float | int]
) -> float:
    # Compute the rank arrays
    rank_array_a = _rank_array(a)
    rank_array_b = _rank_array(b)

    # Compute the rank centroids
    centroid_a = rank_array_a.mean(axis=0).reshape(1, -1)
    centroid_b = rank_array_b.mean(axis=0).reshape(1, -1)

    # Calculate distances from the rank arrays to the centroids
    centroid_distance_a_array_a = np.sqrt(
        np.square(np.subtract(rank_array_a, centroid_a)).sum(axis=1)
    )
    centroid_distance_b_array_a = np.sqrt(
        np.square(np.subtract(rank_array_a, centroid_b)).sum(axis=1)
    )

    centroid_distance_a_array_b = np.sqrt(
        np.square(np.subtract(rank_array_b, centroid_a)).sum(axis=1)
    )
    centroid_distance_b_array_b = np.sqrt(
        np.square(np.subtract(rank_array_b, centroid_b)).sum(axis=1)
    )

    # Calculate the rank centroid distance difference
    dist_diff_a = centroid_distance_a_array_a - centroid_distance_b_array_a
    dist_diff_b = centroid_distance_a_array_b - centroid_distance_b_array_b

    # Calculate the accuracy
    total_samples = a.shape[0] + b.shape[0]
    correct_samples = (dist_diff_a > 0.0).sum() + (dist_diff_b <= 0.0).sum()

    return correct_samples / total_samples

    pass


# endregion Classification rate functions

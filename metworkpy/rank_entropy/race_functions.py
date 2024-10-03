"""
Functions for computing the Rank Correlation Entropy (RACE)
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from itertools import combinations
from typing import Optional, Union, Callable, Tuple

# Enternal Imports
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import gaussian_kde, kendalltau

# Local Imports
from metworkpy.rank_entropy._bootstrap_pvalue import _bootstrap_rank_entropy_p_value

# region Main Functions


def race_gene_set_entropy(
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
    Calculate the difference in rank correlation entropy, and it's significance

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
    :param bw_method: Bandwidth method, see `scipy.stats.gaussian_kde <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`_ for details
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
    :return: Tuple of the difference in rank correlation entropy, and the significance level found via bootstrapping
    :rtype: Tuple[float,float]
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_group1=sample_group1,
        sample_group2=sample_group2,
        gene_network=gene_network,
        rank_entropy_fun=_race_differential_entropy,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


# endregion Main Functions


# region Rank Correlation Functions
def _rank_correlation_mean(input_array: NDArray[int | float]) -> float:
    sum = 0.0
    count = 0
    for a, b in combinations(range(input_array.shape[0]), 2):
        sum += ((kendalltau(input_array[a], input_array[b]).statistic * -1) + 1) / 2
        count += 1
    return sum / count


def _race_differential_entropy(
    a: NDArray[int | float], b: NDArray[int | float]
) -> float:
    return np.abs(_rank_correlation_mean(a) - _rank_correlation_mean(b))


# endregion Rank Correlation Functions

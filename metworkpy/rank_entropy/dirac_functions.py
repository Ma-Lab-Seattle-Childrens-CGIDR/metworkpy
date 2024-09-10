"""
Functions for computing differential rank conservation (DIRAC)
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Union, Optional, Callable, Tuple

# External Imports
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Local Imports
from metworkpy.rank_entropy._bootstrap_pvalue import _bootstrap_rank_entropy_p_value


# region Main Functions


def dirac_gene_set_entropy(
    expression_data: np.ndarray[float | int] | pd.DataFrame,
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
    Calculate the difference in rank conservation indices, and it's significance

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
    :return: Tuple of the difference in rank conservation index, and the significance level found via bootstrapping
    :rtype: Tuple[float,float]
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_group1=sample_group1,
        sample_group2=sample_group2,
        gene_network=gene_network,
        rank_entropy_fun=_dirac_differential_entropy,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


# endregion Main Functions

# region Rank Vector


def _rank_vector(in_vector: np.ndarray[int | float]) -> np.ndarray[int]:
    rank_array = np.repeat(in_vector.reshape(1, -1), len(in_vector), axis=0)
    diff_array = rank_array - rank_array.T
    return (diff_array[np.triu_indices(len(in_vector), k=1)] > 0).astype(int)


def _rank_array(in_array: np.ndarray[int | float]) -> np.ndarray[int]:
    return np.apply_along_axis(_rank_vector, axis=1, arr=in_array)


def _rank_matching_score(in_array: np.ndarray[int]) -> np.ndarray[float]:
    rank_array = _rank_array(in_array)
    rank_template_array = np.repeat(
        (rank_array.mean(axis=0) > 0.5).astype(int).reshape(1, -1), rank_array.shape[0]
    )
    return (rank_array == rank_template_array).mean(axis=1)


def _rank_conservation_index(in_array: np.ndarray[int]) -> float:
    return _rank_matching_score(in_array).mean()


def _dirac_differential_entropy(
    a: np.ndarray[float | int], b: np.ndarray[float | int]
) -> float:
    return np.abs(_rank_conservation_index(a) - _rank_conservation_index(b))


# endregion Rank Vector

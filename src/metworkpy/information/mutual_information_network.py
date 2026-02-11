"""
Functions for computing the Mutual Information Network for a Metabolic Model"""

# Standard Library Imports
from __future__ import annotations

import itertools
from typing import cast, Literal, Optional, Tuple, TypeVar, Union

# External Imports
import joblib  # type: ignore
import numpy as np
import pandas as pd
import scipy  # type: ignore
import tqdm  # type: ignore
from numpy.typing import ArrayLike

# Local Imports
from .mutual_information_functions import mutual_information


# region Main Function
T = TypeVar("T", np.ndarray, pd.DataFrame, ArrayLike)


def mi_network_adjacency_matrix(
    samples: T,
    cutoff: Optional[float] = None,
    cutoff_quantile: Optional[float] = None,
    processes: int = -1,
    progress_bar: bool = False,
    **kwargs,
) -> T:
    """
    Create a Mutual Information Network Adjacency matrix from flux samples.
    Uses kth nearest neighbor method for estimating mutual information.

    Parameters
    ----------
    samples : ArrayLike or DataFrame or NDArray
        ArrayLike containing the samples, columns
        should represent different reactions while rows should represent
        different samples
    cutoff : float, optional
        Lower bound for mutual information, all values smaller than this are
        set to 0
    cutoff_quantile : float, Optional
        Lower bound for mutual information as a quantile, must be a value
        between 0 and 1 representing the quantile to use as a cutoff.
        Any values below this quantile will be set to 0.
    processes : int
        Number of processes to use when calculating the mutual information
    progress_bar : bool
        Whether a progress bar should be displayed
    kwargs
        Keyword arguments passed to the mutual_information function

    Returns
    -------
    mutual_information_ : ArrayLike or DataFrame or NDArray
        The mutual information adjacency matrix, will share a type with the ArrayLike passed in, and
        be a square symmetrical array, with the value at the ith row, jth column representing the mutual
        information between the ith and jth columns of the input samples dataset

    See Also
    --------

    1. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E, 69(6), 066138.
         Method for estimating mutual information between samples from two continuous distributions.
    """
    # Wraps mi_pariwise, here for backward compatibility
    return mi_pairwise(
        dataset=samples,
        cutoff=cutoff,
        cutoff_quantile=cutoff_quantile,
        processes=processes,
        progress_bar=progress_bar,
        **kwargs,
    )


# endregion Main Function


# region Pairwise Mutual Information
def mi_pairwise(
    dataset: T,
    calculate_pvalue: bool = False,
    alternative: Literal["less", "greater", "two-sided"] = "greater",
    permutations: int = 9999,
    cutoff: Optional[float] = None,
    cutoff_quantile: Optional[float] = None,
    cutoff_significance: Optional[float] = None,
    processes: int = -1,
    progress_bar: bool = False,
    **kwargs,
) -> Union[T, Tuple[T, T]]:
    """
    Calculate all pairwise values of mutual information for columns in dataset

    Parameters
    ----------
    dataset : ArrayLike or DataFrame or NDArray
        The dataset to calculate pairwise mutual information values for,
        should be a 2-dimensional array or Dataframe
    calculate_pvalue : bool
         Whether to calculate a p-value for the mutual information using
         a permutation test
    alternative : 'less', 'greater', or 'two-sided'
         The alternative to use, passed to SciPy's `permutation_test<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html>`_
    permutations : int
         The number of permuatations to use when calculating the p-value
    cutoff : float, optional
        Lower bound for mutual information, all values smaller than this are
        set to 0
    cutoff_quantile : float, optional
        Lower bound for mutual information as a quantile, must be a value
        between 0 and 1 representing the quantile to use as a cutoff.
        Any values below this quantile will be set to 0.
    cutoff_significance : float, optional
        Upper bound for the significance of the mutual information,
        any mutual information values with p-values above this
        cutoff will have their mutual information set to 0.
        Requires that calculate_pvalue is True.
    processes : int, default=-1
        The number of processes to use for calculating the pairwise mutual information
    progress_bar : bool, default=False
        Whether a progress bar is desired
    kwargs
        Keyword arguments passed into the mutual_information function

    Returns
    -------
    DataFrame or NDArray or Tuple of DataFrame or NDArray
        The mutual information between every pair of columns in dataset. If dataset
        is a numpy NDArray or an Arraylike, will return a numpy NDArray. If dataset
        is a pandas DataFrame, will return a pandas DataFrame. If calculate_pvalue
        is True, will instead return a tuple of the appropriate array type,
        with the first element being the mutual information array, and the second
        being the p-values

    Notes
    -----
    The parallelization uses joblib, and so can be configured with joblib's parallel_config context manager
    """
    # Check that if cutoff_significance is not None, calculate_pvalue is True
    if cutoff_significance is not None and calculate_pvalue:
        raise ValueError(
            "If cutoff_significance is not None, calculate_pvalue must be True"
        )
    # Add the keywords related to the p-value to the kwargs dict
    kwargs["calculate_pvalue"] = calculate_pvalue
    kwargs["alternative"] = alternative
    kwargs["permutations"] = permutations
    if isinstance(dataset, pd.DataFrame):
        mi_result = pd.DataFrame(
            0.0, index=dataset.columns, columns=dataset.columns
        )
        if calculate_pvalue:
            pvalue_result = pd.DataFrame(
                1.0, index=dataset.columns, columns=dataset.columns
            )
        num_combinations = scipy.special.comb(dataset.shape[1], 2)
        for idx1, idx2, ret_value in tqdm.tqdm(
            joblib.Parallel(n_jobs=processes, return_as="generator")(
                joblib.delayed(_mi_single_pair)(
                    dataset[i], dataset[j], i, j, **kwargs
                )
                for i, j in itertools.combinations(dataset.columns, 2)
            ),
            disable=not progress_bar,
            total=num_combinations,
        ):
            if not calculate_pvalue:
                mi = ret_value
                mi_result.loc[idx1, idx2] = mi
                mi_result.loc[idx2, idx1] = mi
            else:
                mi, pvalue = ret_value
                mi_result.loc[idx1, idx2] = mi
                mi_result.loc[idx2, idx1] = mi
                pvalue_result.loc[idx1, idx2] = pvalue
                pvalue_result.loc[idx2, idx1] = pvalue
        # Apply the significance cutoff if it exists
        if cutoff_significance is not None:
            mi_result.loc[pvalue_result > cutoff_significance] = 0.0
        # Apply the cutoff if it exists
        if cutoff_quantile is not None:
            cutoff = np.quantile(
                mi_result,
                cutoff_quantile,
                overwrite_input=False,
            )
        if cutoff is not None:
            mi_result.loc[pvalue_result < cutoff] = 0.0
    else:
        dataset = np.array(dataset)  # Coerce arraylike into array
        mi_result = np.zeros((dataset.shape[1], dataset.shape[1]))
        if calculate_pvalue:
            pvalue_result = np.ones((dataset.shape[1], dataset.shape[1]))
        num_combinations = scipy.special.comb(dataset.shape[1], 2)
        for idx1, idx2, ret_value in tqdm.tqdm(
            joblib.Parallel(n_jobs=processes, return_as="generator")(
                joblib.delayed(_mi_single_pair)(
                    dataset[:, i], dataset[:, j], i, j, **kwargs
                )
                for i, j in itertools.combinations(range(dataset.shape[1]), 2)
            ),
            disable=not progress_bar,
            total=num_combinations,
        ):
            if not calculate_pvalue:
                mi = ret_value
                mi_result[idx1, idx2] = mi
                mi_result[idx2, idx1] = mi
            else:
                mi, pvalue = ret_value
                mi_result[idx1, idx2] = mi
                mi_result[idx2, idx1] = mi
                pvalue_result[idx1, idx2] = pvalue
                pvalue_result[idx2, idx1] = pvalue
        # Apply the significance cutoff if it exists
        if cutoff_significance is not None:
            mi_result[pvalue_result > cutoff_significance] = 0.0
        # Apply cutoff if necessary
        if cutoff_quantile is not None:
            cutoff = np.quantile(
                mi_result,
                cutoff_quantile,
                overwrite_input=False,
            )
        if cutoff is not None:
            mi_result[mi_result < cutoff] = 0.0
    mi_result = cast(T, mi_result)
    if not calculate_pvalue:
        return mi_result
    return mi_result, pvalue_result


U = TypeVar("U", int, str)
V = TypeVar("V", int, str)


def _mi_single_pair(
    item1: ArrayLike, item2: ArrayLike, idx1: U, idx2: V, **kwargs
) -> Tuple[U, V, Union[float, Tuple[float, float]]]:
    """
    Calculate the mutual information for a single pair of features

    Parameters
    ----------
    item1, item2 : ArrayLike
        The pair to calculate the mutual information for, must be coercable into 1-D arrays
    idx1, idx2 : int or str
        The index of pair for which the mutual information is being calculated (just passed through
        but simplified the mi_pairwise function)
    kwargs
        Keyword arguments passed through to the mutual_information function

    Returns
    -------
    A tuple of the first index, the second index, and the mutual information between
    the two items
    """
    return idx1, idx2, mutual_information(item1, item2, **kwargs)


# endregion Pairwise Mutual Information

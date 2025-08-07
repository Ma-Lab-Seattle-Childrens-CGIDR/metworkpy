"""Functions for computing the Mutual Information Network for a Metabolic Model"""

# Standard Library Imports
from __future__ import annotations

import itertools
from typing import Tuple, TypeVar

# External Imports
import joblib
import numpy as np
import pandas as pd
import scipy
import tqdm
from numpy.typing import ArrayLike
from .mutual_information_functions import mutual_information

# Local Imports


# region Main Function
T = TypeVar("T", np.ndarray, pd.DataFrame, ArrayLike)


def mi_network_adjacency_matrix(
    samples: T, processes: int = -1, progress_bar: bool = False, **kwargs
) -> T:
    """Create a Mutual Information Network Adjacency matrix from flux samples. Uses kth nearest neighbor method
    for estimating mutual information.

    Parameters
    ----------
    samples : ArrayLike
        ArrayLike containing the samples, columns
        should represent different reactions while rows should represent
        different samples
    processes : int
        Number of processes to use when calculating the mutual information
    progress_bar : bool
        Whether a progress bar should be displayed
    kwargs
        Keyword arguments passed to the mutual_information function

    Returns
    -------
    mutual_information_ : ArrayLike
        The mutual information adjacency matrix, will share a type with the ArrayLike passed in, and
        be a square symmetrical array, with the value at the ith row, jth column representing the mutual
        information between the ith and jth columns of the input samples dataset

    See Also
    --------

    1. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E, 69(6), 066138.
         Method for estimating mutual information between samples from two continuous distributions.
    """
    # Wraps mi_pariwise, here for backward compatibility
    mi_adj_mat = mi_pairwise(
        dataset=samples, processes=processes, progress_bar=progress_bar, **kwargs
    )
    if isinstance(mi_adj_mat, pd.DataFrame):
        mi_adj_mat = mi_adj_mat.to_numpy()
    else:
        mi_adj_mat = np.array(mi_adj_mat)
    return mi_adj_mat


# endregion Main Function


# region Pairwise Mutual Information
def mi_pairwise(
    dataset: T, processes: int = -1, progress_bar: bool = False, **kwargs
) -> T:
    """
    Calculate all pairwise values of mutual information for columns in dataset

    Parameters
    ----------
    dataset : ArrayLike
        The dataset to calculate pairwise mutual information values for,
        should be a 2-dimensional array or dataframe
    processes : int, default=-1
        The number of processes to use for calculating the pairwise mutual information
    progress_bar : bool, default=False
        Whether a progress bar is desired
    kwargs
        Keyword arguments passed into the mutual_information function

    Notes
    -----
    The parallelization uses joblib, and so can be configured with joblib's parallel_config context manager
    """
    if isinstance(dataset, pd.DataFrame):
        mi_result = pd.DataFrame(0.0, index=dataset.columns, columns=dataset.columns)
        num_combinations = scipy.special.comb(dataset.shape[1], 2)
        for idx1, idx2, mi in tqdm.tqdm(
            joblib.Parallel(n_jobs=processes, return_as="generator")(
                joblib.delayed(_mi_single_pair)(dataset[i], dataset[j], i, j, **kwargs)
                for i, j in itertools.combinations(dataset.columns, 2)
            ),
            disable=not progress_bar,
            total=num_combinations,
        ):
            mi_result.loc[idx1, idx2] = mi
            mi_result.loc[idx2, idx1] = mi
    else:
        dataset = np.array(dataset)
        mi_result = np.zeros((dataset.shape[1], dataset.shape[1]))
        num_combinations = scipy.special.comb(dataset.shape[1], 2)
        for idx1, idx2, mi in tqdm.tqdm(
            joblib.Parallel(n_jobs=processes, return_as="generator")(
                joblib.delayed(_mi_single_pair)(
                    dataset[:, i], dataset[:, j], i, j, **kwargs
                )
                for i, j in itertools.combinations(range(dataset.shape[1]), 2)
            ),
            disable=not progress_bar,
            total=num_combinations,
        ):
            mi_result[idx1, idx2] = mi
            mi_result[idx2, idx1] = mi
    return mi_result


U = TypeVar("U", int, str)
V = TypeVar("V", int, str)


def _mi_single_pair(
    item1: ArrayLike, item2: ArrayLike, idx1: U, idx2: V, **kwargs
) -> Tuple[U, V, float]:
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
        Keyword arguments passed through to the mutual_information_function

    Returns
    -------
    A tuple of the first index, the second index, and the mutual information between
    the two items
    """
    return idx1, idx2, mutual_information(item1, item2, **kwargs)


# endregion Pairwise Mutual Information

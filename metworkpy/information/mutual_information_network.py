"""
Functions for computing the Mutual Information Network for a Metabolic Model
"""

# Standard Library Imports
from __future__ import annotations
import functools
import math
import itertools
from multiprocessing import shared_memory, Pool, cpu_count
from typing import Tuple

# External Imports
import numpy as np
import pandas as pd
import tqdm

# Local Imports
from metworkpy.utils._parallel import _create_shared_memory_numpy_array
from metworkpy.information.mutual_information_functions import _mi_cont_cont_cheb_only


# region Main Function
def mi_network_adjacency_matrix(
    samples: pd.DataFrame | np.ndarray,
    n_neighbors: int = 5,
    processes: int = 1,
    progress_bar: bool = False,
) -> np.ndarray:
    """
    Create a Mutual Information Network Adjacency matrix from flux samples. Uses kth nearest neighbor method
    for estimating mutual information.
    :param samples: Numpy array or Pandas DataFrame containing the samples, columns should represent different reactions
        while rows should represent different samples
    :type samples: np.ndarray|pd.DataFrame
    :param n_neighbors: Number of neighbors to use for the Mutual Information estimation
    :type n_neighbors: int
    :param processes: Number of processes to use when
    :type processes: int
    :param progress_bar: Whether a progress bar should be displayed
    :type progress_bar: bool
    :return: Square numpy array with values at i,j representing the mutual information between the ith and jth columns
        in the original samples array. This array is symmetrical since mutual information is symmetrical.
    :rtype: np.ndarray

    .. seealso::

       1. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. Physical Review E, 69(6), 066138.
            Method for estimating mutual information between samples from two continuous distributions.
    """
    if isinstance(samples, pd.DataFrame):
        samples_array = samples.to_numpy()
    elif isinstance(samples, np.ndarray):
        samples_array = samples
    else:
        raise ValueError(
            f"samples is of an invalid type, expected numpy ndarray or "
            f"pandas DataFrame but received {type(samples)}"
        )
    processes = min(processes, cpu_count())
    (
        shared_nrows,
        shared_ncols,
        shared_dtype,
        shared_mem_name,
    ) = _create_shared_memory_numpy_array(samples_array)
    shm = shared_memory.SharedMemory(name=shared_mem_name)
    # Wrapped in try finally so that upon an error, the shared memory will be released
    try:
        # Currently this maps over a results matrix, returns the indices and uses those to write the results to
        # A matrix in the main process
        # It could be more memory efficient to put the results array into shared memory, and have each
        # process write its results there without returning, depending on if this means the main
        # process needs to hold on to a list of the returned values, or if it eagerly writes the results...
        mi_array = np.zeros((shared_ncols, shared_ncols), dtype=float)
        with Pool(processes=processes) as pool, tqdm.tqdm(
            total=math.comb(shared_ncols, 2), disable=not progress_bar
        ) as pbar:
            for x, y, mi in pool.imap_unordered(
                functools.partial(
                    _mi_network_worker,
                    shared_nrows=shared_nrows,
                    shared_ncols=shared_ncols,
                    shared_dtype=shared_dtype,
                    shared_mem_name=shared_mem_name,
                    n_neighbors=n_neighbors,
                ),
                itertools.combinations(range(shared_ncols), 2),
                chunksize=shared_ncols // processes,
            ):
                if progress_bar:
                    pbar.update()
                    pbar.refresh()
                # Set the value in the results matrix
                mi_array[x, y] = mi
                mi_array[y, x] = mi
    finally:
        shm.unlink()
    return mi_array


# endregion Main Function


# region Worker Function
def _mi_network_worker(
    index: Tuple[int, int],
    shared_nrows: int,
    shared_ncols: int,
    shared_dtype: np.dtype,
    shared_mem_name: str,
    n_neighbors: int,
) -> Tuple[int, int, float]:
    """
    Calculate the mutual information between two columns in the shared numpy array
    :param index: Tuple representing the index of the two columns
    :type index: Tuple[int, int]
    :param shared_nrows: Number of rows in the shared numpy array
    :type shared_nrows: int
    :param shared_ncols: Number of columns in the shared numpy array
    :type shared_ncols: int
    :param shared_dtype: Data type of the shared numpy array
    :type shared_dtype: np.dtype
    :param shared_mem_name: Name of the shared memory
    :type shared_mem_name: str
    :param n_neighbors: Number of neighbors to use for estimating the mutual information
    :type n_neighbors: int
    :return: Tuple of (column 1, column 2, mutual information between two columns)
    :rtype: Tuple[int, int, float]
    """
    # Get access to the shared memory, and create array from it
    shm = shared_memory.SharedMemory(name=shared_mem_name)
    shared_array = np.ndarray(
        (shared_nrows, shared_ncols), dtype=shared_dtype, buffer=shm.buf
    )
    # Get the x and y columns
    xcol, ycol = index
    x = shared_array[:, (xcol,)]
    y = shared_array[:, (ycol,)]
    return xcol, ycol, _mi_cont_cont_cheb_only(x=x, y=y, n_neighbors=n_neighbors)


# endregion Worker Function

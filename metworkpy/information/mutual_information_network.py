"""
Functions for computing the Mutual Information Network for a Metabolic Model
"""

# Standard Library Imports
from __future__ import annotations

import functools
import itertools
from multiprocessing import shared_memory, Pool
from typing import Tuple

# External Imports
import numpy as np
import pandas as pd
import scipy


# Local Imports


# region Main Function
def mi_network_adjacency_matrix(
    samples: pd.DataFrame | np.ndarray, n_neighbors: int = 5, processes: int = 1
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
        with Pool(processes=processes) as pool:
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
    # Get access to the shred memory, and create array from it
    shm = shared_memory.SharedMemory(name=shared_mem_name)
    shared_array = np.ndarray(
        (shared_nrows, shared_ncols), dtype=shared_dtype, buffer=shm.buf
    )

    # Get the x and y columns
    xcol, ycol = index
    x = shared_array[:, (xcol,)]
    y = shared_array[:, (ycol,)]

    # Stack x and y to form the z space
    z = np.hstack((x, y))

    # Create KDTrees for querying neighbors (this is overkill, but reflects the more generalized mutual information
    # code)
    x_tree = scipy.spatial.KDTree(x)
    y_tree = scipy.spatial.KDTree(y)
    z_tree = scipy.spatial.KDTree(z)

    # Find the distances from z to n_neighbor point
    # In the MI calculations, the z distance is always an inf norm
    r, _ = z_tree.query(z, k=[n_neighbors + 1], p=np.inf)
    r = r.squeeze()

    # Find the number of neighbors within radius r
    r = np.nextafter(r, 0)  # Shrink r to exclude kth neighbor
    x_neighbors = x_tree.query_ball_point(x=x, r=r, p=np.inf, return_length=True) - 1
    y_neighbors = y_tree.query_ball_point(x=y, r=r, p=np.inf, return_length=True) - 1

    # Calculate the Mutual Information based on equation (8) from Kraskov, Stogbauer, and Grassberger 2004
    mi = (
        scipy.special.digamma(n_neighbors)
        - np.mean(
            scipy.special.digamma(x_neighbors + 1)
            + scipy.special.digamma(y_neighbors + 1)
        )
        + scipy.special.digamma(z.shape[0])
    )
    return xcol, ycol, mi


# endregion Worker Function


# region Helper Functions
def _create_shared_memory_numpy_array(
    input_array, name: str | None = None
) -> Tuple[int, int, np.dtype, str]:
    """
    Create a numpy array in shared memory from `input_array`
    :param input_array: Numpy array to create shared memory array from
    :type input_array: np.ndarray
    :param name: Name to give shared memory
    :type name: str
    :return: Tuple of (number of rows, number of columns, dtype, name)
    :rtype: Tuple[int, int, np.dtype, str]
    """
    shm = shared_memory.SharedMemory(name=name, create=True, size=input_array.nbytes)
    shared_array = np.ndarray(
        input_array.shape, dtype=input_array.dtype, buffer=shm.buf
    )
    shared_array[:] = input_array
    nrow, ncol = input_array.shape
    return nrow, ncol, input_array.dtype, shm.name


# endregion Helper Functions

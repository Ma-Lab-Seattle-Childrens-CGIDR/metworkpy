"""
Calculate the divergence between paired columns in two arrays
"""

# Imports
# Standard Library Imports
from __future__ import annotations

from functools import partial
from multiprocessing import shared_memory, Pool, cpu_count
from typing import Callable

# External Imports
import numpy as np
import pandas as pd

# Local Imports
from metworkpy.utils._arguments import _parse_metric
from metworkpy.utils._parallel import _create_shared_memory_numpy_array


def _divergence_array(
    p: pd.DataFrame | np.ndarray,
    q: pd.DataFrame | np.ndarray,
    divergence_function: Callable[[np.ndarray, np.ndarray, int, float], float],
    n_neighbors: int = 5,
    metric: float | str = 2.0,
    processes: int = 1,
) -> np.ndarray | pd.Series:
    processes = min(processes, cpu_count())
    if isinstance(p, pd.DataFrame):
        p_array = p.to_numpy()
    elif isinstance(p, np.ndarray):
        p_array = p
    else:
        raise ValueError(
            f"p is of an invalid type, expected numpy ndarray or "
            f"pandas DataFrame but received {type(p)}"
        )
    if isinstance(q, pd.DataFrame):
        q_array = q.to_numpy()
    elif isinstance(q, np.ndarray):
        q_array = q
    else:
        raise ValueError(
            f"q is of an invalid type, expected numpy ndarray or "
            f"pandas DataFrame but received {type(q)}"
        )
    if p_array.shape[1] != q_array.shape[1]:
        raise ValueError(
            f"p and q must have the same number of columns, but p has {p.shape[1]} columns "
            f"and q has {q.shape[1]} columns"
        )
    metric = _parse_metric(metric=metric)
    ncols = p_array.shape[1]
    (
        p_nrows,
        p_ncols,
        p_dtype,
        p_shared_name,
    ) = _create_shared_memory_numpy_array(p_array)
    (
        q_nrows,
        q_ncols,
        q_dtype,
        q_shared_name,
    ) = _create_shared_memory_numpy_array(q_array)
    p_shm = shared_memory.SharedMemory(name=p_shared_name)
    q_shm = shared_memory.SharedMemory(name=q_shared_name)
    try:
        divergence_array = np.zeros(ncols, dtype=float)
        with Pool(processes=processes) as pool:
            for col, div in pool.imap_unordered(
                partial(
                    _divergence_array_worker,
                    divergence_function=divergence_function,
                    p_nrows=p_nrows,
                    p_ncols=p_ncols,
                    p_dtype=p_dtype,
                    p_shared_name=p_shared_name,
                    q_nrows=q_nrows,
                    q_ncols=q_ncols,
                    q_dtype=q_dtype,
                    q_shared_name=q_shared_name,
                    n_neighbors=n_neighbors,
                    metric=metric,
                ),
                range(ncols),
                chunksize=ncols // processes,
            ):
                divergence_array[col] = div
    finally:
        p_shm.unlink()
        q_shm.unlink()
    if isinstance(p, pd.DataFrame):
        return pd.Series(divergence_array, index=p.columns)
    if isinstance(q, pd.DataFrame):
        return pd.Series(divergence_array, index=q.columns)
    return divergence_array


def _divergence_array_worker(
    col: int,
    divergence_function: Callable[[np.ndarray, np.ndarray, int, float], float],
    p_nrows: int,
    p_ncols: int,
    p_dtype: np.dtype,
    p_shared_name: str,
    q_ncols: int,
    q_nrows: int,
    q_dtype: np.dtype,
    q_shared_name: str,
    n_neighbors: int = 5,
    metric: float = 2.0,
):
    p_shm = shared_memory.SharedMemory(name=p_shared_name)
    p_array = np.ndarray((p_nrows, p_ncols), dtype=p_dtype, buffer=p_shm.buf)
    q_shm = shared_memory.SharedMemory(name=q_shared_name)
    q_array = np.ndarray((q_nrows, q_ncols), dtype=q_dtype, buffer=q_shm.buf)

    # Get the two columns
    p = p_array[:, (col,)]
    q = q_array[:, (col,)]

    # Calculate the JS Divergence
    return col, divergence_function(p, q, n_neighbors, metric)

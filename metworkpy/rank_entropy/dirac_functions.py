"""
Functions for computing differential rank conservation (DIRAC)
"""

# Imports
# Standard Library Imports
from __future__ import annotations

# External Imports
import numpy as np

# Local Imports

# region Main Function

# endregion Main Function

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

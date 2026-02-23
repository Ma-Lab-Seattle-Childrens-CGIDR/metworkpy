"""Calculate the divergence between paired columns in two arrays"""

# Imports
# Standard Library Imports
from __future__ import annotations

from multiprocessing import cpu_count
from typing import Protocol, Tuple, Union

# External Imports
import joblib
import numpy as np
import pandas as pd

# Local Imports
from ._main_wrapper import DivergenceResult


class DivergenceFunction(Protocol):
    def __call__(
        self,
        p: np.typing.ArrayLike,
        q: np.typing.ArrayLike,
        **kwargs,
    ) -> Union[float, DivergenceResult]: ...


ArrayInput = Union[pd.DataFrame, np.ndarray]
Array1D = Union[
    pd.Series,
    np.ndarray[Tuple[int], np.dtype[Union[np.float32, np.float64]]],
]


def _divergence_array(
    p: ArrayInput,
    q: ArrayInput,
    divergence_function: DivergenceFunction,
    calculate_pvalue: bool = False,
    # The axis to slice along (i.e. 1 represents calculating the divergence for every column)
    axis: int = 1,
    processes: int = 1,
    **kwargs,
) -> Union[Array1D, Tuple[Array1D, Array1D]]:
    processes = min(processes, cpu_count())
    # Check that if either p or q are DataFrames, they both are, and that
    # their column indices align
    if isinstance(p, pd.DataFrame) or isinstance(q, pd.DataFrame):
        assert isinstance(p, pd.DataFrame) and isinstance(q, pd.DataFrame), (
            "If p or q is a DataFrame, they both must be DataFrames"
        )
        if axis == 1:
            assert p.columns.equals(q.columns), (
                "p and q must have the same index along specified axis"
            )
        elif axis == 0:
            assert p.index.equals(q.index), (
                "p and q must have the same index along specified axis"
            )
        p_array, q_array = p.to_numpy(), q.to_numpy()
    else:
        p_array, q_array = np.array(p), np.array(q)
    if p_array.shape[axis] != q_array.shape[axis]:
        raise ValueError(
            f"p and q must have the same length along specified axis, but p has {p.shape[axis]} columns "
            f"and q has {q.shape[axis]} columns"
        )
    if np.any(p_array.shape == 0) or (np.any(q_array.shape == 0)):
        raise ValueError(
            "All input array dimensions must be non-zero, but at least 1 dimension of the input arrays has a size of 0."
        )
    divergence_result = np.zeros((p_array.shape[axis],))
    if calculate_pvalue:
        pvalue_result = np.zeros((p_array.shape[axis],))
    # Add calculate p-value to the kwargs dict
    kwargs["calculate_pvalue"] = calculate_pvalue
    for i, res_value in joblib.Parallel(
        n_jobs=processes, return_as="generator_unordered"
    )(
        joblib.delayed(_divergence_array_worker)(
            index=i,
            p=p_array,
            q=q_array,
            axis=axis,
            divergence_function=divergence_function,
            **kwargs,
        )
        for i in range(p_array.shape[axis])
    ):
        if not calculate_pvalue:
            divergence_result[i] = res_value
        else:
            divergence, pvalue = res_value
            divergence_result[i] = divergence
            pvalue_result[i] = pvalue
    if isinstance(p, pd.DataFrame):
        if not calculate_pvalue:
            return pd.Series(divergence_result, index=p.columns)
        return pd.Series(divergence_result, index=p.columns), pd.Series(
            pvalue_result, index=p.columns
        )
    if not calculate_pvalue:
        return divergence_result
    return divergence_result, pvalue_result


def _divergence_array_worker(
    index: int,
    p: np.typing.ArrayLike,
    q: np.typing.ArrayLike,
    axis: int,
    divergence_function: DivergenceFunction,
    **kwargs,
) -> Union[Tuple[int, float], Tuple[int, float, float]]:
    return index, divergence_function(
        p=p.take(indices=index, axis=axis),
        q=q.take(indices=index, axis=axis),
        **kwargs,
    )

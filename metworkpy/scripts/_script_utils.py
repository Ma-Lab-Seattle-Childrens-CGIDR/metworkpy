"""
Utility functions for command line scripts
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy._typing import ArrayLike

from metworkpy.utils._arguments import _parse_str_args_dict


def _parse_samples(samples_str: str) -> list[int]:
    """
    Parse a samples specification string to a list of sample rows
    :param samples_str: Samples specification string
    :type samples_str: str
    :return: List of sample rows
    :rtype: list[int]
    """
    if not samples_str:
        return []
    sample_list = []
    for val in samples_str.split(","):
        if ":" not in val:
            sample_list.append(int(val))
            continue
        start, stop = val.split(":")
        sample_list += list(range(int(start), int(stop)+1))
    return sample_list


def _parse_quantile(quantile_str: str) -> tuple[float, float]:
    """
    Parse a quantile specification string to a tuple of floats
    :param quantile_str: The string specifying desired quantiles
    :type quantile_str: str
    :return: The parsed quantiles
    :rtype: tuple[float,float]
    """
    if "," not in quantile_str:
        q = float(quantile_str)
        return q, 1 - q
    low_q, high_q = quantile_str.split(",")
    return float(low_q), float(high_q)


def _parse_aggregation_method(aggregation_method_str: str) -> Callable[[ArrayLike], float]:
    aggregation_method_str = _parse_str_args_dict(aggregation_method_str,
                                                  {
                                                      "min": ["minimum"],
                                                      "max": ["maximum"],
                                                      "median": ["median"],
                                                      "mean": ["mean", "average"]
                                                  })
    if aggregation_method_str == "min":
        return np.min
    elif aggregation_method_str == "max":
        return np.max
    elif aggregation_method_str == "median":
        return np.median
    elif aggregation_method_str == "mean":
        return np.mean
    else:
        raise ValueError(f"Couldn't Parse Aggregation Method: {aggregation_method_str}, please use "
                         f"min, max, median, or mean")

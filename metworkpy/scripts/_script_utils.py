"""
Utility functions for command line scripts
"""

from __future__ import annotations
import re
from typing import Callable, Optional

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
        sample_list += list(range(int(start), int(stop) + 1))
    return sample_list


def _parse_sample_groups_and_names(
    groups_str: str, names_str: Optional[str] = None
) -> tuple[list[list[int]], list[str]]:
    """
    Parse a sample groups specification string to a list of sample groups, and parse a group names specification
        string to

    :param groups_str: Sample group specification string
    :type groups_str: str
    :param names_str: Sample group names specification string
    :type names_str: str
    :return: A tuple, where the first value is the sample groups (list of lists of sample rows),
        and the second value is the sample group names (list of strings)
    :rtype: tuple[list[list[int]],list[str]]
    """
    if not groups_str:
        return []
    group_pattern = re.compile(r"\(([\d,:]+)\)")
    sample_groups = [_parse_samples(m) for m in group_pattern.findall(groups_str)]
    if names_str:
        group_names = names_str.split(",")
    else:
        group_names = [f"s{i}" for i in range(1, len(sample_groups) + 1)]
    if len(sample_groups) != len(group_names):
        raise ValueError(
            f"Number of provided sample group names must match number of sample groups, "
            f"but {len(sample_groups)} sample groups were provided, and {len(group_names)} "
            f"group names were provided."
        )
    return sample_groups, group_names


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


def _parse_aggregation_method(
    aggregation_method_str: str,
) -> Callable[[ArrayLike], float]:
    aggregation_method_str = _parse_str_args_dict(
        aggregation_method_str,
        {
            "min": ["minimum"],
            "max": ["maximum"],
            "median": ["median"],
            "mean": ["mean", "average"],
        },
    )
    if aggregation_method_str == "min":
        return np.min
    elif aggregation_method_str == "max":
        return np.max
    elif aggregation_method_str == "median":
        return np.median
    elif aggregation_method_str == "mean":
        return np.mean
    else:
        raise ValueError(
            f"Couldn't Parse Aggregation Method: {aggregation_method_str}, please use "
            f"min, max, median, or mean"
        )


def _parse_format_to_extension(format_str: str) -> str:
    extension = _parse_str_args_dict(
        format_str,
        {
            "json": ["json"],
            "yml": ["yaml", "yml"],
            "xml": ["xml", "sbml"],
            "mat": ["matlab"],
        },
    )
    return extension

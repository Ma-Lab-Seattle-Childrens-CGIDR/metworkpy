"""
Simple helper functions for parsing arguments.
"""

# Standard Library Imports
import re
from typing import Dict, List, Union

import numpy as np


def _parse_str_args_list(arg: str, arg_list: List[str]) -> str:
    """
    Function to decide which string argument from a list of arguments is trying to be
    passed

    :param arg: Argument to parse
    :type arg: str
    :param arg_list: List of possible arguments to match against
    :type arg_list: list
    :return: Matched argument
    :rtype: str
    :raises ValueError: If more than one argument could be matched
    """
    arg_regex = re.compile(r"^" + arg, re.IGNORECASE)
    filtered_args = [arg for arg in arg_list if arg_regex.match(arg)]
    if len(filtered_args) == 1:
        return filtered_args[0]
    raise ValueError(
        f"Argument could match more than one argument in arg_list: {filtered_args}"
    )


def _parse_str_args_dict(arg: str, arg_dict: Dict[str, List]) -> str:
    """
    Function to decide which group of arguments given as values in a dictionary the
    argument matches, returns the key for that group.

    :param arg: Argument to match
    :type arg: str
    :param arg_dict: Dictionary defining argument groups
    :type arg_dict: Dict[str, List]
    :return: Key from the dictionary representing the group
    :rtype: str
    :raises ValueError: If more than one argument group could be matched
    """
    arg_regex = re.compile(r"^" + arg, re.IGNORECASE)
    filtered_groups = [
        key
        for key, arg_list in arg_dict.items()
        if len([arg for arg in arg_list if arg_regex.match(arg)]) >= 1
    ]
    if len(filtered_groups) == 1:
        return filtered_groups[0]
    raise ValueError(
        f"Argument could match more than one argument group in arg_dict: "
        f"{filtered_groups}"
    )


def _match_abbr(word: str, *args, **kwargs) -> re.Pattern:
    """
    Creates a regex pattern which will match any word abbreviations.
    :param to_match: String to match abbreviations of
    :type to_match: str
    :param args: Arguments passed to re.compile when creating the pattern
    :param kwargs: Key word arguments passed to re.compile when creating the pattern
    :return: Regex pattern for matching abbreviations
    :rtype:re.Pattern
    """
    if len(word) < 2:
        raise ValueError("word must be at least two characters long")
    if re.match(r"\s", word):
        raise ValueError("word must not contain whitespace")
    pattern = r"(?:" + word[-1] + ")?"
    for char in word[-2:0:-1]:
        pattern = r"(?:" + char + pattern + r")?"
    pattern = word[0] + pattern
    return re.compile(pattern, *args, **kwargs)


def _parse_metric(metric: Union[str, float]):
    if isinstance(metric, int):
        metric = float(metric)
    if isinstance(metric, float):
        if metric >= 1.0:
            return metric
        else:
            raise ValueError(
                "If metric is a float, must be in the range [1,inf] as it represents a Minkowski p-norm"
            )
    try:
        arg = _parse_str_args_dict(
            metric,
            {
                "euclidean": ["euclidean"],
                "manhattan": ["manhattan", "absolute", "taxicab"],
                "chebyshev": ["chebyshev", "tchebyshev", "maxmimum"],
            },
        )
    except ValueError as err:
        raise ValueError(
            "Metric string couldn't be understood, should be Euclidean, Manhattan, Taxicab, or Chebyshev"
        ) from err
    if arg == "euclidean":
        return 2.0
    if arg == "manhattan":
        return 1.0
    if arg == "chebyshev":
        return np.inf

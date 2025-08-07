"""
Submodule containing functions which will calculate the divergence between two dataframes for
groups of columns
"""

# Standard Library Imports
from typing import Literal, Hashable

# External Imports
import joblib
import numpy as np
import pandas as pd

# Local Imports
from .kl_divergence_functions import kl_divergence
from .js_divergence_functions import js_divergence


def calculate_divergence_grouped(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    divergence_groups: dict[str, list[Hashable]],
    divergence_type: Literal["kl", "js"] = "kl",
    processes: int = -1,
    **kwargs,
) -> pd.Series:
    """
    Calculate the divergence between data in two dataframes for a set of groups of columns

    Parameters
    ----------
    dataset1,dataset2 : pd.DataFrame
        Datasets to calculate the divergence between, rows should represent different samples, and
        columns should represent different features
    divergence_groups : dict of str to list of Hashable
        The groups to calculate divergence for, indexed by name of the group, with values
        of lists of features that belong to the group (the feature names must match names
        of columns in the dataframes)
    divergence_type : 'kl' or 'js'
        The type of divergence to calculate, either kl for Kullback-Leibler (default) or
        js for Jenson-Shannon
    processes : int
        The number of processes to use for the calculation
    kwargs
        Keyword arguments passed into the divergence method function

    Returns
    -------
    divergence : pd.Series
        A pandas series indexed by group name, with values representing the divergence of that group between the two
        dataframes

    Notes
    -----
    The parallelization uses joblib, and so can be configured with joblib's parallel_config context manager
    """
    if divergence_type == "kl":
        divergence_function = kl_divergence
    elif divergence_type == "js":
        divergence_function = js_divergence
    else:
        raise ValueError(
            f"Invalid divergence type, must be either kl or js, but received {divergence_type}"
        )
    divergence_results = pd.Series(np.nan, index=pd.Index(divergence_groups.keys()))
    for idx, calculated_divergence in enumerate(
        joblib.Parallel(n_jobs=processes, return_as="generator")(
            joblib.delayed(divergence_function)(
                dataset1[divergence_groups[group_name]],
                dataset2[divergence_groups[group_name]],
                **kwargs,
            )
            for group_name in divergence_results.index
        )
    ):
        divergence_results.iloc[idx] = calculated_divergence
    return divergence_results

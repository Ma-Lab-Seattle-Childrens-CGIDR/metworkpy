"""
Submodule containing functions which will calculate the divergence between two
dataframes for groups of columns
"""

# Standard Library Imports
from typing import Hashable, Literal, Tuple, Union
from warnings import warn

# External Imports
import joblib  # type: ignore   # Missing stubs
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
    calculate_pvalue: bool = False,
    processes: int = 1,
    **kwargs,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Calculate the divergence between data in two dataframes for a set of
    groups of columns

    Parameters
    ----------
    dataset1,dataset2 : pd.DataFrame
        Datasets to calculate the divergence between, rows should represent
        different samples, and columns should represent different features
    divergence_groups : dict of str to list of Hashable
        The groups to calculate divergence for, indexed by name of the group,
        with values of lists of features that belong to the group (the feature
        names must match names of columns in the dataframes)
    divergence_type : 'kl' or 'js'
        The type of divergence to calculate, either kl for Kullback-Leibler
        (default) or js for Jenson-Shannon
    processes : int
        The number of processes to use for the calculation
    kwargs
        Keyword arguments passed into the divergence method function
        either `kl_divergence` or `js_divergence` depending on
        `divergence_type`

    Returns
    -------
    divergence : pd.Series or tuple of pd.Series,pd.Series
        A pandas series indexed by group name, with values representing the
        divergence of that group between the two
        dataframes. If `calculate_pvalue` is True, then instead returns a tuple,
        of two pandas Series, the first being the divergence results, and the
        second being the p-values.

    Notes
    -----
    The parallelization uses joblib, and so can be configured with joblib's
    parallel_config context manager
    """
    if divergence_type == "kl":
        divergence_function = kl_divergence
    elif divergence_type == "js":
        divergence_function = js_divergence
    else:
        raise ValueError(
            f"Invalid divergence type, must be either kl or js, "
            f"but received {divergence_type}"
        )
    divergence_groups_new = {}
    for name, group in divergence_groups.items():
        if len(group) < 1:
            warn(
                f"The divergence group {name} has no "
                f"members, dropping from calculation."
            )
        else:
            divergence_groups_new[name] = group
    divergence_groups = divergence_groups_new
    divergence_results = pd.Series(
        np.nan, index=pd.Index(divergence_groups.keys())
    )
    if calculate_pvalue:
        pvalue_results = pd.Series(
            np.nan, index=pd.Index(divergence_groups.keys())
        )
    kwargs["calculate_pvalue"] = calculate_pvalue
    for idx, ret_value in enumerate(
        joblib.Parallel(n_jobs=processes, return_as="generator")(
            joblib.delayed(divergence_function)(
                dataset1[divergence_groups[group_name]],
                dataset2[divergence_groups[group_name]],
                **kwargs,
            )
            for group_name in divergence_results.index
        )
    ):
        if not calculate_pvalue:
            divergence_results.iloc[idx] = ret_value
        else:
            div, pval = ret_value
            divergence_results.iloc[idx] = div
            pvalue_results.iloc[idx] = pval
    if not calculate_pvalue:
        return divergence_results
    return divergence_results, pvalue_results

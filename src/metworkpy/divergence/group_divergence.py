"""
Submodule containing functions which will calculate the divergence between two
dataframes for groups of columns
"""

# Standard Library Imports
from typing import Hashable, Literal, Optional, Tuple, Union
from warnings import warn

# External Imports
import cobra
import joblib  # type: ignore   # Missing stubs
import numpy as np
import pandas as pd

# Local Imports
from .kl_divergence_functions import kl_divergence
from .js_divergence_functions import js_divergence
from metworkpy.network.network_construction import create_reaction_network
from metworkpy.network.neighborhoods import graph_neighborhoods


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


def calculate_reaction_neighborhood_divergence(
    model: cobra.Model,
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    divergence_type: Literal["kl", "js"] = "kl",
    directed: bool = False,
    nodes_to_remove: Optional[list[str]] = None,
    radius: int = 2,
    calculate_pvalue: bool = False,
    processes: int = 1,
    **kwargs,
):
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
    directed : bool
        Whether the reaction network created to find reaction neighborhoods
        should be directed or not
    nodes_to_remove : list[str] | None
        List of any metabolites or reactions that should be removed from
        the final network. This can be used to remove metabolites that
        participate in a large number of reactions, but are not desired
        in downstream analysis such as water, or ATP, or pseudo
        reactions like biomass. Each metabolite/reaction should be the
        string ID associated with them in the cobra model.
    radius : int
        The radius determining the sizes of the neighborhoods
    calculate_pvalue : bool
        Whether to calculate the p-value of the divergence
        difference using permutation testing
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
    # Construct a reaction network from the model
    rxn_network = create_reaction_network(
        model=model,
        weighted=False,
        directed=directed,
        nodes_to_remove=nodes_to_remove,
    )
    # Get the reaction neighborhoods from the network
    reaction_neighborhoods = graph_neighborhoods(rxn_network, radius=radius)
    return calculate_divergence_grouped(
        dataset1,
        dataset2,
        divergence_groups=reaction_neighborhoods,  # type: ignore
        divergence_type=divergence_type,
        calculate_pvalue=calculate_pvalue,
        processes=processes,
        **kwargs,
    )

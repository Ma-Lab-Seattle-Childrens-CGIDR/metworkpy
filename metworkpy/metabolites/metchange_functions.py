"""Module Implementing the Metchange Algorithm"""

# Imports
# Standard Library Imports
from __future__ import annotations
from functools import reduce
import hashlib
from typing import Optional, Iterable

# External Imports
import cobra
import numpy as np
import pandas as pd
from tqdm import tqdm


# Local Imports


# region Metchange


def metchange(
    model: cobra.Model,
    reaction_weights: dict[str, float] | pd.Series,
    metabolites: Optional[Iterable[str]] = None,
    objective_tolerance: float = 0.05,
    progress_bar: bool = False,
) -> pd.Series:
    """Use the Metchange algorithm to find the inconsistency scores for a set of
    metabolites based on reaction weights.

    Parameters
    ----------
    model : cobra.Model
        Cobra model to use for performing the metchange algorithm
    metabolites : Iterable[str]
        Metabolites to calculate consistency scores for, if None
        (default) will calculate for all metabolites in the model
    reaction_weights : dict[str, float] | pd.Series
        Weights for the reactions in the model, should correspond to the
        probability that reaction should not be active.
    objective_tolerance : float
        The tolerance for the objective value during the second
        optimization step, where the inner product of reaction weights
        and reaction fluxes is minimized. The flux for each metabolite
        will be constrained to be within objective-tolerance*objective-
        value of the unconstrained objective value for a metabolite.
        Defaults to 0.05.
    progress_bar : bool
        Whether a progress bar should be displayed

    Returns
    -------
    pd.Series
        Series of inconsistency scores for all the `metabolites`

    Notes
    -----
    This algorithm seeks to find an inconsistency score for metabolites based
    on gene expression. The gene expression is represented by reaction weights,
    which can be calculated by combining
    :func:`metworkpy.utils.expression_utils.expr_to_metchange_gene_weights` and
    :func:`metworkpy.gpr.gpr_functions.gene_to_rxn_weights`. The algorithm calculates the
    inconsistency score through a two part optimization. First, for a given
    metabolite, the maximum metabolite production is found. Then the metabolite
    production is constrained to stay within objective_tolerance*maximum-production of
    the maximum metabolite production, and the inner product of reaction weights,
    and reaction fluxes is minimized.
    This minimum inner product is the inconsistency score.
    """
    if isinstance(reaction_weights, dict):
        reaction_weights = pd.Series(reaction_weights)
    # If reaction weights is empty, set it to be 0 for all metabolites
    # And raise warning
    if len(reaction_weights) == 0:
        raise ValueError("Reaction weights is empty, must have at least one weight.")
    if metabolites is None:
        metabolites = model.metabolites.list_attr("id")
    elif isinstance(metabolites, str):
        metabolites = metabolites.split(sep=",")
    res_series = pd.Series(np.nan, index=metabolites)
    for metabolite in tqdm(metabolites, disable=not progress_bar):
        with model as m:
            add_metchange_objective_constraint_(
                model=m,
                metabolite=metabolite,
                reaction_weights=reaction_weights,
                objective_tolerance=objective_tolerance,
            )
            res_series[metabolite] = m.slim_optimize()
    return res_series


# endregion Metchange


# region helper functions
def add_metchange_objective_constraint_(
    model: cobra.Model,
    metabolite: str,
    reaction_weights: pd.Series,
    objective_tolerance: float = 0.05,
):
    """
    Add a metchange objective to a cobra Model

    Parameters
    ----------
    model : cobra.Model
        Cobra model to add the metchange objective to
    metabolite : str
        String ID of metabolite to add metchange objective for
    reaction_weights : pd.Series
        Weights for each reaction, representing the probability of that
        reaction being missing. A lower value indicates that the
        reaction is more likely to be present.
    objective_tolerance : float default=0.05
        The tolerance for the objective value during the second
        optimization step, where the inner product of reaction weights
        and reaction fluxes is minimized. The flux for each metabolite
        will be constrained to be within objective-tolerance*objective-
        value of the unconstrained objective value for a metabolite.

    Returns
    -------
    added_reaction : str
        id of the added sink reaction
    """
    if (reaction_weights == 0.0).all():
        raise ValueError(
            "At least one weight must be non-zero, but all weights "
            "in reaction_weights are zero."
        )
    metabolite_hash = hashlib.md5(metabolite.encode("utf-8")).hexdigest()[-8:]
    metabolite_sink_rxn_id = f"{metabolite}_sink_metchange{metabolite_hash}"
    metabolite_sink_rxn = cobra.Reaction(
        id=metabolite_sink_rxn_id, name=f"{metabolite} sink reaction", lower_bound=0.0
    )
    metabolite_sink_rxn.add_metabolites({model.metabolites.get_by_id(metabolite): -1.0})
    model.add_reactions([metabolite_sink_rxn])
    model.objective = metabolite_sink_rxn_id
    model.objective_direction = "max"
    obj_max = model.slim_optimize()
    model.reactions.get_by_id(metabolite_sink_rxn_id).lower_bound = obj_max - (
        obj_max * objective_tolerance
    )
    rxn_vars = []
    to_add = []
    for rxn, weight in reaction_weights.items():
        # If the weight is 0., doesn't need to be added
        if weight == 0.0:
            continue
        abs_var, *constr = cobra.util.solver.add_absolute_expression(
            model=model,
            expression=model.reactions.get_by_id(rxn).flux_expression,
            name=f"abs_var_{rxn}_{metabolite}",
            add=False,
        )
        to_add.append(abs_var)
        to_add.extend(constr)
        rxn_vars.append(abs_var * weight)
    # Add needed constraints and variables to model
    model.add_cons_vars(to_add)
    # Create objective of weight*abs value of rxn flux
    model.objective = reduce(lambda x, y: x + y, rxn_vars)
    model.objective_direction = "min"
    return metabolite_sink_rxn_id


# endregion helper functions

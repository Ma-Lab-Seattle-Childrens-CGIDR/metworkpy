"""
Submodule with functions for creating a context specific model using iMAT
"""

# Standard Library Imports
from __future__ import annotations
from typing import Optional, Union

# External Imports
import cobra
import numpy as np
import pandas as pd
import sympy as sym

# Local Imports
from metworkpy.imat.imat_functions import (
    imat,
    add_imat_constraints,
    add_imat_objective_,
)
from metworkpy.utils import _arguments

# define defaults for the iMAT functions
DEFAULTS = {
    "epsilon": 1.0,
    "objective_tolerance": 5e-2,
    "threshold": 1e-1,
    "tolerance": 1e-7,
}


# region: Main Model Creation Function
def generate_model(
    model: cobra.Model,
    rxn_weights: Union[pd.Series, dict],
    imat_solution: Optional[cobra.Solution] = None,
    method: str = "imat_restrictions",
    epsilon: float = DEFAULTS["epsilon"],
    threshold: float = DEFAULTS["threshold"],
    objective_tolerance: float = DEFAULTS["objective_tolerance"],
    **kwargs,
):
    """
    Generate a context specific model using iMAT.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param imat_solution: A pre-existing IMAT solution passed to the model creation methods,
        will only impact simple and subset as FVA and MILP solve altered versions of the
        IMAT problem.
    :type imat_solution: cobra.Solution
    :param method: The method to use for generating the context specific
        model. Valid methods are:
        'imat_restrictions', 'simple_bounds', 'subset',
        'fva', 'milp'.
    :type method: str
    :param epsilon: The epsilon value to use for iMAT (default: 1).
        Represents the minimum flux for a reaction to  be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-1).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :param objective_tolerance: The tolerance for the objective value
        (used for imat_restrictions and fva methods). The objective will
        be restricted to be within objective_tolerance*objective_value of the
        optimal objective value. (default: 5e-2)
    :type objective_tolerance: float
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. seealso::
        | :func:`imat_constraint_model` for more information on the
            imat_restrictions method.
        | :func:`simple_bounds_model` for more information on the
            simple_bounds method.
        | :func:`subset_model` for more information on the
            subset method.
        | :func:`fva_model` for more information on the fva method.
        | :func:`milp_model` for more information on the milp method.
    """
    method = _parse_method(method)
    if method == "imat_constraint":
        return imat_constraint_model(
            model, rxn_weights, epsilon, threshold, objective_tolerance, **kwargs
        )
    elif method == "simple_bounds":
        return simple_bounds_model(
            model=model,
            rxn_weights=rxn_weights,
            epsilon=epsilon,
            threshold=threshold,
            imat_solution=imat_solution,
            **kwargs,
        )
    elif method == "subset":
        return subset_model(
            model=model,
            rxn_weights=rxn_weights,
            epsilon=epsilon,
            threshold=threshold,
            imat_solution=imat_solution,
            **kwargs,
        )
    elif method == "fva":
        return fva_model(
            model, rxn_weights, epsilon, threshold, objective_tolerance, **kwargs
        )
    elif method == "milp":
        return milp_model(model, rxn_weights, epsilon, threshold, **kwargs)
    else:
        raise ValueError(
            f"Invalid method: {method}. Valid methods are: 'simple_bounds', \
            'imat_restrictions', "
            f"'subset', 'fva', 'milp'."
        )


# endregion: Main Model Creation Function


# region: Model Creation methods
def imat_constraint_model(
    model, rxn_weights, epsilon, threshold, objective_tolerance, **kwargs
):
    """
    Generate a context specific model by adding iMAT constraints, and
    ensuring iMAT objective value is near optimal.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-1).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :param objective_tolerance: The tolerance for the objective value.
        The objective will be restricted to be within
        objective_tolerance*objective_value of the optimal objective value.
        (default: 5e-2)
    :type objective_tolerance: float
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This function first solves the iMAT problem, then adds a constraint
        to ensure that the iMAT objective value is within
        objective_tolerance*objective_value of the optimal objective value.
        This model will include integer constraints, and so can not be used
        for sampling. If you want to use the model for sampling,
        use any of the other methods.

    """
    original_objective = model.objective
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    imat_solution = imat_model.optimize()
    imat_obj_val = imat_solution.objective_value
    rh = rxn_weights[rxn_weights > 0].index.tolist()
    rl = rxn_weights[rxn_weights < 0].index.tolist()
    rh_obj = []
    rl_obj = []
    for rxn in rh:  # For each highly expressed reaction
        # Get the forward and reverse variables from the model
        forward_variable = imat_model.solver.variables[f"y_pos_{rxn}"]
        reverse_variable = imat_model.solver.variables[f"y_neg_{rxn}"]
        # Adds the two variables to the rh list which will be used for sum
        rh_obj += [forward_variable, reverse_variable]
    for rxn in rl:  # For each lowly expressed reaction
        variable = imat_model.solver.variables[f"y_pos_{rxn}"]
        rl_obj += [variable]  # Note: Only one variable for lowly expressed reactions
    imat_obj_constraint = imat_model.solver.interface.Constraint(
        sym.Add(*rh_obj) + sym.Add(*rl_obj),
        lb=imat_obj_val - objective_tolerance * imat_obj_val,
        name="imat_obj_constraint",
    )
    imat_model.solver.add(imat_obj_constraint)
    imat_model.objective = original_objective
    return imat_model


def simple_bounds_model(
    model,
    rxn_weights: dict | pd.Series,
    epsilon: float,
    threshold: float,
    imat_solution: Optional[cobra.Solution] = None,
    **kwargs,
):
    """
    Generate a context specific model by setting bounds on reactions based on
    iMAT solution.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-1).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :param imat_solution: A preexisting IMAT solution, if not provided will be computed when function is called
    :type imat_solution: cobra.Solution
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This method first solves the iMAT solution, then for reactions found
        to be lowly expressed (weight<0), and inactive (flux<threshold),
        the reaction bounds are set to (-threshold, threshold),
        (0, threshold), or (-threshold, 0) depending on reversibility.
        For reactions found to be highly expressed and active in the
        forward direction (weight>0, flux>epsilon), the reaction bounds are
        set to (epsilon, ub), or (lb, ub) if lb>epsilon. For reactions found
        to be highly expressed and active in the reverse direction
        (weight>0, flux<-epsilon), the reaction bounds are set to
        (lb, -epsilon), or (lb, ub) if ub<-epsilon.  This model will not
        include integer constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    if not imat_solution:
        imat_solution = imat(model, rxn_weights, epsilon, threshold)
    if imat_solution.status != "optimal":
        raise ValueError("No optimal solution found for IMAT problem")
    fluxes = imat_solution.fluxes
    rl = rxn_weights[rxn_weights < 0].index.tolist()
    rh = rxn_weights[rxn_weights > 0].index.tolist()
    inactive_reactions = fluxes[(fluxes.abs() <= threshold) & (fluxes.index.isin(rl))]
    forward_active_reactions = fluxes[(fluxes >= epsilon) & (fluxes.index.isin(rh))]
    reverse_active_reactions = fluxes[(fluxes <= -epsilon) & (fluxes.index.isin(rh))]
    for rxn in inactive_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = _inactive_bounds(
            reaction.lower_bound, reaction.upper_bound, threshold
        )
    for rxn in forward_active_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = _active_bounds(
            reaction.lower_bound, reaction.upper_bound, epsilon, forward=True
        )
    for rxn in reverse_active_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = _active_bounds(
            reaction.lower_bound, reaction.upper_bound, epsilon, forward=False
        )
    return updated_model


def subset_model(
    model: cobra.Model,
    rxn_weights: dict | pd.Series,
    epsilon: float,
    threshold: float,
    imat_solution: Optional[cobra.Solution] = None,
    **kwargs,
):
    """
    Generate a context specific model by knocking out reactions found to
    be inactive by iMAT.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-1).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :param imat_solution: A preexisting IMAT solution, if not provided will be computed when function is called
    :type imat_solution: cobra.Solution
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This method first solves the iMAT solution, then for reactions found
        to be lowly expressed (weight<0), and  inactive (flux<threshold), the
        reaction bounds are set to (-threshold, threshold). This model will
        not include integer constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    if not imat_solution:
        imat_solution = imat(model, rxn_weights, epsilon, threshold)
    if imat_solution.status != "optimal":
        raise ValueError("No optimal solution found for IMAT problem")
    fluxes = imat_solution.fluxes
    rl = rxn_weights[rxn_weights < 0].index.tolist()
    inactive_reactions = fluxes[(fluxes.abs() <= threshold) & (fluxes.index.isin(rl))]
    for rxn in inactive_reactions.index:
        reaction = updated_model.reactions.get_by_id(rxn)
        # Force reaction to be below threshold
        reaction.bounds = _inactive_bounds(*reaction.bounds, threshold=threshold)
    return updated_model


def fva_model(
    model,
    rxn_weights,
    epsilon,
    threshold,
    objective_tolerance,
    loopless: bool = True,
    **kwargs,
):
    """
    Generate a context specific model by setting bounds on reactions based on
    FVA for an iMAT model.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-1).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :param objective_tolerance: The tolerance for the objective value.
        The objective will be restricted to be within
        objective_tolerance*objective_value of the optimal objective value.
        (default: 5e-2)
    :type objective_tolerance: float
    :param loopless: Whether to use the loopless FVA method (default: True).
        If False, the standard FVA method will be used.
    :type loopless: bool
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This method first creates a model with the iMAT constraints, and
        objective and then performs FVA to find the minimum and maximum
        flux for each reaction which allow for the objective to be within
        tolerance of optimal. These values are then set as the reaction
        bounds. This model is not guaranteed to have fluxes consistent
        with the optimal iMAT objective. This model will not include integer
        constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    reactions = rxn_weights[~np.isclose(rxn_weights, 0)].index.tolist()
    fva_res = cobra.flux_analysis.flux_variability_analysis(
        imat_model,
        fraction_of_optimum=(1 - objective_tolerance),
        loopless=loopless,
        reaction_list=reactions,
        **kwargs,
    ).dropna()
    for rxn in reactions:
        reaction = updated_model.reactions.get_by_id(rxn)
        reaction.bounds = (
            fva_res.loc[rxn, "minimum"],
            fva_res.loc[rxn, "maximum"],
        )
    return updated_model


def milp_model(model, rxn_weights, epsilon, threshold, **kwargs):
    """
    Generate a context specific model by setting bounds on reactions based on
    a set of mixed integer linear programming problems.

    :param model: A cobra.Model object to use for iMAT
    :type model: cobra.Model
    :param rxn_weights: A dictionary or pandas series of reaction weights.
    :type rxn_weights: dict | pandas.Series
    :param epsilon: The epsilon value to use for iMAT (default: 1).
        Represents the minimum flux for a reaction to be considered on.
    :type epsilon: float
    :param threshold: The threshold value to use for iMAT (default: 1e-1).
        Represents the maximum flux for a reaction to be considered off.
    :type threshold: float
    :return: A context specific cobra.Model.
    :rtype: cobra.Model

    .. note::
        This method first creates a model with the iMAT constraints, and
        objective and then solves a set of mixed integer linear programming
        problems, where each reaction is set to be inactive, active in the
        forward direction, and active in the reverse direction. The
        reaction bounds are then set based on the results of the MILP
        problems. This model is not guaranteed to have fluxes consistent
        with the optimal iMAT objective. This model will not include integer
        constraints, and so can be used for sampling.
    """
    updated_model = model.copy()
    imat_model = add_imat_constraints(model, rxn_weights, epsilon, threshold)
    add_imat_objective_(imat_model, rxn_weights)
    reactions = rxn_weights[~np.isclose(rxn_weights, 0)].index.tolist()
    milp_results = pd.DataFrame(
        np.nan,
        columns=["inactive", "forward", "reverse"],
        index=reactions,
        dtype=float,
    )
    for rxn in reactions:
        with imat_model as ko_model:  # Knock out the reaction
            reaction = ko_model.reactions.get_by_id(rxn)
            if reaction.lower_bound > threshold or reaction.upper_bound < -threshold:
                # Reaction can't be inactive, so skip
                ko_solution = -1
            else:
                reaction.bounds = _inactive_bounds(
                    reaction.lower_bound, reaction.upper_bound, threshold
                )
                ko_solution = ko_model.slim_optimize(error_value=-1.0)
        with imat_model as forward_model:
            reaction = forward_model.reactions.get_by_id(rxn)
            if reaction.upper_bound < epsilon:
                # Reaction can't be forced forward, so skip
                forward_solution = -1
            else:
                reaction.bounds = _active_bounds(
                    reaction.lower_bound,
                    reaction.upper_bound,
                    epsilon,
                    forward=True,
                )
                forward_solution = forward_model.slim_optimize(error_value=-1.0)
        with imat_model as reverse_model:
            reaction = reverse_model.reactions.get_by_id(rxn)
            if reaction.lower_bound > -epsilon:
                # Reaction can't be forced reverse, so skip
                reverse_solution = -1
            else:
                reaction.bounds = _active_bounds(
                    reaction.lower_bound,
                    reaction.upper_bound,
                    epsilon,
                    forward=False,
                )
                reverse_solution = reverse_model.slim_optimize(error_value=-1.0)
        milp_results.loc[rxn, :] = [
            ko_solution,
            forward_solution,
            reverse_solution,
        ]
    final_results = milp_results.apply(_milp_eval, axis=1)
    # Now 0 is inactive, 1 is forward, -1 is reverse, nan is under determined
    for rxn in reactions:
        if pd.isna(final_results[rxn]):  # skip under-determined reactions
            continue
        reaction = updated_model.reactions.get_by_id(rxn)
        if final_results[rxn] == 0:  # inactive
            reaction.bounds = _inactive_bounds(
                reaction.lower_bound, reaction.upper_bound, threshold
            )
        elif final_results[rxn] == 1:  # forward
            reaction.bounds = _active_bounds(
                reaction.lower_bound,
                reaction.upper_bound,
                epsilon,
                forward=True,
            )
        elif final_results[rxn] == -1:  # reverse
            reaction.bounds = _active_bounds(
                reaction.lower_bound,
                reaction.upper_bound,
                epsilon,
                forward=False,
            )
    return updated_model


# endregion Model Creation methods


# region Helper Functions
# noinspection PyProtectedMember
def _parse_method(method: str) -> str:
    """
    Parse the method string to a valid method name.

    :param method: The method to parse.
    :type method: str
    :return: The parsed method name.
    :rtype: str
    """
    try:
        return _arguments._parse_str_args_dict(
            method,
            {
                "simple_bounds": [
                    "simple bounds",
                    "simple-bounds",
                    "simple_bounds",
                ],
                "imat_constraint": [
                    "imat",
                    "imat_restrictions",
                    "imat-restrictions",
                    "imat restrictions",
                    "ir",
                    "imat constraints",
                    "imat-constraints",
                    "imat_constraints",
                    "ic",
                ],
                "subset": [
                    "subset-ko",
                    "subset_ko",
                    "eliminate-below-threshold",
                    "eliminate_below_threshold",
                    "subset-model",
                    "subset_model",
                    "subset model",
                ],
                "fva": [
                    "fva",
                    "flux_variability_analysis",
                    "flux-variability-analysis",
                    "flux variability analysis",
                    "fva_model",
                    "fva-model",
                    "fva model",
                ],
                "milp": [
                    "milp",
                    "milp-model",
                    "milp_model",
                    "milp model" "mixed_integer_linear_programming",
                    "mixed integer linear programming",
                    "mixed-integer-linear-programming",
                ],
            },
        )
    except ValueError as err:
        raise ValueError(
            f"Invalid method: {method}. Valid methods are: 'simple_bounds', "
            f"'imat_restrictions', "
            f"'eliminate_below_threshold', 'fva', 'milp'."
        ) from err


def _inactive_bounds(lb: float, ub: float, threshold: float) -> tuple[float, float]:
    """
    Find the new bounds for the reaction if it is inactive.
    """
    if lb > threshold:
        raise ValueError(
            "Lower bound is greater than threshold, reaction can not be \
                inactive."
        )
    if ub < -threshold:
        raise ValueError(
            "Upper bound is less than negative threshold, reaction can not be \
                inactive."
        )
    if lb > ub:
        raise ValueError("Lower bound is greater than upperbound")
    new_lb = max(lb, -threshold)
    new_ub = min(ub, threshold)
    return new_lb, new_ub


def _active_bounds(
    lb: float, ub: float, epsilon: float, forward: bool
) -> tuple[float, float]:
    """
    Find the new bounds for the reaction if it is active.
    """
    if lb > ub:
        raise ValueError("Lower bound is greater than upperbound")
    if forward:
        if ub < epsilon:
            raise ValueError(
                "Upper bound is less than epsilon, reaction can not be active \
                    in forward direction."
            )
        new_lb = max(lb, epsilon)
        new_ub = ub
    else:
        if lb > -epsilon:
            raise ValueError(
                "Lower bound is greater than negative epsilon, reaction can \
                    not be active in reverse "
                "direction."
            )
        new_lb = lb
        new_ub = min(ub, -epsilon)
    return new_lb, new_ub


def _milp_eval(milp_res: pd.Series) -> float:
    """
    Function for evaluating the results of the MILP method, to determine if a \
        reaction should be considered active or inactive. Returns 1 if
        forced forward, -1 if forced reverse, 0 if inactive, and NaN if
        under determined.
    """
    if pd.isna(milp_res).any():
        return np.nan
    if len(np.unique(milp_res)) == 3:  # Find which is max, then return based on that
        max_ind = np.argmax(milp_res)
        if max_ind > 1:
            res = -1  # reverse is the max
        else:
            res = max_ind  # Index corresponds to which is max
        return res
    elif (milp_res["forward"] == milp_res["reverse"]) and (
        milp_res["inactive"] > milp_res["forward"]
    ):
        # Forced forward, and reverse are the same, and inactive is
        # greater than both, so inactive
        return 0
    elif (milp_res["inactive"] == milp_res["reverse"]) and (
        milp_res["forward"] > milp_res["inactive"]
    ):
        # Forced reverse, and inactive are the same, and forward is
        # greater than both, so forward
        return 1
    elif (milp_res["inactive"] == milp_res["forward"]) and (
        milp_res["reverse"] > milp_res["inactive"]
    ):
        # Forced inactive, and forward are the same, and reverse is
        # greater than both, so reverse
        return -1
    else:
        # Under-determined case, return nan
        return np.nan


# endregion Helper Functions

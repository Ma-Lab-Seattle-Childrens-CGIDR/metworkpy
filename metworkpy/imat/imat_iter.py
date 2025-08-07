"""Submodule provided an iterator over IMAT solutions by employing integer cut constraints"""

# Standard Library Imports
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from typing import Union, Literal, Optional, Any

# External Imports
import cobra
import numpy as np
import optlang
import pandas as pd
from cobra import Configuration

from metworkpy.imat import model_creation

# Local Imports
from metworkpy.imat.imat_functions import (
    add_imat_objective_,
    add_imat_constraints_,
    _get_rxn_imat_binary_variable_name,
)

# define defaults for the iMAT functions
DEFAULTS = {
    "epsilon": 1,
    "threshold": 1e-2,
    "tolerance": Configuration().tolerance,
    "objective_tolerance": 5e-2,
    "max_iter": 20,
}


# region Reaction Activity Enum
class ReactionActivity(Enum):
    """An enum for representing reaction activity, with values of Inactive, ActiveReverse, ActiveForward, and Other"""

    Inactive = 0
    ActiveReverse = -1
    ActiveForward = 1
    Other = 404


# endregion Reaction Activity Enum


# region Main iMAT Iterator
# This class is for convenient dispatch to the different underlying iterators
class ImatIter:
    """Iterator for stepping through different possible iMAT solutions

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    output : Literal["model", "binary-variables", "reaction-activity"]
        What type of output is desired from the iterator, can be
        'model', 'binary-variables', or 'reaction-state', see notes for
        details.
    max_iter : int
        Maximum number of iMAT iterations to perform
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-2). Represents
        the maximum flux for a reaction to be considered off.
    objective_tolerance : float
        The tolerance for a solution to be considered optimal, the
        iterator will continue until no solution can be found which has
        a value of the iMAT objective function which is at least
        (1-`objective_tolerance`)*`optimal_imat_objective`. For example,
        a value of 0.05 (the default) indicates that the iterator will
        continue until no solution is found that is within 5% of the
        optimal objective.
    """

    def __init__(
        self,
        model: cobra.Model,
        rxn_weights: Union[pd.Series, dict],
        output: Literal["model", "binary-variables", "reaction-activity"] = "model",
        max_iter: int = DEFAULTS["max_iter"],
        epsilon: float = DEFAULTS["epsilon"],
        threshold: float = DEFAULTS["threshold"],
        objective_tolerance: float = DEFAULTS["objective_tolerance"],
        **kwargs,
    ):
        # Save all provided parameter to pass to specific iterator
        self.model = model
        self.rxn_weights = rxn_weights
        if output not in ["model", "binary-variables", "reaction-activity"]:
            raise ValueError(
                f'output parameter must be one of "model", "binary-variables", "reaction-activity", '
                f"but {output} was received instead"
            )
        self.output = output
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.threshold = threshold
        self.objective_tolerance = objective_tolerance
        self.kwarg_dict = kwargs

    def __iter__(self):
        if self.output == "model":
            iter_class = ImatIterModels
        elif self.output == "binary-variables":
            iter_class = ImatIterBinaryVariables
        elif self.output == "reaction-activity":
            iter_class = ImatIterReactionActivities
        else:
            raise ValueError(
                f'output parameter must be one of "model", "binary-variables", "reaction-activity", '
                f"but {self.output} was received instead"
            )
        return iter_class(
            model=self.model,
            rxn_weights=self.rxn_weights,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            threshold=self.threshold,
            objective_tolerance=self.objective_tolerance,
            **self.kwarg_dict,
        )


# endregion Main iMAT Iterator


# region Iterative Sampling


def imat_iter_flux_sample(
    model: cobra.Model,
    rxn_weights: pd.Series[float],
    model_generation_method: Literal["simple", "subset"] = "simple",
    max_iter: int = DEFAULTS["max_iter"],
    epsilon: float = DEFAULTS["epsilon"],
    threshold: float = DEFAULTS["threshold"],
    objective_tolerance: float = DEFAULTS["objective_tolerance"],
    sampler: Optional[cobra.sampling.HRSampler] = None,
    thinning: int = 100,
    num_samples: int = 1_000,
    sampler_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> pd.DataFrame[float]:
    """Generate a flux sample from a Model by iterating over multiple optimal (or near-optimal depending on
    objective tolerance) iMAT solutions, and sampling from each

    Parameters
    ----------
    model : cobra.Model
        The base cobra Model to use for generating iMAT models which are
        then sampled from
    rxn_weights : pd.Series[float]
        The qualitative weights indicating which reactions are high
        expression, or low expression, with values of 1 indicating high
        expression reactions, -1 indicating low expression reactions,
        and 0 indicating in between or unknown (see
        :func:`metworkpy.gpr.gpr_functions.gene_to_rxn_weights` for help
        generating this from qualitative gene weights, and :func:`metwor
        kpy.utils.expression_utils.expr_to_imat_gene_weights` for
        converting expression data into gene weights).
    model_generation_method : Literal["simple", "subset"]
        Method to use when creating an updated model based on the
        results of the iMAT, can be either 'simple', or 'subset', see
        note for details.
    max_iter : int
        Maximum number of iMAT updated models to iterate through
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-2). Represents
        the maximum flux for a reaction to be considered off.
    objective_tolerance : float
        The tolerance for a solution to be considered optimal, the
        iterator will continue until no solution can be found which has
        a value of the iMAT objective function which is at least
        (1-`objective_tolerance`)*`optimal_imat_objective`. For example,
        a value of 0.05 (the default) indicates that the iterator will
        continue until no solution is found that is within 5% of the
        optimal objective.
    sampler : cobra.sampling.HRSampler
        Sampler class from cobra to use for sampling (defaults to
        OptGPSampler), should be the class, not an instance of the
        class, so pass `OptGPSampler`, not `OptGPSampler()`. Keywords to
        the __init__function can be passed as a dict to
        `sampler_kwargs`.
    thinning : int
        Thinning factor representing how often the sampler returns a
        sample, for example a value of 100 (the default) indicates that
        the sampler will return a sample every 100 steps. See `Cobrapy
        documentation
        <https://cobrapy.readthedocs.io/en/latest/sampling.html>`_ Will
        be overwritten by value in sampler_kwargs if thinning is
        provided there.
    num_samples : int
        Number of samples to take per iMAT updated model so the total
        number of samples will be (number of iMAT updated models
        generated)*num_samples
    sampler_kwargs : dict[str, Any]
        Keyword arguments passed to the __init__ function of the sampler
        class
    **kwargs : dict[str, Any]
        Additional keyword arguments passed to the sampler's sample
        function

    Returns
    -------
    pd.DataFrame[float]
        Flux samples from several iMAT updated models
    """
    # Create a list to hold the results
    flux_sample_df_list = []
    # Set up the sampler if needed
    if sampler is None:
        sampler = cobra.sampling.OptGPSampler
    if sampler_kwargs is None:
        sampler_kwargs = {}
    if thinning not in sampler_kwargs:
        sampler_kwargs["thinning"] = thinning
    # Iterate through the iMAT updated models
    for updated_model in ImatIterModels(
        model=model,
        rxn_weights=rxn_weights,
        method=model_generation_method,
        max_iter=max_iter,
        epsilon=epsilon,
        threshold=threshold,
        objective_tolerance=objective_tolerance,
    ):
        # Create the sampler
        imat_sampler = sampler(model=updated_model, **sampler_kwargs)
        # Sample from the updated model
        flux_samples = imat_sampler.sample(n=num_samples, **kwargs)
        # Validate the flux samples
        # noinspection PyTypeChecker
        valid_flux_samples = flux_samples[imat_sampler.validate(flux_samples) == "v"]
        # Add the valid samples to the results list
        flux_sample_df_list.append(valid_flux_samples)
    # Return the combined flux samples
    return pd.concat(flux_sample_df_list, axis=0)


# endregion Iterative Sampling


# region Imat Iterator Base Class
class ImatIterBase(ABC):
    """Iterator for stepping through different possible iMAT solutions

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    max_iter : int
        Maximum number of iMAT iterations to perform
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-2). Represents
        the maximum flux for a reaction to be considered off.
    objective_tolerance : float
        The tolerance for a solution to be considered optimal, the
        iterator will continue until no solution can be found which has
        a value of the iMAT objective function which is at least
        (1-`objective_tolerance`)*`optimal_imat_objective`. For example,
        a value of 0.05 (the default) indicates that the iterator will
        continue until no solution is found that is within 5% of the
        optimal objective.
    """

    def __init__(
        self,
        model: cobra.Model,
        rxn_weights: Union[pd.Series, dict],
        max_iter: int = DEFAULTS["max_iter"],
        epsilon: float = DEFAULTS["epsilon"],
        threshold: float = DEFAULTS["threshold"],
        objective_tolerance: float = DEFAULTS["objective_tolerance"],
    ):
        self.in_model = model
        self._imat_model = (
            model.copy()
        )  # Create a new model to actually add the constraints and objective
        # Save the values into the iterator
        self._epsilon = epsilon
        self._threshold = threshold
        self._rxn_weights = rxn_weights
        self._objective_tolerance = objective_tolerance
        self._max_iter = max_iter
        # Start a counter for the maximum number of iterations
        self._counter = 0
        # Set up the imat_model with needed constraints and objective
        add_imat_constraints_(
            self._imat_model,
            rxn_weights=rxn_weights,
            epsilon=self._epsilon,
            threshold=self._threshold,
        )
        add_imat_objective_(self._imat_model, rxn_weights=rxn_weights)
        # Create a variable to hold the optimal objective value
        self._optimal_imat_objective = None  # This will be set on first iteration

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    # Some Helper methods available for all subclasses
    def _get_high_expr_rxns(self) -> list[str]:
        """Get a list of reaction ids of all the high expression reactions

        Returns
        -------
        list[str]
            The list of high expression reaction ids.
        """
        return list(self._rxn_weights[self._rxn_weights > 0.5].index)

    def _get_low_expr_rxns(self) -> list[str]:
        """Get a list of reaction ids of all the low expression reactions

        Returns
        -------
        list[str]
            The list of low expression reaction ids
        """
        return list(self._rxn_weights[self._rxn_weights < -0.5].index)

    def _get_high_expr_pos_variables(self) -> dict[str, optlang.Variable]:
        """Get a dict of all the y_pos variables for high expression reactions, keyed by reaction id

        Returns
        -------
        list[optlang.Variable]
            The list of all the y_pos variables for high expression
            reactions
        """
        high_expr_pos_variables = {}
        for rxn in self._get_high_expr_rxns():
            high_expr_pos_variables[rxn] = self._imat_model.variables.get(
                _get_rxn_imat_binary_variable_name(
                    rxn, expression_weight="high", which="positive"
                )
            )
        return high_expr_pos_variables

    def _get_high_expr_neg_variables(self) -> dict[str, optlang.Variable]:
        """Get a dict of all the y_neg variables for high expression reactions, keyed by reaction id

        Returns
        -------
        dict[str,optlang.Variable]
            The dict of all the y_pos variables for high expression
            reactions
        """
        high_expr_neg_variables = {}
        for rxn in self._get_high_expr_rxns():
            high_expr_neg_variables[rxn] = self._imat_model.variables.get(
                _get_rxn_imat_binary_variable_name(
                    rxn, expression_weight="high", which="negative"
                )
            )
        return high_expr_neg_variables

    def _get_low_expr_variables(self) -> dict[str, optlang.Variable]:
        """Get a dict of all the y_pos variables for low expression reactions, keyed by reaction id

        Returns
        -------
        dict[str, optlang.Variable]
            The dict of all the y_pos variables for low expression
            reactions
        """
        low_expr_pos_variables = {}
        for rxn in self._get_low_expr_rxns():
            low_expr_pos_variables[rxn] = self._imat_model.variables.get(
                _get_rxn_imat_binary_variable_name(
                    rxn, expression_weight="low", which="positive"
                )
            )
        return low_expr_pos_variables

    def _get_high_expr_variables(self) -> dict[str, dict[str, optlang.Variable]]:
        """Get a nested dict of all the variables associated with high expression reactions, keyed by reaction id,
        and then by 'pos'/'neg' for positive and negative variables respectively

        Returns
        -------
        dict[str,dict[str,optlang.Variable]]
            The dict of all the variables associated with high
            expression reactions
        """
        high_expr_variables = {}
        for rxn, y_pos in self._get_high_expr_pos_variables().items():
            high_expr_variables[rxn] = {"pos": y_pos}
        for rxn, y_neg in self._get_high_expr_neg_variables().items():
            high_expr_variables[rxn]["neg"] = y_neg
        return high_expr_variables

    def _get_binary_variables_state(self) -> pd.Series[ReactionActivity]:
        """Get a pandas Series describing the state of the weighted reactions in the iMAT solution

        Returns
        -------
        pd.Series[ReactionActivity]
            Series with reaction ids as the index, and
            :class:`ReactionActivity`

        Notes
        -----
        The activity of all unweighted reactions is Other, as is the activity of all reactions
        which have weights but don't have their binary variables 'on', i.e. high expression reactions
        which are not active (in either the forward, or reverse direction) and
        low expression reactions which are not inactive (so have activity above the threshold) will have
        a reaction activity of Other.

        Warnings
        --------
        This function requires that the model has been optimized, so that all the variables actually
        have primal values.
        """
        # Create a pandas series to hold the state
        reaction_activities = pd.Series(
            ReactionActivity.Other, index=self._rxn_weights.index
        )
        # Go through all high expression reactions
        for rxn, variables in self._get_high_expr_variables().items():
            if np.isclose(variables["pos"].primal, 1.0):
                reaction_activities[rxn] = ReactionActivity.ActiveForward
            elif np.isclose(variables["neg"].primal, 1.0):
                reaction_activities[rxn] = ReactionActivity.ActiveReverse
        # Next through all the low expression reactions
        for rxn, y_neg in self._get_low_expr_variables().items():
            if np.isclose(y_neg.primal, 1.0):
                reaction_activities[rxn] = ReactionActivity.Inactive
        return reaction_activities

    def _get_all_binary_variables(self) -> list[optlang.Variable]:
        """Get all the binary variables associated with the underlying iMAT model

        Returns
        -------
        list[optlang.Variable]
            List of all the binary variables in the iMAT model
        """
        all_binary_variables = []
        # Start with the high expr reactions
        for rxn in self._get_high_expr_rxns():
            # Get the y_pos
            all_binary_variables.append(
                self._imat_model.variables.get(
                    _get_rxn_imat_binary_variable_name(
                        rxn, expression_weight="high", which="positive"
                    )
                )
            )
            # and the y_neg
            all_binary_variables.append(
                self._imat_model.variables.get(
                    _get_rxn_imat_binary_variable_name(
                        rxn, expression_weight="high", which="negative"
                    )
                )
            )
        # Then all the low expr reactions
        for rxn in self._get_low_expr_rxns():
            all_binary_variables.append(
                self._imat_model.variables.get(
                    _get_rxn_imat_binary_variable_name(
                        rxn, expression_weight="low", which="positive"
                    )
                )
            )
        return all_binary_variables

    def _iter_update(self):
        """Function called during  __next__ method of subclasses, used to update counters, evaluate current
        iMAT solution, and raise the StopIteration if needed
        """
        # Start by calculating a solution
        imat_objective = self._imat_model.slim_optimize(error_value=np.nan)
        # If the model is infeasible, stop iteration
        if np.isnan(imat_objective):
            raise StopIteration
        # If this is the first iteration, this value is the optimal iMAT objective
        if self._optimal_imat_objective is None:
            self._optimal_imat_objective = imat_objective
        # If this value is more than `objective_tolerance` away from the optimal, stop iteration
        if (
            imat_objective
            < (1 - self._objective_tolerance) * self._optimal_imat_objective
        ):
            raise StopIteration
        # Check if the number of iterations has exceeded max_iter
        if self._counter >= self._max_iter:
            raise StopIteration
        # Update the counter of iterations
        self._counter += 1

    def _add_integer_cut(self) -> None:
        """Add an integer cut to the _imat_model which restricts a repeat of the current solution"""
        # Create a list to hold the expressions for variables which are "on", i.e. equal to 1
        on_variables = []
        # Create a list to hold the expressions for variables which are "off", i.e. equal to 0
        off_variables = []
        # Iterate through all binary variables
        for var in self._get_all_binary_variables():
            # Check if the variable is on or off
            if np.isclose(var.primal, 1.0):
                on_variables.append(1 - var)
            elif np.isclose(var.primal, 0.0):
                off_variables.append(var)
            else:
                raise RuntimeError(
                    f"Binary variable {var} has a value that is neither 0 nor 1"
                )
        # Add together the on and off expressions
        if len(on_variables) == 0 and len(off_variables) != 0:
            icut_expr = sum(off_variables)
        elif len(off_variables) == 0 and len(on_variables) != 0:
            icut_expr = sum(on_variables)
        elif len(off_variables) == 0 and len(on_variables) == 0:
            raise RuntimeError(
                "No binary variables found in iMAT model, were all the reaction weights 0?"
            )
        else:
            icut_expr = sum(on_variables) + sum(off_variables)
        icut_constraint = self._imat_model.solver.interface.Constraint(
            icut_expr, lb=1.0, name=f"integer_cut_{self._counter}"
        )
        self._imat_model.solver.add(icut_constraint)


# endregion Imat Iterator Base Class

# region Binary Variable Values iMAT Iterator

ImatBinaryVariables = namedtuple(
    "ImatBinaryVariables", ["rh_y_pos", "rh_y_neg", "rl_y_pos"]
)


class ImatIterBinaryVariables(ImatIterBase):
    """Iterator for stepping through different possible iMAT solutions, returning a named tuple of pandas Series describing
    the state of all binary variables in the iMAT problem

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    max_iter : int
        Maximum number of iMAT iterations to perform
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-2). Represents
        the maximum flux for a reaction to be considered off.
    objective_tolerance : float
        The tolerance for a solution to be considered optimal, the
        iterator will continue until no solution can be found which has
        a value of the iMAT objective function which is at least
        (1-`objective_tolerance`)*`optimal_imat_objective`. For example,
        a value of 0.05 (the default) indicates that the iterator will
        continue until no solution is found that is within 5% of the
        optimal objective.

    Returns
    -------
    unknown
        A named tuple with 3 fields

        * rh_y_pos: A pandas Series indexed by reaction id with the values indicating the state of the y+ variables
          associated with the high expression reactions. A value of 1 indicates that the reaction is **active** in the
          forward direction.
        * rh_y_neg: A pandas Series indexed by reaction id with the values indicating the state of the y- variables
          associated with the high expression reactions. A value of 1 indicates that the reaction is active in the
          reverse direction.
        * rl_y_pos: A pandas Series indexed by reaction id with the values indicating the state of the y+ variables
          associated with the low expression reactions. A value of 1 indicates that the reaction is **inactive**.
    """

    def __init__(
        self,
        model: cobra.Model,
        rxn_weights: Union[pd.Series, dict],
        max_iter: int = DEFAULTS["max_iter"],
        epsilon: float = DEFAULTS["epsilon"],
        threshold: float = DEFAULTS["threshold"],
        objective_tolerance: float = DEFAULTS["objective_tolerance"],
    ):
        super().__init__(
            model=model,
            rxn_weights=rxn_weights,
            max_iter=max_iter,
            epsilon=epsilon,
            threshold=threshold,
            objective_tolerance=objective_tolerance,
        )

    def __next__(self):
        # Call the base classes iter_update method to update the iter state
        self._iter_update()
        # Create the needed pandas series
        rh_y_pos = pd.Series(0.0, index=pd.Index(self._get_high_expr_rxns()))
        rh_y_neg = pd.Series(0.0, index=rh_y_pos.index)
        rl_y_pos = pd.Series(0.0, index=pd.Index(self._get_low_expr_rxns()))
        # Iterate through the different groups of binary variables to determine the values
        for rxn in rh_y_pos.index:
            rh_y_pos[rxn] = self._imat_model.variables.get(
                _get_rxn_imat_binary_variable_name(
                    rxn, expression_weight="high", which="positive"
                )
            ).primal
            rh_y_neg[rxn] = self._imat_model.variables.get(
                _get_rxn_imat_binary_variable_name(
                    rxn, expression_weight="high", which="negative"
                )
            ).primal
        for rxn in rl_y_pos.index:
            rl_y_pos[rxn] = self._imat_model.variables.get(
                _get_rxn_imat_binary_variable_name(
                    rxn, expression_weight="low", which="positive"
                )
            ).primal
        # Add the integer cut constraint
        self._add_integer_cut()
        # Return the tuple of binary variable
        return ImatBinaryVariables(
            rh_y_pos=rh_y_pos, rh_y_neg=rh_y_neg, rl_y_pos=rl_y_pos
        )


# endregion Binary Variable Values iMAT Iterator

# region Reaction Activities iMAT Iterator


class ImatIterReactionActivities(ImatIterBase):
    """Iterator for stepping through different possible iMAT solutions, returning the reaction state of
    reactions with non-zero iMAT weights

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    max_iter : int
        Maximum number of iMAT iterations to perform
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-2). Represents
        the maximum flux for a reaction to be considered off.
    objective_tolerance : float
        The tolerance for a solution to be considered optimal, the
        iterator will continue until no solution can be found which has
        a value of the iMAT objective function which is at least
        (1-`objective_tolerance`)*`optimal_imat_objective`. For example,
        a value of 0.05 (the default) indicates that the iterator will
        continue until no solution is found that is within 5% of the
        optimal objective.

    Returns
    -------
    Iterable[pd.Series[ReactionActivity]]
        Every iteration returns a pandas Series of
        :class:`ReactionActivity` describing the activity of reactions
        in the iMAT Model.
    """

    def __init__(
        self,
        model: cobra.Model,
        rxn_weights: Union[pd.Series, dict],
        max_iter: int = DEFAULTS["max_iter"],
        epsilon: float = DEFAULTS["epsilon"],
        threshold: float = DEFAULTS["threshold"],
        objective_tolerance: float = DEFAULTS["objective_tolerance"],
    ):
        super().__init__(
            model=model,
            rxn_weights=rxn_weights,
            max_iter=max_iter,
            epsilon=epsilon,
            threshold=threshold,
            objective_tolerance=objective_tolerance,
        )

    def __next__(self) -> pd.Series[ReactionActivity]:
        # Call the base classes iter_update method to update the iter state
        self._iter_update()
        # Get the binary series of reaction activities to return
        rxn_activities = self._get_binary_variables_state()
        # Add the integer cut constraint
        self._add_integer_cut()
        # Return the reaction activities
        return rxn_activities


# endregion Reaction Activities iMAT Iterator


# region iMAT Model Iterator
class ImatIterModels(ImatIterBase):
    """Iterator for stepping through different possible iMAT solutions, returning an updated model for each
    iMAT solution, with modified reaction bounds to make it conform to the iMAT solution.

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    method : Literal["simple", "subset]
        Which method to use to create the returned iMAT model, can be
        either 'simple', or 'subset', see notes for details.
    max_iter : int
        Maximum number of iMAT iterations to perform
    epsilon : float
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float
        The threshold value to use for iMAT (default: 1e-2). Represents
        the maximum flux for a reaction to be considered off.
    objective_tolerance : float
        The tolerance for a solution to be considered optimal, the
        iterator will continue until no solution can be found which has
        a value of the iMAT objective function which is at least
        (1-`objective_tolerance`)*`optimal_imat_objective`. For example,
        a value of 0.05 (the default) indicates that the iterator will
        continue until no solution is found that is within 5% of the
        optimal objective.

    Returns
    -------
    Iterable[pd.Series[ReactionActivity]]
        Every iteration returns a pandas Series of
        :class:`ReactionActivity` describing the activity of reactions
        in the iMAT Model.

    Notes
    -----
    When creating an updated model based on the solution to the iMAT problem, two different methods can
    be selected, either

    * simple: This method enforces the activity constraints found during the iMAT solution, so
      reactions found to be active in the forward direction are forced to be active in the forward
      direction, and reactions found active in the reverse direction are forced to be active in the
      reverse direction, and reactions found to be inactive are forced to be inactive.
    * subset: This method instead finds which subset of reactions the iMAT problem indicates are not inactive,
      and allows only those reactions to carry flux (essentially inactive reactions are forced off).

    The simple method can lead to the model being infeasible, and can also lead to reactions being considered
    essential because their knockout leads to forced active reactions no longer being able to carry flux. The
    subset method shouldn't lead to as much infeasibility when performing essentiality analysis, but is a
    much lighter restriction on the model so may not fully incorporate the information provided by the
    gene expression weights.
    """

    def __init__(
        self,
        model: cobra.Model,
        rxn_weights: Union[pd.Series, dict],
        method: Literal["simple", "subset"] = "simple",
        max_iter: int = DEFAULTS["max_iter"],
        epsilon: float = DEFAULTS["epsilon"],
        threshold: float = DEFAULTS["threshold"],
        objective_tolerance: float = DEFAULTS["objective_tolerance"],
    ):
        super().__init__(
            model=model,
            rxn_weights=rxn_weights,
            max_iter=max_iter,
            epsilon=epsilon,
            threshold=threshold,
            objective_tolerance=objective_tolerance,
        )
        self._method = method

    # noinspection PyProtectedMember
    def __next__(self) -> cobra.Model:
        # Call the base classes iter_update method to update the iter state
        self._iter_update()
        # Create the model to be returned
        updated_model = self.in_model.copy()
        # Get the reaction activities of the underlying problem
        reaction_activities = self._get_binary_variables_state()
        active_forward_reactions = reaction_activities[
            reaction_activities == ReactionActivity.ActiveForward
        ].index
        active_reverse_reactions = reaction_activities[
            reaction_activities == ReactionActivity.ActiveReverse
        ].index
        inactive_reactions = reaction_activities[
            reaction_activities == ReactionActivity.Inactive
        ].index
        for rxn in inactive_reactions:
            reaction = updated_model.reactions.get_by_id(rxn)
            # noinspection PyProtectedMember
            reaction.bounds = model_creation._inactive_bounds(
                reaction.lower_bound, reaction.upper_bound, self._threshold
            )
        if self._method == "simple":
            for rxn in active_forward_reactions:
                reaction = updated_model.reactions.get_by_id(rxn)
                # noinspection PyProtectedMember
                reaction.bounds = model_creation._active_bounds(
                    reaction.lower_bound,
                    reaction.upper_bound,
                    self._epsilon,
                    forward=True,
                )
            for rxn in active_reverse_reactions:
                reaction = updated_model.reactions.get_by_id(rxn)
                reaction.bounds = model_creation._active_bounds(
                    reaction.lower_bound,
                    reaction.upper_bound,
                    self._epsilon,
                    forward=False,
                )
        # Add the integer cut constraint
        self._add_integer_cut()
        # Return the updated_model
        return updated_model


# endregion iMAT Model Iterator

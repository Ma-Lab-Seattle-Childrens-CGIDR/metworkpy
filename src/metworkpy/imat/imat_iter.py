"""Submodule provided an iterator over IMAT solutions by employing integer cut constraints"""

# Standard Library Imports
from __future__ import annotations
import warnings

from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, Union, Literal, Optional, Any

# External Imports
import cobra
from cobra.exceptions import OptimizationError
from docrep import DocstringProcessor
import numpy as np
import optlang
import pandas as pd
import tqdm

from metworkpy.imat import model_creation

# Local Imports
from metworkpy.imat.imat_functions import (
    add_imat_objective_,
    add_imat_constraints_,
    _get_rxn_imat_binary_variable_name,
)
from metworkpy.utils.metworkpy_defaults import IMAT_DEFAULTS

# Make sure optlang has Variable
assert "Variable" in optlang.__dir__()

# Setup docrep
docs = DocstringProcessor()


# region Reaction Activity Enum
class ReactionActivity(Enum):
    """An enum for representing reaction activity, with values of Inactive, ActiveReverse, ActiveForward, and Other"""

    Inactive = 0
    ActiveReverse = -1
    ActiveForward = 1
    Other = 404


# endregion Reaction Activity Enum


# region Imat Iterator Base Class
@docs.get_sections(
    base="ImatIterBase", sections=["Parameters", "Notes", "References"]
)
@docs.dedent
class ImatIterBase(ABC):
    """
    Iterator for stepping through different possible iMAT solutions

    Parameters
    ----------
    model : cobra.Model
        A cobra.Model object to use for iMAT
    rxn_weights : dict | pandas.Series
        A dictionary or pandas series of reaction weights.
    iter_method : {'icut', 'maxdist', 'corner'}
        The method to use when iterating,

        * 'icut': Adds an integer cut constraint ensuring
          that the next iteration will return a different
          iMAT solution, and then maximizes the iMAT objective
          (ensuring that it is within 'objective_tolerance'
          of the initial iMAT solution without)
        * 'maxdist': Similar to icut, but additionally maximizes
          the distances between iMAT solutions (by maximizing
          the number of iMAT binary variables which are different
          between successive solutions)
        * 'corner': Similar to icut, but at each iteration maximizes a randomly
          constructed objective, see `metworkpy.sampling.corners.corner_sampling`
          for more information
    max_iter : int, default=20
        Maximum number of iMAT iterations to perform, set to None
        to remove limit on iterations
    epsilon : float, default=1.0
        The epsilon value to use for iMAT (default: 1). Represents the
        minimum flux for a reaction to be considered on.
    threshold : float, default=1e-2
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
    reaction_list : list[str], optional
        Used if `iter_method` is 'corner'. The set of reactions which could be selected
        to be a part of the objective. Must be a list of reaction ids.
    fva_scale : bool, default=True
        Used if `iter_method` is 'corner'. Whether to scale the weights assigned to
        reactions in the randomized objectives by the maximum (absolute) flux
        value the associated reaction could achieve.
    seed : int or np.random.Generator, optional
        Used if `iter_method` is 'corner'. Optional seed to use for selection
        of reactions/weights. Note that this doesn't garuntee the generated
        solutions will be the same, only that the objectives selected to generate
        each will be (so it will depend on solver consistancy if the resulting
        yielded values are identical).
    fva_kwargs : dict of str to Any
        Used if `iter_method` is 'corner', and `fva_scale` is True. Key word arguments
        passed to
        `cobra.flux_analysis.flux_variability_analysis <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/variability/index.html#cobra.flux_analysis.variability.flux_variability_analysis>`_,
        by default this will have the `fraction_of_optimum` set to 0.0,
        so that the objective function doesn't impact the sampling.

    Notes
    -----
    The integer cut and maxdist methods of iteration are based on [1]_, the corner method is
    based on the ideas of iMAT enumeration found in [1]_ and the method of corner sampling
    found in [2]_.

    References
    ----------
    .. [1] Rodríguez-Mier, P.; Poupin, N.; Blasio, C. de; Cam, L. L.; Jourdan, F. DEXOM: Diversity-Based
       Enumeration of Optimal Context-Specific Metabolic Networks. PLOS Computational Biology 2021, 17 (2),
       e1008730. https://doi.org/10.1371/journal.pcbi.1008730.
    .. [2] (1) Galuzzi, B. G.; Milazzo, L.; Damiani, C. Adjusting for False Discoveries in
       Constraint-Based Differential Metabolic Flux Analysis. Journal of Biomedical Informatics
       2024, 150, 104597. https://doi.org/10.1016/j.jbi.2024.104597.
    """

    def __init__(
        self,
        model: cobra.Model,
        rxn_weights: Union[pd.Series, dict],
        iter_method: Literal["icut", "maxdist", "corner"] = "icut",
        max_iter: Optional[int] = IMAT_DEFAULTS.max_iter,
        epsilon: float = IMAT_DEFAULTS.epsilon,
        threshold: float = IMAT_DEFAULTS.threshold,
        objective_tolerance: float = IMAT_DEFAULTS.objective_tolerance,
        reaction_list: Optional[list[str]] = None,
        fva_scale: bool = True,
        fva_kwargs: Optional[dict[str, Any]] = None,
        seed: Union[None, int, np.random.Generator] = None,
    ):
        self.in_model = model
        self._imat_model = (
            model.copy()
        )  # Create a new model to actually add the constraints and objective
        # Save the values into the iterator
        self._epsilon = epsilon
        self._threshold = threshold
        self._rxn_weights = pd.Series(rxn_weights)
        self._objective_tolerance = objective_tolerance
        self._max_iter = max_iter
        self._iter_method = iter_method
        self._rng = np.random.default_rng(seed)
        self._fva_scale = fva_scale
        if reaction_list is not None:
            self._reaction_list = reaction_list
        else:
            self._reaction_list: list[str] = (
                self._imat_model.reactions.list_attr("id")
            )

        # If the iter_method is corner, and fva_scale is True,
        # then we need to perform the FVA
        self._fva_max = None
        if iter_method == "corner" and fva_scale is True:
            default_fva_kwargs: dict[str, Any] = {
                "fraction_of_optimum": 0.0,
            }
            if fva_kwargs is None:
                fva_kwargs = default_fva_kwargs
            else:
                default_fva_kwargs.update(fva_kwargs)
                fva_kwargs = default_fva_kwargs
            fva_res = cobra.flux_analysis.flux_variability_analysis(
                self._imat_model,
                reaction_list=reaction_list,  # type:ignore
                **fva_kwargs,
            )
            fva_max = fva_res.abs().max(axis=1)
            # remove all reactions which have an fva_max of 0.0 from
            # possible selection
            zero_reactions = set(
                fva_max[
                    np.isclose(
                        fva_max, 0.0, atol=cobra.Configuration().tolerance
                    )
                ].index
            )
            self._reaction_list = list(
                set(self._reaction_list) - zero_reactions
            )
            self._fva_max = fva_max
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
        # Solve the iMAT problem and add it as a constraint to the model
        cobra.util.fix_objective_as_constraint(
            self._imat_model, fraction=1 - self._objective_tolerance
        )

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise NotImplementedError(
            "Not implemented in the ImatIterBase class, try ImatIterBinaryVariables, ImatIterReactionActivities, or ImatIterModels instead"
        )

    # Some Helper methods available for all subclasses
    def _get_high_expr_rxns(self) -> list[str]:
        """Get a list of reaction ids of all the high expression reactions

        Returns
        -------
        list[str]
            The list of high expression reaction ids.
        """
        assert isinstance(self._rxn_weights, pd.Series)
        return list(self._rxn_weights[self._rxn_weights > 0.0].index)

    def _get_low_expr_rxns(self) -> list[str]:
        """Get a list of reaction ids of all the low expression reactions

        Returns
        -------
        list[str]
            The list of low expression reaction ids
        """
        return list(self._rxn_weights[self._rxn_weights < 0.0].index)

    def _get_high_expr_pos_variables(self) -> dict[str, optlang.Variable]:  # type: ignore ## Checked on import earlier
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

    def _get_high_expr_neg_variables(self) -> dict[str, optlang.Variable]:  # type: ignore ## Checked on import earlier
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

    def _get_low_expr_variables(self) -> dict[str, optlang.Variable]:  # type: ignore ## Checked on import earlier
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

    def _get_high_expr_variables(
        self,
    ) -> dict[str, dict[str, optlang.Variable]]:  # type: ignore ## Checked on import earlier
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

    def _get_binary_variables_state(self) -> pd.Series[ReactionActivity]:  # type: ignore
        """
        Get a pandas Series describing the state of the weighted reactions in the iMAT solution

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
            if np.isclose(
                variables["pos"].primal,
                1.0,
                atol=cobra.Configuration().tolerance,
            ):
                reaction_activities[rxn] = ReactionActivity.ActiveForward
            elif np.isclose(
                variables["neg"].primal,
                1.0,
                atol=cobra.Configuration().tolerance,
            ):
                reaction_activities[rxn] = ReactionActivity.ActiveReverse
        # Next through all the low expression reactions
        for rxn, y_neg in self._get_low_expr_variables().items():
            if np.isclose(
                y_neg.primal,
                1.0,
                atol=cobra.Configuration().tolerance,
            ):
                reaction_activities[rxn] = ReactionActivity.Inactive
        return reaction_activities

    def _get_all_binary_variables(self) -> list[optlang.Variable]:  # type: ignore ## Checked on import earlier
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

    def _iter_next_start(self):
        """Function called at the begining of __next__ method of subclasses, used to update counters, evaluate current
        iMAT solution, and raise the StopIteration if needed
        """
        # Start by calculating a solution
        model_objective = self._imat_model.slim_optimize(error_value=np.nan)
        # If the model is infeasible, stop iteration
        # This will happen when the iMAT solution is no longer
        # sufficiently close to the starting iMAT solution,
        # since we add a constraint that restricts the model to be within
        # a certain distance of the original iMAT objective
        if np.isnan(model_objective):
            raise StopIteration
        # Check if the number of iterations has exceeded max_iter
        if self._max_iter is not None:
            if self._counter >= self._max_iter:
                raise StopIteration
        # Update the counter of iterations
        self._counter += 1

    def _iter_next_end(self):
        """
        Function called at the end of __next__ method of subclasses (just prior to return)
        in order to set up the next iteration
        """
        # Get the icut expression to use for the constraint,
        # and potentially the objective
        icut_expr = self._get_integer_cut_expression()
        icut_constraint = self._imat_model.solver.interface.Constraint(
            self._get_integer_cut_expression(),
            lb=1.0,
            name=f"integer_cut_{self._counter}",
        )
        self._imat_model.solver.add(icut_constraint)
        if self._iter_method == "maxdist":
            self._imat_model.objective = self._imat_model.problem.Objective(
                icut_expr, direction="max"
            )
        elif self._iter_method == "corner":
            self._add_random_reaction_objective()

    def _get_integer_cut_expression(self):
        # Create a list to hold the expressions for variables which are "on", i.e. equal to 1
        on_variables = []
        # Create a list to hold the expressions for variables which are "off", i.e. equal to 0
        off_variables = []
        # Iterate through all binary variables
        for var in self._get_all_binary_variables():
            # Check if the variable is on or off
            if np.isclose(
                var.primal, 1.0, atol=cobra.Configuration().tolerance
            ):
                on_variables.append(1 - var)
            elif np.isclose(
                var.primal, 0.0, atol=cobra.Configuration().tolerance
            ):
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
        return icut_expr

    def _add_random_reaction_objective(self) -> None:
        # Decide how many reactions from the reaction list to consider
        num_reactions = self._rng.integers(
            1, len(self._reaction_list) + 1
        )  # Pick at least 1 reactions
        # Select reactions
        objective_dict = {}
        for rxn in self._rng.choice(
            self._reaction_list, num_reactions, replace=False
        ):
            weight = self._rng.random() * 2 - 1
            if self._fva_scale:
                assert self._fva_max is not None
                weight /= self._fva_max[rxn]
            objective_dict[self._imat_model.reactions.get_by_id(rxn)] = weight
        self._imat_model.objective = objective_dict
        self._imat_model.objective_direction = self._rng.choice(
            ["max", "min"], 1, replace=False
        )[0]


# endregion Imat Iterator Base Class

# region Binary Variable Values iMAT Iterator


class ImatBinaryVariables(NamedTuple):
    rh_y_pos: pd.Series
    rh_y_neg: pd.Series
    rl_y_pos: pd.Series


@docs.dedent
class ImatIterBinaryVariables(ImatIterBase):
    """
    Iterator for stepping through different possible iMAT solutions, returning a named tuple of pandas Series describing
    the state of all binary variables in the iMAT problem

    Parameters
    ----------
    %(ImatIterBase.parameters)s

    Yields
    ------
    ImatBinaryVariables
        A named tuple with 3 fields

        * rh_y_pos: A pandas Series indexed by reaction id with the values indicating the state of the y+ variables
          associated with the high expression reactions. A value of 1 indicates that the reaction is **active** in the
          forward direction.
        * rh_y_neg: A pandas Series indexed by reaction id with the values indicating the state of the y- variables
          associated with the high expression reactions. A value of 1 indicates that the reaction is **active** in the
          reverse direction.
        * rl_y_pos: A pandas Series indexed by reaction id with the values indicating the state of the y+ variables
          associated with the low expression reactions. A value of 1 indicates that the reaction is **inactive**.

    Notes
    -----
    %(ImatIterBase.notes)s

    References
    ----------
    %(ImatIterBase.references)s
    """

    def __next__(self):
        # Call the base classes iter_update method to update the iter state
        self._iter_next_start()
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
        self._iter_next_end()
        # Return the tuple of binary variable
        return ImatBinaryVariables(
            rh_y_pos=rh_y_pos, rh_y_neg=rh_y_neg, rl_y_pos=rl_y_pos
        )


# endregion Binary Variable Values iMAT Iterator

# region Reaction Activities iMAT Iterator


@docs.dedent
class ImatIterReactionActivities(ImatIterBase):
    """
    Iterator for stepping through different possible iMAT solutions, returning the reaction state of
    reactions with non-zero iMAT weights

    Parameters
    ----------
    %(ImatIterBase.parameters)s

    Yields
    ------
    pd.Series
        Every iteration returns a pandas Series of
        `ReactionActivity` describing the activity of reactions
        in the iMAT Model.

    Notes
    -----
    %(ImatIterBase.notes)s

    References
    ----------
    %(ImatIterBase.references)s
    """

    def __next__(self) -> pd.Series[ReactionActivity]:  # type: ignore ## Checked on import earlier
        # Call the base classes iter_update method to update the iter state
        self._iter_next_start()
        # Get the binary series of reaction activities to return
        rxn_activities = self._get_binary_variables_state()
        # Add the integer cut constraint
        self._iter_next_end()
        # Return the reaction activities
        return rxn_activities


# endregion Reaction Activities iMAT Iterator


# region iMAT Model Iterator
@docs.dedent
class ImatIterModels(ImatIterBase):
    """Iterator for stepping through different possible iMAT solutions, returning an updated model for each
    iMAT solution, with modified reaction bounds to make it conform to the iMAT solution.

    Parameters
    ----------
    %(ImatIterBase.parameters)s
    model_method : {"simple", "subset"}
        Which method to use to create the returned iMAT model, can be
        either 'simple', or 'subset', see notes for details. *KEYWORD ONLY*

    Yields
    ------
    cobra.Model
        Every iteration returns a cobra Model which has been
        updated based on the current iterations iMAT solution,
        see Notes for more details

    Notes
    -----
    %(ImatIterBase.notes)s

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

    References
    ----------
    %(ImatIterBase.references)s
    """

    def __init__(
        self,
        *args,
        model_method: Literal["subset", "simple"] = "simple",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._method = model_method

    def __next__(self) -> cobra.Model:
        # Call the base classes iter_update method to update the iter state
        self._iter_next_start()
        # Create the model to be returned
        updated_model = self.in_model.copy()
        assert isinstance(updated_model, cobra.Model)
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
            assert isinstance(reaction, cobra.Reaction)
            reaction.bounds = model_creation._inactive_bounds(
                reaction.lower_bound,
                reaction.upper_bound,
                self._threshold,
            )
        if self._method == "simple":
            for rxn in active_forward_reactions:
                reaction = updated_model.reactions.get_by_id(rxn)
                assert isinstance(reaction, cobra.Reaction)
                reaction.bounds = model_creation._active_bounds(
                    reaction.lower_bound,
                    reaction.upper_bound,
                    self._epsilon,
                    forward=True,
                )
            for rxn in active_reverse_reactions:
                reaction = updated_model.reactions.get_by_id(rxn)
                assert isinstance(reaction, cobra.Reaction)
                reaction.bounds = model_creation._active_bounds(
                    reaction.lower_bound,
                    reaction.upper_bound,
                    self._epsilon,
                    forward=False,
                )
        # Add the integer cut constraint
        self._iter_next_end()
        # Return the updated_model
        return updated_model


# endregion iMAT Model Iterator


# region Main iMAT Iterator
# This class is for convenient dispatch to the different underlying iterators
@docs.dedent
class ImatIter:
    """
    Iterator for stepping through different possible iMAT solutions

    Parameters
    ----------
    %(ImatIterBase.parameters)s
    output : {'model', 'binary-variables', 'reaction-activity'}
        The output desired for each iteration, see the
        iterators in the `See Also` section for more details. *KEYWORD ONLY*

    See Also
    --------
    ImatIterBinaryVariables : Iterator yielding state of the binary iMAT variables
    ImatIterModel : Iterator yielding models modified to reflect iMAT solution
    ImatIterReactionActivities : Iterator yielding the activities of the reactions in the model

    Notes
    -----
    %(ImatIterBase.notes)s

    References
    ----------
    %(ImatIterBase.references)s
    """

    def __init__(
        self,
        *args,
        output: Literal[
            "model", "binary-variables", "reaction-activity"
        ] = "model",
        **kwargs,
    ):
        if output == "model":
            iter_class = ImatIterModels
        elif output == "binary-variables":
            iter_class = ImatIterBinaryVariables
        elif output == "reaction-activity":
            iter_class = ImatIterReactionActivities
        else:
            raise ValueError(
                f'output parameter must be one of "model", "binary-variables", "reaction-activity", '
                f"but {output} was received instead"
            )
        self._iterator = iter_class(*args, **kwargs)

    def __iter__(self):
        return self._iterator


# endregion Main iMAT Iterator


# region Iterative Sampling


@docs.dedent
def imat_iter_flux_sample(
    *args,
    n_samples: int = 1_000,
    sampler: Optional[type[cobra.sampling.HRSampler]] = None,
    sampler_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> pd.DataFrame[float]:
    """
    Generate a flux sample from a Model by iterating over multiple optimal (or near-optimal depending on
    objective tolerance) iMAT solutions, and sampling from each

    Parameters
    ----------
    %(ImatIterBase.parameters)s
    n_samples : int, default=1000
        Number of samples to generate from *each* iMAT model, (so the total
        number of samples will depend on how many iMAT models are generated).
        *KEYWORD ONLY*
    sampler : cobra.sampling.HRSampler class, default=cobra.sampling.OptGPSampler
        The sampling class to use for generating flux samples from
        each iMAT model, defaults to OptGPSampler. *KEYWORD ONLY*
    sampler_kwargs : dict of str to Any, optional
        Keyword arguments to pass to the sampler's __init__ method. *KEYWORD ONLY*
    kwargs
        Keyword arguments are passed to the `ImatIterModels`
        iterator class

    Returns
    -------
    pd.DataFrame of float
        Flux samples from the iMAT updated models, columns
        are indexed by the model's reaction ids, each row represents
        a flux sample

    See Also
    --------
    ImatIterModels : Underlying iterator used to iterate through the iMAT models

    Notes
    -----
    %(ImatIterBase.notes)s

    References
    ----------
    %(ImatIterBase.references)s
    """
    # Create a list to hold the results
    flux_sample_df_list = []
    # Set up the sampler if needed
    if sampler is None:
        sampler = cobra.sampling.OptGPSampler
    if sampler_kwargs is None:
        sampler_kwargs = {}
    # Iterate through the iMAT updated models
    for updated_model in ImatIterModels(*args, **kwargs):
        # Create the sampler
        imat_sampler = sampler(model=updated_model, **sampler_kwargs)
        # Sample from the updated model
        flux_samples = imat_sampler.sample(n=n_samples)
        # Validate the flux samples
        valid_flux_samples = flux_samples[
            imat_sampler.validate(flux_samples) == "v"  # type: ignore
        ]
        # Add the valid samples to the results list
        flux_sample_df_list.append(valid_flux_samples)
    # Return the combined flux samples
    return pd.concat(flux_sample_df_list, axis=0, ignore_index=True)


# endregion Iterative Sampling


# region Consensus Essentiality
@docs.dedent
def imat_iter_essential(
    *args,
    essential_proportion: float = 0.1,
    processes: Optional[int] = None,
    progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Iterate through iMAT solutions to identify which genes are essential
    for each iMAT model.

    Parameters
    ----------
    %(ImatIterBase.parameters)s
    essential_proportion : float, default=0.1
        Minimal objective flux to be considered viable, in terms of
        proportion of the pre-gene-KO maximum objective flux. That is
        a value of 0.1 indicates that if genes whose knockout results
        in a maximum objective flux less than 10% of the pre-knockout
        level are considered essential. *KEYWORD ONLY*
    processes : int, optional
        Number of parallel processes to use for finding the essential genes. *KEYWORD ONLY*
    progress_bar : bool, default=False
        Whether a progress bar is desired. *KEYWORD ONLY*
    kwargs
        Keyword arguments are passed to `ImatIterModels`

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with a column for each gene in the
        model, each row representing a different iMAT model,
        and boolean indicating if a gene was essential in
        a given iMAT model.
    """
    # Extract the model from args/kwargs
    if "model" in kwargs:
        model = kwargs["model"]
    else:
        model = args[0]
    assert isinstance(model, cobra.Model)
    # Determine the max_iter
    if "max_iter" in kwargs:
        max_iter = kwargs["max_iter"]
    elif len(args) >= 4:
        max_iter = args[3]
    else:
        max_iter = IMAT_DEFAULTS.max_iter
    # Find the genes in the model to construct the
    # results dataframe
    model_gene_list = model.genes.list_attr("id")
    result_df = pd.DataFrame(index=pd.Index(model_gene_list), dtype="boolean")
    # NOTE: Catching these may be unnecesary, not sure
    # if the fragmented data warning occurs with writing columns instead of rows
    # but this will definitely not be the performance bottleneck
    with warnings.catch_warnings():
        warnings.simplefilter(
            action="ignore", category=pd.errors.PerformanceWarning
        )
        # Iterate through the iMAT models
        for idx, imat_model in enumerate(
            tqdm.tqdm(
                ImatIterModels(*args, **kwargs),
                disable=not progress_bar,
                total=max_iter,
            )
        ):
            try:
                # Use COBRApy to find the essential genes for each Model
                ess_genes = [
                    g.id
                    for g in cobra.flux_analysis.variability.find_essential_genes(
                        imat_model,
                        threshold=essential_proportion
                        * imat_model.slim_optimize(),
                        processes=processes,
                    )
                ]
            except OptimizationError as e:
                # If there is a problem with the optimization,
                # just stop and return what we have so far
                # (There might be issues with numerical instability
                # given the number of integer constraints that will
                # be added)
                warnings.warn(
                    f"Optimization Error occured while finding essential genes: {e}"
                )
                break
            result_df[idx] = False
            result_df.loc[ess_genes, idx] = True
    # NOTE: We start with the transpose since writing the results
    # as columns is more efficient
    return result_df.T


@docs.dedent
def consensus_essentiality(
    *args, consensus_method: float = 0.5, **kwargs
) -> pd.Series:
    """
    Iterate through iMAT solutions to identify gene essentiality based
    on consensus of the iMAT models

    Parameters
    ----------
    %(ImatIterBase.parameters)s
    consensus_method : float or str, default=0.5
        The method for determining whether the gene is considered
        essential. A str value of 'all' indicates that genes will only
        be considered essential if it was essential for all the iMAT models.
        A str value of 'any' indicates that if the gene was essential
        in any of the iMAT models it will be considered essential.
        Floats are treated as the proportion of iMAT models in which
        the gene is essential for it to be considered essential by the
        consensus method. For example a value of 0.6 indicates that for
        a gene to be considered essential, it must have been essential in
        at least 60% of the iMAT models. *KEYWORD ONLY*
    kwargs
        Keyword arguments are passed to `imat_iter_essential`

    Returns
    -------
    pd.Series
        A boolean series, indexed by gene id. True indicates the gene
        is essential based on the consensus essentiality approach,
        False indicates it is not considered essential.

    Notes
    -----
    This is basically just a wrapper around `imat_iter_essential`
    adding a final aggregation step after finding the essential
    genes for each iMAT model. If a different aggregation
    method is desired, you can use that function directly
    and then aggregate essentiality across the returned dataframe.
    """
    iter_essential = imat_iter_essential(*args, **kwargs)
    if isinstance(consensus_method, float):
        return iter_essential.sum(axis=0) > consensus_method
    if consensus_method == "any":
        return iter_essential.any(axis=0)
    elif consensus_method == "all":
        return iter_essential.all(axis=0)
    raise ValueError(
        f"Unrecognized consensus_method, value should be 'all', 'any' or a float but received {consensus_method}"
    )


# endregion Consensus Essentiality

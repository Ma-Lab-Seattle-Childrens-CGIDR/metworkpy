"""
Module Implementing the Metchange Algorithm
"""
# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Iterable

# External Imports
import cobra
import numpy as np
import pandas as pd

# Local Imports


# region Metchange

def metchange(model: cobra.Model,
              metabolites: Iterable[str],
              reaction_weights: dict[str, float] | pd.Series,
              proportion: float
              ) -> pd.Series:
    """
    Use the Metchange algorithm to find the inconsistency scores for a set of
    metabolites based on reaction weights.

    :param model: Cobra model to use for performing the metchange algorithm
    :type model: cobra.Model
    :param metabolites: Metabolites to calculate consistency scores for
    :type metabolites: Iterable[str]
    :param reaction_weights: Weights for the reactions in the model, should
        correspond to the probability that reaction should not be active.
    :type reaction_weights: dict[str, float] | pd.Series
    :param proportion: Proportion of the maximum to constrain the metabolite
        production to be above.
    :type proportion: float
    :return: Series of inconsistency scores for all the `metabolites`
    :rtype: pd.Series

    .. note:
       This algorithm seeks to find an inconsistency score for metabolites based
       on gene expression. The gene expression is represented by reaction weights,
       which can be calculated by combining
       `metworkpy.metabolites.metchange_functions.expr_to_metchange_gene_weights`_ and
       `metworkpy.parse.gpr.gene_to_rxn_weights`_ . The algorithm calculates the
       inconsistency score through a two part optimization. First, for a given
       metabolite, the maximum metabolite production is found. Then the metabolite
       production is constrained to stay above proportion*maximum metabolite production,
       and the inner product of reaction weights, and reaction fluxes is minimized.
       This minimum inner product is the inconsistency score.
    """
    if isinstance(reaction_weights, dict):
        reaction_weights = pd.Series(reaction_weights)
    res_series = pd.Series(np.nan, index=metabolites)
    for metabolite in metabolites:
        with MetchangeObjectiveConstraint(model=model,
                                          metabolite=metabolite,
                                          reaction_weights=reaction_weights,
                                          proportion=proportion) as m:
            res_series[metabolite] = m.slim_optimize()
    return res_series


# endregion Metchange

# region Context Manager

class MetchangeObjectiveConstraint:
    """
    Context Manager for creating the Metchange objective

    :param model: Cobra model to add metchange objective to
    :type model: cobra.Model
    :param metabolite: String ID of metabolite to add metchange objective for
    :type metabolite: str
    :param reaction_weights: Weights for each reaction, representing the probability
        of that reaction being missing. A lower value indicates that the reaction is
        more likely to be present.
    :type reaction_weights: pd.Series
    :param proportion: Proportion of maximum metabolite production required as a
        constraint during the second optimization, where the inner product of
        reaction weights and reaction fluxes is minimized.
    :type proportion: float
    """

    def __init__(self, model: cobra.Model,
                 metabolite: str,
                 reaction_weights: pd.Series,
                 proportion: float):
        self.added_sink = f"tmp_{metabolite}_sink"
        self.metabolite = metabolite
        self.model = model
        self.rxn_weights = reaction_weights
        self.proportion = proportion

    def __enter__(self):
        self.original_objective = self.model.objective
        self.original_objective_direction = self.model.objective_direction
        met_sink_reaction = cobra.Reaction(id=self.added_sink,
                                           name=f"Temporary {self.metabolite} sink",
                                           lower_bound=0.)
        met_sink_reaction.add_metabolites({
            self.model.metabolites.get_by_id(self.metabolite): -1
        })
        self.model.add_reactions([met_sink_reaction])
        self.model.objective = self.added_sink
        self.model.objective_direction = 'max'
        obj_max = self.model.slim_optimize()
        self.model.reactions.get_by_id(self.added_sink).lower_bound = (obj_max *
                                                                       self.proportion)
        self.model.objective = {self.model.reactions.get_by_id(rxn): weight for
                                rxn, weight in self.rxn_weights.items()}
        self.model.objective_direction = "min"
        return self.model

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.model.objective = self.original_objective
        self.model.objective_direction = self.original_objective_direction
        self.model.remove_reactions(self.added_sink)


# endregion Context Manager

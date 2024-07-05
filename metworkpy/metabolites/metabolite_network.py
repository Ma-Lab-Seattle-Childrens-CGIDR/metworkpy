"""
Module with functions for finding metabolite networks in cobra Models
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Literal

# External Imports
import cobra
import pandas as pd
from tqdm import tqdm

# Local Imports
from metworkpy.utils import reaction_to_gene_df


# region Main Functions


def find_metabolite_network_reactions(
    model: cobra.Model,
    method: Literal["pfba", "essential"] = "pfba",
    pfba_proportion: float = 0.95,
    essential_proportion: float = 0.05,
    progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame[bool | float]:
    """
    Find which reactions are used to generate each metabolite in the model

    :param model: Cobra Model used to find which reactions are associated with which
        metabolite
    :type model: cobra.Model
    :param method: Which method to use to associate reactions with metabolites. Either

        1. 'pfba'(default):
            Use parsimonious flux analysis with the metabolite as the
            objective to find reaction-metabolite associations.
            Each reaction is associated with a flux for generating a particular
            metabolite.
        2. 'essential':
            Use essentiality to find reaction-metabolite associations.
            Find which reactions are essential for each metabolite.
    :type method: Literal["pfba", "essential"]
    :param pfba_proportion: Proportion to use for pfba analysis. This represents the
        fraction of optimum constraint applied before minimizing the sum of fluxes
        during pFBA.
    :type pfba_proportion: float
    :param essential_proportion: Proportion to use for essentiality, gene knockouts
        which result in an objective function value less than
        `essential_proportion * maximum_objective` are considered
        essential.
    :type essential_proportion: float
    :param progress_bar: Whether a progress bar should be displayed
    :type progress_bar: bool
    :param kwargs: Keyword arguments passed to `cobra.flux_analysis.variability.find_essential_genes`,
        or to `cobra.flux_analysis.pfba` depending on the chosen method.
    :type kwargs: dict
    :return: A dataframe with reactions as the index and metabolites as the columns,
        containing either

        1. Flux values if pfba is used.
           For a given reaction and metabolite,
           this represents the reaction flux found during pFBA required to maximally
           produce the metabolite.
        2. Boolean values if essentiality is used. For a given reaction and metabolite,
           this represents whether the reaction is essential for producing the
           metabolite.
    :rtype: pd.DataFrame[bool|float]

    .. seealso:
       | :func: `find_metabolite_network_reactions` for equivalent method with reactions
    """
    if method == "pfba":
        res_dtype = "float"
    elif method == "essential":
        res_dtype = "bool"
    else:
        raise ValueError(
            f"Method must be 'pfba' or 'essential' but received" f"{method}"
        )
    res_df = pd.DataFrame(
        None,
        columns=model.metabolites.list_attr("id"),
        index=model.reactions.list_attr("id"),
        dtype=res_dtype,
    )
    for metabolite in tqdm(res_df.columns, disable=not progress_bar):
        with MetaboliteObjective(model=model, metabolite=metabolite) as m:
            if method == "essential":
                ess_rxns = [
                    rxn.id
                    for rxn in (
                        cobra.flux_analysis.variability.find_essential_reactions(
                            model=m,
                            threshold=essential_proportion * m.slim_optimize(),
                            **kwargs,
                        )
                    )
                    if rxn.id != f"tmp_{metabolite}_sink_rxn"
                ]
                res_df.loc[ess_rxns, metabolite] = True
                res_df.loc[~res_df.index.isin(ess_rxns), metabolite] = False
            elif method == "pfba":
                pfba_sol = (
                    cobra.flux_analysis.pfba(
                        model=m,
                        objective=m.objective,
                        fraction_of_optimum=pfba_proportion,
                        **kwargs,
                    )
                ).fluxes
                pfba_sol.drop(f"tmp_{metabolite}_sink_rxn", inplace=True)
                res_df.loc[pfba_sol.index, metabolite] = pfba_sol
            else:
                raise ValueError(
                    f"Method must be 'pfba' or 'essential' but received" f"{method}"
                )
    return res_df


def find_metabolite_network_genes(
    model: cobra.Model,
    method: Literal["pfba", "essential"] = "pfba",
    pfba_proportion: float = 0.95,
    essential_proportion: float = 0.05,
    progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame[bool | float]:
    """
    Find which genes are used to generate each metabolite in the model

    :param model: Cobra Model used to find which genes are associated with which
        metabolite
    :type model: cobra.Model
    :param method: Which method to use to associate genes with metabolites. Either

        1. 'pfba'(default):
            Use parsimonious flux analysis with the metabolite as the
            objective to find genes-metabolite associations.
            Each reaction is associated with a flux for generating a particular
            metabolite. This is then translated to genes by finding the maximal
            (in terms of absolute value)
            flux for a reaction associated with a particular gene.
        2. 'essential':
            Use essentiality to find gene-metabolite associations.
            Find which genes are essential for each metabolite.
    :type method: Literal["pfba", "essential"]
    :param pfba_proportion: Proportion to use for pfba analysis. This represents the
        fraction of optimum constraint applied before minimizing the sum of fluxes
        during pFBA.
    :type pfba_proportion: float
    :param essential_proportion: Proportion to use for essentiality, gene knockouts
        which result in an objective function value less than
        `essential_proportion * maximum_objective` are considered
        essential.
    :type essential_proportion: float
    :param progress_bar: Whether to display a progress bar
    :type progress_bar: bool
    :param kwargs: Keyword arguments passed to `cobra.flux_analysis.variability.find_essential_genes`,
        or to `cobra.flux_analysis.pfba` depending on the chosen method.
    :type kwargs: dict
    :return: A dataframe with genes as the index and metabolites as the columns,
        containing either

        1. Flux values if pfba is used. For a given gene and metabolite,
           this represents the maximum of reaction fluxes associated with a gene,
           found during pFBA required to maximally produce the metabolite.
        2. Boolean values if essentiality is used. For a given reaction and metabolite,
           this represents whether the reaction is essential for producing the
           metabolite.
    :rtype: pd.DataFrame[bool|float]

    .. note:
       For converting from the reaction fluxes to gene fluxes, the gene is assigned
       a value corresponding to the maximum magnitude flux the gene is associated
       with (but the value assigned keeps the sign). For example, if a gene was
       associated with reactions which had parsimonious flux values of -10, and 1 the
       gene would be assigned a value of -10.

    .. seealso:
       | :func: `find_metabolite_network_reactions` for equivalent method with reactions
    """
    if method == "pfba":
        res_dtype = "float"
    elif method == "essential":
        res_dtype = "bool"
    else:
        raise ValueError(
            f"Method must be 'pfba' or 'essential' but received" f"{method}"
        )
    res_df = pd.DataFrame(
        None,
        columns=model.metabolites.list_attr("id"),
        index=model.genes.list_attr("id"),
        dtype=res_dtype,
    )
    for metabolite in tqdm(res_df.columns, disable=not progress_bar):
        with MetaboliteObjective(model=model, metabolite=metabolite) as m:
            if method == "essential":
                ess_genes = [
                    gene.id
                    for gene in (
                        cobra.flux_analysis.variability.find_essential_genes(
                            model=m,
                            threshold=essential_proportion * m.slim_optimize(),
                            **kwargs,
                        )
                    )
                ]
                res_df.loc[ess_genes, metabolite] = True
                res_df.loc[~res_df.index.isin(ess_genes), metabolite] = False
            elif method == "pfba":
                pfba_sol = (
                    cobra.flux_analysis.pfba(
                        model=m, fraction_of_optimum=pfba_proportion, **kwargs
                    )
                ).fluxes
                pfba_sol.name = "fluxes"
                gene_fluxes = reaction_to_gene_df(
                    model, pfba_sol.to_frame()
                ).reset_index()
                # Set the values of res_df such that the value reflects the
                # maximum value in terms of magnitude, but sign is maintained,
                # i.e. if a gene is
                gene_fluxes_max = gene_fluxes.groupby("genes").max()["fluxes"]
                gene_fluxes_min = gene_fluxes.groupby("genes").min()["fluxes"]
                res_df.loc[
                    gene_fluxes_max.abs() >= gene_fluxes_min.abs(), metabolite
                ] = gene_fluxes_max[gene_fluxes_max.abs() >= gene_fluxes_min.abs()]
                res_df.loc[
                    gene_fluxes_max.abs() < gene_fluxes_min.abs(), metabolite
                ] = gene_fluxes_min[gene_fluxes_max.abs() < gene_fluxes_min.abs()]
            else:
                raise ValueError(
                    f"Method must be 'pfba' or 'essential' but received" f"{method}"
                )
    return res_df


# endregion Main Functions

# region Context Manager


class MetaboliteObjective:
    """
    Context Manager for adding a metabolite sink reaction as the objective reaction

    :param model: Cobra model to add metabolite sink objective reaction to
    :type model: cobra.Model
    :param metabolite: Metabolite to create sink reaction objective function for
    :type metabolite: str
    """

    def __init__(self, model: cobra.Model, metabolite: str):
        self.metabolite = metabolite
        self.model = model

    def __enter__(self):
        self.rxn_added = f"tmp_{self.metabolite}_sink_rxn"
        self.original_objective = self.model.objective
        self.original_objective_direction = self.model.objective_direction
        met_sink_reaction = cobra.Reaction(
            id=self.rxn_added, name=f"Temporary {self.metabolite} sink", lower_bound=0.0
        )
        met_sink_reaction.add_metabolites(
            {self.model.metabolites.get_by_id(self.metabolite): -1}
        )
        self.model.add_reactions([met_sink_reaction])
        self.model.objective = self.rxn_added
        self.model.objective_direction = "max"
        return self.model

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.model.objective = self.original_objective
        self.model.objective_direction = self.original_objective_direction
        self.model.remove_reactions([self.model.reactions.get_by_id(self.rxn_added)])


# region Context Manager

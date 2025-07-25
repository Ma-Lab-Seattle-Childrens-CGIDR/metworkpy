"""Module with functions for finding metabolite networks in cobra Models"""

# Imports
from __future__ import annotations

# Standard Library Imports
import hashlib
from typing import Literal

# External Imports
import cobra
import pandas as pd
import sympy
from tqdm import tqdm

# Local Imports
from metworkpy.utils import reaction_to_gene_df, reaction_to_gene_list


# region Main Functions


def find_metabolite_synthesis_network_reactions(
    model: cobra.Model,
    method: Literal["pfba", "essential"] = "pfba",
    pfba_proportion: float = 0.95,
    essential_proportion: float = 0.05,
    progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame[bool | float]:
    """Find which reactions are used to generate each metabolite in the model

    Parameters
    ----------
    model : cobra.Model
        Cobra Model used to find which reactions are associated with
        which metabolite
    method : Literal["pfba", "essential"]
        Which method to use to associate reactions with metabolites.
        Either

        1. 'pfba'(default):
            Use parsimonious flux analysis with the metabolite as the
            objective to find reaction-metabolite associations.
            Each reaction is associated with a flux for generating a particular
            metabolite.
        2. 'essential':
            Use essentiality to find reaction-metabolite associations.
            Find which reactions are essential for each metabolite.

    pfba_proportion : float
        Proportion to use for pfba analysis. This represents the
        fraction of optimum constraint applied before minimizing the sum
        of fluxes during pFBA.
    essential_proportion : float
        Proportion to use for essentiality, gene knockouts which result
        in an objective function value less than `essential_proportion *
        maximum_objective` are considered essential.
    progress_bar : bool
        Whether a progress bar should be displayed
    **kwargs : dict
        Keyword arguments passed to
        `cobra.flux_analysis.variability.find_essential_genes`, or to
        `cobra.flux_analysis.pfba` depending on the chosen method.

    Returns
    -------
    pd.DataFrame[bool|float]
        A dataframe with reactions as the index and metabolites as the
        columns, containing either

        1. Flux values if pfba is used.
           For a given reaction and metabolite,
           this represents the reaction flux found during pFBA required to maximally
           produce the metabolite.
        2. Boolean values if essentiality is used. For a given reaction and metabolite,
           this represents whether the reaction is essential for producing the
           metabolite.

    See Also
    --------
    find_metabolite_synthesis_network_genes : Equivalent method with genes
    """
    if method == "pfba":
        res_dtype = "float"
    elif method == "essential":
        res_dtype = "bool"
    else:
        raise ValueError(f"Method must be 'pfba' or 'essential' but received{method}")
    res_df = pd.DataFrame(
        None,
        columns=model.metabolites.list_attr("id"),
        index=model.reactions.list_attr("id"),
        dtype=res_dtype,
    )
    for metabolite in tqdm(res_df.columns, disable=not progress_bar):
        with model as m:
            metabolite_sink_reaction_id = add_metabolite_objective_(m, metabolite)
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
                    if rxn.id != metabolite_sink_reaction_id
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
                pfba_sol.drop(metabolite_sink_reaction_id, inplace=True)
                res_df.loc[pfba_sol.index, metabolite] = pfba_sol
            else:
                raise ValueError(
                    f"Method must be 'pfba' or 'essential' but received {method}"
                )
    return res_df


def find_metabolite_synthesis_network_genes(
    model: cobra.Model,
    method: Literal["pfba", "essential"] = "pfba",
    pfba_proportion: float = 0.95,
    essential_proportion: float = 0.05,
    progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame[bool | float]:
    """Find which genes are used to generate each metabolite in the model

    Parameters
    ----------
    model : cobra.Model
        Cobra Model used to find which genes are associated with which
        metabolite
    method : Literal["pfba", "essential"]
        Which method to use to associate genes with metabolites.
        Either

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

    pfba_proportion : float
        Proportion to use for pfba analysis. This represents the
        fraction of optimum constraint applied before minimizing the sum
        of fluxes during pFBA.
    essential_proportion : float
        Proportion to use for essentiality, gene knockouts which result
        in an objective function value less than `essential_proportion *
        maximum_objective` are considered essential.
    progress_bar : bool
        Whether to display a progress bar
    **kwargs : dict
        Keyword arguments passed to
        `cobra.flux_analysis.variability.find_essential_genes`, or to
        `cobra.flux_analysis.pfba` depending on the chosen method.

    Returns
    -------
    pd.DataFrame[bool|float]
        A dataframe with genes as the index and metabolites as the
        columns, containing either

        1. Flux values if pfba is used. For a given gene and metabolite,
           this represents the maximum of reaction fluxes associated with a gene,
           found during pFBA required to maximally produce the metabolite.
        2. Boolean values if essentiality is used. For a given reaction and metabolite,
           this represents whether the reaction is essential for producing the
           metabolite.

    Notes
    -----
    For converting from the reaction fluxes to gene fluxes, the gene is assigned
    a value corresponding to the maximum magnitude flux the gene is associated
    with (but the value assigned keeps the sign). For example, if a gene was
    associated with reactions which had parsimonious flux values of -10, and 1 the
    gene would be assigned a value of -10.


    See Also
    --------
    find_metabolite_synthesis_network_reactions : Equivalent method with reactions
    """
    if method == "pfba":
        res_dtype = "float"
    elif method == "essential":
        res_dtype = "bool"
    else:
        raise ValueError(f"Method must be 'pfba' or 'essential' but received {method}")
    res_df = pd.DataFrame(
        None,
        columns=model.metabolites.list_attr("id"),
        index=model.genes.list_attr("id"),
        dtype=res_dtype,
    )
    for metabolite in tqdm(res_df.columns, disable=not progress_bar):
        with model as m:
            _ = add_metabolite_objective_(m, metabolite)
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
                    f"Method must be 'pfba' or 'essential' but received {method}"
                )
    return res_df


def find_metabolite_consuming_network_reactions(
    model: cobra.Model,
    reaction_proportion: float = 0.05,
    check_reverse: bool = True,
    progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame[bool]:
    """Find reactions which consume a metabolite, or its derivatives

    Parameters
    ----------
    model : cobra.Model
        Cobra Model used to find which reactions are associated with which metabolites
    reaction_proportion : float
        Proportion used to judge if a reaction consumes a metabolite or its derivatives,
        if the maximum flux for a reaction drops below reaction_proportion * maximum flux
        when a metabolite is forced into a sink psuedo-reaction then that reaction
        will be considered to be a consumer of a metabolite.
    check_reverse : bool, default=True
        Whether to check for metabolite consumption in reverse reactions,
        i.e. whether to check if a metabolite or its derivatives is a
        product of a reversible reaction
    progress_bar : bool
        Whether to display a progress bar
    kwargs
        Keyword arguments passed to `cobra.flux_analysis.variability.flux_variability_analysis`,
        which is used to find changes in maximal reaction flux.

    Returns
    -------
    metabolite_network : pd.DataFrame[bool]
        A dataframe with reactions as the index and metabolites as the columns,
        a True value indicates that a particular reaction consumes a metabolite
        or one of its derivatives
    """
    res_df = pd.DataFrame(
        False,
        columns=model.metabolites.list_attr("id"),
        index=model.reactions.list_attr("id"),
        dtype=bool,
    )
    for metabolite in tqdm(res_df.columns, disable=not progress_bar):
        with model as m:
            # Remove maintenance reactions to avoid issues with infeasibility
            eliminate_maintenance_requirements_(m)
            # Perform FVA for the model
            fva_results = cobra.flux_analysis.variability.flux_variability_analysis(
                m, fraction_of_optimum=0.0, **kwargs
            )
            # Add the absorbing reaction
            add_metabolite_absorb_reaction_(m, metabolite)
            fva_results_remove_metabolite = (
                cobra.flux_analysis.variability.flux_variability_analysis(
                    m, fraction_of_optimum=0.0, **kwargs
                )
            )
            # Now determine which reactions consume the metabolite
            for rxn in res_df.index:
                rxn_max = fva_results.loc[rxn, "maximum"]
                rxn_min = fva_results.loc[rxn, "minimum"]
                rxn_max_no_met = fva_results_remove_metabolite.loc[rxn, "maximum"]
                rxn_min_no_met = fva_results_remove_metabolite.loc[rxn, "minimum"]
                if (rxn_max > 0.0) and (rxn_max_no_met < rxn_max * reaction_proportion):
                    res_df.loc[rxn, metabolite] = True
                if (
                    check_reverse
                    and (rxn_min < 0.0)
                    and (rxn_min_no_met > rxn_min * reaction_proportion)
                ):
                    res_df.loc[rxn, metabolite] = True
    return res_df


def find_metabolite_consuming_network_genes(
    model: cobra.Model,
    reaction_proportion: float = 0.05,
    essential: bool = False,
    progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame[bool]:
    """
    Find genes associated with reactions which consume a metabolite or its derivatives

    Parameters
    ----------
    model : cobra.Model
        Cobra model used to find which reactions are association with which metabolites
    reaction_proportion: float
        Proportion used to judge if a reaction consumes a metabolite or its derivatives,
        if the maximum flux for a reaction drops below reaction_proportion * maximum flux
        when a metabolite is forced into a sink psuedo-reaction then that reaction
        will be considered to be a consumer of a metabolite.
    essential : bool
        Whether to only include genes which are essential for the reactions that consume the
        metabolite or its derivatives
    progres_bar : bool
        Whether to display a progress bar
    kwargs
        Keyword arguments passed to `cobra.flux_analysis.variability.flux_variability_analysis`,
        which is used to find changes in maximal reaction flux.

    Returns
    -------
    metabolite_network : pd.DataFrame[bool]
        A dataframe with genes as the index, and metabolites as the columns,
        a True value indicates that a particular gene is associated with a reaction
        that consumes a metabolite or one of its derivatives
    """
    res_df = pd.DataFrame(
        False,
        columns=model.metabolites.list_attr("id"),
        index=model.genes.list_attr("id"),
        dtype=bool,
    )
    metabolite_reaction_network = find_metabolite_consuming_network_reactions(
        model=model,
        reaction_proportion=reaction_proportion,
        progress_bar=progress_bar,
        **kwargs,
    )
    for metabolite in metabolite_reaction_network.columns:
        gene_list = reaction_to_gene_list(
            model=model,
            reaction_list=list(
                metabolite_reaction_network[
                    metabolite_reaction_network[metabolite]
                ].index
            ),
            essential=essential,
        )
        res_df.loc[gene_list, metabolite] = True
    return res_df


# endregion Main Functions

# region Helper Functions


def add_metabolite_objective_(model: cobra.Model, metabolite: str) -> str:
    """
    Adds a sink reaction for a metabolite, and sets it as the objective function

    Parameters
    ----------
    model : cobra.Model
        The model to update
    metabolite : str
        The id of the metabolite to set as the objective

    Returns
    -------
    reaction : str
        id of the added reaction

    Note
    ----
    If used within a model context everything this function alters
    will be reset upon leaving the context
    """
    metabolite_hash = hashlib.md5(metabolite.encode("utf-8")).hexdigest()[-8:]
    metabolite_sink_rxn_id = f"{metabolite}_sink_objective_{metabolite_hash}"
    metabolite_sink_rxn = cobra.Reaction(
        id=metabolite_sink_rxn_id, name=f"{metabolite} sink reaction", lower_bound=0.0
    )
    metabolite_sink_rxn.add_metabolites({model.metabolites.get_by_id(metabolite): -1.0})
    model.add_reactions([metabolite_sink_rxn])
    model.objective = metabolite_sink_rxn_id
    model.objective_direction = "max"
    return metabolite_sink_rxn_id


def add_metabolite_absorb_reaction_(
    model: cobra.Model,
    metabolite: str,
) -> str:
    """Add a reaction which consumes the metabolite, and constrain it to consume all
    the metabolite which is generated, stopping it from being used for any other reactions

    Parameters
    ----------
    model : cobra.Model
        The model to add the absorbing reaction to
    metabolite : str
        The metabolite id for the metabolite to add the absorbing reaction for

    Returns
    -------
    reaction_id : str
        The id of the added reaction
    """
    # Start by adding the sink reaction for the metabolite to the model
    met_to_absorb = model.metabolites.get_by_id(metabolite)
    # Create a name for the reaction (<metabolite>_absorb_<partial hash>)
    metabolite_hash = hashlib.md5(met_to_absorb.id.encode("utf-8")).hexdigest()[-8:]
    absorbing_reaction_id = f"{met_to_absorb.id}_absorb_reaction_{metabolite_hash}"
    absorbing_reaction = cobra.Reaction(
        id=absorbing_reaction_id,
        name=f"{met_to_absorb.id} Absorbing Reaction",
        lower_bound=0.0,
    )
    absorbing_reaction.add_metabolites(
        {
            met_to_absorb: -1,
        }
    )
    # Add the absorbing reaction to the model
    model.add_reactions([absorbing_reaction])
    # Now get a list consisting of optlang expressions (variable and coefficient)
    # which will be summed to equal the total amount of the metabolite being produced
    metabolite_gen_exprs = []
    for rxn in met_to_absorb.reactions:
        if rxn.id == absorbing_reaction_id:
            continue  # Don't want to add the absorbing reaction
        met_coef = rxn.metabolites[met_to_absorb]
        if met_coef > 0.0:
            # The metabolite is generated by the forward reaction
            forward_var = rxn.forward_variable
            if forward_var is None:
                raise ValueError(
                    "Metabolite is associated with reaction not found in model"
                )
            metabolite_gen_exprs.append(met_coef * forward_var)
        elif met_coef < 0.0:
            # The metabolite is generated by the reverse reaction
            reverse_var = rxn.reverse_variable
            if reverse_var is None:
                raise ValueError(
                    "Metabolite is associated with reaction not found in model"
                )
            metabolite_gen_exprs.append(abs(met_coef) * reverse_var)
        else:
            pass  # If the value is exactly 0.0, doesn't actually get generated
    # Create a constraint such that the sum of all the metabolite generated, minus the amount
    # being consumed by the absorbing reaction is 0
    absorbing_constraint_name = f"{met_to_absorb}_absorb_constraint_{metabolite_hash}"
    absorbing_constraint = model.problem.Constraint(
        sympy.Add(*metabolite_gen_exprs, -1 * absorbing_reaction.forward_variable),
        name=absorbing_constraint_name,
        lb=0.0,
        ub=0.0,  # Allows for the maintenance reactions to run
    )
    model.add_cons_vars(absorbing_constraint)
    return absorbing_reaction_id


def eliminate_maintenance_requirements_(model: cobra.Model):
    """
    Change bounds of maintenance reactions to remove maintenance requirements

    Parameters
    ----------
    model : cobra.Model
        Model to eliminate maintenance requirements from

    Note
    ----
    When used within a model context, all changes will be reversed on leaving the model context
    """
    for rxn in model.reactions:
        if rxn.lower_bound > 0.0:
            rxn.lower_bound = 0.0
        elif rxn.upper_bound < 0.0:
            rxn.upper_bound = 0.0
        else:
            pass


# endregion Helper Functions

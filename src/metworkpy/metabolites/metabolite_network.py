"""Module with functions for finding metabolite networks in cobra Models"""

# Imports
from __future__ import annotations

# Standard Library Imports
from collections.abc import Iterable
import hashlib
from typing import Literal, Optional
import warnings

# External Imports
import cobra
from cobra.exceptions import OptimizationError
import numpy as np
import pandas as pd
import sympy
from scipy import stats
from tqdm import tqdm

# Local Imports
from metworkpy.utils import (
    reaction_to_gene_list,
    get_reaction_to_gene_translation_dict,
    fisher_enrichment,
)


# region Main Functions
def find_metabolite_synthesis_network_reactions(
    model: cobra.Model,
    method: Literal["pfba", "gfba", "essential"] = "pfba",
    return_type: Literal["dict", "DataFrame"] = "DataFrame",
    metabolites: Optional[Iterable[str]] = None,
    add_sinks: bool = False,
    pfba_proportion: float = 0.95,
    essential_proportion: float = 0.05,
    progress_bar: bool = False,
    **kwargs,
) -> (
    pd.DataFrame[bool | float]
    | dict[str, list[str]]
    | dict[str, dict[str, float]]
):
    """Find which reactions are used to generate each metabolite in the model

    Parameters
    ----------
    model : cobra.Model
        Cobra Model used to find which reactions are associated with
        which metabolite
    method : Literal["pfba", "essential"]
        Which method to use to associate reactions with metabolites.
        Either

        #. 'pfba'(default):
            Use parsimonious flux analysis with the metabolite as the
            objective to find reaction-metabolite associations.
            Each reaction is associated with a flux for generating a particular
            metabolite.
        #. 'gfba'(default):
            Use geometric flux analysis with the metabolite as the
            objective to find reaction-metabolite associations.
            Each reaction is associated with a flux for generating a particular
            metabolite.
        #. 'essential':
            Use essentiality to find reaction-metabolite associations.
            Find which reactions are essential for each metabolite.

    return_type : {'DataFrame', 'dict'}, default='DataFrame'
        How to return the networks, either a dataframe or dict
        (see returns for more information).
    metabolites : iterable of str, optional
        Which metabolites to find the synthesis networks for, if not provided will
        find the networks for all the metabolites in the model
    add_sinks : bool, default=False
        Add sinks for all the metabolites in the model to allow for reactions
        to be active without all of their products having a final destination.
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
        `cobra.flux_analysis.variability.find_essential_genes`,
        `cobra.flux_analysis.geometric_fba`, or to
        `cobra.flux_analysis.pfba` depending on the chosen method.

    Returns
    -------
    pd.DataFrame[bool|float] or dict
        If `return_type` is 'DataFrame' (the default), returns
        a dataframe with reactions as the index and metabolites as the
        columns, containing either

        #. Flux values if pfba or gfba are used.
           For a given reaction and metabolite,
           this represents the reaction flux found during pFBA required to maximally
           produce the metabolite.
        #. Boolean values if essentiality is used. For a given reaction and metabolite,
           this represents whether the reaction is essential for producing the
           metabolite.

        If `return_type` is 'dict', returns a dict keyed by metabolite. If
        method is 'essential', then the values are lists of reaction ids which
        are required to produce the metabolite. If method is 'pfba', or 'gfba',
        the the values are another dict, keyed by reaction id, with values corresponding
        to the flux associated with that reaction in the parsimonious/geometric
        FBA solution when optimizing for the production of the metabolite.

    See Also
    --------
    find_metabolite_synthesis_network_genes : Equivalent method with genes
    """
    if method == "pfba" or method == "gfba":
        res_dtype = "float"
    elif method == "essential":
        res_dtype = "bool"
    else:
        raise ValueError(
            f"Method must be 'pfba', 'gfba', or 'essential' but received{method}"
        )
    if metabolites is None:
        metabolites = model.metabolites.list_attr("id")
    res_df = pd.DataFrame(
        None,
        columns=pd.Index(metabolites),
        index=model.reactions.list_attr("id"),
        dtype=res_dtype,
    )
    for metabolite in tqdm(res_df.columns, disable=not progress_bar):
        with model as m:
            metabolite_sink_reaction_id = add_metabolite_objective_(
                m, metabolite
            )
            if add_sinks:
                add_all_metabolite_sinks_(m)
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
            elif method == "pfba" or method == "gfba":
                if method == "pfba":
                    flux_series = (
                        cobra.flux_analysis.pfba(
                            model=m,
                            objective=m.objective,
                            fraction_of_optimum=pfba_proportion,
                            **kwargs,
                        )
                    ).fluxes
                elif method == "gfba":
                    try:
                        flux_series = (
                            cobra.flux_analysis.geometric.geometric_fba(
                                model=m, **kwargs
                            ).fluxes
                        )
                    except RuntimeError:
                        flux_series = pd.Series(np.nan, res_df.index)

                assert isinstance(flux_series, pd.Series), (
                    "Invalid return from COBRApy pfba or geometric_fba function"
                )
                flux_series.drop(metabolite_sink_reaction_id, inplace=True)
                res_df.loc[flux_series.index, metabolite] = flux_series
            else:
                raise ValueError(
                    f"Method must be 'pfba', 'gfba', or 'essential' but received {method}"
                )
    if return_type == "DataFrame":
        return res_df
    elif return_type == "dict":
        if method == "essential":
            return_dict = {}
            for metabolite, rxn_id_series in res_df.items():
                return_dict[metabolite] = list(
                    rxn_id_series[rxn_id_series].index
                )
            return return_dict
        elif method == "pfba" or method == "gfba":
            return res_df.to_dict()
        else:
            raise ValueError(
                f"Method must be 'pfba', 'gfba', or 'essential' but received {method}"
            )
    else:
        raise ValueError(
            f"Expected return type to be 'DataFrame', or 'dict' but received {return_type}"
        )


def find_metabolite_synthesis_network_genes(
    model: cobra.Model,
    method: Literal["pfba", "gfba", "essential"] = "pfba",
    return_type: Literal["DataFrame", "dict"] = "DataFrame",
    metabolites: Optional[Iterable[str]] = None,
    add_sinks: bool = False,
    pfba_proportion: float = 0.95,
    essential_proportion: float = 0.05,
    progress_bar: bool = False,
    essential: bool = False,
    **kwargs,
) -> pd.DataFrame[bool | float]:
    """Find which genes are used to generate each metabolite in the model

    Parameters
    ----------
    model : cobra.Model
        Cobra Model used to find which genes are associated with which
        metabolite
    method : Literal["pfba", "gfba", "essential"]
        Which method to use to associate genes with metabolites.
        Either

        #. 'pfba'(default):
            Use parsimonious flux analysis with the metabolite as the
            objective to find genes-metabolite associations.
            Each reaction is associated with a flux for generating a particular
            metabolite. This is then translated to genes by finding the maximal
            (in terms of absolute value)
            flux for a reaction associated with a particular gene.
        #. 'gfba'(default):
            Use geometric flux analysis with the metabolite as the
            objective to find genes-metabolite associations.
            Each reaction is associated with a flux for generating a particular
            metabolite. This is then translated to genes by finding the maximal
            (in terms of absolute value)
            flux for a reaction associated with a particular gene.
        #. 'essential':
            Use essentiality to find gene-metabolite associations.
            Find which genes are essential for each metabolite.


    return_type : {'DataFrame', 'dict'}, default='DataFrame'
        How to return the networks, either a dataframe or dict
        (see returns for more information).
    metabolites : iterable of str, optional
        Which metabolites to find the synthesis networks for, if not provided will
        find the networks for all the metabolites in the model
    add_sinks : bool, default=False
        Add sinks for all the metabolites in the model to allow for reactions
        to be active without all of their products having a final destination.
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
    essential : bool
        When translating from reactions to genes to get the
        gene metabolite network for the pFBA method, whether
        to only include genes which are essential for a reaction
        in the genes associated with said reaction.
    kwargs : dict
        Keyword arguments passed to
        `cobra.flux_analysis.variability.find_essential_genes`,
        `cobra.flux_analysis.geometric_fba` or to
        `cobra.flux_analysis.pfba` depending on the chosen method.

    Returns
    -------
    pd.DataFrame[bool|float] or dict
        If `return_type` is 'DataFrame' (the default), returns
        a dataframe with genes as the index and metabolites as the
        columns, containing either

        #. Flux values if pfba or gfba is used. For a given gene and metabolite,
           this represents the maximum of reaction fluxes associated with a gene,
           found during parsimoniou/geometric FBA required to maximally produce the
           metabolite.
        #. Boolean values if essentiality is used. For a given reaction and metabolite,
           this represents whether the reaction is essential for producing the
           metabolite.

        If `return_type` is 'dict', returns a dict keyed by metabolite. If
        method is 'essential', then the values are lists of gene ids which
        are required to produce the metabolite. If method is 'pfba', or 'gfba',
        the the values are another dict, keyed by gene id, with values
        representing the maximum of reaction fluxes associated with a gene,
        found during parsimonious/geometric FBA required to maximally
        produce the metabolite.

        with values corresponding
        to the flux associated with that reaction in the parsimonious/geometric
        FBA solution when optimizing for the production of the metabolite.

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
    if method == "pfba" or method == "gfba":
        res_dtype = "float"
    elif method == "essential":
        res_dtype = "bool"
    else:
        raise ValueError(
            f"Method must be 'pfba' or 'essential' but received {method}"
        )
    if metabolites is None:
        metabolites = model.metabolites.list_attr("id")
    res_df = pd.DataFrame(
        None,
        columns=pd.Index(metabolites),
        index=model.genes.list_attr("id"),
        dtype=res_dtype,
    )
    for metabolite in tqdm(res_df.columns, disable=not progress_bar):
        with model as m:
            _ = add_metabolite_objective_(m, metabolite)
            if add_sinks:
                add_all_metabolite_sinks_(m)
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
            elif method == "pfba" or method == "gfba":
                if method == "pfba":
                    flux_series = (
                        cobra.flux_analysis.pfba(
                            model=m,
                            fraction_of_optimum=pfba_proportion,
                            **kwargs,
                        )
                    ).fluxes
                elif method == "gfba":
                    flux_series = cobra.flux_analysis.geometric_fba(
                        model=m, **kwargs
                    ).fluxes
                flux_series.name = "fluxes"
                # Create a dataframe indexed by reaction, with a column for genes
                rxn_to_gene_frame = (
                    pd.Series(
                        get_reaction_to_gene_translation_dict(
                            model=model, essential=essential
                        ),
                    )
                    .explode()
                    .to_frame(name="gene")
                )
                pfba_frame = flux_series.to_frame(name="fluxes")
                gene_fluxes = pfba_frame.merge(
                    rxn_to_gene_frame,
                    how="left",
                    left_index=True,
                    right_index=True,
                ).reset_index(drop=True)
                # Set the values of res_df such that the value reflects the
                # maximum value in terms of magnitude, but sign is maintained,
                gene_fluxes_max = gene_fluxes.groupby("gene").max()["fluxes"]
                gene_fluxes_min = gene_fluxes.groupby("gene").min()["fluxes"]
                res_df.loc[
                    gene_fluxes_max.abs() >= gene_fluxes_min.abs(), metabolite
                ] = gene_fluxes_max[
                    gene_fluxes_max.abs() >= gene_fluxes_min.abs()
                ]
                res_df.loc[
                    gene_fluxes_max.abs() < gene_fluxes_min.abs(), metabolite
                ] = gene_fluxes_min[
                    gene_fluxes_max.abs() < gene_fluxes_min.abs()
                ]
            else:
                raise ValueError(
                    f"Method must be 'pfba', 'gfba', or 'essential' but received {method}"
                )
    if return_type == "DataFrame":
        return res_df
    elif return_type == "dict":
        if method == "essential":
            return_dict = {}
            for metabolite, gene_id_series in res_df.items():
                return_dict[metabolite] = list(
                    gene_id_series[gene_id_series].index
                )
            return return_dict
        elif method == "pfba" or method == "gfba":
            return res_df.to_dict()
        else:
            raise ValueError(
                f"Method must be 'pfba', 'gfba', or 'essential' but received {method}"
            )
    else:
        raise ValueError(
            f"Expected return type to be 'DataFrame', or 'dict' but received {return_type}"
        )


def find_metabolite_consuming_network_reactions(
    model: cobra.Model,
    metabolites: Optional[Iterable[str]] = None,
    return_type: Literal["DataFrame", "dict"] = "DataFrame",
    reaction_proportion: float = 0.05,
    add_sinks: bool = False,
    check_reverse: bool = True,
    progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame[bool]:
    """Find reactions which consume a metabolite, or its derivatives

    Parameters
    ----------
    model : cobra.Model
        Cobra Model used to find which reactions are associated with which metabolites
    metabolites : iterable of str, optional
        Which metabolites to find the consuming networks for, if not provided will
        find the networks for all the metabolites in the model
    return_type : {'DataFrame', 'dict'}, default='DataFrame'
        How to return the networks, either a dataframe or dict
        (see returns for more information).
    reaction_proportion : float
        Proportion used to judge if a reaction consumes a metabolite or its derivatives,
        if the maximum flux for a reaction drops below reaction_proportion * maximum flux
        when a metabolite is forced into a sink psuedo-reaction then that reaction
        will be considered to be a consumer of a metabolite.
    add_sinks : bool, default=False
        Whether to add sinks for all metabolites in the model. This stops
        reactions from being included in consumption networks when they
        don't consume the metabolite, but their metabolite products don't have
        anywhere to go when finding the consumption network for another metabolite.
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
    metabolite_network : pd.DataFrame[bool] or dict
        If `return_type` is 'DataFrame' (the default), returns a dataframe with
        reactions as the index and metabolites as the columns, a True value
        indicates that a particular reaction consumes a metabolite or one
        of its derivatives.

        If `return_type` is 'dict', returns the network as a dict instead. The
        dictionary keyed by metabolite id, with values that are lists of
        the ids of reactions which consume a metabolite or its derivatives.
    """
    if metabolites is None:
        metabolites = model.metabolites.list_attr("id")
    res_df = pd.DataFrame(
        False,
        columns=pd.Index(metabolites),
        index=model.reactions.list_attr("id"),
        dtype=bool,
    )
    with model as m:
        # Remove maintenance reactions to avoid issues with infeasibility
        eliminate_maintenance_requirements_(m)
        if add_sinks:
            add_all_metabolite_sinks_(m)
        # Perform FVA for the model
        fva_results = (
            cobra.flux_analysis.variability.flux_variability_analysis(
                m, fraction_of_optimum=0.0, **kwargs
            )
        )
    for metabolite in tqdm(res_df.columns, disable=not progress_bar):
        with model as m:
            # Remove maintenance reactions to avoid issues with infeasibility
            eliminate_maintenance_requirements_(m)
            # Add all metabolite sinks if desired
            if add_sinks:
                add_all_metabolite_sinks_(m)
            # Add the absorbing reaction
            add_metabolite_absorb_reaction_(m, metabolite)
            try:
                fva_results_remove_metabolite = (
                    cobra.flux_analysis.variability.flux_variability_analysis(
                        m, fraction_of_optimum=0.0, **kwargs
                    )
                )
            except OptimizationError:
                warnings.warn(
                    f"Optimization error occurred when finding consuming reactions "
                    f"for metabolite {metabolite}, no reactions will be marked as consuming "
                    f"this metabolite."
                )
                continue
            # Now determine which reactions consume the metabolite
            for rxn in res_df.index:
                rxn_max = fva_results.loc[rxn, "maximum"]
                rxn_min = fva_results.loc[rxn, "minimum"]
                rxn_max_no_met = fva_results_remove_metabolite.loc[
                    rxn, "maximum"
                ]
                rxn_min_no_met = fva_results_remove_metabolite.loc[
                    rxn, "minimum"
                ]
                if (rxn_max > 0.0) and (
                    rxn_max_no_met < rxn_max * reaction_proportion
                ):
                    res_df.loc[rxn, metabolite] = True
                if (
                    check_reverse
                    and (rxn_min < 0.0)
                    and (rxn_min_no_met > rxn_min * reaction_proportion)
                ):
                    res_df.loc[rxn, metabolite] = True
    if return_type == "DataFrame":
        return res_df
    elif return_type == "dict":
        return {m: list(rs[rs].index) for m, rs in res_df.items()}
    else:
        raise ValueError(
            f"Expected 'DataFrame' or 'dict' as 'return_type', received {return_type}"
        )


def find_metabolite_consuming_network_genes(
    model: cobra.Model,
    metabolites: Optional[Iterable[str]] = None,
    return_type: Literal["DataFrame", "dict"] = "DataFrame",
    reaction_proportion: float = 0.05,
    add_sinks: bool = False,
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
    metabolites : iterable of str, optional
        Which metabolites to find the consuming networks for, if not provided will
        find the networks for all the metabolites in the model
    return_type : {'DataFrame', 'dict'}, default='DataFrame'
        How to return the networks, either a dataframe or dict
        (see returns for more information).
    reaction_proportion: float
        Proportion used to judge if a reaction consumes a metabolite or its derivatives,
        if the maximum flux for a reaction drops below reaction_proportion * maximum flux
        when a metabolite is forced into a sink psuedo-reaction then that reaction
        will be considered to be a consumer of a metabolite.
    add_sinks : bool, default=False
        Whether to add sinks for all metabolites in the model. This stops
        reactions from being included in consumption networks when they
        don't consume the metabolite, but their metabolite products don't have
        anywhere to go when finding the consumption network for another metabolite.
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
    if metabolites is None:
        metabolites = model.metabolites.list_attr("id")
    res_df = pd.DataFrame(
        False,
        columns=pd.Index(metabolites),
        index=model.genes.list_attr("id"),
        dtype=bool,
    )
    metabolite_reaction_network = find_metabolite_consuming_network_reactions(
        model=model,
        reaction_proportion=reaction_proportion,
        progress_bar=progress_bar,
        add_sinks=add_sinks,
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

    if return_type == "DataFrame":
        return res_df
    elif return_type == "dict":
        return {m: list(gs[gs].index) for m, gs in res_df.items()}
    else:
        raise ValueError(
            f"Expected 'DataFrame' or 'dict' as 'return_type', received {return_type}"
        )
    return res_df


def find_metabolite_network_enrichment(
    metabolite_networks: pd.DataFrame,
    target_set=Iterable[str],
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    fdr: bool = True,
    **kwargs,
) -> pd.Series:
    """
    Find the enrichment of the target set within the metabolite networks

    Parameters
    ----------
    metabolite_networks : pd.DataFrame
        DataFrame describing the metabolite networks. Columns should
        be the metabolite ids, and the index should include the values
        from the target_set
    target_set : Iterable of str
        Set to evaluate the enrichment for, if genes the
        metabolite_networks should be indexed by gene, and
        if reactions the metabolite_networks should be indexed
        reaction. This will be filtered to only include elements of the
        index of the metabolite_networks dataframe prior to
        enrichment test.
    alternative : {"two-sided", "less", "greater"}, default="greater"
        Alternative hypothesis for the enrichment test
    fdr : bool
        Whether to perform false discovery rate adjustment on the p-values
    kwargs
        Key word arguments are passed to SciPy's false_discovery_control
        function if fdr is True, ignored otherwise

    Returns
    -------
    enrichment_results : pd.Series
        The p-values for the enrichment of the target set within
        all of the metabolites
    """
    results_series = pd.Series(np.nan, index=metabolite_networks.columns)
    assert isinstance(target_set, Iterable)
    target_set = set(target_set) & set(metabolite_networks.index)
    total_count = len(metabolite_networks.index)
    for metabolite, metabolite_network in metabolite_networks.items():
        metabolite_net_set = set(metabolite_network[metabolite_network].index)
        results_series[metabolite] = fisher_enrichment(
            group1=target_set,
            group2=metabolite_net_set,
            total_count=total_count,
            alternative=alternative,
        ).pvalue
    if fdr:
        results_series = stats.false_discovery_control(results_series)
    return results_series


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

    Notes
    -----
    If used within a model context everything this function alters
    will be reset upon leaving the context
    """
    metabolite_sink_rxn_id = add_metabolite_sink_(
        model=model, metabolite=metabolite
    )
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
    assert isinstance(met_to_absorb, cobra.Metabolite)
    # Create a name for the reaction (<metabolite>_absorb_<partial hash>)
    metabolite_hash = hashlib.md5(
        met_to_absorb.id.encode("utf-8")
    ).hexdigest()[-8:]
    absorbing_reaction_id = (
        f"{met_to_absorb.id}_absorb_reaction_{metabolite_hash}"
    )
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
    absorbing_constraint_name = (
        f"{met_to_absorb}_absorb_constraint_{metabolite_hash}"
    )
    absorbing_constraint = model.problem.Constraint(
        sympy.Add(
            *metabolite_gen_exprs,
            (-1) * absorbing_reaction.forward_variable,  # type:ignore
        ),
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

    Notes
    -----
    When used within a model context, all changes will be reversed on leaving the model context
    """
    for rxn in model.reactions:
        if rxn.lower_bound > 0.0:
            rxn.lower_bound = 0.0
        elif rxn.upper_bound < 0.0:
            rxn.upper_bound = 0.0
        else:
            pass


def _get_metabolite_sink_id(metabolite: str) -> str:
    """Get an ID for a sink reaction involving a metabolite"""
    metabolite_hash = hashlib.md5(metabolite.encode("utf-8")).hexdigest()[-8:]
    return f"{metabolite}_sink_reaction_{metabolite_hash}"


def add_metabolite_sink_(model: cobra.Model, metabolite: str) -> str:
    """
    Add a metabolite sink reaction to the model
    """
    # Get the id for the reaction to add
    met_sink_id = _get_metabolite_sink_id(metabolite)
    # Check if this is already in the model
    if met_sink_id in model.reactions:
        return met_sink_id
    met_sink_rxn = cobra.Reaction(
        id=met_sink_id, name=f"{metabolite} sink reaction", lower_bound=0.0
    )
    # Get the metabolite object
    met_obj = model.metabolites.get_by_id(metabolite)
    assert isinstance(met_obj, cobra.Metabolite), (
        "Failed to get metabolite from Model!"
    )
    # Add the metabolite sink to the reaction
    met_sink_rxn.add_metabolites({met_obj: -1.0})
    model.add_reactions([met_sink_rxn])
    # Return the ID
    return met_sink_id


def add_all_metabolite_sinks_(model: cobra.Model):
    """
    Add a sink for every metabolite in the model
    """
    for met_id in model.metabolites.list_attr("id"):
        add_metabolite_sink_(model, met_id)


# endregion Helper Functions

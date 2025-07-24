"""Functions for performing fast gene and reaction essentiality analysis"""

# Standard Library Imports
from __future__ import annotations

# External Imports
import cobra
import numpy as np

# Local Imports
from metworkpy.utils.translate import reaction_to_gene_list


# Function for finding essential reactions
def fast_ess_rxn(
    model: cobra.Model,
    essentiality_threshold=0.1,
    pfba_tolerance: float = 1e-7,
    **kwargs,
) -> list[str]:
    """Find which reactions are essential for the model to grow

    Parameters
    ----------
    model : cobra.Model
        The genome scale metabolic model for which to find essential
        reactions
    essentiality_threshold : float
        The proportion of optimal growth, below which a reaction
        knockout is considered essential (default 0.1)
    pfba_tolerance : 1e-7
        Value below which reaction fluxes are considered 0 (default
        1e-7)
    **kwargs
        Keyword arguments passed to COBRApy's single_reaction_deletion

    Returns
    -------
    unknown
        List of reactions which are essential for the model to be able
        to grow

    Notes
    -----
    Uses parsimonious flux balance analysis filter out non-essential reactions,
    then iterates through remaining reactions to check if they are essential.
    """
    # First use pFBA to identify candidate reactions
    pfba_res = cobra.flux_analysis.parsimonious.pfba(
        model=model, fraction_of_optimum=essentiality_threshold
    )
    pfba_fluxes = pfba_res.fluxes
    potentially_essential_reactions = list(
        pfba_fluxes[pfba_fluxes.abs() >= pfba_tolerance].index
    )
    # NOTE: Format of the deletion results is a dataframe,
    # with the index being a frozenset of str, a column for 'growth',
    # and another column for 'status'
    deletion_results = cobra.flux_analysis.deletion.single_reaction_deletion(
        model=model,
        reaction_list=potentially_essential_reactions,
        method="fba",
        **kwargs,
    ).replace(np.nan, 0.0)
    essential_reactions = [
        rxn.__iter__().__next__()
        for rxn in deletion_results[
            deletion_results["growth"]
            <= (model.slim_optimize() * essentiality_threshold)
        ]["ids"]
    ]
    return essential_reactions


# Function for finding essential genes
def fast_ess_genes(
    model: cobra.Model,
    essentiality_threshold=0.1,
    pfba_tolerance: float = 1e-7,
    **kwargs,
) -> list[str]:
    """Find which genes are essential for the model to grow

    Parameters
    ----------
    model : cobra.Model
        The genome scale metabolic model for which to find essential
        genes
    essentiality_threshold : float
        The proportion of optimal growth, below which a gene knockout is
        considered essential (default 0.1)
    pfba_tolerance : 1e-7
        Value below which reaction fluxes are considered 0 (default
        1e-7)
    **kwargs
        Keyword arguments passed to COBRApy's single_gene_deletion
        function

    Returns
    -------
    unknown
        List of reactions which are essential for the model to be able
        to grow

    Notes
    -----
    Uses parsimonious flux balance analysis filter out non-essential genes,
    then iterates through remaining genes to check if they are essential.
    """
    # First use pFBA to identify candidate reactions
    pfba_res = cobra.flux_analysis.parsimonious.pfba(
        model=model, fraction_of_optimum=essentiality_threshold
    )
    pfba_fluxes = pfba_res.fluxes
    potentially_essential_reactions = list(
        pfba_fluxes[pfba_fluxes.abs() >= pfba_tolerance].index
    )
    potentially_essential_genes = reaction_to_gene_list(
        model=model, reaction_list=potentially_essential_reactions, essential=True
    )
    # NOTE: Format of the deletion results is a dataframe,
    # with the index being a frozenset of str, a column for 'growth',
    # and another column for 'status'
    deletion_results = cobra.flux_analysis.deletion.single_gene_deletion(
        model=model,
        gene_list=potentially_essential_genes,
        method="fba",
        **kwargs,
    ).replace(np.nan, 0.0)
    essential_genes = [
        gene.__iter__().__next__()
        for gene in deletion_results[
            deletion_results["growth"]
            <= (model.slim_optimize() * essentiality_threshold)
        ]["ids"]
    ]
    return essential_genes

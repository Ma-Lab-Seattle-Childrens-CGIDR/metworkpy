"""
Submodule for finding fuzzy sets of reactions
"""

# Standard Library Imports
from __future__ import annotations
from typing import Iterable, Optional, Protocol, Union

# External Imports
import cobra
from joblib import Parallel, delayed
import networkx as nx
import pandas as pd

# Local Imports
from metworkpy.network.neighborhoods import _get_rxn_to_gene_set


class FuzzyMembershipFunction(Protocol):
    """
    Protocol for fuzzy membership functions, which should take a reaction,
    a network, a set of target genes, and a reaction to gene set mapping,
    and return a float between 0 and 1. This method can also take in
    additional parameters as kwargs, which will be passed through form
    the calling functions.
    """

    def __call__(
        self,
        reaction: cobra.Reaction,
        network: nx.Graph,
        gene_set: set[str],
        reaction_to_gene_dict: dict[str, set[str]],
        **kwargs,
    ) -> float: ...


def fuzzy_reaction_set(
    metabolic_network: Union[nx.Graph, nx.DiGraph],
    metabolic_model: cobra.Model,
    gene_set: Iterable[str],
    membership_fn: Union[str, FuzzyMembershipFunction],
    essential: bool = False,
    processes: Optional[int] = None,
    **kwargs,
) -> pd.Series:
    """
    Convert from a gene set to a fuzzy reaction set

    Parameters
    ----------
    metabolic_network : nx.Graph or nx.DiGraph
        Metabolic reaction network represented by a networkx Graph or DiGraph.
        DiGraphs will be converted to Graphs before processing.
    metabolic_model : cobra.Model
        Metabolic model from which the metabolic network was constructed
        (used for translating reactions to genes)
    gene_set : Iterable of str
        Set of genes to convert into a fuzzy reaction set
    membership_fn : str or `FuzzyMembershipFunction`
        The membership function to use, can be a string giving the
        functions name, or the function itself which must match the
        signature of `FuzzyMembershipFunction`
    essential : bool
        Whether, when translating from reactions to genes, only
        genes required for a reaction to function should be associated
        with a particular reaction.
    processes : int, optional
        Number of processes to use for parallel processing
    kwargs
        Keyword arguments passed to the membership_fn

    Returns
    -------
    reaction_set : pd.Series
        The fuzzy reaction set, described by a pandas series. The index
        is the reaction id, and the values are the set membership.
    """
    # Get the correct membership function
    if not callable(membership_fn):
        raise ValueError("Received invalid membership function")
    # Convert directed graphs into undirected graphs
    if isinstance(metabolic_network, nx.DiGraph):
        metabolic_network = nx.to_undirected(metabolic_network)
    if not isinstance(metabolic_network, nx.Graph):
        raise ValueError(
            f"metabolic_network must be a networkx Graph or "
            f"DiGraph but received {type(metabolic_network)}"
        )
    # Ensure the gene_set is a set of genes
    gene_set = set(gene_set)
    # Get a mapping from reactions to genes
    rxn_to_gene_dict: dict[str, set[str]] = _get_rxn_to_gene_set(
        model=metabolic_model, essential=essential
    )
    # Create the results series
    rxn_set = pd.Series(
        0.0, index=pd.Index(metabolic_model.reactions.list_attr("id"))
    )
    for rxn, membership in zip(
        rxn_set.index,
        Parallel(n_jobs=processes, return_as="generator")(
            delayed(membership_fn)(
                rxn,
                network=metabolic_network,
                gene_set=gene_set,
                reaction_to_gene_dict=rxn_to_gene_dict,
                **kwargs,
            )
            for rxn in rxn_set.index
        ),
    ):
        rxn_set[rxn] = membership
    return rxn_set


# region Membership Functions


# endregion Membership Functions

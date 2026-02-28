"""Functions for finding and working with neighborhoods in metabolic networks"""

# Standard Library Imports
from __future__ import annotations
from typing import cast, Hashable, Iterator

# External Imports
import cobra  # type:ignore     # Cobra doesn't have py.typed marker
import networkx as nx

# Local Imports
from metworkpy.utils.translate import get_reaction_to_gene_translation_dict

# region Graph Neighborhoods


def graph_neighborhoods(
    network: nx.Graph, radius: int
) -> dict[Hashable, set[Hashable]]:
    """
    Find the neighborhoods of a graph

    Parameters
    ----------
    network : nx.Graph
        The network whose neighborhoods will be identified
    radius : int
        The radius determining the sizes of the neighborhoods

    Returns
    -------
    neighborhoods : dict of nodes to sets of nodes
        Dict describing the nodes in the graph, keyed by
        node with values of sets of nodes in the neighborhood
        of the node (including the node itself)
    """
    return {
        n: neighborhood
        for n, neighborhood in graph_neighborhood_iter(
            network=network, radius=radius
        )
    }


def graph_gene_neighborhoods(
    network: nx.Graph, model: cobra.Model, radius: int
) -> dict[Hashable, set[str]]:
    """
    Find the neighborhoods of a graph

    Parameters
    ----------
    network : nx.Graph
        The network whose neighborhoods will be identified
    model : cobra.Model
        The cobra model associated with the metabolic network
    radius : int
        The radius determining the sizes of the neighborhoods

    Returns
    -------
    neighborhoods : dict of nodes to sets of gene ids
        Dict describing the nodes in the graph, keyed by
        node with values of sets of gene ids in the neighborhood
        of the node
    """
    return {
        n: neighborhood
        for n, neighborhood in graph_gene_neighborhood_iter(
            network=network, model=model, radius=radius
        )
    }


# region neighborhood iterators


def graph_neighborhood_iter(
    network: nx.Graph, radius: int
) -> Iterator[tuple[Hashable, set[Hashable]]]:
    """
    Iterator over neighborhoods in a graph

    Parameters
    ----------
    network : nx.Graph
        The network whose neighborhoods will be iterated over
    radius : int
        The radius determining the size of the neighborhood

    Yields
    ------
    tuple of Hashable and set of Hashable
        Tuple of node and neighborhood
    """
    for node in network.nodes:
        yield (
            node,
            _graph_neighborhood(network=network, radius=radius, node=node),
        )


def graph_gene_neighborhood_iter(
    network: nx.Graph,
    model: cobra.Model,
    radius: int,
    essential: bool = False,
):
    """
    Iterator over gene neighborhoods in a graph

    Parameters
    ----------
    network : nx.Graph
        The network whose neighborhoods will be iterated over
    model : cobra.Model
        The cobra model associated with the metabolic network
    radius : int
        The radius determining the size of the neighborhood
    essential : bool
        Whether to only include genes essential for reactions in the
        neighborhood

    Yields
    ------
    tuple of Hashable and set of str
        Tuple of node and gene ids in neighborhood
    """
    rxn_to_gene_set_dict = get_reaction_to_gene_translation_dict(
        model=model, essential=False
    )
    for node, neighborhood in graph_neighborhood_iter(
        network=network, radius=radius
    ):
        yield (
            node,
            _graph_gene_neighborhood(
                network=network,
                radius=radius,
                node=cast(str, node),
                rxn_to_gene_set_dict=rxn_to_gene_set_dict,
            ),
        )


# endregion neighborhood iterator


def _graph_neighborhood(
    network: nx.Graph, radius: int, node: Hashable
) -> set[Hashable]:
    """Get the neighborhood around a node in the network"""
    neighborhood = {node}
    for _, successors in nx.bfs_successors(
        network, source=node, depth_limit=radius
    ):
        neighborhood.update(successors)
    return neighborhood


def _graph_gene_neighborhood(
    network: nx.Graph,
    radius: int,
    node: str,
    rxn_to_gene_set_dict: dict[str, set[str]],
) -> set[str]:
    """Get the neighborhood of genes around a node in the network"""
    neighborhood = set()
    for rxn_id in _graph_neighborhood(
        network=network, radius=radius, node=node
    ):
        if rxn_id in rxn_to_gene_set_dict:
            neighborhood |= rxn_to_gene_set_dict[rxn_id]  # type:ignore
    return neighborhood


# endregion Graph Neighborhoods

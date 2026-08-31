"""Functions for finding and working with neighborhoods in metabolic networks"""

# Standard Library Imports
from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import NamedTuple, TypeVar, cast

# External Imports
import cobra
import joblib
import networkx as nx
import numpy as np
from scipy import stats

# Local Imports
from metworkpy.utils.translate import get_reaction_to_gene_translation_dict

# region Graph Neighborhoods


def get_graph_neighborhoods(
    network: nx.Graph | nx.DiGraph, radius: int
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


def get_graph_gene_neighborhoods(
    network: nx.Graph,
    model: cobra.Model,
    radius: int,
    essential: bool = False,
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
    essential : bool
        Whether to only include genes essential for reactions in the
        neighborhood

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
            network=network, model=model, radius=radius, essential=essential
        )
    }


# region neighborhood iterators


def graph_neighborhood_iter(
    network: nx.Graph | nx.DiGraph, radius: int
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
            get_graph_neighborhood(network=network, radius=radius, node=node),
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
        model=model, essential=essential
    )
    for node in network:
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


def get_graph_neighborhood(
    network: nx.Graph | nx.DiGraph, radius: int, node: Hashable
) -> set[Hashable]:
    """
    Get the neighborhood around a node in the network

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network to find the neighborhood in
    radius : int
        The radius of the neighborhood
    node : Hashable
        The node to find the neighborhood around

    Returns
    -------
    neighborhood : set of Hashable
        The neighborhood around `node` in `network`
    """
    neighborhood = {node}
    for _, successors in nx.bfs_successors(
        network, source=node, depth_limit=radius
    ):
        neighborhood.update(successors)
    return neighborhood


def get_group_graph_neighborhood(
    network: nx.Graph | nx.DiGraph, radius: int, nodes: set[Hashable]
) -> set[Hashable]:
    """
    Get the neighborhood of a group of nodes, that is all nodes reachable
    within a distance of `radius` from a node in `nodes`

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network to find the neighborhood in
    radius : int
        The radius of the neighborhood
    node : set of Hashable
        The group of nodes to find the neighborhood for

    Returns
    -------
    neighborhood : set of Hashable
        The neighborhood around the `nodes` in `network`
    """
    return functools.reduce(
        operator.or_,
        (
            get_graph_neighborhood(network=network, radius=radius, node=n)
            for n in nodes
        ),
        set(),
    )


def _graph_gene_neighborhood(
    network: nx.Graph,
    radius: int,
    node: str,
    rxn_to_gene_set_dict: dict[str, set[str]],
) -> set[str]:
    """Get the neighborhood of genes around a node in the network"""
    neighborhood = set()
    for rxn_id in get_graph_neighborhood(
        network=network, radius=radius, node=node
    ):
        if rxn_id in rxn_to_gene_set_dict:
            rxn_id = cast(str, rxn_id)
            neighborhood |= rxn_to_gene_set_dict.get(rxn_id, set())
    return neighborhood


# endregion Graph Neighborhoods


########################
### Neighborhood Map ###
########################
NodeType = TypeVar("NodeType")
T = TypeVar("T")


def neighborhood_map(
    fn: Callable[set[NodeType], T],
    network: nx.Graph | nx.DiGraph,
    radius: float = 2,
    nodes: Iterable[NodeType] | None = None,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
    weight: str | None = None,
    include_node: bool = True,
    processes: int | None = None,
) -> dict[NodeType, T]:
    """
    Map a function across neighborhoods in a network

    Parameters
    ----------
    fn : Callable of set of node ids -> Any
        Function to map over the neighborhoods of the network,
        should accept a set of node ids and return a single value
    network : nx.Graph or nx.DiGraph
        The network to map over
    radius : float
        The size of the neighborhood to map over.
        Any nodes within radius distance of the central node will be included
        in the central nodes neighborhood. A radius of 0
        means that only the central node will be included in the neighborhood
        (assuming `include_node` is True, other it would just be the empty set).
    nodes : Iterable of node id, optional
        Nodes to use as neighborhood centers, other nodes will still be included
        in neighborhoods but will not act as neighborhood centers.
    node_filter : callable of node id->bool or set of node ids, optional
        Filter nodes in the network to consider when finding neighborhoods.
        If a Callable, should take node ids as the only argument and return
        a bool, if True the node will be considered in neighborhoods,
        if False it will not be. If a set, only nodes in the set will be included
        in neighborhoods.
    weight : str, optional
        If provided indicates the edge parameter to be used as weights
        when finding distances from a central node to
        define a neighborhood. If None, all edges are treated as having a
        weight of 1.
    include_node : bool, default=True
        Whether to include the central node in a neighborhood
    processes : int, optional
        The number of processes to use for parallel mapping of a
        the function

    Returns
    -------
    dict of node id to result
        Dictionary of central nodes to the result of applying the passed function `fn`
        to the neighborhood around it.
    """
    if callable(node_filter):
        filter_set = {node for node in network if not node_filter(node)}  # ty: ignore[call-top-callable]
    elif isinstance(node_filter, set):
        filter_set = set(network.nodes) - node_filter
    else:
        filter_set = set()

    if nodes is None:
        nodes = network.nodes

    map_res: dict[NodeType, T] = {}
    for node_idx, ret_value in joblib.Parallel(
        n_jobs=processes, return_as="generator_unordered"
    )(
        joblib.delayed(_neighborhood_map_worker)(
            node=node,
            fn=fn,
            network=network,
            radius=radius,
            filter_set=filter_set,
            weight=weight,
            include_node=include_node,
        )
        for node in nodes
    ):
        map_res[node_idx] = ret_value

    return map_res


def _neighborhood_map_worker(
    node: NodeType,
    fn: Callable[set[NodeType], T],
    network: nx.Graph | nx.DiGraph,
    radius: float,
    filter_set: set[str],
    weight: str | None = None,
    include_node: bool = True,
) -> tuple[NodeType, T]:
    # Find the neighborhood around the node
    if include_node:
        neighborhood: set[NodeType] = {node}
    else:
        neighborhood: set[NodeType] = set()
    if weight is None:
        for _, successors in nx.bfs_successors(
            network, source=node, depth_limit=int(radius)
        ):
            neighborhood.update(successors)
    else:
        neighborhood.update(
            nx.single_source_dijkstra_path_length(
                network, source=node, cutoff=radius, weight=weight
            ).keys()
        )
    return node, fn(neighborhood - filter_set)


def gene_neighborhood_map(
    fn: Callable[set[str], T],
    network: nx.Graph | nx.DiGraph,
    model: cobra.Model | None = None,
    reaction_to_gene_set_dict: Mapping[NodeType, set[str]] | None = None,
    radius: float = 2,
    essential: bool = False,
    nodes: Iterable[NodeType] | None = None,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
    weight: str | None = None,
    include_node: bool = True,
    processes: int | None = None,
) -> dict[NodeType, T]:
    """
    Map a function across gene neighborhoods in a network

    Parameters
    ----------
    fn : Callable of set of node ids -> Any
        Function to map over the gene neighborhoods of the network,
        should accept a set of gene ids and return a single value
    network : nx.Graph or nx.DiGraph
        The metabolic network to map over
    model : cobra.Model, optional
        Metabolic model that was used to create the metabolic network, used
        to map reaction ids to gene id sets if `reaction_to_gene_set_dict`
        is not provided. Must provide at least one of
        `model` or `reaction_to_gene_set_dict`, `reaction_to_gene_set_dict`
        takes precedence if both are provided.
    reaction_to_gene_set_dict : dict of reaction id to sets of gene ids, optional
        Map between reaction ids and sets of gene ids. Must provide at least one of
        `model` or `reaction_to_gene_set_dict`, `reaction_to_gene_set_dict`
        takes precedence if both are provided.
    radius : float
        The size of the neighborhood to map over.
        Any nodes within radius distance of the central node will be included
        in the central nodes neighborhood. A radius of 0
        means that only the central node will be included in the neighborhood
        (assuming `include_node` is True, other it would just be the empty set).
    nodes : Iterable of node id, optional
        Nodes to use as neighborhood centers, other nodes will still be included
        in neighborhoods but will not act as neighborhood centers.
    node_filter : callable of node id->bool or set of node ids, optional
        Filter nodes in the network to consider when finding neighborhoods.
        If a Callable, should take node ids as the only argument and return
        a bool, if True the node will be considered in neighborhoods,
        if False it will not be. If a set, only nodes in the set will be included
        in neighborhoods.
    weight : str, optional
        If provided indicates the edge parameter to be used as weights
        when finding distances from a central node to
        define a neighborhood. If None, all edges are treated as having a
        weight of 1.
    include_node : bool, default=True
        Whether to include the central node in a neighborhood
    processes : int, optional
        The number of processes to use for parallel mapping of a
        the function

    Returns
    -------
    dict of node id to result
        Dictionary of central nodes to the result of applying the passed function `fn`
        to the neighborhood around it.
    """
    filter_set = _create_filter_set(network=network, node_filter=node_filter)

    if nodes is None:
        nodes = network.nodes

    # Get a dict of reaction to gene set
    rxn_to_gene_dict = _create_rxn_to_gene_set_dict(
        model=model,
        reaction_to_gene_set_dict=reaction_to_gene_set_dict,
        essential=essential,
    )
    map_res: dict[NodeType, T] = {}
    for node_idx, ret_value in joblib.Parallel(
        n_jobs=processes, return_as="generator_unordered"
    )(
        joblib.delayed(_gene_neighborhood_worker)(
            node=node,
            fn=fn,
            network=network,
            rxn_to_gene_dict=rxn_to_gene_dict,
            radius=radius,
            filter_set=filter_set,
            weight=weight,
            include_node=include_node,
        )
        for node in nodes
    ):
        map_res[node_idx] = ret_value

    return map_res


def _gene_neighborhood_worker(
    node: NodeType,
    fn: Callable[set[NodeType], T],
    network: nx.Graph | nx.DiGraph,
    rxn_to_gene_dict: dict[NodeType, set[str]],
    radius: float,
    filter_set: set[str],
    weight: str | None = None,
    include_node: bool = True,
):
    # Find the neighborhood around the node
    if include_node:
        neighborhood: set[str] = rxn_to_gene_dict.get(node, set())
    else:
        neighborhood: set[str] = set()
    if weight is None:
        for _, successors in nx.bfs_successors(
            network, source=node, depth_limit=int(radius)
        ):
            for n in set(successors) & filter_set:
                neighborhood.update(rxn_to_gene_dict.get(n, set()))
    else:
        for n in (
            set(
                nx.single_source_dijkstra_path_length(
                    network, source=node, cutoff=radius, weight=weight
                ).keys()
            )
            & filter_set
        ):
            neighborhood.update(rxn_to_gene_dict.get(n, set()))

    return node, fn(neighborhood)


#########################
### Stouffer's method ###
#########################
class CombinePvaluesResult(NamedTuple):
    statistics: dict[Hashable, float]
    pvalues: dict[Hashable, float]


def combine_neighborhood_pvalues(
    network: nx.Graph | nx.DiGraph,
    gene_pvalues: Mapping[str, float],
    gene_weights: Mapping[str, float] | None = None,
    model: cobra.Model | None = None,
    reaction_to_gene_set_dict: Mapping[NodeType, set[str]] | None = None,
    radius: float = 2,
    essential: bool = False,
    nodes: Iterable[NodeType] | None = None,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
    weight: str | None = None,
    include_node: bool = True,
    processes: int | None = None,
    **kwargs,
) -> CombinePvaluesResult:
    """
    Map a function across gene neighborhoods in a network

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The metabolic network to map over
    gene_pvalues : dict of str to float
        P-values assigned to each gene, any genes with ids not in this dict
        will be treated as having a p-value of NaN, the handling of which
        can be modified by passing `nan_policy` as a keyword argument
        (which will be passed to SciPy stats `combine_pvalues` function).
    gene_weights : dict of str to float, optional
        Optional weights to apply if using "stouffer" method
        `scipy.stats.combine_pvalues <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html>`_,
        should be a dict of gene id to a weight for each gene.
    model : cobra.Model, optional
        Metabolic model that was used to create the metabolic network, used
        to map reaction ids to gene id sets if `reaction_to_gene_set_dict`
        is not provided. Must provide at least one of
        `model` or `reaction_to_gene_set_dict`, `reaction_to_gene_set_dict`
        takes precedence if both are provided.
    reaction_to_gene_set_dict : dict of reaction id to sets of gene ids, optional
        Map between reaction ids and sets of gene ids. Must provide at least one of
        `model` or `reaction_to_gene_set_dict`, `reaction_to_gene_set_dict`
        takes precedence if both are provided.
    radius : float
        The size of the neighborhood to map over.
        Any nodes within radius distance of the central node will be included
        in the central nodes neighborhood. A radius of 0
        means that only the central node will be included in the neighborhood
        (assuming `include_node` is True, other it would just be the empty set).
    nodes : Iterable of node id, optional
        Nodes to use as neighborhood centers, other nodes will still be included
        in neighborhoods but will not act as neighborhood centers.
    node_filter : callable of node id->bool or set of node ids, optional
        Filter nodes in the network to consider when finding neighborhoods.
        If a Callable, should take node ids as the only argument and return
        a bool, if True the node will be considered in neighborhoods,
        if False it will not be. If a set, only nodes in the set will be included
        in neighborhoods.
    weight : str, optional
        If provided indicates the edge parameter to be used as weights
        when finding distances from a central node to
        define a neighborhood. If None, all edges are treated as having a
        weight of 1.
    include_node : bool, default=True
        Whether to include the central node in a neighborhood
    processes : int, optional
        The number of processes to use for parallel mapping of a
        the function
    kwargs
        Keyword arguments are passed to
        `scipy.stats.combine_pvalues <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html>`_

    Returns
    -------
    dict of node id to result
        Dictionary of central nodes to the result of applying the passed function `fn`
        to the neighborhood around it.
    """

    def combine_pvals(gene_ids: set[str]):
        return stats.combine_pvalues(
            [gene_pvalues.get(g, np.nan) for g in gene_ids],
            weights=[gene_weights.get(g, np.nan) for g in gene_ids]
            if gene_weights is not None
            else None,
            **kwargs,
        )

    res_dict = gene_neighborhood_map(
        fn=combine_pvals,
        network=network,
        model=model,
        reaction_to_gene_set_dict=reaction_to_gene_set_dict,
        radius=radius,
        essential=essential,
        nodes=nodes,
        node_filter=node_filter,
        weight=weight,
        include_node=include_node,
        processes=processes,
    )
    # Split the results into statistics and pvalues
    stats_dict = {}
    pvals_dict = {}
    for node, (stat, pval) in res_dict.items():
        stats_dict[node] = stat
        pvals_dict[node] = pval
    return CombinePvaluesResult(stats_dict, pvals_dict)


########################
### Helper Functions ###
########################
def _create_filter_set(
    network: nx.Graph | nx.DiGraph,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
):
    if callable(node_filter):
        filter_set = {node for node in network if not node_filter(node)}  # ty: ignore[call-top-callable]
    elif isinstance(node_filter, set):
        filter_set = set(network.nodes) - node_filter
    else:
        filter_set = set()
    return filter_set


def _create_rxn_to_gene_set_dict(
    model: cobra.Model | None = None,
    reaction_to_gene_set_dict: Mapping[NodeType, set[str]] | None = None,
    essential: bool = False,
):
    # Get a dict of reaction to gene set
    if reaction_to_gene_set_dict is None:
        if model is not None:
            rxn_to_gene_dict = get_reaction_to_gene_translation_dict(
                model=model, essential=essential
            )
        else:
            raise ValueError(
                "Must provide at least one of model or reaction_to_gene_set_dict, but received None"
            )
    else:
        rxn_to_gene_dict = reaction_to_gene_set_dict
    return rxn_to_gene_dict

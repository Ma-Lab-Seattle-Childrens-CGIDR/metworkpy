"""Functions for finding and working with neighborhoods in metabolic networks"""

# Standard Library Imports
from __future__ import annotations

import functools
import operator
from collections import defaultdict
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

NodeType = TypeVar("NodeType")
EdgeWeight = TypeVar("EdgeWeight")


def get_graph_neighborhoods(
    network: nx.Graph | nx.DiGraph,
    radius: int,
    *,
    include_node: bool = True,
    weight: str | None = None,
) -> dict[Hashable, set[Hashable]]:
    """
    Find the neighborhoods of a graph

    Parameters
    ----------
    network : nx.Graph
        The network whose neighborhoods will be identified
    radius : int
        The radius determining the sizes of the neighborhoods
    include_node : bool, default=True
        Whether to include central nodes in the neighborhoods
    weight : str,optional
        Edge attribute to use as weight, if None
        each edge has a weight of 1.

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
            network=network,
            radius=radius,
            include_node=include_node,
            weight=weight,
        )
    }


def get_graph_gene_neighborhoods(
    network: nx.Graph,
    model: cobra.Model,
    radius: int,
    *,
    essential: bool = False,
    include_node: bool = True,
    weight: str | None = None,
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
    include_node : bool, default=True
        Whether to include central nodes in the neighborhoods
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
            network=network,
            model=model,
            radius=radius,
            essential=essential,
            include_node=include_node,
            weight=weight,
        )
    }


# region neighborhood iterators


def graph_neighborhood_iter(
    network: nx.Graph | nx.DiGraph,
    radius: int,
    *,
    include_node: bool = True,
    weight: str | None = None,
) -> Iterator[tuple[Hashable, set[Hashable]]]:
    """
    Iterator over neighborhoods in a graph

    Parameters
    ----------
    network : nx.Graph
        The network whose neighborhoods will be iterated over
    radius : int
        The radius determining the size of the neighborhood
    include_node : bool, default=True
        Whether to include central nodes in the neighborhoods
    weight : str,optional
        Edge attribute to use as weight, if None
        each edge has a weight of 1.

    Yields
    ------
    tuple of Hashable and set of Hashable
        Tuple of node and neighborhood
    """
    for node in network.nodes:
        yield (
            node,
            get_graph_neighborhood(
                network=network,
                radius=radius,
                node=node,
                include_node=include_node,
                weight=weight,
            ),
        )


def graph_gene_neighborhood_iter(
    network: nx.Graph,
    model: cobra.Model,
    radius: int,
    *,
    essential: bool = False,
    include_node: bool = True,
    weight: str | None = None,
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
    include_node : bool, default=True
        Whether to include `node` in the neighborhood
    weight : str, optional
        The edge attribute to use as the weight of an edge,
        if not provided all edges have weight of 1
        (weights are interpreted as distances for finding neighborhoods)

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
            graph_gene_neighborhood(
                network=network,
                radius=radius,
                node=cast(str, node),
                rxn_to_gene_set_dict=rxn_to_gene_set_dict,
                model=None,
                include_node=include_node,
                weight=weight,
            ),
        )


# endregion neighborhood iterator


def get_graph_neighborhood(
    network: nx.Graph | nx.DiGraph,
    node: Hashable,
    radius: float,
    *,
    include_node: bool = True,
    weight: str | None = None,
) -> set[NodeType]:
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
    include_node : bool, default=True
        Whether to include `node` in the neighborhood
    weight : str,optional
        Edge attribute to use as weight, if None
        each edge has a weight of 1.

    Returns
    -------
    neighborhood : set of Hashable
        The neighborhood around `node` in `network`
    """
    neighborhood = {node} if include_node else set()
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
    return neighborhood


def get_target_set_graph_neighborhood(
    network: nx.Graph | nx.DiGraph,
    nodes: set[Hashable],
    radius: int,
    *,
    include_node: bool = True,
    weight: str | None = None,
) -> set[Hashable]:
    """
    Get the neighborhood of a target set of nodes, that is all nodes reachable
    within a distance of `radius` from a node in `nodes`

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network to find the neighborhood in
    node : set of Hashable
        The target set of nodes to find the neighborhood for
    radius : int
        The radius of the neighborhood
    include_node : bool, default=True
        Whether to include `node` in the neighborhood
    weight : str,optional
        Edge attribute to use as weight, if None
        each edge has a weight of 1.

    Returns
    -------
    neighborhood : set of Hashable
        The neighborhood around the `nodes` in `network`
    """
    return functools.reduce(
        operator.or_,
        (
            get_graph_neighborhood(
                network=network,
                radius=radius,
                node=n,
                include_node=include_node,
                weight=weight,
            )
            for n in nodes
        ),
        set(),
    )


def graph_gene_neighborhood(
    network: nx.Graph,
    node: str,
    radius: int,
    *,
    model: cobra.Model | None = None,
    rxn_to_gene_set_dict: dict[str, set[str]] | None = None,
    essential: bool = False,
    include_node: bool = True,
    weight: str | None = None,
) -> set[str]:
    """
    Get the neighborhood of genes around a node in the network

    Parameters
    ----------
    network : nx.Graph
        The network whose neighborhoods will be identified
    node : Hashable
        The node to find the neighborhood around
    radius : int
        The radius determining the sizes of the neighborhoods
    model : cobra.Model, optional
        The cobra model associated with the metabolic network.
        Either `model` of `rxn_to_gene_set_dict` must be provided
        for mapping between reactions and genes, if both are provided
        `rxn_to_gene_set_dict` takes priority.
    rxn_to_gene_set_dict : dict of reaction id to set of gene ids, optional
        A dictionary mapping reaction ids to sets of gene ids which
        are associated with the reaction. Either `model` of `rxn_to_gene_set_dict`
        must be provided for mapping between reactions and genes, if both are
        provided `rxn_to_gene_set_dict` takes priority.
    essential : bool
        Whether to only include genes essential for reactions in the
        neighborhood
    include_node : bool, default=True
        Whether to include `node` in the neighborhood
    weight : str, optional
        The edge attribute to use as the weight of an edge,
        if not provided all edges have weight of 1
        (weights are interpreted as distances for finding neighborhoods)

    Returns
    -------
    neighborhood : set of str
        The ids of genes in the neighborhood around `node` in `network`
    """
    if rxn_to_gene_set_dict is None:
        if model is None:
            raise ValueError(
                "At least one of 'model' or 'rxn_to_gene_set_dict' must be provided, but both are None"
            )
        rxn_to_gene_set_dict = get_reaction_to_gene_translation_dict(
            model=model, essential=essential
        )
    neighborhood = set()
    for rxn_id in get_graph_neighborhood(
        network=network,
        radius=radius,
        node=node,
        include_node=include_node,
        weight=weight,
    ):
        if rxn_id in rxn_to_gene_set_dict:
            rxn_id = cast(str, rxn_id)
            neighborhood |= rxn_to_gene_set_dict.get(rxn_id, set())
    return neighborhood


# endregion Graph Neighborhoods


########################
### Neighborhood Map ###
########################
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
    filter_set = _create_filter_set(network, node_filter)

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
    filter_set: set[NodeType],
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
    essential : bool,default=False
        Whether, when finding which genes are associated with the
        reaction nodes in the network, the mapping should require
        a gene to be essential for the reaction to function.
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
    # Filtering out empty sets, since if a key isn't
    # found an empty set is assumed
    rxn_to_gene_dict = {
        r: gs
        for r, gs in _create_rxn_to_gene_set_dict(
            model=model,
            reaction_to_gene_set_dict=reaction_to_gene_set_dict,
            essential=essential,
        ).items()
        if len(gs) > 0
    }
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
        neighborhood: set[str] = rxn_to_gene_dict.get(node, set()).copy()
    else:
        neighborhood: set[str] = set()
    if weight is None:
        for _, successors in nx.bfs_successors(
            network, source=node, depth_limit=int(radius)
        ):
            for n in set(successors) - filter_set:
                neighborhood.update(rxn_to_gene_dict.get(n, set()))
    else:
        for n in (
            set(
                nx.single_source_dijkstra_path_length(
                    network, source=node, cutoff=radius, weight=weight
                ).keys()
            )
            - filter_set
        ):
            neighborhood.update(rxn_to_gene_dict.get(n, set()))
    return node, fn(neighborhood)


#####################
### Neighbors Map ###
#####################


# These functions map over direct neighbors, but
# also provide the weights of the edges
def weighted_neighbor_map(
    fn: Callable[[NodeType, dict[NodeType, EdgeWeight]], T],
    network: nx.Graph | nx.DiGraph,
    nodes: Iterable[NodeType] | None = None,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
    weight: str | None = None,
    processes: int | None = None,
) -> dict[NodeType, T]:
    """
    Map a function across all groups of node neighbors in a network,
    weighted by the edge weight between the node and its neighbor.

    Parameters
    ----------
    fn : Callable of (node id, dict of node id to edge weight) -> Any
        Function to map over weighted neighbors in the network,
        the function receives a node id of the central node, and then a
        a dict of node id to weight of the edge between the central node
        and the neighboring node.
    network : nx.Graph or nx.DiGraph
        The network to map over
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
        The edge attribute to use as weight, if None all edges are
        given a weight of 1. The weight is passed as the
        value of the dict in the second argument of `fn`.
    processes : int, optional
        The number of processes to use for parallel mapping of a
        the function

    Returns
    -------
    dict of node id to result
        Dictionary of central node ids to the result of applying the passed function `fn`
        to the neighboring nodes, weighted by an edge attribute.

    Notes
    -----
    This function maps a function across all the direct neighbors
    of nodes in a network, weighted by an edge attribute. That is, the
    function is given the central node id, and then a dict keyed by
    the neighboring node ids, with values equal to the `weight` of
    the edge between the central node and the neighbor.
    """
    filter_set = _create_filter_set(network, node_filter)

    if nodes is None:
        nodes = network.nodes

    map_res: dict[NodeType, T] = {}
    for node_idx, ret_value in joblib.Parallel(
        n_jobs=processes, return_as="generator_unordered"
    )(
        joblib.delayed(_weighted_neighbor_map_worker)(
            node=node,
            fn=fn,
            network=network,
            filter_set=filter_set,
            weight=weight,
        )
        for node in nodes
    ):
        map_res[node_idx] = ret_value

    return map_res


def _weighted_neighbor_map_worker(
    node: NodeType,
    fn: Callable[[NodeType, dict[NodeType, EdgeWeight]], T],
    network: nx.Graph | nx.DiGraph,
    filter_set: set[NodeType],
    weight: str | None = None,
):
    neighbors: dict[NodeType, EdgeWeight] = {}
    for n, edata in network[node].items():
        if n in filter_set:
            continue
        neighbors[n] = edata[weight] if weight is not None else 1  # ty: ignore[invalid-assignment]
    return fn(node, neighbors)


def weighted_gene_neighbor_map(
    fn: Callable[[set[str], dict[str, EdgeWeight]], T],
    network: nx.Graph | nx.DiGraph,
    model: cobra.Model | None = None,
    reaction_to_gene_set_dict: Mapping[NodeType, set[str]] | None = None,
    essential: bool = False,
    nodes: Iterable[NodeType] | None = None,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
    weight: str | None = None,
    weight_combine_fn: Callable[[list[EdgeWeight]], EdgeWeight] = max,  # ty: ignore[invalid-parameter-default]
    processes: int | None = None,
):
    """
    Map a function across all groups of gene neighbors in a network,
    weighted by the edge weight between the node and its neighbor.

    Parameters
    ----------
    fn : Callable of (set of gene ids, dict of gene id to edge weight) -> Any
        Function to map over weighted gene neighbors in the network,
        the function receives a set of gene ids of the central node, and then a
        a dict of neighboring genes to weight of the edge between the central node
        and the neighboring node.
    network : nx.Graph or nx.DiGraph
        The network to map over
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
    essential : bool,default=False
        Whether, when finding which genes are associated with the
        reaction nodes in the network, the mapping should require
        a gene to be essential for the reaction to function.
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
        The edge attribute to use as weight, if None all edges are
        given a weight of 1. The weight is passed as the
        value of the dict in the second argument of `fn`.
    weight_combine_fn : Callable of (list of edge weights)->single weight, default=maximum
        If a gene is associated with multiple reactions in a neighborhood, how
        should the weights be combined. Receives a list of the edge weights between
        the central node and the neighboring nodes associated with the gene,
        and should return a single weighting.
    processes : int, optional
        The number of processes to use for parallel mapping of a
        the function

    Returns
    -------
    dict of node id to result
        Dictionary of central node ids to the result of applying the passed function `fn`
        to the neighboring genes, weighted by an edge attribute.

    Notes
    -----
    This function maps a function across all the direct neighbors
    of nodes in a network, weighted by an edge attribute. That is, the
    function is given the central node id, and then a dict keyed by
    the neighboring node ids, with values equal to the `weight` of
    the edge between the central node and the neighbor.
    """
    filter_set = _create_filter_set(network=network, node_filter=node_filter)

    if nodes is None:
        nodes = network.nodes

    # Get a dict of reaction to gene set
    # Filtering out empty sets, since if a key isn't
    # found an empty set is assumed
    rxn_to_gene_dict = {
        r: gs
        for r, gs in _create_rxn_to_gene_set_dict(
            model=model,
            reaction_to_gene_set_dict=reaction_to_gene_set_dict,
            essential=essential,
        ).items()
        if len(gs) > 0
    }
    map_res: dict[NodeType, T] = {}
    for node_idx, ret_value in joblib.Parallel(
        n_jobs=processes, return_as="generator_unordered"
    )(
        joblib.delayed(_weighted_gene_neighborhor_map_worker)(
            node=node,
            fn=fn,
            cmb_fn=weight_combine_fn,
            network=network,
            rxn_to_gene_dict=rxn_to_gene_dict,
            filter_set=filter_set,
            weight=weight,
        )
        for node in nodes
    ):
        map_res[node_idx] = ret_value

    return map_res


def _weighted_gene_neighborhor_map_worker(
    node: NodeType,
    fn: Callable[[set[str], dict[str, EdgeWeight]], T],
    cmb_fn: Callable[[list[EdgeWeight]], EdgeWeight],
    network: nx.Graph | nx.DiGraph,
    rxn_to_gene_dict: dict[NodeType, set[str]],
    filter_set: set[str],
    weight: str | None = None,
):
    neighbors: dict[str, list[EdgeWeight]] = defaultdict(list)
    for n, edata in network[node].items():
        if n in filter_set:
            continue
        for g in rxn_to_gene_dict.get(n, ()):
            neighbors[g].append(edata[weight] if weight is not None else 1)  # ty: ignore[invalid-argument-type]
    neighbors_reduced = {
        g: cmb_fn(weights) for g, weights in neighbors.items()
    }
    return fn(rxn_to_gene_dict.get(node, set()), neighbors_reduced)


#########################
### Stouffer's method ###
#########################
class CombinePvaluesResult(NamedTuple):
    statistics: dict[Hashable, float]
    pvalues: dict[Hashable, float]


def combine_neighborhood_pvalues_weighted(
    gene_pvalues: Mapping[str, float],
    network: nx.Graph | nx.DiGraph,
    model: cobra.Model | None = None,
    reaction_to_gene_set_dict: Mapping[NodeType, set[str]] | None = None,
    essential: bool = False,
    nodes: Iterable[NodeType] | None = None,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
    weight: str | None = None,
    central_genes_proportion: float = 0.5,
    weight_combine_fn: Callable[[list[EdgeWeight]], EdgeWeight] = max,  # ty: ignore[invalid-parameter-default]
    processes: int | None = None,
    **kwargs,
) -> CombinePvaluesResult:
    """
    Map a function across all groups of gene neighbors in a network,
    weighted by the edge weight between the node and its neighbor.

    Parameters
    ----------
    gene_pvalues : dict of str to float
        P-values assigned to each gene, any genes with ids not in this dict
        will be treated as having a p-value of NaN, the handling of which
        can be modified by passing `nan_policy` as a keyword argument
        (which will be passed to SciPy stats `combine_pvalues` function).
    network : nx.Graph or nx.DiGraph
        The network to map over
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
    essential : bool,default=False
        Whether, when finding which genes are associated with the
        reaction nodes in the network, the mapping should require
        a gene to be essential for the reaction to function.
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
        The edge attribute to use as weight, if None all edges are
        given a weight of 1. The weight is passed as the
        value of the dict in the second argument of `fn`.
    central_genes_proportion : float, default=0.5
        The proportion of the total weighting of the p-values to be
        taken by the genes associated with the central node. If the central
        node is not associated with any genes, the weights of the
        surrounding nodes are not scaled (as it wouldn't actually impact the
        result).
    weight_combine_fn : Callable of (list of edge weights)->single weight, default=maximum
        If a gene is associated with multiple reactions in a neighborhood, how
        should the weights be combined. Receives a list of the edge weights between
        the central node and the neighboring nodes associated with the gene,
        and should return a single weighting.
    processes : int, optional
        The number of processes to use for parallel mapping of a
        the function

    Returns
    -------
    dict of node id to result
        Dictionary of central node ids to the result of applying the passed function `fn`
        to the neighboring genes, weighted by an edge attribute.

    Notes
    -----
    This function maps a function across all the direct neighbors
    of nodes in a network, weighted by an edge attribute. That is, the
    function is given the central node id, and then a dict keyed by
    the neighboring node ids, with values equal to the `weight` of
    the edge between the central node and the neighbor.
    """

    def combine_pvals(
        central_genes: set[str],
        weighted_gene_dict: dict[str, float],
    ):
        if len(central_genes) > 0:
            pvalues = [gene_pvalues.get(g, np.nan) for g in central_genes] + [
                gene_pvalues.get(g, np.nan) for g in weighted_gene_dict
            ]
            weights = [
                w * (1 - central_genes_proportion)
                for w in weighted_gene_dict.values()
            ]
            weights = [
                sum(weights) * central_genes_proportion / len(central_genes)
            ] * len(central_genes) + weights
        else:
            pvalues = [gene_pvalues.get(g, np.nan) for g in weighted_gene_dict]
            weights = [w for w in weighted_gene_dict.values()]
        return stats.combine_pvalues(
            pvalues=pvalues,
            weights=weights,
            method="stouffer",
            **kwargs,
        )

    res_dict = weighted_gene_neighbor_map(
        fn=combine_pvals,  # ty: ignore[invalid-argument-type]
        network=network,
        model=model,
        reaction_to_gene_set_dict=reaction_to_gene_set_dict,
        essential=essential,
        nodes=nodes,
        node_filter=node_filter,
        weight=weight,
        weight_combine_fn=weight_combine_fn,  # ty: ignore[invalid-argument-type]
        processes=processes,
    )
    stats_dict = {}
    pvals_dict = {}
    for node, (stat, pval) in res_dict.items():
        stats_dict[node] = stat
        pvals_dict[node] = pval
    return CombinePvaluesResult(stats_dict, pvals_dict)


def combine_neighborhood_pvalues(
    gene_pvalues: Mapping[str, float],
    network: nx.Graph | nx.DiGraph,
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
    Combine the p-values for genes in neighborhoods of `network`

    Parameters
    ----------
    gene_pvalues : dict of str to float
        P-values assigned to each gene, any genes with ids not in this dict
        will be treated as having a p-value of NaN, the handling of which
        can be modified by passing `nan_policy` as a keyword argument
        (which will be passed to SciPy stats `combine_pvalues` function).
    network : nx.Graph or nx.DiGraph
        The metabolic network to map over
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
        `scipy.stats.combine_pvalues <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html>`_,
        don't pass 'method' since that is required to be 'stouffer' to allow for the weighting.

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
            method="stouffer",
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

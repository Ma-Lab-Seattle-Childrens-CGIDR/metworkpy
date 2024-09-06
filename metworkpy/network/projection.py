"""
Module to project bipartite graphs onto the node sets
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Iterable, Optional
from typing import Callable

# External Imports
import networkx as nx
from networkx.algorithms.bipartite import projected_graph, weighted_projected_graph

# Local Imports
from metworkpy.utils._arguments import _parse_str_args_dict


# region Main Function
def bipartite_project(
    network: nx.Graph | nx.DiGraph,
    node_set: Iterable,
    directed: Optional[bool] = None,
    weight: str | Callable[[float, float], float] | None = None,
    weight_attribute: str = "weight",
    reciprocal: bool = False,
) -> nx.Graph | nx.DiGraph:
    """
    Function to project a bipartite graph onto the specified set of nodes

    :param network: Network to project
    :type network: nx.Graph | nx.DiGraph
    :param node_set: Nodes to project the graph onto
    :type node_set: Iterable
    :param directed: Whether the projected graph should be directed. If the
        `network` argument is not directed this is ignored. A value of None will have
        the directedness of the output match the directedness of the input network.
    :type directed: bool | None
    :param weight: How to weight the projected graph. If None, the projected graph
        will not be weighted. If "ratio", the edges will be weighted based on the
        ratio between actual shared neighbors and maximum possible shared neighbors.
        If "count", the edges will be weighted by the number of shared neighbors.
        A function can also be provided, which takes two float arguments (the
        weights of two edges), and returns a float.
    :type weight: str | Callable[[float, float], float] | None
    :param weight_attribute: Which edge attribute in the original network to use for
        weighting. Default is 'weight'.
    :type weight_attribute: str
    :param reciprocal: If converting from a directed graph to an undirected one,
        whether to only keep edges that appear in both directions in the original
        directed network.
    :type reciprocal: bool
    :return: Projected network
    :rtype: nx.Graph | nx.DiGraph
    """
    if directed is not None:
        if not directed and isinstance(network, nx.DiGraph):
            network = network.to_undirected(reciprocal=reciprocal)
    if weight is None:
        return projected_graph(network, nodes=node_set)
    if isinstance(weight, str):
        weight = _parse_str_args_dict(
            weight, {"ratio": ["ratio", "proportion"], "count": ["count", "number"]}
        )
        if weight == "ratio":
            ratio = True
        elif weight == "count":
            ratio = False
        else:
            raise ValueError("Couldn't parse weight string")
        return weighted_projected_graph(network, nodes=node_set, ratio=ratio)
    if isinstance(network, nx.DiGraph):
        return _directed_projection(
            network=network, node_set=node_set, weight=weight, attr=weight_attribute
        )
    return _unirected_projection(
        network=network, node_set=node_set, weight=weight, attr=weight_attribute
    )


# endregion Main Function


# region Directed Projection
def _directed_projection(
    network: nx.DiGraph,
    node_set: Iterable,
    weight: Callable[[float, float], float],
    attr: str = "weight",
) -> nx.DiGraph:
    res_graph = network.subgraph(nodes=node_set).copy()
    for node in network.nodes():
        if node in res_graph:
            continue
        for pred in network.predecessors(node):
            for suc in network.successors(node):
                if pred == suc:
                    continue
                new_weight = weight(
                    network.get_edge_data(pred, node)[attr],
                    network.get_edge_data(node, suc)[attr],
                )
                if res_graph.has_edge(pred, suc):
                    new_weight = weight(
                        new_weight, res_graph.get_edge_data(pred, suc)[attr]
                    )
                res_graph.add_edge(pred, suc, **{attr: new_weight})
    return res_graph


# endregion Directed Projection


# region Undirected Projection
def _unirected_projection(
    network: nx.Graph,
    node_set: Iterable,
    weight: Callable[[float, float], float],
    attr: str = "weight",
) -> nx.Graph:
    res_graph = network.subgraph(nodes=node_set).copy()
    for node in network.nodes():
        if node in res_graph:
            continue
        for pred in network.neighbors(node):
            for suc in network.neighbors(node):
                if pred == suc:
                    continue
                new_weight = weight(
                    network.get_edge_data(pred, node)[attr],
                    network.get_edge_data(node, suc)[attr],
                )
                if res_graph.has_edge(pred, suc):
                    new_weight = weight(
                        new_weight, res_graph.get_edge_data(pred, suc)[attr]
                    )
                res_graph.add_edge(pred, suc, **{attr: new_weight})
    return res_graph


# endregion Undirected Projection

"""
Functions for identifying subnetworks which include
a provided set of nodes
"""

import itertools
from collections.abc import Iterable

import cobra
import networkx as nx

from metworkpy.gpr.gpr_functions import gene_group_to_reaction_list

from .neighborhoods import NodeType, get_graph_neighborhood


def get_subnetwork(
    network: nx.Graph | nx.DiGraph,
    nodes: Iterable[NodeType],
    *,
    k: int | None = 1,
    radius: float | None = None,
    weight: str | None = None,
) -> nx.Graph | nx.DiGraph:
    """
    Extract a subnetwork from `network` which includes `nodes`, and
    (optionally) paths between those nodes, and neighborhoods around the nodes

    Parameters
    ----------
    network : nx.Graph
        The network to extract the subnetwork from
    nodes : iterable of node ids
        The nodes to include in the subnetwork
    k : int or None,default=1
        The number of shortest paths between each pair
        of nodes in the network to include in the subnetwork.
        If None, no pathways between nodes are included.
    radius : int or None, default=None
        The radius determining the sizes of the neighborhoods
        around `nodes` to include in the subnetwork.
        If None (or 0) no neighborhoods are included.
    weight : str, optional
        The edge attribute to use as the weight of an edge,
        if not provided all edges have weight of 1
        (weights are interpreted as distances for finding neighborhoods)

    Returns
    -------
    subnetwork : nx.Graph or nx.Digraph
        A subgraph *view* of `network`, that includes the `nodes`, as well
        as the paths between the nodes, and neighborhoods around the nodes.
        Use `subnetwork.copy()` to get a new graph object with its own copy of
        the graph data.
    """
    subnetwork_node_set = set(nodes)
    if k is not None and k > 0:
        for u, v in itertools.combinations(nodes, 2):
            try:
                for path in itertools.islice(
                    nx.shortest_simple_paths(network, source=u, target=v), k
                ):
                    subnetwork_node_set.update(path[1:-1])
            except nx.NetworkXNoPath:
                pass
            if network.is_directed():
                try:
                    for path in itertools.islice(
                        nx.shortest_simple_paths(network, source=u, target=v),
                        k,
                    ):
                        subnetwork_node_set.update(path[1:-1])
                except nx.NetworkXNoPath:
                    pass
    if radius is not None and radius > 0:
        for n in nodes:
            subnetwork_node_set |= get_graph_neighborhood(
                network=network,
                node=n,
                radius=radius,
                include_node=False,
                weight=weight,
            )
    return network.subgraph(subnetwork_node_set)


def get_gene_subnetwork(
    network: nx.Graph | nx.DiGraph,
    model: cobra.Model,
    genes: Iterable[str],
    *,
    k: int | None = 1,
    radius: float | None = 0,
    weight: str | None = None,
    essential: bool = False,
):
    """
    Extract a subnetwork from `network` which includes `nodes`, and
    (optionally) paths between those nodes, and neighborhoods around the nodes

    Parameters
    ----------
    network : nx.Graph
        The network to extract the subnetwork from
    model : cobra.Model, optional
        The cobra model associated with the metabolic network.
    nodes : iterable of node ids
        The nodes to include in the subnetwork
    k : int or None,default=1
        The number of shortest paths between each pair
        of nodes in the network to include in the subnetwork.
        If None, no pathways between nodes are included.
    radius : int or None, default=None
        The radius determining the sizes of the neighborhoods
        around `nodes` to include in the subnetwork.
        If None (or 0) no neighborhoods are included.
    weight : str, optional
        The edge attribute to use as the weight of an edge,
        if not provided all edges have weight of 1
        (weights are interpreted as distances for finding neighborhoods)
    essential : bool
        When translating from `genes` to nodes in the network,
        whether to only include nodes representing reactions which
        require the genes in genes.

    Returns
    -------
    subnetwork : nx.Graph or nx.Digraph
        A subgraph *view* of `network`, that includes the `nodes`, as well
        as the paths between the nodes, and neighborhoods around the nodes.
        Use `subnetwork.copy()` to get a new graph object with its own copy of
        the graph data.
    """
    nodes = set(network.nodes) & set(
        gene_group_to_reaction_list(
            model=model, gene_list=genes, essential=essential
        )
    )
    return get_subnetwork(
        network=network, nodes=nodes, k=k, radius=radius, weight=weight
    )

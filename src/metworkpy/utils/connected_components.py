"""
Module for finding the connected components of a graph
"""

# Standard Library imports
from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import TypeVar, Protocol


# Define the type for the nodes
class HashableComparable(Protocol):
    """Protocol for annotating Comparable and Hashable types."""

    @abstractmethod
    def __lt__(self: HCT, other: HCT, /) -> bool: ...

    @abstractmethod
    def __gt__(self: HCT, other: HCT, /) -> bool: ...

    @abstractmethod
    def __hash__(self: HCT, /): ...


# The nodes need to be hashable (since the methods use dicts and sets)
# and must be comparable (they are sorted using min)
HCT = TypeVar("HCT", bound=HashableComparable)


def find_connected_components(
    node_list: list[HCT], edge_list: list[tuple[HCT, HCT]]
) -> list[set[HCT]]:
    """
    Find the connected components of a graph

    Parameters
    ----------
    node_list : list of Hashable
        List of the nodes which are in the graph (nodes must be hashable)
    edge_list : list of tuple of Hashable
        List of edges in the graph

    Returns
    -------
    connected_components : list of sets of Hashable
        List of the connected components of the graph
    """
    # Create a parents dict
    parents: dict[HCT, HCT] = {node: node for node in node_list}
    # For each edge, ensure that the nodes have a common parent
    for n1, n2 in edge_list:
        try:
            p1 = _get_parent(n1, parents)
            p2 = _get_parent(n2, parents)
        except KeyError as e:
            raise ValueError(
                "Edge list contains nodes not found in node list!"
            ) from e
        if p1 != p2:
            common_parent = min(p1, p2)  # Using min to select a single parent
            parents[n1] = common_parent
            parents[n2] = common_parent
    # The parents for every node can now be evaluated
    components: defaultdict[HCT, set[HCT]] = defaultdict(set)
    for node in node_list:
        p = _get_parent(node, parents)
        components[p].add(node)
    # Return the list of the component sets
    return list(components.values())


def _get_parent(node: HCT, parents: dict[HCT, HCT]) -> HCT:
    """
    Get the (root) parent of a node

    Parameters
    ----------
    node : str
        The node to determine the parent of
    parents : dict of str to str
        Dictionary describing the parent of each node
    """
    if node == parents[node]:
        return node
    else:
        parents[node] = _get_parent(parents[node], parents)
        return parents[node]


def find_degree(
    node_list: list[HCT], edge_list: list[tuple[HCT, HCT]]
) -> dict[HCT, int]:
    """
    Find the degree of nodes within a graph

    Parameters
    ----------
    node_list : list of Hashable
        List of the nodes which are in the graph (nodes must be hashable)
    edge_list : list of tuple of Hashable
        List of edges in the graph

    Returns
    -------
    degree_dict : dict of Hashable to int
        Dictionary keyed by node, with value corresponding to degree of
        the node
    """
    degree: defaultdict[HCT, int] = defaultdict(int)
    for n1, n2 in edge_list:
        degree[n1] += 1
        degree[n2] += 1
    for n in node_list:
        degree[n] += 0
    return degree


def find_neighbors(
    node_list: list[HCT], edge_list: list[tuple[HCT, HCT]]
) -> dict[HCT, set[HCT]]:
    """
    Find the neighbors of nodes within a graph

    Parameters
    ----------
    node_list : list of Hashable
        List of the nodes which are in the graph (nodes must be hashable)
    edge_list : list of tuple of Hashable
        List of edges in the graph

    Returns
    -------
    neighors : dict of Hashable to set of Hashable
        A dictionary keyed by node, with values of sets of neighbors to
        the key node
    """
    neighbors: dict[HCT, set[HCT]] = defaultdict(set)
    for n1, n2 in edge_list:
        neighbors[n1].add(n2)
        neighbors[n2].add(n1)
    for n in node_list:
        # This just forces an empty set to be created if
        # it wasn't already created during edge iteration
        _ = neighbors[n]
    return neighbors


def find_representative_nodes(
    node_list: list[HCT], edge_list: list[tuple[HCT, HCT]]
) -> dict[HCT, set[HCT]]:
    """
    Find representative nodes in a graph by selecting the node of highest degree
    from each component of the graph

    Parameters
    ----------
    node_list : list of Hashable
        List of the nodes in the graph (nodes must be hashable)
    edge_list : list of tuple of Hashable
        List of edges in the graph, in the form of a tuple of nodes which it connects

    Returns
    -------
    representative_dict : dict of Hashable to set of Hashable
        A dictionary with the representative nodes as the keys,
        and a set of the nodes they represent as the values

    Note
    ----
    The sets of nodes in the returned dictionary will include the representative node
    """
    representative_nodes: dict[HCT, set[HCT]] = defaultdict(set)
    components = find_connected_components(
        node_list=node_list, edge_list=edge_list
    )
    degree_dict = find_degree(node_list=node_list, edge_list=edge_list)
    for component in components:
        representative_nodes[max(component, key=lambda n: degree_dict[n])] = (
            component
        )
    return representative_nodes

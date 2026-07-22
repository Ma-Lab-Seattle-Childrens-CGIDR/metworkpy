"""
Class representing Hypergraphs
"""

import itertools
from typing import Any, Hashable


class HyperGraph:
    """
    Representation of a Hypergraph, with nodes connected by edges that can
    contain more than two nodes
    """

    def __init__(self, **attr):
        # Edges, which include an edge ID mapped to the nodes the edge includes
        self._edges: dict[Hashable, tuple[Hashable, ...]] = {}
        # Nodes, which map node IDs to node properties
        self._nodes: dict[Hashable, dict[Hashable, Any]] = {}
        # Create a adjacency map, which maps from nodes to neighbors
        # This will be a nested dict,
        # The First layer maps the node source to dicts,
        # The second layer maps these neighbors to sets of edge ids,
        # which indicate which edges connect the two nodes
        self._adj: dict[Hashable, dict[Hashable, set[Hashable]]] = {}
        # Edge properties, which could include weights
        self._edge_properties: dict[Hashable, dict[Hashable, Any]] = {}
        # Any attributes of the graph itself
        self._attributes = attr

    def _add_node(self, node: Hashable, properties: dict[Hashable, Any]):
        """Add a node to the graph with associated properties"""
        self._nodes[node] = properties

    def _add_edge(
        self,
        edge: tuple[Hashable, ...],
        edge_id: Hashable,
        properties: dict[Hashable, Any],
    ):
        """Add an edge to the graph with associated properties"""
        # Add nodes as needed
        for n in edge:
            if n not in self._nodes:
                self._add_node(n, {})
        # Add edge to the edge dict
        self._edges[edge_id] = edge
        # Add Edge properties
        self._edge_properties[edge_id] = properties
        # Update the adjacency
        for n1, n2 in itertools.combinations(edge, 2):
            self._add_adj(n1, n2, edge_id)

    def _add_adj(self, n1, n2, edge_id):
        """Add a connection between n1<->n2 (undirected), can overwrite in derived
        classes to create a directed version"""
        # One direction
        if n1 not in self._adj:
            self._adj[n1] = {}
        if n2 not in self._adj[n1]:
            self._adj[n1][n2] = {edge_id}
        else:
            self._adj[n1][n2].add(edge_id)

        # Then the opposite direction
        if n2 not in self._adj:
            self._adj[n2] = {}
        if n1 not in self._adj[n2]:
            self._adj[n2][n1] = {edge_id}
        else:
            self._adj[n2][n1].add(edge_id)

    def _remove_edge(self, edge_id):
        """Remove an edge from the HyperGraph"""
        _ = self._edges.pop(edge_id)
        _ = self._edge_properties.pop(edge_id)
        self._remove_edge_adj(edge_id)

    def _remove_node(self, node, remove_associated_edges: bool):
        """Remove a node from the HyperGraph"""
        # Remove from the node dict
        try:
            _ = self._nodes.pop(node)
        except KeyError:
            raise ValueError(
                f"Tried to remove {node}, but it wasn't in the Hypergraph"
            )
        # Remove from edges/remove edges
        edges_with_node = []
        for edge_id, edge in self._edges.items():
            if node in edge:
                edges_with_node.append(edge_id)
        if remove_associated_edges:
            for edge_id in edges_with_node:
                self._remove_edge(edge_id)
        else:
            for edge_id in edges_with_node:
                # Remove the node from the edge
                self._edges[edge_id] = tuple(
                    n for n in self._edges[edge_id] if n != node
                )
            self._remove_node_adj(node)

    def _remove_node_adj(self, node):
        """Remove a node from the adjacency map"""
        # Silently ignore if the node isn't present, since it might not have
        # any neighbors
        _ = self._adj.pop(node, None)
        for _, n_dict in self._adj.items():
            _ = n_dict.pop(node, None)

    def _remove_edge_adj(self, edge_id):
        """Remove an edge from the adjacency map"""
        for _, n_dict in self._adj.items():
            n_dict = {
                n: {e for e in es if e != edge_id} for n, es in n_dict.items()
            }

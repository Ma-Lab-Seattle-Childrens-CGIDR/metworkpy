"""
Class representing a directed HyperGraph
"""

from networkx.classes.coreviews import AdjacencyView


import itertools
from typing import Any, Hashable, Iterable, Optional, Union

from .hypergraph import HyperGraph
from .views import ReadOnlyDictView


class DiHyperGraph(HyperGraph):
    """
    Representation of a Directed Hypergraph, with nodes connected by
    edges which contain a set of source nodes, and a set of target nodes.
    """

    def __init__(self, **attr):
        super().__init__(**attr)
        self._edges: dict[
            Hashable, tuple[tuple[Hashable, ...], tuple[Hashable, ...]]
        ] = {}
        # Successors will be treated as adjacency
        self._succ: dict[Hashable, dict[Hashable, set[Hashable]]] = self._adj
        self._pred: dict[Hashable, dict[Hashable, set[Hashable]]] = {}

    def add_edge(
        self,
        id: Hashable,
        u: Iterable[Hashable],
        v: Optional[Iterable[Hashable]],
        **kwargs,
    ):
        """
        Add an edge from u->v in the HyperGraph

        Parameters
        ----------
        id : Hashable
            The id of the edge, can be any Hashable
        u : iterable of node ids
            An iterable of source nodes in the edge, if the nodes are
            not already in the HyperGraph they will be added.
        v : iterable of node ids
            An iterable of target nodes in the edge, if the nodes are
            not already in the HyperGraph they will be added.
        kwargs
            Keyword arguments are added as properties of the edge
        """
        if v is None:
            raise ValueError(
                "For directed HyperGraphs, target nodes are required"
            )
        self._add_edge(id, (tuple(u), tuple(v)), kwargs)

    def add_edges_from(  # type: ignore  ## Have to change directional handling
        self,
        edge_group: Iterable[
            Union[
                tuple[
                    Hashable,
                    Iterable[Hashable],
                    Iterable[Hashable],
                ],
                tuple[
                    Hashable,
                    Iterable[Hashable],
                    Iterable[Hashable],
                    dict[str, Any],
                ],
            ]
        ],
        **kwargs,
    ):
        """
        Add a group of edges to the directed HyperGraph

        Parameters
        ----------
        edge_group : tuple of (id, sources, targets) or (id, source, targets, properties)
            The edges to add to the directed HyperGraph, should be an iterable of (id, source, targets),
            where id is the id of the edge (any Hashable), and sources/targets are the source/target
            nodes in the edge (a container of node ids). Alternatively, can be an iterable of
            (id, sources, targets, properties) where proprties is a dict of edge proprties.
        kwargs
            Keyword arguments are used as default properties for the edges,
            which can be overwritten if proprty dicts are passed as part
            of the `edge_group`
        """
        for e in edge_group:
            len_e = len(e)
            if len_e == 4:
                edge_id, sources, targets, proprties = e  # type: ignore
            elif len_e == 3:
                edge_id, sources, targets = e  # type: ignore
                properties = {}
            else:
                raise ValueError("Invalid edge group, must be a 3 or 4 tuple")
            if edge_id in self._edge_properties:
                edge_property_dict = self._edge_properties[edge_id]
            else:
                edge_property_dict = {}
            edge_property_dict.update(kwargs)
            edge_property_dict.update(properties)
            self._add_edge(
                edge_id, (tuple(sources), tuple(targets)), edge_property_dict
            )

    def _add_edge(  # type: ignore  ## This has to be different
        self,
        edge_id: Hashable,
        edge: tuple[tuple[Hashable, ...], tuple[Hashable, ...]],
        properties: dict[str, Any],
    ):
        """Add an edge to the HyperGraph with associated properties"""
        source, target = edge
        # Add nodes as needed
        for s in source:
            if s not in self._nodes:
                self._add_node(s, {})
        for t in target:
            if t not in self._nodes:
                self._add_node(t, {})
        # Add the edge to the edge dict
        self._edges[edge_id] = edge
        # Add Edge properties
        if edge_id not in self._edge_properties:
            self._edge_properties[edge_id] = properties
        else:
            self._edge_properties[edge_id].update(properties)
        # Update the adjacency
        for s, t in itertools.product(source, target):
            self._add_adj(s, t, edge_id)

    def _add_adj(self, u, v, edge_id):
        """Add a connection between u->v (directed)"""
        # First the successors
        if u not in self._succ:
            self._succ[u] = {}
        if v not in self._succ[u]:
            self._succ[u][v] = {edge_id}
        else:
            self._succ[u][v].add(edge_id)
        # Then the predecessors
        if v not in self._pred:
            self._pred[v] = {}
        if u not in self._pred[v]:
            self._pred[v][u] = {edge_id}
        else:
            self._pred[v][u].add(edge_id)

    def _remove_edge(self, edge_id):
        """Remove an edge from the HyperGraph"""
        _ = self._edges.pop(edge_id)
        _ = self._edge_properties.pop(edge_id)
        self._remove_edge_adj(edge_id, self._succ)
        self._remove_edge_adj(edge_id, self._pred)

    def _remove_node(self, node, remove_associated_edges: bool):
        """Remove a node from the Directed HyperGraph"""
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
            source, target = edge
            if node in source or node in target:
                edges_with_node.append(edge_id)
        if remove_associated_edges:
            for edge_id in edges_with_node:
                self._remove_edge(edge_id)
        else:
            for edge_id in edges_with_node:
                source, target = self._edges[edge_id]
                source = tuple(n for n in source if n != node)
                target = tuple(n for n in target if n != node)
                if len(source) < 2 or len(target) < 2:
                    self._remove_edge(edge_id)
                else:
                    self._edges[edge_id] = (source, target)
        self._remove_node_adj(node, self._succ)
        self._remove_node_adj(node, self._pred)

    @property
    def successors(self):
        """
        Directed HyperGraph adjacency object, holding sucessors of a node (nodes which are
        the targets of at least one edge where the node is a source).

        This is a read-only dict of dicts of dicts, that is the first level
        dict maps nodes to their successors, and the second level maps
        those successors to a set describing the edges which contain the
        first and second level.
        """
        return AdjacencyView(self._succ)

    @property
    def predecessors(self):
        """
        Directed HyperGraph adjacency object, holding predecessors of a node (nodes which are
        the sources of at least one edge where the node is a target).

        This is a read-only dict of dicts of dicts, that is the first level
        dict maps nodes to their predecessors, and the second level maps
        those predecessors to a set describing the edges which contain the
        first and second level.
        """
        return AdjacencyView(self._pred)

    @property
    def edges(self):
        """
        A read-only dict of edge-id to tuple of tuples (sources, targets)
        of nodes included in the edge
        """
        return ReadOnlyDictView(self._edges)

    #################
    ### Traversal ###
    #################
    def _get_neighbors_of_node(self, node_id: Hashable) -> set[Hashable]:
        return set(self._succ[node_id].keys())

    def _get_neighbors_of_edge(self, edge_id: Hashable) -> set[Hashable]:
        neighbors = set()
        _, targets = self._edges[edge_id]
        for n in targets:
            for edge_set in self._succ[n].values():
                neighbors |= edge_set
        return neighbors

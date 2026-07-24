"""
Class representing Hypergraphs
"""

from collections import deque
import itertools
from typing import cast, Any, Hashable, Iterable, Optional, Union

from .views import AtlasView, AdjacencyView, ReadOnlyDictView


class HyperGraph:
    """
    Representation of a Hypergraph, with nodes connected by edges that can
    contain more than two nodes
    """

    def __init__(self, **attr):
        # Edges, which include an edge ID mapped to the nodes the edge includes
        self._edges: dict[Hashable, tuple[Hashable, ...]] = {}
        # Nodes, which map node IDs to node properties
        self._nodes: dict[Hashable, dict[str, Any]] = {}
        # Create a adjacency map, which maps from nodes to neighbors
        # This will be a nested dict,
        # The First layer maps the node source to dicts,
        # The second layer maps these neighbors to sets of edge ids,
        # which indicate which edges connect the two nodes
        self._adj: dict[Hashable, dict[Hashable, set[Hashable]]] = {}
        # Edge properties, which could include weights
        self._edge_properties: dict[Hashable, dict[str, Any]] = {}
        # Any attributes of the graph itself
        self._attributes = attr

    def add_node(self, node: Hashable, **kwargs):
        """
        Add a node to the HyperGraph

        Parameters
        ----------
        node : Hashable
            The node to add
        kwargs
            Keyword arguments are added as properties of the node
        """
        self._add_node(node, kwargs)

    def add_edge(
        self,
        id: Hashable,
        u: Iterable[Hashable],
        v: Optional[Iterable[Hashable]],
        **kwargs,
    ):
        """
        Add an edge (u<->v) to the HyperGraph

        Parameters
        ----------
        id : Hashable
            The id of the edge, can be any Hashable
        u : iterable of node ids
            An iterable of source nodes in the edge, if the nodes are
            not already in the HyperGraph they will be added.
            In undirected HyperGraphs, nodes in u and v will
            all be included in the edge with no difference between
            the nodes.
        v : iterable of node ids, optional
            An iterable of target nodes in the edge, if the nodes are
            not already in the HyperGraph they will be added.
            In undirected HyperGraphs, nodes in u and v will
            all be included in the edge with no difference between
            the nodes.
        kwargs
            Keyword arguments are added as properties of the edge
        """
        if v is not None:
            edge = tuple(n for n in itertools.chain(u, v))
        else:
            edge = tuple(u)
        self._add_edge(id, edge, kwargs)

    def add_nodes_from(
        self,
        node_group: Iterable[Union[Hashable, tuple[Hashable, dict[str, Any]]]],
        **kwargs,
    ):
        """
        Add a group of nodes to the HyperGraph

        Parameters
        ----------
        node_group : Iterable of node_id, or (node_id,proprties)
            The nodes to add, can be an iterable of the node ids, or an iterable of
            tuples of (node_id, properties) where properties is a dict of
            proprties that will be added to the node
        kwargs
            Keyword arguments are used as default properties for the
            nodes in `node_group`
        """
        for ns in node_group:
            try:
                id, props = ns  # type: ignore
                id = cast(Hashable, id)
                props = cast(dict[str, Any], props)
                newdict = kwargs.copy()
                newdict.update(props)
            except TypeError:
                id = ns
                newdict = kwargs
            self._add_node(id, newdict)

    def add_edges_from(
        self,
        edge_group: Iterable[
            Union[
                tuple[Hashable, Iterable[Hashable]],
                tuple[Hashable, Iterable[Hashable], dict[str, Any]],
            ]
        ],
        **kwargs,
    ):
        """
        Add a group of edges to the HyperGraph

        Parameters
        ----------
        edge_group : tuple of (id, nodes) or (id, nodes, properties)
            The edges to add to the HyperGraph, should be an iterable of (id, nodes),
            where id is the id of the edge (any Hashable), and nodes is the nodes in the edge
            (a container of node ids). Alternatively, can be an iterable of (id, nodes, properties)
            where proprties is a dict of edge proprties.
        kwargs
            Keyword arguments are used as default properties for the edges,
            which can be overwritten if proprty dicts are passed as part
            of the `edge_group`
        """
        for e in edge_group:
            len_e = len(e)
            if len_e == 3:
                edge_id, edge, proprties = e  # type: ignore
            elif len_e == 2:
                edge_id, edge = e  # type: ignore
                properties = {}
            else:
                raise ValueError("Invalid edge group, must be a 2 or 3 tuple")
            if edge_id in self._edge_properties:
                edge_property_dict = self._edge_properties[edge_id]
            else:
                edge_property_dict = {}
            edge_property_dict.update(kwargs)
            edge_property_dict.update(properties)
            self._add_edge(edge_id, tuple(edge), edge_property_dict)

    def _add_node(self, node: Hashable, properties: dict[str, Any]):
        """Add a node to the graph with associated properties"""
        self._nodes[node] = properties

    def _add_edge(
        self,
        edge_id: Hashable,
        edge: tuple[Hashable, ...],
        properties: dict[str, Any],
    ):
        """Add an edge to the graph with associated properties"""
        # Add nodes as needed
        for n in edge:
            if n not in self._nodes:
                self._add_node(n, {})
        # Add edge to the edge dict
        self._edges[edge_id] = edge
        # Add Edge properties
        if edge_id not in self._edge_properties:
            self._edge_properties[edge_id] = properties
        else:
            self._edge_properties[edge_id].update(properties)
        # Update the adjacency
        for n1, n2 in itertools.combinations(edge, 2):
            self._add_adj(n1, n2, edge_id)

    def _add_adj(self, u, v, edge_id):
        """Add a connection between n1<->n2 (undirected), can overwrite in derived
        classes to create a directed version"""
        # One direction
        if u not in self._adj:
            self._adj[u] = {}
        if v not in self._adj[u]:
            self._adj[u][v] = {edge_id}
        else:
            self._adj[u][v].add(edge_id)

        # Then the opposite direction
        if v not in self._adj:
            self._adj[v] = {}
        if u not in self._adj[v]:
            self._adj[v][u] = {edge_id}
        else:
            self._adj[v][u].add(edge_id)

    def _remove_edge(self, edge_id):
        """Remove an edge from the HyperGraph"""
        _ = self._edges.pop(edge_id)
        _ = self._edge_properties.pop(edge_id)
        self._remove_edge_adj(edge_id, self._adj)

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
                # Check if the edge needs to be removed (i.e.
                # the edge has fewer than two nodes
                if len(self._edges[edge_id]) < 2:
                    self._remove_edge(edge_id)
        self._remove_node_adj(node, self._adj)

    def _remove_node_adj(self, node, adj_map):
        """Remove a node from the adjacency map"""
        # Silently ignore if the node isn't present, since it might not have
        # any neighbors
        _ = adj_map.pop(node, None)
        for _, n_dict in adj_map.items():
            _ = n_dict.pop(node, None)

    def _remove_edge_adj(self, edge_id, adj_map):
        """Remove an edge from the adjacency map"""
        for _, n_dict in adj_map.items():
            cutoff_neighbors = []
            for target_id, edge_set in n_dict.items():
                edge_set.discard(edge_id)
                if len(edge_set) == 0:
                    cutoff_neighbors.append(target_id)
            for t in cutoff_neighbors:
                _ = n_dict.pop(t, None)

    @property
    def adj(self):
        """
        HyperGraph adjacency object, holding neighbors (nodes which are together
        in at least one edge).

        This is a read-only dict of dicts of dicts, that is the first level
        dict maps nodes to their neighbors, and the second level maps
        those neighbors to a set describing the edges which contain the
        first and second level.
        """
        return AdjacencyView(self._adj)

    @property
    def nodes(self):
        return AtlasView(self._nodes)

    @property
    def edges(self):
        """A read-only dict of edge-id to tuple of nodes included in the edge"""
        return ReadOnlyDictView(self._edges)

    @property
    def edge_proprties(self):
        return AtlasView(self._edge_properties)

    def __contains__(self, n: Hashable) -> bool:
        """
        Check if a node is in the HyperGraph (e.g. `n in HG`)
        """
        try:
            return n in self._nodes
        except TypeError:
            return False

    def __len__(self) -> int:
        """
        Return number of nodes in the HyperGraph
        """
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, n) -> AtlasView:
        return self.adj[n]

    #################
    ### Traversal ###
    #################
    def _get_neighbors_of_node(self, node_id: Hashable) -> set[Hashable]:
        return set(self._adj[node_id].keys())

    def _get_neighbors_of_edge(self, edge_id: Hashable) -> set[Hashable]:
        neighbors = set()
        for n in self._edges[edge_id]:
            for edge_set in self._adj[n].values():
                neighbors |= edge_set
        return neighbors

    # Based on code from HyperGraphX, licensed under the BSD-3-clause
    # see: https://github.com/HGX-Team/hypergraphx?tab=License-1-ov-file
    def _bfs_nodes(
        self, start, max_depth: Optional[int] = None
    ) -> set[Hashable]:
        """
        Perform a breadth first search for neighboring nodes
        from `start`

        Parameters
        ----------
        start : Hashable
            The starting node
        max_depth : int,optional
            The maximum distance from the starting node to explore
            (a value of 0 includes only the starting node, 1 includes
            the starting node and its direct neighbors, etc.)

        Returns
        -------
        set of Hashable
            The nodes found within max_distance of `start`
        """
        if start not in self._nodes:
            raise ValueError(f"Node {start} not found in HyperGraph")
        visited = set()
        queue = deque([(start, 0)])

        while queue:
            node, depth = queue.popleft()
            if node not in visited:
                visited.add(node)
                if max_depth is None or depth < max_depth:
                    queue.extend(
                        (n, depth + 1)
                        for n in self._get_neighbors_of_node(node)
                        if n not in visited
                    )
        return visited

    # Based on code from HyperGraphX, licensed under the BSD-3-clause
    # see: https://github.com/HGX-Team/hypergraphx?tab=License-1-ov-file
    def _bfs_edges(
        self, start, max_depth: Optional[int] = None
    ) -> set[Hashable]:
        """
        Perform a breadth first search for neighboring edges (edges which share
        at least 1 node) from `start`

        Parameters
        ----------
        start : Hashable
            The starting edge
        max_depth : int,optional
            The maximum distance from the starting edge to explore
            (a value of 0 includes only the starting edge, 1 includes
            the starting edge and its direct neighbors, etc.)

        Returns
        -------
        set of Hashable
            The edges found within max_distance of `start`
        """
        if start not in self._edges:
            raise ValueError(f"Edge {start} not found in HyperGraph")
        visited = set()
        queue = deque([(start, 0)])

        while queue:
            edge, depth = queue.popleft()
            if edge not in visited:
                visited.add(edge)
                if max_depth is None or depth < max_depth:
                    queue.extend(
                        (e, depth + 1)
                        for e in self._get_neighbors_of_edge(edge)
                        if e not in visited
                    )
        return visited

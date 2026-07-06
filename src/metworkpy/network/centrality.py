# The code for closeness_centrality_subset, betweenness_centrality_subset,
# _rescale, and _accumulate_subset is a modified
# version of the NetworkX closeness_centrality, betweenness_centrality_subset,
# _rescale, and _accumulate_subset functions.
# NetworkX is licensed under the BSD-3-Clause license
# Reproduced below
# Copyright (c) 2004-2025, NetworkX Developers
# Aric Hagberg <hagberg@lanl.gov>
# Dan Schult <dschult@colgate.edu>
# Pieter Swart <swart@lanl.gov>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
#  * Neither the name of the NetworkX Developers nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Submodule containing functions for finding centrality
of nodes in a network
"""

from collections.abc import Iterable
import functools
from typing import cast, Hashable, Optional, Union

import networkx as nx
from networkx.algorithms.centrality.betweenness import (
    _single_source_dijkstra_path_basic as dijkstra,  # type: ignore
)
from networkx.algorithms.centrality.betweenness import (
    _single_source_shortest_path_basic as shortest_path,  # type: ignore
)


def closeness_centrality_subset(
    G: Union[nx.Graph, nx.DiGraph],
    targets: Optional[Iterable[Hashable]] = None,
    u: Optional[Hashable] = None,
    distance: Optional[Hashable] = None,
    wf_improved: bool = True,
) -> Union[float, dict[Hashable, float]]:
    r"""Compute closeness centrality for nodes, considering only paths
    to a subset of other nodes.

    Subset closeness centrality, based on closeness centrality [1]_, of a
    node `u` is the reciprocal of the avergage shortest path distance
    to `u` over all `n-1` reachable nodes which are in `targets`

    .. math::

        C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    where `d(v, u)` is the shortest-path distance between `v` and `u`, where
    `v` is in `targets`, and `n-1` is the number of targets reachable from `u`.
    Notice that the closeness distance function computes the incoming
    distance to `u` for directed graphs. To use outward distance, act
    on `G.reverse()`.

    Notice that higher values of closeness indicate higher centrality.

    Wasserman and Faust propose an improved formula for graphs with
    more than one connected component. The result is "a ratio of the
    fraction of actors in the group who are reachable, to the average
    distance" from the reachable actors [2]_. You might think this
    scale factor is inverted but it is not. As is, nodes from small
    components receive a smaller closeness value. Letting `N` denote
    the number of nodes in the graph,

    .. math::

        C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    Parameters
    ----------
    G : graph
      A NetworkX graph

    targets : list of nodes, optional
        The nodes to use as targets for the shortest paths in closeness

    u : node, optional
      Return only the value for node u

    distance : edge attribute key, optional (default=None)
      Use the specified edge attribute as the edge distance in shortest
      path calculations.  If `None` (the default) all edges have a distance of 1.
      Absent edge attributes are assigned a distance of 1. Note that no check
      is performed to ensure that edges have the provided attribute.

    wf_improved : bool, optional (default=True)
      If True, scale by the fraction of nodes reachable. This gives the
      Wasserman and Faust improved formula. For single component graphs
      it is the same as the original formula.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.

    Notes
    -----
    This function is the
    `closeness_centrality <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html>`_
    function from
    `NetworkX <https://networkx.org/>`_ modified
    to only compute distances to a subset of the nodes in the graph. NetworkX
    is licensed under a
    `BSD-3-Clause license <https://github.com/networkx/networkx?tab=License-1-ov-file>`_.

    The closeness centrality is normalized to `(n-1)/(|T|-1)` where
    `n` is the number of targets in the connected part of graph
    containing the node, and `|T|` is the total number of targets. If the graph
    is not completely connected, this algorithm computes the closeness centrality
    for each connected part separately scaled by the number of targets in that parts.

    If the 'distance' keyword is set to an edge attribute key then the
    shortest-path length will be computed using Dijkstra's algorithm with
    that edge attribute as the edge weight.

    The closeness centrality uses *inward* distance to a node, not outward.
    If you want to use outword distances apply the function to `G.reverse()`

    References
    ----------
    .. [1] Linton C. Freeman: Centrality in networks: I.
       Conceptual clarification. Social Networks 1:215-239, 1979.
       https://doi.org/10.1016/0378-8733(78)90021-7
    .. [2] pg. 201 of Wasserman, S. and Faust, K.,
       Social Network Analysis: Methods and Applications, 1994,
       Cambridge University Press.
    """
    if G.is_directed():
        assert isinstance(G, nx.DiGraph), "NetworkX is_directed failed"
        G = G.reverse()
    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(
            nx.single_source_dijkstra_path_length, weight=distance
        )
    else:
        path_length = nx.single_source_shortest_path_length
    if u is None:
        nodes = G.nodes
    else:
        nodes = [u]
    assert isinstance(nodes, Iterable), (
        "Can't iterate over nodes in the network"
    )
    if targets is None:
        targets = cast(Iterable, G.nodes)
    targets_set = set(targets)
    closeness_dict = {}
    target_count = len(targets_set)
    for n in nodes:
        sp = path_length(G, n)
        totsp = 0.0
        reachable_targets = 0.0
        for id, length in sp.items():
            if id in targets_set:
                totsp += length
                reachable_targets += 1.0
        _closeness_centrality = 0.0
        if totsp > 0.0 and target_count > 1:
            _closeness_centrality = (reachable_targets - 1.0) / totsp
            # normalize to number of nodes-1 in connected part
            if wf_improved:
                s = (reachable_targets - 1.0) / (target_count - 1)
                _closeness_centrality *= s
        closeness_dict[n] = _closeness_centrality
    if u is not None:
        return closeness_dict[u]
    return closeness_dict


def betweenness_centrality_subset(
    G: Union[nx.Graph, nx.DiGraph],
    targets: Optional[Iterable[Hashable]] = None,
    normalized=True,
    weight=None,
):
    r"""Compute betweenness centrality for a subset of nodes.

    .. math::

       c_B(v) =\sum_{s,t \in T} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where :math:`T` is the set of targets,
    :math:`\sigma(s, t)` is the number of shortest :math:`(s, t)`-paths,
    and :math:`\sigma(s, t|v)` is the number of those paths
    passing through some  node :math:`v` other than :math:`s, t`.
    If :math:`s = t`, :math:`\sigma(s, t) = 1`,
    and if :math:`v \in {s, t}`, :math:`\sigma(s, t|v) = 0` [2]_.

    The normalization is slightly different from NetworkX,
    as it normalizes only to the possible (s,t) pairs in targets,
    rather than to all possible (s,t) pairs in the network.


    Parameters
    ----------
    G : graph
      A NetworkX graph.

    targets: list of nodes
      Nodes to use as sources/targets for shortest paths in betweenness

    normalized : bool, optional
      If True the betweenness values are normalized by :math:`2/((n-1)(n-2))`
      for graphs, and :math:`1/((n-1)(n-2))` for directed graphs where :math:`n`
      is the number of nodes in targets.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      Weights are used to calculate weighted shortest paths, so they are
      interpreted as distances.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value.

    Notes
    -----
    The basic algorithm is from [1]_.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The normalization might seem a little strange but it is
    designed to make betweenness_centrality(G) be the same as
    betweenness_centrality_subset(G,sources=G.nodes(),targets=G.nodes()).

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    are easy to count. Undirected paths are tricky: should a path
    from "u" to "v" count as 1 undirected path or as 2 directed paths?

    References
    ----------
    .. [1] Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """
    betweenness_dict = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    if targets is None:
        targets = G.nodes
    for s in targets:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = shortest_path(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = dijkstra(G, s, weight)
        betweenness_dict = _accumulate_subset(
            betweenness_dict, S, P, sigma, s, targets
        )
    betweenness_dict = _rescale(
        betweenness_dict,
        set(targets),
        normalized=normalized,
        directed=G.is_directed(),
    )
    return betweenness_dict


def betweenness_centrality_bipartite_subset(
    G: Union[nx.Graph, nx.DiGraph],
    node_partition: Iterable[Hashable],
    targets: Optional[Iterable[Hashable]] = None,
    normalized=True,
    weight=None,
):
    r"""
    Compute betweenness centrality for a subset of
    nodes on a bipartite network, where the node subset
    comes from one of the partitions and nodes in the
    other partitions are treated as edges

    .. math::

       c_B(v) =\sum_{s,t \in T} \frac{\sigma(s, t|v)}{\sigma(s, t)}

    where :math:`T` is the set of targets, :math:`\sigma(s, t)` is the number of
    shortest :math:`(s, t)`-paths, and :math:`\sigma(s, t|v)` is the number of
    those paths passing through some  node :math:`v` other than :math:`s, t`.
    If :math:`s = t`, :math:`\sigma(s, t) = 1`, and if :math:`v \in {s, t}`,
    :math:`\sigma(s, t|v) = 0` [2]_.

    The betweenness can also be further normalized to
    the number of possible pairs of s and t.

    Parameters
    ----------
    G : graph
      A NetworkX graph, should be a bipartite graph (this condition is not checked).

    node_partition : Iterable[Hashable]
        One of the two sets of nodes in the bipartite graph, specifically
        the set which contains all the targets

    targets: list of nodes, optional
      Nodes to use as sources/targets for shortest paths in betweenness,
      all of these should fall into a single partition of the bipartite
      graph (this condition is not checked). If None, uses all nodes
      in the node_partition

    normalized : bool, optional
      If True the betweenness values are normalized by :math:`2/((n-1)(n-2))`
      for graphs, and :math:`1/((n-1)(n-2))` for directed graphs where :math:`n`
      is the number of nodes in targets.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with betweenness centrality as the value. This
       includes betweenness values for all the nodes in the Graph
       (in both sets of the partition).

    Notes
    -----
    The basic algorithm is from [1]_.

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    are easy to count. Undirected paths are tricky: should a path
    from "u" to "v" count as 1 undirected path or as 2 directed paths?

    For betweenness_centrality we report the number of undirected
    paths when G is undirected.

    For betweenness_centrality_subset the reporting is different.
    If the source and target subsets are the same, then we want
    to count undirected paths. But if the source and target subsets
    differ -- for example, if sources is {0} and targets is {1},
    then we are only counting the paths in one direction. They are
    undirected paths but we are counting them in a directed way.
    To count them as undirected paths, each should count as half a path.

    References
    ----------
    .. [1] Ulrik Brandes: A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """
    # All nodes start with 0.0 betweenness
    betweenness_dict = dict.fromkeys(G, 0.0)
    # Project the network onto the node partition
    projected_graph = nx.bipartite.projected_graph(G, node_partition)
    # If no targets provided, use all nodes in the node_partition
    if targets is None:
        targets = projected_graph.nodes
    # For each source,
    for s in targets:
        # single source shortest paths
        S, P, sigma, _ = shortest_path(projected_graph, s)
        betweenness_dict = _accumulate_bipartite_subset(
            G, betweenness_dict, S, P, sigma, s, targets
        )
    betweenness_dict = _rescale(
        betweenness_dict,
        set(targets),
        normalized=normalized,
        directed=G.is_directed(),
    )
    return betweenness_dict


def _common_neighbors(
    G: Union[nx.Graph, nx.DiGraph], n1: Hashable, n2: Hashable
) -> set[Hashable]:
    if G.is_directed():
        assert isinstance(G, nx.DiGraph)
        return set(G.successors(n1)) & set(G.predecessors(n2))  # type: ignore # ty error
    return set(G.neighbors(n1)) & set(G.neighbors(n2))


# NOTE:
# G is the non-projected graph, used for finding common neighbors
# Betweenness dict is the betweenness to update
# S is a stack of nodes visited on the BFS
# P is a dict of node to list of predecessors (on ANY shortest path)
# Sigma[v] is a count of shortest paths from source to v
# delta[v] is the dependency of source on v \in V (starts at 0.0)
def _accumulate_bipartite_subset(
    G: Union[nx.Graph, nx.DiGraph],
    betweenness_dict: dict[Hashable, float],
    S,  # Nodes on paths
    P,  # Predecessors
    sigma: dict[Hashable, float],  # Path counts
    source: Hashable,  # Source Node
    targets: Iterable[Hashable],  # Set of targets
):
    delta = dict.fromkeys(G, 0.0)
    target_set = set(targets) - {source}
    # S is a stack that we will go through to
    # visit the nodes from the BFS in reverse order
    while S:
        # w is the current node
        w = S.pop()
        for v in P[w]:
            if w in target_set:
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
            else:
                c = delta[w] * sigma[v] / sigma[w]
            for u in _common_neighbors(G, v, w):
                betweenness_dict[u] += c
            delta[v] += c
        if w != source:
            betweenness_dict[w] += delta[w]
    return betweenness_dict


def _rescale(
    betweenness: dict[Hashable, float],
    subset: set[Hashable],
    normalized: bool,
    directed=False,
):
    """
    betweenness_centrality_subset helper.

    Uses different normalization that the default in networkx
    """
    len_subset = len(subset)
    if normalized:
        if len_subset < 2:
            subset_scale = None  # no normalization b=0 for all nodes
            non_subset_scale = None
        elif len_subset == 2:
            subset_scale = (
                1.0  # No normalization for subset, since it's only endpoints
            )
            non_subset_scale = 1.0 / ((len_subset) * (len_subset - 1))
        else:
            subset_scale = 1.0 / ((len_subset - 1) * (len_subset - 2))
            non_subset_scale = 1.0 / ((len_subset) * (len_subset - 1))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            non_subset_scale = 0.5
            subset_scale = 0.5
        else:
            non_subset_scale = None
            subset_scale = None
    if (non_subset_scale is not None) and (subset_scale is not None):
        for v in betweenness:
            betweenness[v] *= subset_scale if v in subset else non_subset_scale
    return betweenness


def _accumulate_subset(betweenness, S, P, sigma, s, targets):
    delta = dict.fromkeys(S, 0.0)
    target_set = set(targets) - {s}
    while S:
        w = S.pop()
        if w in target_set:
            coeff = (delta[w] + 1.0) / sigma[w]
        else:
            coeff = delta[w] / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness

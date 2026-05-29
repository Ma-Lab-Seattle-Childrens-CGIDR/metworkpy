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

    where $S$ is the set of sources, $T$ is the set of targets,
    $\sigma(s, t)$ is the number of shortest $(s, t)$-paths,
    and $\sigma(s, t|v)$ is the number of those paths
    passing through some  node $v$ other than $s, t$.
    If $s = t$, $\sigma(s, t) = 1$,
    and if $v \in {s, t}$, $\sigma(s, t|v) = 0$ [2]_.

    The normalization is slightly different from NetworkX,
    as it normalizes only to the possible paths between
    nodes in targets, not to all nodes in the network.


    Parameters
    ----------
    G : graph
      A NetworkX graph.

    targets: list of nodes
      Nodes to use as sources/targets for shortest paths in betweenness

    normalized : bool, optional
      If True the betweenness values are normalized by $2/((n-1)(n-2))$
      for graphs, and $1/((n-1)(n-2))$ for directed graphs where $n$
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
    .. [1] Ulrik Brandes, A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [2] Ulrik Brandes: On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """
    b = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    if targets is None:
        targets = G.nodes
    for s in targets:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = shortest_path(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = dijkstra(G, s, weight)
        b = _accumulate_subset(b, S, P, sigma, s, targets)
    b = _rescale(
        b, set(targets), normalized=normalized, directed=G.is_directed()
    )
    return b


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
        if len_subset <= 2:
            subset_scale = None  # no normalization b=0 for all nodes
            not_subset_scale = None
        else:
            not_subset_scale = 1.0 / ((len_subset) * (len_subset - 1))
            subset_scale = 1.0 / ((len_subset - 1) * (len_subset - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            not_subset_scale = 0.5
            subset_scale = 0.5
        else:
            not_subset_scale = None
            subset_scale = None
    if (subset_scale is not None) and (not_subset_scale is not None):
        for v in betweenness:
            betweenness[v] *= subset_scale if v in subset else not_subset_scale
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

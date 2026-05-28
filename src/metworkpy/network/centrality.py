# The code for closeness_centrality_subset is a modified
# version of the NetworkX closeness_centrality function
# which is licensed under the BSD-3-Clause license
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

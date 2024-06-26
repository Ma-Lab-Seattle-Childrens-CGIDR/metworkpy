"""
Module for finding the density of labels on a graph.
"""

# region Imports
# Standard Library Imports
from __future__ import annotations
from typing import Hashable

# External Imports
import networkx as nx
import numpy as np
import pandas as pd


# Local Imports

# endregion Imports

# region Main Functions


def label_density(
    network: nx.Graph | nx.DiGraph,
    labels: list[Hashable] | dict[Hashable, float | int] | pd.Series,
    radius: int = 3,
) -> pd.Series:
    """
    Find the label density for different nodes in the graph. See note for details.

    :param network: Networkx network (directed or undirected) to find the label
        density of. Directed graphs are converted to undirected, and edge weights
        are currently ignored.
    :type network: nx.DiGraph | nx.Graph
    :param labels: Labels to find density of. Can be a list of nodes in the network
        where are labeled nodes will be treated equally, or a dict or Series keyed
        by nodes in the network which can specify a label weight (such as multiple
        labels for a single node). If a dict or Series, values should be ints or
        floats.
    :type labels: list | dict | pd.Series
    :param radius: Radius to use for finding density. Specifies how far out from a
        given node labels are counted towards density. A radius of 0 only counts the
        single node, and so will just return the `labels` values back unchanged. Default
        value of 3.
    :type radius: int
    :return: The label density for the nodes in the network
    :rtype: pd.Series

    .. note:
       For each node in a network, neighboring nodes up to a distance of `radius`
       away are checked for labels. The total number of labels, or the sum of the labels
       found (in the case of dict or Series input) divided by the number of nodes
       within that radius is the density for a particular node.
    """
    if isinstance(network, nx.DiGraph):
        # copy of original graph
        network = network.to_undirected()
    if not isinstance(network, nx.Graph):
        raise ValueError(
            f"Network must be a networkx network, but received {type(network)}"
        )
    if isinstance(labels, list):
        labels = pd.Series(1, index=list)
    elif isinstance(labels, dict):
        labels = pd.Series(labels)
    density_dict = dict()
    for node in network:
        density_dict[node] = _node_density(
            network=network, labels=labels, node=node, radius=radius
        )
    return pd.Series(density_dict)


def find_dense_clusters(
    network: nx.Graph | nx.DiGraph,
    labels: list[Hashable] | dict[Hashable, float | int] | pd.Series,
    radius: int = 3,
    quantile_cutoff: float = 0.20,
) -> pd.DataFrame:
    """
    Find the clusters within a network with high label density

    :param network: Network to find clusters from
    :type network: nx.Graph | nx.DiGraph
    :param labels: Labels to find density of. Can be a list of nodes in the network
        where are labeled nodes will be treated equally, or a dict or Series keyed
        by nodes in the network which can specify a label weight (such as multiple
        labels for a single node). If a dict or Series, values should be ints or
        floats.
    :type labels: list | dict | pd.Series
    :param radius: Radius to use for finding density. Specifies how far out from a
        given node labels are counted towards density. A radius of 0 only counts the
        single node, and so will just return the `labels` values back unchanged. Default
        value of 3.
    :type radius: int
    :param quantile_cutoff: Quantile cutoff for defining high density,
        the nodes within the top 100*`quantile`% of label density are considered
        high density. Must be between 0 and 1.
    :type quantile_cutoff:  float
    :return: A dataframe indexed by reaction, with columns for density and cluster.
        The clusters are assigned integers starting from 0 to differentiate them. The
        clusters are not ordered.
    :rtype: pd.DataFrame

    .. note:
       This method finds the label density of the graph, then defines high density
       nodes as those in the top `quantile` (so if quantile = 0.15, the top 15%
       of nodes in terms of density will be defined as high density). Following this,
       the low density nodes are removed (doesn't impact `network` which is copied), and
       the connected components of the graph that remains. These components are the
       high density components which are returned.
    """
    if isinstance(network, nx.DiGraph):
        network = network.to_undirected()
    if not isinstance(network, nx.Graph):
        raise ValueError(
            f"Network must be a networkx network, but received {type(network)}"
        )
    density = label_density(network=network, labels=labels, radius=radius)
    # Find which nodes are below the quantile density cutoff
    cutoff = np.quantile(density, 1 - quantile_cutoff)
    low_density = density[density < cutoff].index
    # Copy the network, and remove all low density nodes
    high_density_network = network.copy()
    high_density_network.remove_nodes_from(low_density)
    # Create a dataframe for the results
    res_df = pd.DataFrame(
        None,
        index=density[density >= cutoff].index,
        columns=["density", "cluster"],
        dtype="float",
    )
    # Find the connected components, and assign each to a cluster
    current_cluster = 0
    for comp in nx.connected_components(high_density_network):
        nodes = list(comp)
        res_df.loc[nodes, "density"] = density[nodes]
        res_df.loc[nodes, "cluster"] = current_cluster
        current_cluster += 1
    res_df["cluster"] = res_df["cluster"].astype("int")
    return res_df


# endregion Main Functions


# region Node Density
def _node_density(
    network: nx.Graph, labels: pd.Series, node: Hashable, radius: int
) -> float:
    node_count = 1
    if node in labels.index:
        label_sum = float(labels[node])
    else:
        label_sum = 0.0
    # Iterate through connected nodes
    # bfs_successors used as it allows for depth limit, and
    for predecessors, successors in nx.bfs_successors(
        network, source=node, depth_limit=radius
    ):
        for n in successors:
            node_count += 1
            if n in labels.index:
                label_sum += labels[n]
    return label_sum / node_count


# endregion Node Density

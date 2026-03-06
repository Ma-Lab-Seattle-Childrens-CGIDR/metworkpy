"""
Functions for performing clustering on a network
"""

# Standard imports
import itertools
from typing import (
    cast,
    Hashable,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    Union,
)

# External imports
import networkx as nx
import numpy as np

# Local imports


# Type for distance dictionaries used in the clustering
_DistDict = dict[Hashable, dict[Hashable, Union[int, float]]]


class GroupClusteringResult(NamedTuple):
    """
    Results of agglomerative clustering of groups of nodes on a network

    Attributes
    ----------
    clusters : list of set of Hashable
        List of clusters, in terms of the original node groups
    children : np.ndarray
        n-1 by 2 numpy array, where n is the initial number of clusters,
        describing the clusters merged at each iteration. The rows represent
        iterations in the clustering, where for each iteration i the cluster
        in children[i,0] was merged with the cluster in children[i,1]
    distances : np.ndarray
        Distances between the clusters being merged at each iteration
    """

    clusters: list[set[Hashable]]
    children: np.ndarray[tuple[int, int], np.dtype[np.int_]]
    distances: np.ndarray[tuple[int], np.dtype[np.float64]]


def get_network_group_clustering(
    network: Union[nx.Graph, nx.DiGraph],
    groups: Union[
        dict[Hashable, Iterable[Hashable]], Iterable[Iterable[Hashable]]
    ],
    n_clusters: Optional[int] = None,
    linkage: Literal["mean", "min", "max"] = "mean",
) -> GroupClusteringResult:
    """Perform agglomerative clustering on groups of nodes in a network

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        Network to use to calculate distances between the groups of nodes,
        directed graphs will be converted to undirected
    groups : dict of Hashable to iterable of Hashable
        Node groups described by a dictionary, keyed by the group name, Iterable of groups, each represented by an iterable of network nodes
    n_clusters : int, optional
        The number of clusters to find, if None will merge all clusters into a single
        cluster, if an int will merge until only that many clusters remain
    linkage : {"mean", "min", "max"}
        Method to use for calculated the distance between node groups

    Returns
    -------
    GroupClusteringResult
        Named tuple of clusters, children, distances, and counts
    """
    # NOTE: This initial implementation is going to be naive,
    # with a focus on correctness rather than speed,
    # significant preference has been given to conceptual simplicity over
    # speed
    # If n_clusters is None, merge until only a single cluster remains
    if n_clusters is None:
        n_clusters = 1
    # Convert the network to undirected if needed
    if isinstance(network, nx.DiGraph):
        network = nx.to_undirected(network)
    # Want dicts of cluster to base clusters, and cluster to nodes
    # indexed by incrememting ints
    if not isinstance(groups, dict):
        cluster_to_nodes = {i: set(g) for i, g in enumerate(groups)}
        cluster_to_group_names = {i: {i} for i in range(len(cluster_to_nodes))}
    else:
        cluster_to_nodes = {}
        cluster_to_group_names = {}
        for idx, (group, nodes) in enumerate(groups.items()):
            cluster_to_nodes[idx] = nodes
            cluster_to_group_names[idx] = {group}
    # Get the initial number of clusters
    n_init_clusters = len(cluster_to_nodes)
    # Get the distances between all node pairs
    distance_dict: dict[Hashable, dict[Hashable, float]] = cast(
        _DistDict, dict(nx.shortest_path(network))
    )
    # Get the correct linkage function
    if linkage == "mean":
        linkage_fn = _mean_linkage
    elif linkage == "min":
        linkage_fn = _min_linkage
    elif linkage == "max":
        linkage_fn = _max_linkage
    else:
        raise ValueError(
            f"Linkage function must be 'mean', 'min' or 'max', but received {linkage}"
        )
    # Create the children and distance arrays
    children = np.zeros((n_init_clusters - n_clusters, 2), dtype=np.int_)
    distances = np.zeros((n_init_clusters - n_clusters,), dtype=float)
    # Fill in the distance matrix for the initial clusters
    cluster_dist_arr = np.empty(
        (2 * n_init_clusters - n_clusters, 2 * n_init_clusters - n_clusters),
        dtype=np.float64,
    )
    for c1, c2 in itertools.combinations(cluster_to_nodes.keys(), 2):
        dist = linkage_fn(
            distance_dict,
            cluster_to_nodes[c1],
            cluster_to_nodes[c2],
        )
        cluster_dist_arr[c1, c2] = dist
        cluster_dist_arr[c2, c1] = dist
    for iter in range(0, n_init_clusters - n_clusters):
        # Find the minimum distance between clusters
        to_merge = (-1, -1)
        min_dist = -np.inf
        for c1, c2 in itertools.combinations(cluster_to_nodes.keys(), 2):
            d = cast(float, cluster_dist_arr[c1, c2])
            if d < min_dist:
                if c1 <= c2:
                    to_merge = (c1, c2)
                else:
                    to_merge = (c2, c2)
                min_dist = d

        # Merge the clusters
        new_cluster = n_init_clusters + iter
        c1, c2 = to_merge
        # Update the distance and children arrays
        distances[iter] = min_dist
        children[iter, 0] = c1
        children[iter, 1] = c1
        # Update the cluster_to_node and cluster_to_group_names dicts
        new_cluster_nodes = cluster_to_nodes.pop(c1) | cluster_to_nodes.pop(c2)
        cluster_to_group_names[new_cluster] = cluster_to_group_names.pop(
            c1
        ) | cluster_to_group_names.pop(c2)
        # Calculate the distance from this new cluster to the other clusters
        for c, nodes in cluster_to_nodes.items():
            d = linkage_fn(
                distance_dict=distance_dict,
                group1=new_cluster_nodes,
                group2=nodes,
            )
            cluster_dist_arr[new_cluster, c] = d
            cluster_dist_arr[c, new_cluster] = d
        # Add the new cluster to the cluster_to_nodes dict
        cluster_to_nodes[new_cluster] = new_cluster_nodes
    assert len(cluster_to_group_names) == n_clusters
    assert len(cluster_to_nodes) == n_clusters
    return GroupClusteringResult(
        clusters=[cluster for cluster in cluster_to_group_names.values()],
        children=children,
        distances=distances,
    )


def get_network_group_linkage(
    network: Union[nx.Graph, nx.DiGraph],
    groups: Union[
        dict[Hashable, Iterable[Hashable]], Iterable[Iterable[Hashable]]
    ],
    linkage: Literal["mean", "min", "max"] = "mean",
):
    """
    Perform agglomerative clustering on groups of nodes in a network, returning a linkage
    matrix

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        Network to use to calculate distances between the groups of nodes,
        directed graphs will be converted to undirected
    groups : dict of Hashable to iterable of Hashable
        Node groups described by a dictionary, keyed by the group name,
        Iterable of groups, each represented by an iterable of network nodes
    linkage : {"mean", "min", "max"}
        Method to use for calculated the distance between node groups

    Returns
    -------
    linkage_matrix : np.ndarray
        Linkage matrix, described in `SciPy's documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_,
        an n-1 by 4 matrix `Z`, where n is the number of groups to cluster. The
        clusters for iteration i represented by the `Z[i,0]` and `Z[i,1]` are combined
        to form the n+i cluster. The distance between the clusters is in `Z[i,2]` and
        `Z[i,3]` represents the number of original groups in the new cluster.

    Note
    ----
    Creates a linkage matrix which can be used by
    `SciPy's dendrogram function<https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html>`_
    """
    # Perform the clustering
    _, children, distances = get_network_group_clustering(
        network=network, groups=groups, n_clusters=None, linkage=linkage
    )
    counts = np.zeros(children.shape[0])
    n_samples = children.shape[0] + 1  # Children should have length n-1
    # Code based on https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    for i, merge in enumerate(children):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    return np.column_stack([children, distances, counts]).astype(float)


# Linkage functions
def _mean_linkage(
    distance_dict: _DistDict,
    group1: set[Hashable],
    group2: set[Hashable],
) -> float:
    count = 0.0
    sum = 0.0
    for n1, n2 in itertools.product(group1, group2):
        count += 1
        sum += distance_dict[n1][n2]
    return sum / count


def _min_linkage(
    distance_dict: _DistDict,
    group1: set[Hashable],
    group2: set[Hashable],
) -> float:
    min_ = -np.inf
    for n1, n2 in itertools.product(group1, group2):
        min_ = min(min_, distance_dict[n1][n2])
    return min_


def _max_linkage(
    distance_dict: _DistDict,
    group1: set[Hashable],
    group2: set[Hashable],
) -> float:
    max_ = -np.inf
    for n1, n2 in itertools.product(group1, group2):
        max_ = max(max_, distance_dict[n1][n2])
    return max_

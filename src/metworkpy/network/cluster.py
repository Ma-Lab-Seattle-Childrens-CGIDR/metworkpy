"""
Functions for performing clustering on a network
"""

# Standard imports
from __future__ import annotations

import itertools
from collections.abc import Hashable, Iterable
from typing import (
    Literal,
    NamedTuple,
    cast,
)

# External imports
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

from metworkpy.network.sparray_utils import (
    allclose_array,
    normalize_array,
    prune_array,
)

# Local imports
from metworkpy.utils._notebook import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# Type for distance dictionaries used in the clustering
_DistDict = dict[Hashable, dict[Hashable, int | float]]


class TargetSetClusteringResult(NamedTuple):
    """
    Results of agglomerative clustering of target sets of nodes on a network

    Attributes
    ----------
    clusters : list of set of Hashable
        List of clusters, in terms of the original node target sets
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


def get_network_target_set_clustering(
    network: nx.Graph | nx.DiGraph,
    target_sets: dict[Hashable, Iterable[Hashable]]
    | Iterable[Iterable[Hashable]],
    n_clusters: int | None = None,
    linkage: Literal["mean", "min", "max"] = "mean",
) -> TargetSetClusteringResult:
    """Perform agglomerative clustering on target sets of nodes in a network

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        Network to use to calculate distances between the target sets of nodes,
        directed graphs will be converted to undirected
    target_sets : dict of Hashable to iterable of Hashable or iterable of Hashable
        Node target sets described by a dictionary, keyed by the target set name or Iterable
        of target sets, each represented by an iterable of network nodes
    n_clusters : int, optional
        The number of clusters to find, if None will merge all clusters into a single
        cluster, if an int will merge until only that many clusters remain
    linkage : {"mean", "min", "max"}
        Method to use for calculated the distance between node target sets

    Returns
    -------
    TargetSetClusteringResult
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
    # Input network must be connected
    if not nx.is_connected(network):
        raise ValueError("Network isn't connected, can't perform clustering")
    # Want dicts of cluster to base clusters, and cluster to nodes
    # indexed by incrememting ints
    if not isinstance(target_sets, dict):
        cluster_to_nodes = {i: set(g) for i, g in enumerate(target_sets)}
        cluster_to_group_names = {i: {i} for i in range(len(cluster_to_nodes))}
    else:
        assert isinstance(target_sets, dict)
        cluster_to_nodes: dict[int, set[Hashable]] = {}
        cluster_to_group_names = {}
        for idx, (group, nodes) in enumerate(target_sets.items()):
            nodes = cast(Iterable[Hashable], nodes)
            cluster_to_nodes[idx] = set(nodes)
            cluster_to_group_names[idx] = {group}
    # Get the initial number of clusters
    n_init_clusters = len(cluster_to_nodes)
    # Get the distances between all node pairs
    distance_dict: dict[Hashable, dict[Hashable, float]] = cast(
        _DistDict, dict(nx.shortest_path_length(network))
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
    for idx in range(n_init_clusters - n_clusters):
        # Find the minimum distance between clusters
        to_merge = (-1, -1)
        min_dist = np.inf
        for c1, c2 in itertools.combinations(cluster_to_nodes.keys(), 2):
            d = cast(float, cluster_dist_arr[c1, c2])
            if d < min_dist:
                if c1 <= c2:
                    to_merge = (c1, c2)
                else:
                    to_merge = (c2, c1)
                min_dist = d

        # Merge the clusters
        new_cluster = n_init_clusters + idx
        c1, c2 = to_merge
        # Update the distance and children arrays
        distances[idx] = min_dist
        children[idx, 0] = c1
        children[idx, 1] = c2
        # Update the cluster_to_node and cluster_to_group_names dicts
        new_cluster_nodes = cluster_to_nodes.pop(c1) | cluster_to_nodes.pop(c2)
        cluster_to_group_names[new_cluster] = cluster_to_group_names.pop(
            c1
        ) | cluster_to_group_names.pop(c2)  # ty: ignore[invalid-assignment]
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
    return TargetSetClusteringResult(
        clusters=[cluster for cluster in cluster_to_group_names.values()],  # ty: ignore[invalid-argument-type]
        children=children,
        distances=distances,
    )


def get_network_target_set_linkage(
    network: nx.Graph | nx.DiGraph,
    target_sets: Iterable[Iterable[Hashable]],
    linkage: Literal["mean", "min", "max"] = "mean",
):
    """
    Perform agglomerative clustering on target sets of nodes in a network, returning a linkage
    matrix

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        Network to use to calculate distances between the target sets of nodes,
        directed graphs will be converted to undirected
    target_sets : Iterable to iterable of Hashable
        Node target sets described by an iterable, with each entry representing a target set.
        Each target set should be an iterable of nodes in the network.
    linkage : {"mean", "min", "max"}
        Method to use for calculated the distance between node target sets

    Returns
    -------
    linkage_matrix : np.ndarray
        Linkage matrix, described in `SciPy's documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_,
        an n-1 by 4 matrix `Z`, where n is the number of target sets to cluster. The
        clusters for iteration i represented by the `Z[i,0]` and `Z[i,1]` are combined
        to form the n+i cluster. The distance between the clusters is in `Z[i,2]` and
        `Z[i,3]` represents the number of original target sets in the new cluster.

    Note
    ----
    Creates a linkage matrix which can be used by
    `SciPy's dendrogram function<https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html>`_
    """
    # Perform the clustering
    _, children, distances = get_network_target_set_clustering(
        network=network,
        target_sets=target_sets,
        n_clusters=None,
        linkage=linkage,
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
    min_ = np.inf
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


def get_distance_matrix(
    network: nx.Graph | nx.DiGraph, **kwargs
) -> pd.DataFrame:
    """
    Get a matrix of the shortest path distances between each pair of
    nodes in a network

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network to find the distance matrix for
    kwargs
        Keyword arguments passed to the
        `nx.shortest_path_length function <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path_length.html>`_

    Returns
    -------
    distance_matrix : pd.DataFrame
        A pandas DataFrame with the nodes of the network as the index and columns, with
        each i,j entry being the shortest-path distance from node i to node j
    """
    node_index = pd.Index(network.nodes)
    distance_matrix = pd.DataFrame(0.0, index=node_index, columns=node_index)
    # NOTE: The shortest path returns iterable if source/target not specified
    for source, target_dict in nx.shortest_path_length(network, **kwargs):  # ty: ignore[not-iterable]
        for target, dist in target_dict.items():
            distance_matrix.loc[source, target] = dist
    return distance_matrix


def get_mcl_clustering(
    network: nx.Graph
    | nx.DiGraph
    | pd.DataFrame
    | np.ndarray
    | sparse.sparray,
    *,
    weight: str | None = "weight",
    expansion: int = 2,
    inflation: float = 2.0,
    loop_val: float = 1,
    prune_threshold: float = 1e-5,
    prune_freq: int = 1,
    max_iter: int = 100,
    convergence_atol: float = 1e-8,
    convergence_rtol: float = 1e-5,
    ascending_size: bool = True,
    progress_bar: bool = False,
):
    """
    Perform MCL Clustering on a network

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph or np.ndarray or pd.DataFrame
        The network to be clusteredsimmilarity matrix of the graph to be clustered
    weight : str,optional
        If `network` is a networkx Graph or DiGraph, which edge attribute holds the
        simmilarity between nodes
    expansion : int,default=2
        The power to raise the stochastic matrix to when clustering
    inflation : float,default=2.0
        The inflation power parameter
    loop_val : float, default=1
        Weight for self-loops added to the graph
    prune_threshold : float,default=1e-5
        Values below this threshold will be set to 0.0
    prune_freq : int,default=2
        Number of iterations between pruning
    max_iter : int,default=100
        Maximum number of iterations to perform, if convergence fails
        a RuntimeError will be raised
    convergence_atol,convergence_rtol : float
        The convergence absolute and relative tolerance, handled the
        same as `numpy.allclose <https://numpy.org/devdocs//reference/generated/numpy.allclose.html>`_
    ascending_size : bool,default=False
        Whether the cluster should be returned in ascending size,
        if False returns in descending size.
    progress_bar : bool,default=True
        Whether to display a progress bar (will count from 0 to
        `max_iter`, but usually end before the progress bar completes)

    Returns
    -------
    list of sets of Hashable
        The clusters identified via the MCL algorithm as a list of
        sets of nodes containing the clusters

    Notes
    -----
    Markov cluster process (MCL) is a graph clustering algorithm based on
    stochastic flow on a graph [1]_. See [2]_ for more information about
    the algorithm.

    References
    ----------
    .. [1] Van Dongen, S. (2008). Graph Clustering Via a Discrete
           Uncoupling Process. SIAM Journal on Matrix Analysis and
           Applications, 30(1), 121–141. https://doi.org/10.1137/040608635
    .. [2] MCL - a cluster algorithm for graphs. (n.d.). Retrieved November
           16, 2026, from https://micans.org/mcl/
    """
    # Convert the incoming network into a sparse array (in DOK format)
    # since we have to update the self-loop values
    match network:
        case nx.Graph() | nx.DiGraph():
            nodes = list(network.nodes)
            sim = nx.to_scipy_sparse_array(
                network, weight=weight, format="dok"
            )
        case sparse.sparray():
            nodes = list(range(network.shape[0]))  # ty: ignore[unresolved-attribute]
            sim = network.to_dok()  # ty: ignore[unresolved-attribute]
        case pd.DataFrame():
            nodes = list(network.columns)
            # NOTE: Ensuring the order of the index
            sim = sparse.dok_array(network.loc[nodes].to_numpy())
        case np.ndarray():
            nodes = list(range(network.shape[0]))
            sim = sparse.dok_array(network)

    # Ensure we have a square matrix
    if sim.shape[0] != sim.shape[1]:
        raise ValueError(
            f"Expected 'network' to be a square matrix, but found {sim.shape[0]} rows and {sim.shape[1]} columns"
        )
    # Make sure it has a float dtype
    sim = sim.astype(np.float64)

    # Add in the self-loops
    for idx in range(sim.shape[0]):
        sim[idx, idx] = loop_val

    # Normalize the matrix
    sim: sparse.csc_array = normalize_array(sim, axis=0).tocsc()

    # Create a variable holding the previous matrix to check for convergence
    prev_sim: sparse.csc_array | None = None

    # Iterate for at most max_iter
    for iter in tqdm(range(max_iter), disable=not progress_bar):
        # Expand
        sim = sp_linalg.matrix_power(sim, expansion)
        # Inflate
        sim = sim.power(inflation)
        sim = normalize_array(sim, axis=0).tocsc()

        # Prune
        if iter % prune_freq == prune_freq - 1:
            sim = prune_array(sim, prune_threshold)

        # Check for convergence
        if prev_sim is not None and allclose_array(
            sim, prev_sim, rtol=convergence_rtol, atol=convergence_atol
        ):
            break
        prev_sim = sim.copy()
    else:
        raise RuntimeError("Failed to converge!")
    # Final Prune
    sim = prune_array(sim, prune_threshold)
    # Convert the simmilarity matrix into a networkx Graph (since we are looking for
    # connected components)
    cluster_network = nx.from_scipy_sparse_array(sim)
    # Rename the nodes to match the input network
    cluster_network = nx.relabel_nodes(
        cluster_network, {idx: node for idx, node in enumerate(nodes)}
    )

    # Get the connected_components
    return [
        set(c)
        for c in sorted(
            nx.connected_components(cluster_network),
            key=len,
            reverse=not ascending_size,
        )
    ]

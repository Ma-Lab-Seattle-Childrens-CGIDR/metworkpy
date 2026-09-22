"""
Function for finding load points in a Metabolic Network
"""

import itertools
from collections import defaultdict

import networkx as nx
import numpy as np


def find_load_values(
    network: nx.DiGraph | nx.Graph,
    k: int = 1,
    endpoints: bool = False,
) -> dict[str, float]:
    """
    Find the load values of metabolites or reactions in a metabolic network

    Parameters
    ----------
    network : nx.DiGraph or nx.Graph
        The network to find the load values for
    k : int,default=1
        The number of shortest paths between each pair
        of nodes in the network to use for calculating the
        load for a node
    endpoints: bool,default=False
        Whether to include enpoints as being on a path

    Returns
    -------
    dict of str to float
        The load values for each node in the network

    Notes
    -----
    Load values are 'hot spots' in the metabolic network, calculated
    based on the ration between the number of k-shortest paths passing
    through a node, and the number of neighbors the node has (
    compared to the average load in the network) [1].

    References
    ----------
    .. [1] Rahman, S. A.; Schomburg, D. Observing Local and Global
       Properties of Metabolic Pathways: ‘Load Points’ and ‘Choke Points’
       in the Metabolic Networks. Bioinformatics 2006, 22 (14), 1767–1774.
       https://doi.org/10.1093/bioinformatics/btl181
    """

    load_dict = {}

    btwness_count = defaultdict(int)
    num_paths = 0

    # Get the betweenness of nodes
    for u, v in itertools.combinations(network.nodes, 2):
        for p in itertools.islice(
            nx.shortest_simple_paths(network, source=u, target=v), k
        ):
            num_paths += 1
            for n in p if endpoints else p[1:-1]:
                btwness_count[n] += 1
        if network.is_directed():
            for p in itertools.islice(
                nx.shortest_simple_paths(network, source=v, target=u), k
            ):
                num_paths += 1
                for n in p if endpoints else p[1:-1]:
                    btwness_count[n] += 1

    # Get the degree of nodes, and the degree sum
    degree = network.degree()
    degree_sum = len(network.edges) * 2

    # Get average load
    average_load = num_paths / degree_sum

    # Calculate the load
    for n in network:
        node_btwn = btwness_count[n]
        if node_btwn == 0:
            load_dict[n] = 0
            continue
        deg = degree[n]
        load_dict[n] = np.log10((node_btwn / deg) / average_load)

    return load_dict

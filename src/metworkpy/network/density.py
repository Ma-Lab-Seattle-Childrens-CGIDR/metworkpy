"""Module for finding the density of labels on a graph."""

# region Imports
# Standard Library Imports
from __future__ import annotations
from typing import Hashable, Union, Literal, cast
from warnings import warn

# External Imports
import cobra  # type:ignore     # Cobra doesn't have py.typed marker
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


# Local Imports

# endregion Imports

# region Main Functions


def label_density(
    network: nx.Graph | nx.DiGraph,
    labels: list[Hashable] | dict[Hashable, float | int] | pd.Series,
    radius: int = 3,
) -> pd.Series:
    """
    Find the label density for different nodes in the graph. See note for
    details.

    Parameters
    ----------
    network : nx.DiGraph | nx.Graph
        Networkx network (directed or undirected) to find the label
        density of. Directed graphs are converted to undirected, and
        edge weights are currently ignored.
    labels : list | dict | pd.Series
        Labels to find density of. Can be a list of nodes in the network
        where are labeled nodes will be treated equally, or a dict or
        Series keyed by nodes in the network which can specify a label
        weight (such as multiple labels for a single node). If a dict or
        Series, values should be ints or floats.
    radius : int
        Radius to use for finding density. Specifies how far out from a
        given node labels are counted towards density. A radius of 0
        only counts the single node, and so will just return the
        `labels` values back unchanged. Default value of 3.

    Returns
    -------
    pd.Series
        The label density for the nodes in the network

    Notes
    -----
    For each node in a network, neighboring nodes up to a distance of `radius`
    away are checked for labels. The total number of labels, or the sum of the
    labels
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
        labels = pd.Series(1, index=list)  # type: ignore
    elif isinstance(labels, dict):
        labels = pd.Series(labels)
    density_dict = dict()
    for node in network:
        density_dict[node] = _node_density(
            network=network,
            labels=labels,  # type: ignore
            node=node,
            radius=radius,
        )
    return pd.Series(density_dict)


def find_dense_clusters(
    network: nx.Graph | nx.DiGraph,
    labels: list[Hashable] | dict[Hashable, float | int] | pd.Series,
    radius: int = 3,
    quantile_cutoff: float = 0.20,
) -> pd.DataFrame:
    """Find the clusters within a network with high label density

    Parameters
    ----------
    network : nx.Graph | nx.DiGraph
        Network to find clusters from
    labels : list | dict | pd.Series
        Labels to find density of. Can be a list of nodes in the network
        where are labeled nodes will be treated equally, or a dict or
        Series keyed by nodes in the network which can specify a label
        weight (such as multiple labels for a single node). If a dict or
        Series, values should be ints or floats.
    radius : int
        Radius to use for finding density. Specifies how far out from a
        given node labels are counted towards density. A radius of 0
        only counts the single node, and so will just return the
        `labels` values back unchanged. Default value of 3.
    quantile_cutoff : float
        Quantile cutoff for defining high density, the nodes within the
        top 100*`quantile`% of label density are considered high
        density. Must be between 0 and 1.

    Returns
    -------
    pd.DataFrame
        A dataframe indexed by reaction, with columns for density and
        cluster. The clusters are assigned integers starting from 0 to
        differentiate them. The clusters are not ordered.

    Notes
    -----
    This method finds the label density of the graph, then defines high density
    nodes as those in the top `quantile` (so if quantile = 0.15, the top 15%
    of nodes in terms of density will be defined as high density).
    Following this, the low density nodes are removed (doesn't impact `network`
    which is copied), and the connected components of the graph that remains.
    These components are the high density components which are returned.
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
        label_sum = float(labels[node])  # type: ignore
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


# region Gene Target Density
def gene_target_density(
    metabolic_network: Union[nx.Graph, nx.DiGraph],
    metabolic_model: cobra.Model,
    gene_labels: Union[pd.Series, list, dict],
    radius: int = 3,
) -> pd.Series:
    """
    Determine the density of gene targets in the neighborhood of a reaction
    within a metabolic network

    Parameters
    ----------
    metabolic_network : nx.Graph or nx.DiGraph
        Metabolic network in the form of a reaction network, can be
        directed or undirected, but directed graphs will be converted
        to undirected.
    metabolic_model : cobra.Model
        Metabolic model from which the metabolic network was constructed
    gene_labels : pd.Series or list or dict
        Labels/counts of labels for genes associated with reactions in the
        metabolic network. If a list each value should be a gene id, and will
        have equal weight. If a dict, should be keyed by gene id, with values
        corresponding to weight. If a pd.Series, should be indexed by gene id,
        with values corresponding to weight.
    radius : int, default=3
        The radius to use for finding density, specifies how far out from
        a given node labels are counted towards density. A radius of 0 only
        counts the genes associated with the single node.

    Returns
    -------
    target_density : pd.Series
        Pandas series with index corresponding to reactions in the network,
        and values corresponding to the density of gene targets in the
        neighborhood of that reaction node
    """
    if isinstance(metabolic_network, nx.DiGraph):
        metabolic_network = metabolic_network.to_undirected()
    if not isinstance(metabolic_network, nx.Graph):
        raise ValueError(
            f"Metabolic network must be a networkx Graph but received a "
            f"{type(metabolic_network)}"
        )
    if isinstance(gene_labels, list):
        gene_labels = pd.Series(1, index=pd.Index(gene_labels))
    elif isinstance(gene_labels, dict):
        gene_labels = pd.Series(gene_labels)
    density_dict = {}
    for node in metabolic_network:
        density_dict[node] = _node_gene_density(
            node=node,
            network=metabolic_network,
            model=metabolic_model,
            labels=gene_labels,
            radius=radius,
        )

    return pd.Series(density_dict)


def _node_gene_density(
    node: str,
    network: nx.Graph,
    model: cobra.Model,
    labels: pd.Series,
    radius: int,
) -> float:
    """
    Find the gene label density for a single node
    """
    gene_neighborhood = _get_gene_neighborhood(
        node=node, network=network, model=model, radius=radius
    )
    if len(gene_neighborhood) == 0:
        return (
            0.0  # If there are no genes in the neigborhood, the density is 0
        )
    gene_count = len(gene_neighborhood)
    label_weight_sum = 0.0
    for g in gene_neighborhood:
        if g in labels.index:
            label_weight_sum += cast(float, labels[g])
    return label_weight_sum / gene_count


# endregion Gene Target Density

# region Gene Target Enrichment


def gene_target_enrichment(
    metabolic_network: Union[nx.Graph, nx.DiGraph],
    metabolic_model: cobra.Model,
    gene_targets: Union[set[str], list[str]],
    metric: Literal["odds-ratio", "p-value"] = "p-value",
    alternative: Literal["two-sided", "less", "greater"] = "greater",
    radius: int = 3,
) -> pd.Series:
    """
    Determine the enrichment of gene targets in the neighborhood of a reaction
    within a metabolic network

    Parameters
    ----------
    metabolic_network : nx.Graph or nx.DiGraph
        Metabolic network in the form of a reaction network, can be
        directed or undirected, but directed graphs will be converted
        to undirected.
    metabolic_model : cobra.Model
        Metabolic model from which the metabolic network was constructed
    gene_targets : list or set of str
        Targeted genes associated with reactions in the
        metabolic network. Result will be the enrichment in these targeted
        genes in a neighborhood of each reaction in the network
    metric : "odds-ratio" or "p-value", default="p-value"
        The enrichment metric to return in the Series, either the odds-ratio
        or the p-value (default) of the Fisher's exact test used to
        evaluate enrichment
    alternative : "two-sided", "less", or "greater"
        The alternative hypothesis for the Fisher's exact test used to
        evaluate the enrichment
    radius : int, default=3
        The radius to use for defining a neighborhood around the reaction for
        finding enrichment, specifies how far out from a given node labels are
        counted towards enrichment. A radius of 0 only counts the genes
        associated with the single node.

    Returns
    -------
    target_enrichment : pd.Series
        Pandas series with index corresponding to reactions in the network,
        and values corresponding to either the odds-ratio or the enrichment
        p-value (depending on the value of metric)
    """
    if isinstance(metabolic_network, nx.DiGraph):
        metabolic_network = metabolic_network.to_undirected()
    if not isinstance(metabolic_network, nx.Graph):
        raise ValueError(
            f"Metabolic network must be a networkx Graph but received a "
            f"{type(metabolic_network)}"
        )
    if isinstance(gene_targets, list):
        gene_targets = set(gene_targets)
    if not isinstance(gene_targets, set):
        raise ValueError(
            f"Gene labels must be a list or a set but received a "
            f"{type(gene_targets)}"
        )
    if len(gene_targets) < 1:
        warn("No labeled genes, p-values all 1.0, odds-ratio all 0.0")
        if metric == "p-value":
            return pd.Series(1.0, index=pd.Index(metabolic_network.nodes))
        elif metric == "odds-ratio":
            return pd.Series(0.0, index=pd.Index(metabolic_network.nodes))
    total_gene_set = set(metabolic_model.genes.list_attr("id"))
    enrichment_dict = {}
    for node in metabolic_network:
        odds, pval = _node_gene_enrichment(
            node=node,
            network=metabolic_network,
            model=metabolic_model,
            labels=gene_targets,
            total_gene_set=total_gene_set,
            radius=radius,
            alternative=alternative,
        )
        enrichment_dict[node] = pval if metric == "p-value" else odds
    return pd.Series(enrichment_dict)


def _node_gene_enrichment(
    node: str,
    network: nx.Graph,
    model: cobra.Model,
    labels: set[str],
    total_gene_set: set[str],
    radius: int,
    alternative: Literal["two-sided", "less", "greater"],
) -> tuple[float, float]:
    """
    Calculate the enrichment in labels in a neighborhood around `node`

    Parameters
    ----------
    node : str
        Node to evaluate the neighborhood enrichment for
    network : nx.Graph
        The network to use for finding gene neighborhood
    model: cobra.Model
        The cobra Model to use for finding genes associated with
        the reactions in the network
    labels : set of str
        The set of genes considered "labeled"
    total_gene_set : set of str
        The set of genes within the model
    radius : int
        The radius to use for defining a neighborhood around
    """
    gene_neighborhood = _get_gene_neighborhood(
        node=node, network=network, model=model, radius=radius
    )
    if len(gene_neighborhood) == 0:
        return (0.0, 1.0)
    # Create contingency table
    #                      | in labels | not in labels |
    #  in neighborhood     |           |               |
    #  not in neighborhood |           |               |
    contingency_table = np.array(
        [
            [len(gene_neighborhood & labels), len(gene_neighborhood - labels)],
            [
                len((total_gene_set - gene_neighborhood) & labels),
                len((total_gene_set - gene_neighborhood) - labels),
            ],
        ]
    )
    fisher_res = stats.fisher_exact(contingency_table, alternative)
    return (fisher_res.statistic, fisher_res.pvalue)


# endregion Gene Target Enrichment

# region Helper Functions


def _get_gene_neighborhood(
    node: str, network: nx.Graph, model: cobra.Model, radius: int
) -> set[str]:
    """
    Find the genes in a neighborhood around a node
    """
    gene_neighborhood = set()
    gene_neighborhood.update(
        [g.id for g in model.reactions.get_by_id(node).genes]
    )
    for _, successors in nx.bfs_successors(
        network, source=node, depth_limit=radius
    ):
        # For each rxn, get the genes
        for n in successors:
            gene_neighborhood.update(
                [g.id for g in model.reactions.get_by_id(n).genes]
            )
    return gene_neighborhood


# endregion Helper Functions

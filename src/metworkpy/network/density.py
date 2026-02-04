"""Module for finding the density of labels on a graph."""

# region Imports
# Standard Library Imports
from __future__ import annotations
from typing import Hashable, Union, Literal, Iterator, Tuple, Optional
from warnings import warn

# External Imports
import cobra  # type:ignore     # Cobra doesn't have py.typed marker
from joblib import Parallel, delayed
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
    processes: Optional[int] = None,
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
    processes : int, optional
        Number of processes to use for finding the density

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
    results_series = pd.Series(np.nan, index=pd.Index(network.nodes))
    for node, density in Parallel(n_jobs=processes, return_as="generator")(
        delayed(_node_density_worker)(node, neighborhood, labels)
        for node, neighborhood in graph_neighborhood_iter(
            network=network, radius=radius
        )
    ):
        results_series[node] = density
    return results_series


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


# region Gene Target Density
def gene_target_density(
    metabolic_network: Union[nx.Graph, nx.DiGraph],
    metabolic_model: cobra.Model,
    gene_labels: Union[pd.Series, list, dict],
    radius: int = 3,
    processes: Optional[int] = None,
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
    processes : int, optional
        Number of processes to use

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
    density_series = pd.Series(np.nan, index=pd.Index(metabolic_network.nodes))
    for node, density in Parallel(n_jobs=processes, return_as="generator")(
        delayed(_gene_density_worker)(
            node, gene_neighborhood=neighborhood, gene_targets=gene_labels
        )
        for node, neighborhood in graph_neighborhood_iter_genes(
            network=metabolic_network, model=metabolic_model, radius=radius
        )
    ):
        density_series[node] = density
    return density_series


# endregion Gene Target Density

# region Gene Target Enrichment


def gene_target_enrichment(
    metabolic_network: Union[nx.Graph, nx.DiGraph],
    metabolic_model: cobra.Model,
    gene_targets: Union[set[str], list[str]],
    metric: Literal["odds-ratio", "p-value"] = "p-value",
    alternative: Literal["two-sided", "less", "greater"] = "greater",
    radius: int = 3,
    processes: Optional[int] = None,
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
    processes : int, optional
        Number of processes to use

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
    total_genes = len(metabolic_model.genes.list_attr("id"))
    enrichment_series = pd.Series(
        np.nan, index=pd.Index(metabolic_network.nodes)
    )
    for node, odds, pval in Parallel(n_jobs=processes, return_as="generator")(
        delayed(_gene_enrichment_worker)(
            node,
            gene_neighborhood=neighborhood,
            gene_targets=gene_targets,
            total_genes=total_genes,
            alternative=alternative,
        )
        for node, neighborhood in graph_neighborhood_iter_genes(
            network=metabolic_network, model=metabolic_model, radius=radius
        )
    ):
        enrichment_series[node] = pval if metric == "p-value" else odds
    return enrichment_series


# endregion Gene Target Enrichment

# region neighborhood iterators


def graph_neighborhood_iter(
    network: nx.Graph, radius: int
) -> Iterator[tuple[Hashable, set[Hashable]]]:
    """
    Iterator over neighborhoods in a graph

    Parameters
    ----------
    network : nx.Graph
        The network whose neighborhoods will be iterated over
    radius : int
        The radius determining the size of the neighborhood

    Yields
    ------
    tuple of Hashable and set of Hashable
        Tuple of node and neighborhood
    """
    for node in network.nodes:
        neighborhood = {node}
        for _, successors in nx.bfs_successors(
            network, source=node, depth_limit=radius
        ):
            neighborhood.update(successors)
        yield node, neighborhood


def graph_neighborhood_iter_genes(
    network: nx.Graph, model: cobra.Model, radius: int
):
    """
    Iterator over gene neighborhoods in a graph

    Parameters
    ----------
    network : nx.Graph
        The network whose neighborhoods will be iterated over
    model : cobra.Model
        The cobra model associated with the metabolic network
    radius : int
        The radius determining the size of the neighborhood

    Yields
    ------
    tuple of Hashable and set of str
        Tuple of node and gene ids in neighborhood
    """
    model_rxns: set[str] = set(model.reactions.list_attr("id"))
    for node, neighborhood in graph_neighborhood_iter(
        network=network, radius=radius
    ):
        gene_neighborhood = set()
        for n in neighborhood:
            if (
                n in model_rxns
            ):  # Check in case the network includes metabolites
                gene_neighborhood.update(
                    [g.id for g in model.reactions.get_by_id(n).genes]  # type: ignore
                )
        yield node, gene_neighborhood


# endregion neighborhood iterator
# region worker functions
def _node_density_worker(
    node: Hashable, neighborhood: set[Hashable], labels: pd.Series
) -> Tuple[Hashable, float]:
    """
    Calculate the density of labels in a neighborhood
    """
    return node, labels[
        [idx for idx in neighborhood if idx in labels.index]
    ].sum() / len(neighborhood)


def _gene_density_worker(
    node: str,
    gene_neighborhood: set[str],
    gene_targets: pd.Series,
) -> Tuple[str, float]:
    """
    Calculate the density of gene labels in a neighborhood
    """
    if len(gene_neighborhood) == 0:
        return node, 0.0
    return node, gene_targets[
        [idx for idx in gene_neighborhood if idx in gene_targets.index]
    ].sum() / len(gene_neighborhood)


def _gene_enrichment_worker(
    node: str,
    gene_neighborhood: set[str],
    gene_targets: set[str],
    total_genes: int,
    alternative: str = "greater",
) -> Tuple[str, float, float]:
    """
    Calculate the enrichment of gene labels in a neighborhood

    Returns
    -------
    node, odds-ratio, p-value : tuple of str, float, float
    """
    if len(gene_neighborhood) == 0:
        return node, 0.0, 1.0
    # Create contingency table
    #                      | in labels | not in labels |
    #  in neighborhood     |           |               |
    #  not in neighborhood |           |               |
    contingency_table = np.array(
        [
            [
                len(
                    gene_targets & gene_neighborhood
                ),  # In labels and Neighborhood
                len(
                    gene_neighborhood - gene_targets
                ),  # In neighborhood but not labels
            ],
            [
                len(
                    gene_targets - gene_neighborhood
                ),  # In targets but not neighborhood
                total_genes
                - len(gene_targets | gene_neighborhood),  # in neither
            ],
        ]
    )
    fisher_res = stats.fisher_exact(
        table=contingency_table, alternative=alternative
    )
    return node, fisher_res.statistic, fisher_res.pvalue


# endregion worker functions

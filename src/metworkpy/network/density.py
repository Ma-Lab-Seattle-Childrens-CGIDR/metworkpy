"""Module for finding the density of labels on a graph."""

# Standard Library Imports
from __future__ import annotations
from typing import Callable, Hashable, Union, Literal, Tuple, Optional
from warnings import warn

# External Imports
import cobra  # type:ignore     # Cobra doesn't have py.typed marker
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


# Local Imports
from metworkpy.network.neighborhoods import (
    _graph_gene_neighborhood,
    get_graph_neighborhood,
)
from metworkpy.utils.translate import get_reaction_to_gene_translation_dict


# region Main Functions


def node_target_density(
    network: nx.Graph | nx.DiGraph,
    targets: list[Hashable] | dict[Hashable, float | int] | pd.Series,
    radius: int = 3,
    node_filter: Optional[
        Union[Callable[[Hashable], bool], set[Hashable]]
    ] = None,
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
    node_filter : Callable of node id to bool or set of node id, optional
        Filter nodes in the network to consider when calculating density.
        If a Callable, should take node ids as the only argument and return
        a bool, if True the node will be considered in the density,
        if False it will not be. If a set, only nodes in the set will be considered
        when calculating density. Note that the density is still calculated for
        all nodes, but nodes that are not in the filter won't count towards the
        size of the neighborhoods, and won't be checked for being in the target
        set.
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
    if isinstance(targets, list):
        targets = pd.Series(1, index=list)  # type: ignore
    elif isinstance(targets, dict):
        targets = pd.Series(targets)
    if callable(node_filter):
        filter_fn = node_filter
    elif isinstance(node_filter, set):

        def filter_fn(x: Hashable) -> bool:
            return x in node_filter
    else:

        def filter_fn(_: Hashable) -> bool:
            return True

    results_series = pd.Series(np.nan, index=pd.Index(network.nodes))
    for node, density in Parallel(n_jobs=processes, return_as="generator")(
        delayed(_node_density_worker)(
            node,
            network=network,
            labels=targets,
            node_filter=filter_fn,
            radius=radius,
        )
        for node in network.nodes
    ):
        results_series[node] = density
    return results_series


def gene_target_density(
    metabolic_network: Union[nx.Graph, nx.DiGraph],
    metabolic_model: cobra.Model,
    gene_targets: Union[pd.Series, list, dict],
    radius: int = 3,
    essential: bool = False,
    processes: Optional[int] = None,
) -> pd.Series:
    """
    Determine the density of gene targets in the neighborhood of a nodes
    within a metabolic network

    Parameters
    ----------
    metabolic_network : nx.Graph or nx.DiGraph
        Metabolic network in the form of a reaction network, can be
        directed or undirected, but directed graphs will be converted
        to undirected.
    metabolic_model : cobra.Model
        Metabolic model from which the metabolic network was constructed
    gene_targets : pd.Series or list or dict
        Targets/counts of targets for genes associated with reactions in the
        metabolic network. If a list each value should be a gene id, and will
        have equal weight. If a dict, should be keyed by gene id, with values
        corresponding to weight. If a pd.Series, should be indexed by gene id,
        with values corresponding to weight.
    radius : int, default=3
        The radius to use for finding density, specifies how far out from
        a given node targets are counted towards density. A radius of 0 only
        counts the genes associated with the single node.
    essential : bool
        Whether for a gene to be in a neighborhood it should be
        essential for at least 1 reaction in that neighborhood. If
        False, all genes associated with reactions within the radius
        are counted as in the neighborhood. If True, only genes
        which are required for at least 1 reaction within the radius
        are counted as in the neighborhood.
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
    if isinstance(gene_targets, list):
        gene_targets = pd.Series(1, index=pd.Index(gene_targets))
    elif isinstance(gene_targets, dict):
        gene_targets = pd.Series(gene_targets)
    density_series = pd.Series(np.nan, index=pd.Index(metabolic_network.nodes))
    rxn_to_gene_set_dict = get_reaction_to_gene_translation_dict(
        model=metabolic_model, essential=essential
    )
    for node, density in Parallel(n_jobs=processes, return_as="generator")(
        delayed(_gene_density_worker)(
            node,
            network=metabolic_network,
            gene_targets=gene_targets,
            radius=radius,
            rxn_to_gene_set_dict=rxn_to_gene_set_dict,
        )
        for node in metabolic_network.nodes
    ):
        density_series[node] = density
    return density_series


def gene_target_enrichment(
    metabolic_network: Union[nx.Graph, nx.DiGraph],
    metabolic_model: cobra.Model,
    gene_targets: Union[set[str], list[str]],
    metric: Literal["odds-ratio", "p-value"] = "p-value",
    alternative: Literal["two-sided", "less", "greater"] = "greater",
    radius: int = 3,
    essential: bool = False,
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
        finding enrichment, specifies how far out from a given node targets are
        counted towards enrichment. A radius of 0 only counts the genes
        associated with the single node.
    essential : bool
        Whether for a gene to be in a neighborhood it should be
        essential for at least 1 reaction in that neighborhood. If
        False, all genes associated with reactions within the radius
        are counted as in the neighborhood. If True, only genes
        which are required for at least 1 reaction within the radius
        are counted as in the neighborhood.
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
            f"Gene targets must be a list or a set but received a "
            f"{type(gene_targets)}"
        )
    # Filter the gene targets for only those in the model
    gene_targets &= set(metabolic_model.genes.list_attr("id"))
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
    rxn_to_gene_set_dict = get_reaction_to_gene_translation_dict(
        model=metabolic_model, essential=essential
    )
    for node, odds, pval in Parallel(
        n_jobs=processes, return_as="generator_unordered"
    )(
        delayed(_gene_enrichment_worker)(
            node,
            network=metabolic_network,
            gene_targets=gene_targets,
            radius=radius,
            rxn_to_gene_set_dict=rxn_to_gene_set_dict,
            total_genes=total_genes,
        )
        for node in metabolic_network.nodes
    ):
        enrichment_series[node] = pval if metric == "p-value" else odds
    return enrichment_series


def find_dense_clusters(
    network: nx.Graph | nx.DiGraph,
    targets: list[Hashable] | dict[Hashable, float | int] | pd.Series,
    radius: int = 3,
    top_quantile_cutoff: float = 0.20,
    target_type: Literal["genes", "nodes"] = "nodes",
    **kwargs,
) -> pd.DataFrame:
    """Find the clusters within a network with high label density

    Parameters
    ----------
    network : nx.Graph | nx.DiGraph
        Network to find clusters from
    targets : list | dict | pd.Series
        Targets to find density of. Can be a list of nodes in the network
        where are labeled nodes will be treated equally, or a dict or
        Series keyed by nodes in the network which can specify a label
        weight (such as multiple targets for a single node). If a dict or
        Series, values should be ints or floats.
    radius : int
        Radius to use for finding density. Specifies how far out from a
        given node targets are counted towards density. A radius of 0
        only counts the single node, and so will just return the
        `targets` values back unchanged. Default value of 3.
    top_quantile_cutoff : float
        Quantile cutoff for defining high density, the nodes within the
        top 100*`quantile`% of label density are considered high
        density. So a `top_quantile_cutoff` of 0.2 means that the top
        20% of mode dense nodes will be defined as high density. Must be
        between 0 and 1.
    target_type : {'genes', 'nodes'}, default='nodes'
        The type of targets, with 'genes' indicating the targets are
        genes (which will require that a COBRApy model is provided as a kwarg,
        i.e. `model=model`), and so gene target density will be used. If 'nodes',
        then the targets should be nodes in the network.
    kwargs
        Passed to `node_target_density`, or `gene_target_density` functions
        depending on `target_type`

    Returns
    -------
    pd.DataFrame
        A dataframe indexed by node id, with columns for density and
        cluster. The clusters are assigned integers starting from 0 to
        differentiate them. The clusters are not ordered, and so multiple
        calls to this method can results in different labels for the clusters.

    Notes
    -----
    This method finds the target density of the metabolic graph, and then identifies
    nodes with a high target density in their neighborhoods. Nodes without a high
    target densit are dropped from the graph, and then the connected components of
    the graph are then used as the high density clusters.
    """
    if target_type == "nodes":
        density = node_target_density(
            network=network, targets=targets, radius=radius, **kwargs
        )
    elif target_type == "genes":
        density = gene_target_density(
            metabolic_network=network,
            gene_targets=targets,
            radius=radius,
            **kwargs,
        )
    else:
        raise ValueError(
            f"target_type must be 'nodes' or 'genes', but received {target_type}"
        )
    # Find which nodes are below the quantile density cutoff
    cutoff = np.quantile(density, 1 - top_quantile_cutoff)
    low_density_nodes = density[density < cutoff].index
    # Copy the network, and remove all low density nodes
    high_density_network = network.copy()
    high_density_network.remove_nodes_from(low_density_nodes)
    # Create a dataframe for the results
    res_df = pd.DataFrame(
        None,
        index=density[density >= cutoff].index,
        columns=["density", "cluster"],
        dtype="float",
    )
    # Find the connected components, and assign each to a cluster
    for current_cluster, connected_component in enumerate(
        nx.connected_components(high_density_network)
    ):
        nodes = list(connected_component)
        res_df.loc[nodes, "density"] = density[nodes]
        res_df.loc[nodes, "cluster"] = current_cluster
    res_df["cluster"] = res_df["cluster"].astype("int")
    return res_df


# endregion Main Functions


# region worker functions
def _node_density_worker(
    node: Hashable,
    network: nx.Graph,
    targets: pd.Series,
    radius: int,
    node_filter: Callable[[Hashable], bool],
) -> Tuple[Hashable, float]:
    """
    Calculate the density of targets in a neighborhood
    """
    neighborhood = set(
        filter(
            node_filter,
            get_graph_neighborhood(network=network, radius=radius, node=node),
        )
    )
    return node, targets[
        [idx for idx in neighborhood if idx in targets.index]
    ].sum() / len(neighborhood)


def _gene_density_worker(
    node: str,
    network: nx.Graph,
    gene_targets: pd.Series,
    radius: int,
    rxn_to_gene_set_dict: dict[str, set[str]],
) -> Tuple[str, float]:
    """
    Calculate the density of gene targets in a neighborhood
    """
    gene_neighborhood = _graph_gene_neighborhood(
        network=network,
        radius=radius,
        node=node,
        rxn_to_gene_set_dict=rxn_to_gene_set_dict,
    )
    if len(gene_neighborhood) == 0:
        return node, 0.0
    return node, gene_targets[
        [idx for idx in gene_neighborhood if idx in gene_targets.index]
    ].sum() / len(gene_neighborhood)


def _gene_enrichment_worker(
    node: str,
    network: nx.Graph,
    gene_targets: set[str],
    radius: int,
    rxn_to_gene_set_dict: dict[str, set[str]],
    total_genes: int,
    alternative: str = "greater",
) -> Tuple[str, float, float]:
    """
    Calculate the enrichment of gene targets in a neighborhood

    Returns
    -------
    node, odds-ratio, p-value : tuple of str, float, float
        The node and the results of an enrichment test
    """
    gene_neighborhood = _graph_gene_neighborhood(
        network=network,
        radius=radius,
        node=node,
        rxn_to_gene_set_dict=rxn_to_gene_set_dict,
    )
    if len(gene_neighborhood) == 0:
        return node, 0.0, 1.0
    # Create contingency table
    #                      | in targets | not in targets |
    #  in neighborhood     |            |                |
    #  not in neighborhood |            |                |
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

"""Module for finding the density of targets on a graph."""

# Standard Library Imports
from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import (
    Literal,
    cast,
)
from warnings import warn

# External Imports
import cobra
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

# Local Imports
from metworkpy.network.neighborhoods import (
    NodeType,
    _create_filter_set,
    _create_rxn_to_gene_set_dict,
    gene_neighborhood_map,
    neighborhood_map,
)

# region Main Functions

DEFAULT_RADIUS = 2


def node_target_density(
    network: nx.Graph | nx.DiGraph,
    targets: list[Hashable] | dict[Hashable, float | int] | pd.Series,
    radius: int = DEFAULT_RADIUS,
    nodes: Iterable[NodeType] | None = None,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
    weight: str | None = None,
    include_node: bool = True,
    processes: int | None = None,
) -> dict[NodeType, float]:
    """
    Find the target density for different nodes in the graph. See note for
    details.

    Parameters
    ----------
    network : nx.DiGraph | nx.Graph
        Networkx network (directed or undirected) to find the target
        density of.
    targets : list | dict | pd.Series
        Targets to find density of. Can be a list of nodes in the network
        where are targeted nodes will be treated equally, or a dict or
        Series keyed by nodes in the network which can specify a target
        weight (such as multiple targets for a single node). If a dict or
        Series, values should be ints or floats.
    radius : int, default=2
        Radius to use for finding density. Specifies how far out from a
        given node targets are counted towards density. A radius of 0
        only counts the single node, and so will just return the
        `targets` values back unchanged. Default value of 3.
    nodes : iterable of hashable, optional
        Subset of nodes to find the density for, if not provided defaults
        to all of the nodes in the network
    node_filter : callable of node id->bool or set of node ids, optional
        Filter nodes in the network to consider when finding neighborhoods.
        If a Callable, should take node ids as the only argument and return
        a bool, if True the node will be considered in neighborhoods,
        if False it will not be. If a set, only nodes in the set will be included
        in neighborhoods.
    weight : str, optional
        If provided indicates the edge parameter to be used as weights
        when finding distances from a central node to
        define a neighborhood. If None, all edges are treated as having a
        weight of 1.
    include_node : bool, default=True
        Whether to include the central node in a neighborhood
    processes : int, optional
        Number of processes to use for finding the density

    Returns
    -------
    dict of node id to density
        The target density for the nodes in the network

    Notes
    -----
    For each node in a network, neighboring nodes up to a distance of `radius`
    away are checked for targets. The total number of targets, or the sum of the
    targets found (in the case of dict or Series input) divided by the number of nodes
    within that radius is the density for a particular node.
    """
    if isinstance(targets, list):
        targets = {k: 1 for k in targets}
    elif isinstance(targets, pd.Series):
        targets = targets.to_dict()  # type: ignore
    assert isinstance(targets, dict), (
        f"Targets must be a dict, but received: {type(targets)}"
    )

    def _get_density(node_ids: set[NodeType]):
        return float(
            sum(cast(dict, targets).get(n, 0.0) for n in node_ids)
        ) / len(node_ids)

    return neighborhood_map(
        _get_density,
        network=network,
        radius=radius,
        nodes=nodes,
        node_filter=node_filter,
        weight=weight,
        include_node=include_node,
        processes=processes,
    )


def gene_target_density(
    metabolic_network: nx.Graph | nx.DiGraph,
    gene_targets: pd.Series | list[str] | dict[str, float],
    metabolic_model: cobra.Model | None = None,
    reaction_to_gene_set_dict: Mapping[NodeType, set[str]] | None = None,
    radius: int = DEFAULT_RADIUS,
    essential: bool = False,
    nodes: Iterable[NodeType] | None = None,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
    weight: str | None = None,
    include_node: bool = True,
    processes: int | None = None,
) -> dict[NodeType, float]:
    """
    Determine the density of gene targets in the neighborhood of a nodes
    within a metabolic network

    Parameters
    ----------
    metabolic_network : nx.Graph or nx.DiGraph
        Metabolic network in the form of a reaction network, can be
        directed or undirected, but directed graphs will be converted
        to undirected.
    gene_targets : pd.Series or list or dict
        Targets/counts of targets for genes associated with reactions in the
        metabolic network. If a list each value should be a gene id, and will
        have equal weight. If a dict, should be keyed by gene id, with values
        corresponding to weight. If a pd.Series, should be indexed by gene id,
        with values corresponding to weight.
    metabolic_model : cobra.Model, optional
        Metabolic model from which the metabolic network was constructed
    reaction_to_gene_set_dict : dict of reaction id to sets of gene ids, optional
        Map between reaction ids and sets of gene ids. Must provide at least one of
        `model` or `reaction_to_gene_set_dict`, `reaction_to_gene_set_dict`
        takes precedence if both are provided.
    radius : int, default=2
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
    nodes : iterable of hashable, optional
        Subset of nodes to find the density for, if not provided defaults
        to all of the nodes in the network
    node_filter : callable of node id->bool or set of node ids, optional
        Filter nodes in the network to consider when finding neighborhoods.
        If a Callable, should take node ids as the only argument and return
        a bool, if True the node will be considered in neighborhoods,
        if False it will not be. If a set, only nodes in the set will be included
        in neighborhoods.
    weight : str, optional
        If provided indicates the edge parameter to be used as weights
        when finding distances from a central node to
        define a neighborhood. If None, all edges are treated as having a weight of 1.
    include_node : bool, default=True
        Whether to include the central node in a neighborhood
    processes : int, optional
        Number of processes to use

    Returns
    -------
    target_density : dict of node id to gene target density
        Dict with keys corresponding to nodes in the network,
        and values corresponding to the density of gene targets in the
        neighborhood of that node (`nodes` and `node_filter` can be
        used to only )
    """
    if isinstance(gene_targets, list):
        gene_targets = {g: 1 for g in gene_targets}
    elif isinstance(gene_targets, pd.Series):
        gene_targets = gene_targets.to_dict()  # type: ignore

    def _get_density(gene_ids: set[NodeType]):
        return float(
            sum(cast(dict, gene_targets).get(g, 0.0) for g in gene_ids)
        ) / len(gene_ids)

    return gene_neighborhood_map(
        _get_density,
        network=metabolic_network,
        model=metabolic_model,
        reaction_to_gene_set_dict=reaction_to_gene_set_dict,
        radius=radius,
        essential=essential,
        nodes=nodes,
        node_filter=node_filter,
        weight=weight,
        include_node=include_node,
        processes=processes,
    )


def gene_target_enrichment(
    metabolic_network: nx.Graph | nx.DiGraph,
    gene_targets: set[str] | list[str],
    metabolic_model: cobra.Model | None = None,
    reaction_to_gene_set_dict: Mapping[NodeType, set[str]] | None = None,
    radius: int = DEFAULT_RADIUS,
    essential: bool = False,
    nodes: Iterable[NodeType] | None = None,
    node_filter: Callable[[NodeType], bool] | set[NodeType] | None = None,
    weight: str | None = None,
    include_node: bool = True,
    metric: Literal["odds-ratio", "p-value"] = "p-value",
    alternative: Literal["two-sided", "less", "greater"] = "greater",
    processes: int | None = None,
    **kwargs,
) -> dict[NodeType, float]:
    """
    Determine the enrichment of gene targets in the neighborhood of a reaction
    within a metabolic network

    Parameters
    ----------
    metabolic_network : nx.Graph or nx.DiGraph
        Metabolic network in the form of a reaction network, can be
        directed or undirected.
    gene_targets : list or set of str
        Targeted genes associated with reactions in the
        metabolic network. Result will be the enrichment in these targeted
        genes in a neighborhood of each reaction in the network
    metabolic_model : cobra.Model, optional
        Metabolic model from which the metabolic network was constructed
    reaction_to_gene_set_dict : dict of reaction id to sets of gene ids, optional
        Map between reaction ids and sets of gene ids. Must provide at least one of
        `model` or `reaction_to_gene_set_dict`, `reaction_to_gene_set_dict`
        takes precedence if both are provided.
    radius : int, default=2
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
    nodes : iterable of hashable, optional
        Subset of nodes to find the enrichment for, if not provided defaults
        to all of the nodes in the network
    node_filter : callable of node id->bool or set of node ids, optional
        Filter nodes in the network to consider when finding neighborhoods.
        If a Callable, should take node ids as the only argument and return
        a bool, if True the node will be considered in neighborhoods,
        if False it will not be. If a set, only nodes in the set will be included
        in neighborhoods.
    weight : str, optional
        If provided indicates the edge parameter to be used as weights
        when finding distances from a central node to
        define a neighborhood. If None, all edges are treated as having a weight of 1.
    include_node : bool, default=True
        Whether to include the central node in a neighborhood
    metric : "odds-ratio" or "p-value", default="p-value"
        The enrichment metric to return in the Series, either the odds-ratio
        or the p-value (default) of the Fisher's exact test used to
        evaluate enrichment
    alternative : "two-sided", "less", or "greater", default="greater"
        The alternative hypothesis for the Fisher's exact test used to
        evaluate the enrichment
    processes : int, optional
        Number of processes to use
    kwargs
        Keyword arguments are passed to SciPy's
        `stats.fisher_exact <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html>`_
        for performing the enrichment test.

    Returns
    -------
    target_enrichment : dict of node id to enrichment value
        Dict with keys corresponding to nodes in the network,
        and values corresponding to either the odds-ratio or the
        p-value (depending on the `value` of `metric`)
    """
    if isinstance(gene_targets, list):
        gene_targets = set(gene_targets)

    # Get a dict of reaction to gene set
    rxn_to_gene_dict = _create_rxn_to_gene_set_dict(
        model=metabolic_model,
        reaction_to_gene_set_dict=reaction_to_gene_set_dict,
        essential=essential,
    )
    # Get the filter set
    filter_set = _create_filter_set(
        network=metabolic_network, node_filter=node_filter
    )

    # Find the community gene set (possible background)
    rxn_node_set: set[NodeType] = (
        set(metabolic_network.nodes) & set(rxn_to_gene_dict.keys())
    ) - filter_set
    community_gene_set: set[str] = functools.reduce(
        operator.or_,
        (rxn_to_gene_dict.get(r, set()) for r in rxn_node_set),
        set(),
    )
    gene_targets &= community_gene_set
    if len(gene_targets) < 1:
        warn(
            "No targeted genes in metabolic network, p-values all 1.0, odds-ratios all 0.0"
        )
        if nodes is None:
            nodes = metabolic_network.nodes
        match metric:
            case "p-value":
                return {n: 1.0 for n in nodes}
            case "odds-ratio":
                return {n: 0.0 for n in nodes}
            case m:
                raise ValueError(
                    f"Expected either 'p-value' or 'odds-ratio' for metric, received: {m}"
                )
    total_gene_count = len(community_gene_set)

    def _get_enrichment(neighborhood_gene_ids: set[NodeType]):
        fisher_res = stats.fisher_exact(
            [
                [
                    len(neighborhood_gene_ids & gene_targets),
                    len(gene_targets - neighborhood_gene_ids),
                ],
                [
                    len(neighborhood_gene_ids - gene_targets),
                    total_gene_count
                    - len(neighborhood_gene_ids | gene_targets),
                ],
            ],
            alternative=alternative,
            **kwargs,
        )
        match metric:
            case "p-value":
                return fisher_res.pvalue
            case "odds-ratio":
                return fisher_res.statistic
            case m:
                raise ValueError(
                    f"Excpected 'p-value' or 'statistic' for metric, received {m}"
                )

    return gene_neighborhood_map(
        _get_enrichment,
        network=metabolic_network,
        model=None,
        reaction_to_gene_set_dict=rxn_to_gene_dict,
        radius=radius,
        essential=essential,
        nodes=nodes,
        node_filter=node_filter,
        weight=weight,
        include_node=include_node,
        processes=processes,
    )


def find_dense_clusters(
    network: nx.Graph | nx.DiGraph,
    targets: list[Hashable] | dict[Hashable, float | int] | pd.Series,
    radius: int = DEFAULT_RADIUS,
    top_quantile_cutoff: float = 0.20,
    target_type: Literal["genes", "nodes"] = "nodes",
    **kwargs,
) -> pd.DataFrame:
    """Find the clusters within a network with high target density

    Parameters
    ----------
    network : nx.Graph | nx.DiGraph
        Network to find clusters from
    targets : list | dict | pd.Series
        Targets to find density of. Can be a list of nodes or genes, in
        which case all targets will have equal weight, or a dict or
        Series keyed by nodes/genes in the network which can specify
        a target weight. If a dict or Series, values should be ints
        or floats.
    radius : int, default=2
        Radius to use for finding density. Specifies how far out from a
        given node targets are counted towards density. A radius of 0
        only counts the single node, and so will just return the
        `targets` values back unchanged.
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
        density = pd.Series(
            node_target_density(
                network=network, targets=targets, radius=radius, **kwargs
            )
        )
    elif target_type == "genes":
        density = pd.Series(
            gene_target_density(
                metabolic_network=network,
                gene_targets=targets,  # type: ignore
                radius=radius,
                **kwargs,
            )
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

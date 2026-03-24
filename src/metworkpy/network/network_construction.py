"""
Functions for constructing networks based on genome scale metabolic models
"""

# Imports
# Standard Library Imports
from __future__ import annotations
import itertools
from typing import (
    cast,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Optional,
    Union,
)

# External Imports
import cobra  # type: ignore
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# Local Imports
from metworkpy.information.mutual_information_network import (
    mi_pairwise,
)
from metworkpy.network.projection import bipartite_project
from metworkpy.utils import reaction_to_gene_ids, reaction_to_gene_list
from metworkpy.network.neighborhoods import (
    get_graph_neighborhood_group,
)


# region Main Function
def create_mutual_information_network(
    model: Optional[cobra.Model] = None,
    flux_samples: pd.DataFrame | np.ndarray | None = None,
    reaction_names: Iterable[str] | None = None,
    cutoff_significance: Optional[float] = None,
    n_samples: int = 10_000,
    reciprocal_weights: bool = False,
    processes: int = 1,
    **kwargs,
) -> nx.Graph:
    """Create a mutual information network from the provided metabolic model

    Parameters
    ----------
    model : Optional[cobra.Model]
        Metabolic model to construct the mutual information network
        from. Only required if the flux_samples parameter is None
    flux_samples : Optional[pd.DataFrame|np.ndarray]
        Flux samples used to calculate mutual information between
        reactions. If None, the passed model will be sampled to generate
        these flux samples.
    reaction_names : Optional[Iterable[str]]
        Names for the reactions
    cutoff_significance : float, optional
        Upper bound for the significance of the mutual information,
        any mutual information values with p-values above this
        cutoff will have their mutual information set to 0.
        Will calculate this p-value using permutation testing,
        see `mi_pairwise` for more information.
    n_samples : int
        Number of samples to take if flux_samples is None (ignored if
        flux_samples is not None)
    reciprocal_weights : bool
        Whether the non-zero weights in the network should be the
        reciprocal of mutual information.
    processes : int
        Number of processes to use during the flux sampling and
        mutual information calculation
    kwargs
        Keyword arguments passed to the `mi_pairwise` function

    Returns
    -------
    nx.Graph
        A networkx Graph, which nodes representing different reactions
        and edge weights corresponding to estimated mutual information
    """
    if flux_samples is None:
        if model is None:
            raise ValueError(
                "Requires either a metabolic model, or flux samples but received "
                "neither"
            )
        flux_samples = cobra.sampling.sample(
            model=model, n=n_samples, processes=processes
        )
    if isinstance(flux_samples, np.ndarray):
        if not reaction_names:
            if model:
                reaction_names = model.reactions.list_attr("id")
            else:
                reaction_names = [
                    f"rxn_{i}" for i in range(flux_samples.shape[1])
                ]
        sample_df = pd.DataFrame(
            flux_samples, columns=pd.Index(reaction_names)
        )
    elif isinstance(flux_samples, pd.DataFrame):
        sample_df = flux_samples
        if reaction_names is not None:
            sample_df.columns = pd.Index(reaction_names)
    else:
        raise ValueError(
            f"Invalid type for flux samples, requires pandas DataFrame or "
            f"numpy ndarray, but "
            f"received {type(flux_samples)}"
        )
    if cutoff_significance is not None:
        kwargs["calculate_pvalue"] = True
    if not cutoff_significance:
        adj_mat = cast(
            pd.DataFrame,
            mi_pairwise(dataset=sample_df, processes=processes, **kwargs),
        )
    else:
        adj_mat, _ = mi_pairwise(
            dataset=sample_df, processes=processes, **kwargs
        )
        adj_mat = cast(pd.DataFrame, adj_mat)
    if reciprocal_weights:
        # Should be all floats, so no issue with integer division
        adj_mat[adj_mat > 0] = np.reciprocal(adj_mat[adj_mat > 0])
    mi_network = nx.from_pandas_adjacency(
        adj_mat,
        create_using=nx.Graph,
    )
    return mi_network


def create_adjacency_matrix(
    model: cobra.Model,
    weighted: bool,
    directed: bool,
    weight_by: Literal["stoichiometry", "flux"] = "stoichiometry",
    threshold: float = 0.0,
    **kwargs,
) -> pd.DataFrame:
    """
    Create an adjacency matrix representing the metabolic network of a provided
    cobra Model

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    weighted : bool
        Whether the network should be weighted
    directed : bool
        Whether the network should be directed
    weight_by : 'flux' or 'stoichiometry', default='stoichiometry'
        String indicating if the network should be weighted by
        'stoichiometry', or 'flux' (see notes for more information).
        Ignored if `weighted = False`
    threshold : float
        Threshold, below which to consider a (absolute value of a) bound/flux
        to be 0
    kwargs
        Passed to cobra's flux_variability_analysis function if the weight_by
        is flux

    Returns
    -------
    pd.DataFrame
        The adjacency matrix

    Notes
    -----
    When creating a weighted network, the options are to weight the edges based
    on flux, or stoichiometry. If stoichiometry is chosen the edge weight will
    correspond to the stoichiometric coefficient of the metabolite, in a given
    reaction.

    For flux weighting, first flux variability analysis is performed. The edge
    weight is determined by the maximum flux through a reaction in a particular
    direction (forward if the metabolite is a product of the reaction,
    reverse if the metabolite is a substrate) multiplied by the metabolite
    stoichiometry. If the network is unweighted, the maximum of the absolute
    value of the forward and the reverse flux is used instead.
    """
    if not isinstance(model, cobra.Model):
        raise ValueError(
            f"Model must be a cobra.Model, received a {type(model)} instead"
        )
    if threshold < 0.0:
        raise ValueError(
            f"Threshold must be greater than 0.0, but received {threshold}"
        )
    if directed:
        if weighted:
            if weight_by == "stoichiometry":
                return _create_adj_matrix_d_w_stoich(
                    model=model, threshold=threshold
                )
            elif weight_by == "flux":
                return _create_adj_matrix_d_w_flux(
                    model=model, threshold=threshold, **kwargs
                )
            else:
                raise ValueError(
                    f"weight_by must be stoichiometry or flux, but received {weight_by}"
                )
        else:
            return _create_adj_matrix_d_uw(model=model, threshold=threshold)
    else:
        if weighted:
            if weight_by == "stoichiometry":
                return _create_adj_matrix_ud_w_stoich(
                    model=model, threshold=threshold
                )
            elif weight_by == "flux":
                return _create_adj_matrix_ud_w_flux(
                    model=model, threshold=threshold, **kwargs
                )
            else:
                raise ValueError(
                    f"weight_by must be stoichiometry or flux, but received {weight_by}"
                )
        else:
            return _create_adj_matrix_ud_uw(model=model, threshold=threshold)


def create_metabolic_network(
    model: cobra.Model,
    weighted: bool,
    directed: bool,
    weight_by: Literal["stoichiometry", "flux"] = "stoichiometry",
    nodes_to_remove: list[str] | None = None,
    reciprocal_weights: bool = False,
    threshold: float = 0.0,
    **kwargs,
) -> nx.Graph | nx.DiGraph:
    """Create a metabolic network from a cobrapy Model

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    weighted : bool
        Whether the network should be weighted
    directed : bool
        Whether the network should be directed
    weight_by : 'stoichiometry' or 'flux', default='stoichiometry'
        String indicating if the network should be weighted by
        'stoichiometry', or 'flux' (see notes for more information).
        Ignored if `weighted = False`
    nodes_to_remove : list[str] | None
        List of any metabolites or reactions that should be removed from
        the final network. This can be used to remove metabolites that
        participate in a large number of reactions, but are not desired
        in downstream analysis such as water, or ATP, or pseudo
        reactions like biomass. Each metabolite/reaction should be the
        string ID associated with them in the cobra model.
    reciprocal_weights : bool
        Whether to use the reciprocal of the weights, useful if higher
        flux should equate with lower weights in the final network (for
        use with graph algorithms)
    threshold : float
        Threshold, below which to consider a bound to be 0
    kwargs
        Keyword arguments are passed to the cobra flux_variability_analysis method
        when weight_by is flux

    Returns
    -------
    nx.Graph | nx.DiGraph
        A network representing the metabolic network from the provided
        cobrapy model

    Notes
    -----
    When creating a weighted network, the options are to weight the edges based
    on flux, or stoichiometry. If stoichiometry is chosen the edge weight will
    correspond to the stoichiometric coefficient of the metabolite, in a given
    reaction.

    For flux weighting, first flux variability analysis is performed. The edge
    weight is determined by the maximum flux through a reaction in a particular
    direction (forward if the metabolite is a product of the reaction,
    reverse if the metabolite is a substrate) multiplied by the metabolite
    stoichiometry. If the network is unweighted, the maximum of the forward
    and the reverse flux is used instead.
    """
    adjacency_frame = create_adjacency_matrix(
        model=model,
        weighted=weighted,
        directed=directed,
        weight_by=weight_by,
        threshold=threshold,
        **kwargs,
    )

    if reciprocal_weights:
        adjacency_frame.data = np.reciprocal(adjacency_frame.data)

    # Create the base network
    if directed:
        out_network = nx.from_pandas_adjacency(
            adjacency_frame, create_using=nx.DiGraph
        )
    else:
        out_network = nx.from_pandas_adjacency(
            adjacency_frame,
            create_using=nx.Graph,  # type: ignore
        )

    # Remove any metabolites desired
    if nodes_to_remove:
        out_network.remove_nodes_from(nodes_to_remove)
    return out_network


def create_reaction_network(
    model: cobra.Model,
    weighted: bool,
    directed: bool,
    weight_by: Literal["stoichiometry", "flux"] = "stoichiometry",
    nodes_to_remove: list[str] | None = None,
    reciprocal_weights: bool = False,
    threshold: float = 0.0,
    projection_weight: str | Callable[[float, float], float] | None = None,
    projection_weight_combine: str
    | Callable[[float, float], float]
    | None = None,
    **kwargs,
):
    """
    Create a reaction connectivity network from the
    metabolic model

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    weighted : bool
        Whether the network should be weighted
    directed : bool
        Whether the network should be directed
    weight_by : 'stoichiometry' or 'flux', default='stoichiometry'
        String indicating if the network should be weighted by
        'stoichiometry', or 'flux' (see notes for more information).
        Ignored if `weighted = False`
    nodes_to_remove : list[str] | None
        List of any metabolites or reactions that should be removed from
        the final network. This can be used to remove metabolites that
        participate in a large number of reactions, but are not desired
        in downstream analysis such as water, or ATP, or pseudo
        reactions like biomass. Each metabolite/reaction should be the
        string ID associated with them in the cobra model.
    reciprocal_weights : bool
        Whether to use the reciprocal of the weights, useful if higher
        flux should equate with lower weights in the final network (for
        use with graph algorithms)
    threshold : float
        Threshold, below which to consider a bound to be 0
    projection_weight : str | Callable[[float, float], float] | None
        How to weight the projected graph. If None, the projected graph
        will not be weighted. If "ratio", the edges will be weighted
        based on the ratio between actual shared neighbors and maximum
        possible shared neighbors. If "count", the edges will be
        weighted by the number of shared neighbors. A function can also
        be provided, which takes two float arguments (the weights of two
        edges), and returns a float.
    projection_weight_combine : Callable[[list[float]], float], optional
        How to combine multiple projected edges. If two nodes in the set
        being projected onto, share multiple neighbors in the other node set,
        they can have multiple possible edge weights. This function takes in
        a list of possible weights, and returns a single final weight. Python
        builtin `max` and `min` can be used for this. If not provided,
        `max` is used.
    kwargs
        Keyword arguments are passed to the cobra flux_variability_analysis method
        when weight_by is flux
    """
    # Create the metabolic network
    metabolic_network = create_metabolic_network(
        model=model,
        weighted=weighted,
        directed=directed,
        weight_by=weight_by,
        nodes_to_remove=nodes_to_remove,
        reciprocal_weights=reciprocal_weights,
        threshold=threshold,
        **kwargs,
    )
    # Get a list of reactions which are in the model and not in the nodes_to_remove
    if nodes_to_remove:
        rxns_to_remove_set = set(nodes_to_remove)
    else:
        rxns_to_remove_set = set()
    reaction_ids = {
        r.id for r in model.reactions if r.id not in rxns_to_remove_set
    }
    # Project onto only reactions
    return bipartite_project(
        network=metabolic_network,
        node_set=reaction_ids,
        directed=directed,
        weight=projection_weight,
        weight_combine=projection_weight_combine,
        weight_attribute="weight",
        # reciprocal won't actually impact, since the graph will be
        # created with the correct directedness
        reciprocal=False,
    )


def create_metabolite_network(
    model: cobra.Model,
    weighted: bool,
    directed: bool,
    weight_by: Literal["stoichiometry", "flux"] = "stoichiometry",
    nodes_to_remove: list[str] | None = None,
    reciprocal_weights: bool = False,
    threshold: float = 0.0,
    projection_weight: str | Callable[[float, float], float] | None = None,
    projection_weight_combine: str
    | Callable[[float, float], float]
    | None = None,
    **kwargs,
):
    """
    Create a metabolite connectivity network from the
    metabolic model

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    weighted : bool
        Whether the network should be weighted
    directed : bool
        Whether the network should be directed
    weight_by : 'stoichiometry' or 'flux', default='stoichiometry'
        String indicating if the network should be weighted by
        'stoichiometry', or 'flux' (see notes for more information).
        Ignored if `weighted = False`
    nodes_to_remove : list[str] | None
        List of any metabolites or reactions that should be removed from
        the final network. This can be used to remove metabolites that
        participate in a large number of reactions, but are not desired
        in downstream analysis such as water, or ATP, or pseudo
        reactions like biomass. Each metabolite/reaction should be the
        string ID associated with them in the cobra model.
    reciprocal_weights : bool
        Whether to use the reciprocal of the weights, useful if higher
        flux should equate with lower weights in the final network (for
        use with graph algorithms)
    threshold : float
        Threshold, below which to consider a bound to be 0
    projection_weight : str | Callable[[float, float], float] | None
        How to weight the projected graph. If None, the projected graph
        will not be weighted. If "ratio", the edges will be weighted
        based on the ratio between actual shared neighbors and maximum
        possible shared neighbors. If "count", the edges will be
        weighted by the number of shared neighbors. A function can also
        be provided, which takes two float arguments (the weights of two
        edges), and returns a float.
    projection_weight_combine : Callable[[list[float]], float], optional
        How to combine multiple projected edges. If two nodes in the set
        being projected onto, share multiple neighbors in the other node set,
        they can have multiple possible edge weights. This function takes in
        a list of possible weights, and returns a single final weight. Python
        builtin `max` and `min` can be used for this. If not provided,
        `max` is used.
    kwargs
        Keyword arguments are passed to the cobra flux_variability_analysis method
        when weight_by is flux
    """
    # Create the metabolic network
    metabolic_network = create_metabolic_network(
        model=model,
        weighted=weighted,
        directed=directed,
        weight_by=weight_by,
        nodes_to_remove=nodes_to_remove,
        reciprocal_weights=reciprocal_weights,
        threshold=threshold,
        **kwargs,
    )
    # Get a list of reactions which are in the model and not in the nodes_to_remove
    if nodes_to_remove:
        mets_to_remove_set = set(nodes_to_remove)
    else:
        mets_to_remove_set = set()
    met_ids = {
        m.id for m in model.metabolites if m.id not in mets_to_remove_set
    }
    # Project onto only reactions
    return bipartite_project(
        network=metabolic_network,
        node_set=met_ids,
        directed=directed,
        weight=projection_weight,
        weight_combine=projection_weight_combine,
        weight_attribute="weight",
        # reciprocal won't actually impact, since the graph will be
        # created with the correct directedness
        reciprocal=False,
    )


def create_gene_network(
    model: cobra.Model,
    directed: bool,
    nodes_to_remove: list[str] | None,
    essential: bool,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Create a gene connectivity network from the metabolic model,
    see notes for details

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    directed : bool
        Whether the network should be directed. It True,
        the network's edges direction will be decided by the
        directionality of the reaction network, and
        multiple genes associated with a single reaction
        will have two (reciprocal) edges connecting them.
    nodes_to_remove : list[str] or None
        List of any metabolites or reactions to remove
        from the metabolic network prior to projecting
        it onto the reactions and constructing the gene network.
        Each metabolite/reaction to remove should be the string
        id associated with them in the cobra Model
    essential : bool
        Whether a gene should be required for a reaction to function
        in order for that reaction to be used in assigning the
        gene edges

    Returns
    -------
    gene_network : nx.Graph or nx.DiGraph
        Network connecting genes which are neighboring in the
        reaction network together

    Notes
    -----
    The gene network includes nodes for each gene associated with
    a reaction in the network (whether or not essential is True).
    Edges are added by connecting each gene associated with a reaction
    to genes associated with all the neighboring reactions. If the
    graph is directed, then gene nodes are connected to genes associated
    with succcessor reactions. For genes associated with a single reaction
    they are given edges between them (going both directions in the
    case of directed graphs).

    The essential parameter is to decide which genes are associated
    with which reactions in order to determine which genes are neighbors
    in the gene network. If True, genes will only be associated with
    a reaction, when adding edges to the network, if they are required
    for that reaction to function. All genes associated with reactions
    in the network will still be added as nodes even if they are not
    essential for any reactions in the network.
    """
    # NOTE: Only unweighted due to ill-defined nature
    # of connecting multiple genes associated with a reaction,
    # if there is a good way of handling this it can be added.

    # Construct the reaction network
    rxn_network = create_reaction_network(
        model=model,
        weighted=False,
        directed=directed,
        nodes_to_remove=nodes_to_remove,
    )
    # Create the new gene network
    gene_list = reaction_to_gene_list(
        model=model, reaction_list=rxn_network.nodes, essential=False
    )
    # Create the new network
    if not directed:
        gene_network: Union[nx.Graph, nx.DiGraph] = nx.Graph()
    else:
        gene_network = nx.DiGraph()
    gene_network.add_nodes_from(gene_list)

    # Add edges
    for rxn in rxn_network.nodes:
        rxn_gene_set = reaction_to_gene_ids(
            model=model, reaction=rxn, essential=essential
        )
        # This won't run at all if there are not at least 2 genes
        for g1, g2 in itertools.combinations(rxn_gene_set, 2):
            gene_network.add_edge(g1, g2)
            gene_network.add_edge(g2, g1)
        # Go through all neighboring reactions (successors for directed)
        # NOTE: For networkx DiGraphs, neighbors and successors are the same
        for g1, g2 in itertools.product(
            rxn_gene_set,
            reaction_to_gene_list(
                model=model,
                reaction_list=rxn_network.neighbors(rxn),
                essential=essential,
            ),
        ):
            gene_network.add_edge(g1, g2)
    return gene_network


def create_group_neighborhood_network(
    network: Union[nx.Graph, nx.DiGraph],
    groups: dict[Hashable, Iterable[Hashable]],
    max_distance: int = 1,
    weighted: Optional[
        Literal[
            "count",
            "proportion",
            "enrichment",
        ]
    ] = None,
    directed: bool = False,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Create a group connectivity network, see notes for details

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        Network to use when finding neighbors. Edge weights
        will be ignored.
    groups : dict of Hashable to Iterable of Hashable
        Group definitions, must be a map between group names (which
        will be used as nodes in the network), and an iterable of
        group members (which should be nodes in the network)
    max_distance : int, default=1
        Max distance for nodes to be considered neighbors. A value of 0
        will only connect groups with direct overlaps, while a value of 1
        will connect groups which have members that are direct neighbors in the
        network.
    weighted : {'count', 'proportion', 'enrichment'}, optional
        Whether to weight the graph based on the number of connections
        between the groups. If None (default) no weights are added. If
        'count' then the edge weight is the count of connections between
        the two groups. If 'proportion', the edge weight is normalized
        by the maximum possible overlap. If enrichment, node attributes are
        added called pvalue, odds_ratio, and significance. The pvalue and
        odds ratio are the results of performing a Fisher's exact test on
        the enrichment of one group in the neighborhood of the other (in the
        undirected case, it is the minimum p-value/maximum odds_ratio found
        when finding the enrichment of one group in the neighborhood of the
        other). The significance is the -log10 of the p-value. Note that the
        odds_ratio can be infinite.
    directed : bool, default=False
        Whether the resulting connectivity graph should be directed,
        ignored unless the input network is directed.

    Returns
    -------
    group_neighborhood_network : nx.Graph or nx.DiGraph
        The group connectivity graph, which includes nodes for every group
        defined in `group`, with edges connecting groups which are connected
        in `network`, with optional edge weighted. Will be nx.Graph unless
        the input network is a DiGraph, and `directed` is True.

    Notes
    -----
    The group connectivity graph is a graph with a node for each group
    in `groups`, and edges connecting groups which include neighbors
    on the `network`.

    For example, take a graph with:

        * Nodes: {a, b, c, d, e, f, g}
        * Edges: {(a, b), (c,d), (e,f), (a,g)}

    then the group connectivity graph for groups
    {group1: {a,c}, group2:{d,e}, group3:{b,f}, group4:{g}}
    will produce the group connectivity graph (with parameter
    max_distance set to 1):

        * Nodes: {group1, group2, group3, group4}
        * Edges: {(group1, group2), (group1, group3), (group1, group4), (group2, group3)}

    When counting the number of connections, it is determined
    by finding the total neighborhood of one of the groups
    (that is the total node set within radius of a node
    in that group), and counting the number of nodes from
    the other group which are within that neighborhood.
    """
    # If the input network isn't directed, directed must be False
    if not isinstance(network, nx.DiGraph):
        directed = False
    # If the result shouldn't be directed, get an undirected view
    # of the input graph
    if not directed and isinstance(network, nx.DiGraph):
        network = nx.to_undirected(network)
    # Add the expected nodes
    connectivity_network = nx.Graph()
    connectivity_network.add_nodes_from(groups.keys())
    # Convert the iterables into sets for easier comparison
    group_sets = {k: set(v) for k, v in groups.items()}
    # Find the neighborhoods around the groups
    neighborhood_dict = {
        g: get_graph_neighborhood_group(
            network=network, radius=max_distance, nodes=n
        )
        for g, n in group_sets.items()
    }
    for g1, g2 in itertools.combinations(connectivity_network.nodes, 2):
        g1_overlaps_g2 = len(neighborhood_dict[g1] & group_sets[g2])
        g2_overlaps_g1 = len(neighborhood_dict[g2] & group_sets[g1])
        if not directed:
            if g1_overlaps_g2 > 0 or g2_overlaps_g1 > 0:
                if weighted == "count":
                    connectivity_network.add_edge(
                        g1, g2, weight=max(g1_overlaps_g2, g2_overlaps_g1)
                    )
                elif weighted == "proportion":
                    connectivity_network.add_edge(
                        g1,
                        g2,
                        weight=max(g1_overlaps_g2, g2_overlaps_g1)
                        / max(len(group_sets[g1]), len(group_sets[g2])),
                    )
                elif weighted == "enrichment":
                    g1_neighborhood = neighborhood_dict[g1]
                    g2_neighborhood = neighborhood_dict[g2]
                    g1_set = group_sets[g1]
                    g2_set = group_sets[g2]
                    fisher_res1 = stats.fisher_exact(
                        [
                            [
                                len(g1_neighborhood & g2_set),
                                len(g2_set - g1_neighborhood),
                            ],
                            [
                                len(g1_neighborhood - g2_set),
                                len(network.nodes)
                                - len(g1_neighborhood | g2_set),
                            ],
                        ],
                        alternative="greater",
                    )
                    fisher_res2 = stats.fisher_exact(
                        [
                            [
                                len(g2_neighborhood & g1_set),
                                len(g1_set - g2_neighborhood),
                            ],
                            [
                                len(g2_neighborhood - g1_set),
                                len(network.nodes)
                                - len(g2_neighborhood | g1_set),
                            ],
                        ],
                        alternative="greater",
                    )
                    pval = min(fisher_res1.pvalue, fisher_res2.pvalue)
                    odds = max(fisher_res1.statistic, fisher_res2.statistic)
                    connectivity_network.add_edge(
                        g1,
                        g2,
                        pvalue=pval,
                        odds_ratio=odds,
                        significance=-np.log10(pval),
                    )
                else:
                    connectivity_network.add_edge(g1, g2)
            continue
        # Directed Case
        if g1_overlaps_g2 > 0:
            if weighted == "count":
                connectivity_network.add_edge(g1, g2, weight=g1_overlaps_g2)
            elif weighted == "proportion":
                connectivity_network.add_edge(
                    g1, g2, weight=g1_overlaps_g2 / len(group_sets[g2])
                )
            if weighted == "enrichment":
                g1_neighborhood = neighborhood_dict[g1]
                g2_set = group_sets[g2]
                fisher_res = stats.fisher_exact(
                    [
                        [
                            len(g1_neighborhood & g2_set),
                            len(g2_set - g1_neighborhood),
                        ],
                        [
                            len(g1_neighborhood - g2_set),
                            len(network.nodes) - len(g1_neighborhood | g2_set),
                        ],
                    ],
                    alternative="greater",
                )
                connectivity_network.add_edge(
                    g1,
                    g2,
                    pvalue=fisher_res.pvalue,
                    odds_ratio=fisher_res.statistic,
                    significance=-np.log10(fisher_res.pvalue),
                )
            else:
                connectivity_network.add_edge(g1, g2)
        if g2_overlaps_g1 > 0:
            if weighted and weighted == "count":
                connectivity_network.add_edge(g2, g1, weight=g2_overlaps_g1)
            if weighted and weighted == "proportion":
                connectivity_network.add_edge(
                    g2, g1, weight=g2_overlaps_g1 / len(group_sets[g1])
                )
            if weighted == "enrichment":
                g2_neighborhood = neighborhood_dict[g2]
                g1_set = group_sets[g1]
                fisher_res = stats.fisher_exact(
                    [
                        [
                            len(g2_neighborhood & g1_set),
                            len(g1_set - g2_neighborhood),
                        ],
                        [
                            len(g2_neighborhood - g1_set),
                            len(network.nodes) - len(g2_neighborhood | g1_set),
                        ],
                    ],
                    alternative="greater",
                )
                connectivity_network.add_edge(
                    g1,
                    g2,
                    pvalue=fisher_res.pvalue,
                    odds_ratio=fisher_res.statistic,
                    significance=-np.log10(fisher_res.pvalue),
                )
            else:
                connectivity_network.add_edge(g2, g1)
    return connectivity_network


def create_group_distance_adjacency_matrix(
    network: Union[nx.Graph, nx.DiGraph],
    groups: dict[Hashable, Iterable[Hashable]],
    weight: Optional[str] = None,
    linkage: Literal["mean", "min", "max"] = "mean",
    directed: bool = False,
) -> pd.DataFrame:
    """
    Create an adjacency matrix for the distances between the `groups`

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        Network to use when finding distances between nodes
        in the groups. Edge weights are ignored.
    groups : : dict of Hashable to Iterable of Hashable
        Group definitions, must be a map between group names (which
        will be used as index/columns in the matrix), and an iterable of
        group members (which should be nodes in the network)
    weight : str, optional
        Edge attribute to use for weight, if None all edges have weight 1
    linkage : {'mean', 'min', 'max'}
        Method to use when combining pairwise distances between groups
    directed : bool
        Whether the adjacency matrix should be directed or not, ignored
        unless the input network is a nx.DiGraph

    Returns
    -------
    adjacency_matrix : pd.DataFrame
        DataFrame representing the adjacency matrix of the distances
        between the `groups` on the `network`. Index and columns
        are the keys of the `groups` dict, with values representing the
        distances between the groups.

    Notes
    -----
    Constructs the adjacency matrix using the pairwise distances between
    groups. For each pair of groups, finds the distances between their
    nodes and finds the distance between the two groups by aggregating
    these distances, either using the mean, minimum, or maximum of
    the set of pairwise distances between two groups of nodes.
    """
    # Compute the pairwise distances
    distance_dict = dict(nx.shortest_path_length(network))
    # Convert the groups into sets
    group_sets = {s: set(m) for s, m in groups.items()}
    # Get the set of all nodes in the network
    network_node_set = set(network.nodes)
    # Create the adjacency matrix
    adj_mat = pd.DataFrame(
        0.0, index=pd.Index(groups.keys()), columns=pd.Index(groups.keys())
    )
    # Fill in the adjacency matrix
    for g1, g2 in itertools.combinations(group_sets.keys(), 2):
        g1_nodes = group_sets[g1] & network_node_set
        g2_nodes = group_sets[g2] & network_node_set
        if isinstance(network, nx.Graph):
            # Undirected case
            adj_mat.loc[g1, g2] = _get_group_distance(
                distance_dict=distance_dict,
                group1=g1_nodes,
                group2=g2_nodes,
                linkage=linkage,
            )
            adj_mat.loc[g2, g1] = adj_mat.loc[g1, g2]
        if isinstance(network, nx.DiGraph):
            # Directed Case
            d1 = _get_group_distance(
                distance_dict=distance_dict,
                group1=g1_nodes,
                group2=g2_nodes,
                linkage=linkage,
            )
            d2 = _get_group_distance(
                distance_dict=distance_dict,
                group1=g2_nodes,
                group2=g1_nodes,
                linkage=linkage,
            )
            if directed:
                adj_mat.loc[g1, g2] = d1
                adj_mat.loc[g2, g2] = d2
            else:
                adj_mat.loc[g1, g2] = min(d1, d2)
                adj_mat.loc[g2, g1] = min(d1, d2)
    return adj_mat


def create_group_distance_network(
    network: Union[nx.Graph, nx.DiGraph],
    groups: dict[Hashable, Iterable[Hashable]],
    weight: Optional[str] = None,
    linkage: Literal["mean", "min", "max"] = "mean",
    directed: bool = False,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Create an network for the distances between the `groups`

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        Network to use when finding distances between nodes
        in the groups. Edge weights are ignored.
    groups : : dict of Hashable to Iterable of Hashable
        Group definitions, must be a map between group names (which
        will be used as index/columns in the matrix), and an iterable of
        group members (which should be nodes in the network)
    weight : str, optional
        Edge attribute to use for weight, if None all edges have weight 1
    linkage : {'mean', 'min', 'max'}
        Method to use when combining pairwise distances between groups
    directed : bool
        Whether the adjacency matrix should be directed or not, ignored
        unless the input network is a nx.DiGraph

    Returns
    -------
    nx.Graph or nx.DiGraph
        Network with a node for each group, and edges weighted by the distances
        between the `groups` on the `network`.

    Notes
    -----
    Constructs the network using the pairwise distances between
    groups. For each pair of groups, finds the distances between their
    nodes and finds the distance between the two groups by aggregating
    these distances, either using the mean, minimum, or maximum of
    the set of pairwise distances between two groups of nodes.

    """
    if directed:
        group_obj = nx.DiGraph
    else:
        group_obj = nx.Graph
    return group_obj(
        network=network,
        groups=groups,
        weight=weight,
        linkage=linkage,
        directed=directed,
    )


# endregion Main Function


# region Helpers
def _get_group_distance(
    distance_dict,
    group1: set[Hashable],
    group2: set[Hashable],
    linkage: Literal["mean", "min", "max"],
) -> float:
    max_ = -np.inf
    min_ = np.inf
    count = 0
    sum = 0.0
    for g1 in group1:
        for g2 in group2:
            dist = distance_dict[g1][g2] if g1 != g2 else 0.0
            max_ = max(max_, dist)
            min_ = min(min_, dist)
            sum += dist
            count += 1
    if linkage == "mean":
        return sum / count
    elif linkage == "min":
        return min_
    elif linkage == "max":
        return max_


def _get_rxn_attr_series(model: cobra.Model, attr: str) -> pd.Series:
    return pd.Series(
        model.reactions.list_attr(attr),
        index=model.reactions.list_attr("id"),
    )


def _get_lower_bounds(model: cobra.Model) -> pd.Series:
    return _get_rxn_attr_series(model, "lower_bound")


def _get_upper_bounds(model: cobra.Model) -> pd.Series:
    return _get_rxn_attr_series(model, "upper_bound")


def _get_stoichiometric_matrix(model: cobra.Model) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        cobra.util.create_stoichiometric_matrix(
            model=model, array_type="DataFrame"
        ),
    )


def _create_adj_matrix_ud_uw(
    model: cobra.Model, threshold: float
) -> pd.DataFrame:
    stoich_mat = _get_stoichiometric_matrix(model=model)
    lb_series = _get_lower_bounds(model=model)
    ub_series = _get_upper_bounds(model=model)
    product_mat = stoich_mat.copy().clip(lower=0.0)
    substrate_mat = stoich_mat.copy().clip(upper=0.0).abs()
    # Split into reaction gen/consume matrices
    rxn_gen_forward = product_mat.mul(ub_series > threshold)
    rxn_cons_forward = substrate_mat.mul(ub_series > threshold)
    rxn_gen_reverse = substrate_mat.mul(lb_series < -threshold)
    rxn_cons_reverse = product_mat.mul(lb_series < -threshold)
    # Build up the block matrix
    rxn_rxn_block = pd.DataFrame(
        False, columns=stoich_mat.columns, index=stoich_mat.columns
    )
    met_met_block = pd.DataFrame(
        False, columns=stoich_mat.index, index=stoich_mat.index
    )
    met_rxn_block = (
        (rxn_gen_forward > threshold)
        | (rxn_cons_forward > threshold)
        | (rxn_gen_reverse > threshold)
        | (rxn_cons_reverse > threshold)
    )
    rxn_met_block = met_rxn_block.T
    # Combine the blocks
    return pd.concat(
        [
            pd.concat([rxn_rxn_block, rxn_met_block], axis=1),
            pd.concat([met_rxn_block, met_met_block], axis=1),
        ],
        axis=0,
    )


def _create_adj_matrix_d_uw(
    model: cobra.Model, threshold: float
) -> pd.DataFrame:
    stoich_mat = _get_stoichiometric_matrix(model=model)
    lb_series = _get_lower_bounds(model=model)
    ub_series = _get_upper_bounds(model=model)
    product_mat = stoich_mat.copy().clip(lower=0.0)
    substrate_mat = stoich_mat.copy().clip(upper=0.0).abs()
    # Split into reaction gen/consum matrices
    rxn_gen_forward = product_mat.mul(ub_series > threshold)
    rxn_cons_forward = substrate_mat.mul(ub_series > threshold)
    rxn_gen_reverse = substrate_mat.mul(lb_series < -threshold)
    rxn_cons_reverse = product_mat.mul(lb_series < -threshold)
    # Build up the block matrix
    rxn_rxn_block = pd.DataFrame(
        False, columns=stoich_mat.columns, index=stoich_mat.columns
    )
    met_met_block = pd.DataFrame(
        False, columns=stoich_mat.index, index=stoich_mat.index
    )
    rxn_met_block = (
        (rxn_gen_forward > threshold) | (rxn_gen_reverse > threshold)
    ).T
    met_rxn_block = (rxn_cons_forward > threshold) | (
        rxn_cons_reverse > threshold
    )
    # Combine the blocks
    return pd.concat(
        [
            pd.concat([rxn_rxn_block, rxn_met_block], axis=1),
            pd.concat([met_rxn_block, met_met_block], axis=1),
        ],
        axis=0,
    )


def _create_adj_matrix_d_w_stoich(
    model: cobra.Model, threshold: float
) -> pd.DataFrame:
    stoich_mat = _get_stoichiometric_matrix(model=model)
    lb_series = _get_lower_bounds(model=model)
    ub_series = _get_upper_bounds(model=model)
    product_mat = stoich_mat.copy().clip(lower=0.0)
    substrate_mat = stoich_mat.copy().clip(upper=0.0).abs()
    # Split into reaction gen/consum matrices
    rxn_gen_forward = product_mat.mul(ub_series > threshold)
    rxn_cons_forward = substrate_mat.mul(ub_series > threshold)
    rxn_gen_reverse = substrate_mat.mul(lb_series < -threshold)
    rxn_cons_reverse = product_mat.mul(lb_series < -threshold)
    # Build up the block matrix
    rxn_rxn_block = pd.DataFrame(
        0.0, columns=stoich_mat.columns, index=stoich_mat.columns
    )
    met_met_block = pd.DataFrame(
        0.0, columns=stoich_mat.index, index=stoich_mat.index
    )
    rxn_met_block = np.maximum(rxn_gen_forward, rxn_gen_reverse).T
    met_rxn_block = np.maximum(rxn_cons_forward, rxn_cons_reverse)
    # Combine the blocks
    return pd.concat(
        [
            pd.concat([rxn_rxn_block, rxn_met_block], axis=1),
            pd.concat([met_rxn_block, met_met_block], axis=1),
        ],
        axis=0,
    )


def _create_adj_matrix_d_w_flux(
    model: cobra.Model, threshold: float, **kwargs
) -> pd.DataFrame:
    stoich_mat = _get_stoichiometric_matrix(model=model)
    fva_res = cobra.flux_analysis.flux_variability_analysis(
        model=model, **kwargs
    )
    min_series = fva_res["minimum"]
    max_series = fva_res["maximum"]
    product_mat = stoich_mat.copy().clip(lower=0.0)
    substrate_mat = stoich_mat.copy().clip(upper=0.0).abs()
    # Multiply the stoich matrices by the fva series
    # Split into reaction gen/consum matrices
    rxn_gen_forward = product_mat.mul(max_series).clip(lower=threshold)
    rxn_cons_forward = substrate_mat.mul(max_series).clip(lower=threshold)
    rxn_gen_reverse = (
        substrate_mat.mul(min_series).clip(upper=-threshold).abs()
    )
    rxn_cons_reverse = product_mat.mul(min_series).clip(upper=-threshold).abs()
    # Build up the block matrix
    rxn_rxn_block = pd.DataFrame(
        0.0, columns=stoich_mat.columns, index=stoich_mat.columns
    )
    met_met_block = pd.DataFrame(
        0.0, columns=stoich_mat.index, index=stoich_mat.index
    )
    rxn_met_block = np.maximum(rxn_gen_forward, rxn_gen_reverse).T
    met_rxn_block = np.maximum(rxn_cons_forward, rxn_cons_reverse)
    # Combine the blocks
    return pd.concat(
        [
            pd.concat([rxn_rxn_block, rxn_met_block], axis=1),
            pd.concat([met_rxn_block, met_met_block], axis=1),
        ],
        axis=0,
    )


def _create_adj_matrix_ud_w_stoich(
    model: cobra.Model, threshold: float
) -> pd.DataFrame:
    stoich_mat = _get_stoichiometric_matrix(model=model)
    lb_series = _get_lower_bounds(model=model)
    ub_series = _get_upper_bounds(model=model)
    product_mat = stoich_mat.copy().clip(lower=0.0)
    substrate_mat = stoich_mat.copy().clip(upper=0.0).abs()
    # Split into reaction gen/consum matrices
    rxn_gen_forward = product_mat.mul(ub_series > threshold)
    rxn_cons_forward = substrate_mat.mul(ub_series > threshold)
    rxn_gen_reverse = substrate_mat.mul(lb_series < -threshold)
    rxn_cons_reverse = product_mat.mul(lb_series < -threshold)
    # Build up the block matrix
    rxn_rxn_block = pd.DataFrame(
        0.0, columns=stoich_mat.columns, index=stoich_mat.columns
    )
    met_met_block = pd.DataFrame(
        0.0, columns=stoich_mat.index, index=stoich_mat.index
    )
    met_rxn_block = np.maximum(
        np.maximum(rxn_gen_forward, rxn_gen_reverse),
        np.maximum(rxn_cons_forward, rxn_cons_reverse),
    )
    rxn_met_block = met_rxn_block.T
    # Combine the blocks
    return pd.concat(
        [
            pd.concat([rxn_rxn_block, rxn_met_block], axis=1),
            pd.concat([met_rxn_block, met_met_block], axis=1),
        ],
        axis=0,
    )


def _create_adj_matrix_ud_w_flux(
    model: cobra.Model, threshold: float, **kwargs
) -> pd.DataFrame:
    stoich_mat = _get_stoichiometric_matrix(model=model)
    fva_res = cobra.flux_analysis.flux_variability_analysis(
        model=model, **kwargs
    )
    min_series = fva_res["minimum"]
    max_series = fva_res["maximum"]
    product_mat = stoich_mat.copy().clip(lower=0.0)
    substrate_mat = stoich_mat.copy().clip(upper=0.0).abs()
    # Multiply the stoich matrices by the fva series
    # Split into reaction gen/consum matrices
    rxn_gen_forward = product_mat.mul(max_series).clip(lower=threshold)
    rxn_cons_forward = substrate_mat.mul(max_series).clip(lower=threshold)
    rxn_gen_reverse = (
        substrate_mat.mul(min_series).clip(upper=-threshold).abs()
    )
    rxn_cons_reverse = product_mat.mul(min_series).clip(upper=-threshold).abs()
    # Build up the block matrix
    rxn_rxn_block = pd.DataFrame(
        0.0, columns=stoich_mat.columns, index=stoich_mat.columns
    )
    met_met_block = pd.DataFrame(
        0.0, columns=stoich_mat.index, index=stoich_mat.index
    )
    met_rxn_block = np.maximum(
        np.maximum(rxn_gen_forward, rxn_gen_reverse),
        np.maximum(rxn_cons_forward, rxn_cons_reverse),
    )
    rxn_met_block = met_rxn_block.T
    # Combine the blocks
    return pd.concat(
        [
            pd.concat([rxn_rxn_block, rxn_met_block], axis=1),
            pd.concat([met_rxn_block, met_met_block], axis=1),
        ],
        axis=0,
    )


# endregion Helpers

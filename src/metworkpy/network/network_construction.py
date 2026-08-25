"""
Functions for constructing networks based on genome scale metabolic models
"""

# Imports
# Standard Library Imports
from __future__ import annotations

import itertools
from collections.abc import Hashable, Iterable
from typing import (
    Callable,
    Literal,
    cast,
)

# External Imports
import cobra
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse, stats

# Local Imports
from metworkpy.information.mutual_information_network import (
    mi_pairwise,
)
from metworkpy.network.neighborhoods import (
    get_graph_neighborhood_group,
)
from metworkpy.network.projection import bipartite_project
from metworkpy.utils import reaction_to_gene_ids, reaction_to_gene_list

ALMOST_ZERO = 1e-15


##################################
### Mutual Information Network ###
##################################


def create_mutual_information_network(
    model: cobra.Model | None = None,
    flux_samples: pd.DataFrame | np.ndarray | None = None,
    reaction_names: Iterable[str] | None = None,
    cutoff_significance: float | None = None,
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
        raise TypeError(
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


# endregion Main Function


####################################
### Network Generation Functions ###
####################################
def create_metabolic_network(
    model: cobra.Model,
    weight: None
    | Literal["stoichiometry", "fva", "pfba", "gfba"]
    | np.typing.ArrayLike
    | pd.Series
    | tuple[np.typing.ArrayLike, np.typing.ArrayLike]
    | tuple[pd.Series, pd.Series] = None,
    directed: bool = True,
    weight_by_metabolite_stoich: bool = True,
    product_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    reactant_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    nodes_to_remove: Iterable[str] | None = None,
    remove_top_metabolites: int | None = None,
    weight_scale_fn: None | Callable[[np.ndarray], np.ndarray] = None,
    zero_tolerance: float = ALMOST_ZERO,
    **kwargs,
) -> nx.Graph | nx.DiGraph:
    """
    Create a bipartite metabolic network from provided
    cobra Model, with nodes representing both reactions and metabolites

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    weight : {'stoichiometry', "fva", "pfba", "gfba"} or ArrayLike or Series or tuple of ArrayLike or Series, optional
        The reaction weights to use for creating the adjacency matrix. If None the network represented
        by the adjacency matrix will be unweighted (all values will be 0 or 1). If an ArrayLike or Series,
        treated as reaction weights, with positive values being used for forward weights,
        and negative values being used for reverse weights. If a tuple, treated as
        (forward, reverse). For all array arguments, they should be a 1-D array (or
        coercible to a 1-D array), with length equal to the number of reactions in the model.
        Also, all weights (forward and reverse) should be positive.
        See `Notes` for more information.
    directed : bool
        Whether the network should be directed
    weight_by_metabolite_stoich: bool, default=true
        whether the reaction weights should be multiplied by
        a metabolite's stoichiometric coefficient to find
        the edge weight between a reation and a metabolite
        (or a metabolite and a reaction).
    product_scale_fn, reactant_scale_fn : callable of coo_array to coo_array, optional
        if provided function will be called on the reactant and product
        edge weight arrays (both with columns for reactions and rows for
        metabolites). the product array is all the weights of edges connecting a
        reaction to a metabolite, and the reactant array represents all of the
        edges connecting a metabolite to a reaction. these functions must return a
        coo_array of the same dimension of the passed array. this allows for rescaling
        or otherwise modifying the edge weights prior to network construction if that is desired.
    nodes_to_remove : Iterable of str, optional
        Iterable of nodes which will be removed from the network before it is returned
    remove_top_metabolites : int, optional
        Number of top most connected metabolites to remove. This can be useful to remove
        common currency metabolites such as ATP, or solvent metabolites like H20.
    weight_scale_fn : callable taking np.ndarray and returning np.ndarray, optional
        Optional function for scaling the weights, called with a 1-D numpy array of all the
        weights in the network, and must return a 1-D numpy array of the same size.
        This could be used to make the weights all fall in a specific range
        (e.g. use a minmax scalar so they are all between 0 and 1),
        or to invert the direction of the weights (so larger weights become smaller) by
        taking the reciprocal of all the weights.
    zero_tolerance : float
        Threshold, below which to consider a (absolute value of a) bound/flux
        to be 0
    kwargs
        Passed to COBRApy functions depending on value of `weight`.

            * `flux_variability_analysis <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/index.html#cobra.flux_analysis.flux_variability_analysis>`_ if `weight` is 'fva'
            * `pfba <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/index.html#cobra.flux_analysis.pfba>`_ if `weight` is 'pfba'
            * `geometric_fba <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/geometric/index.html#cobra.flux_analysis.geometric.geometric_fba>`_ if `weight` is 'gfba'

    Returns
    -------
    nx.Graph or nx.DiGraph
        The bipartite network constructed from the provided `cobra.Model`,
        with nodes for reactions and metabolites (using the reaction/metabolite id
        as the node id).

    Notes
    -----
    When creating a weighted network, for each (reaction, metabolite) edge the weight
    is the reaction weight multiplied by the stoichiometric coefficient of the metabolite.
    Each reaction is allowed a forward, and a reverse weight. The forward weights
    are used to connect reactions to their products, and the reverse weights are
    used to connect reactions to their reactants.

    As an example, take a reaction named rxn1 with formula 2A + B -> 3C, a forward weight of
    2.5, and a reverse weight of 5.0. The reaction will connect to the A,B and C
    metabolites, and the edges will have weights 10.0, 5.0, and 7.5 respectively.

    For the weights parameter, these forward and reverse weights can be supplied
    directly as a tuple of (forward, reverse), where forward and reverse can be
    either numpy arrays or pandas series (they should have length equal to the number
    of reactions in the model). Alternatively, they can be supplied as a single
    numpy array or series, where each reaction has only a forward or (exclusive) a
    reverse weight. In this case positive values will be treated as the forward
    weight, and negative values will be treated as reverse weights (but their
    absolute value will be the actual weight value).

    Another option is to use the stoichiometry directly as weights, this is equivalent
    to supplying 1 for all forward weights for reactions which can run in the forward
    direction, and 0 for all reactions that can't. Simmilarly for the reverse weights,
    values of 1 for all reactions which can run in reverse, and 0 for all reactions
    that can't.

    Alternatively, several strategies of using flux to weight to edges can be employed,
    specifically flux variability analysis (fva), parsimonious flux balance analysis (pfba),
    or geometric flux balance analysis (gfba).

    For fva, the maximum possible positive flux through a reaction is used as its forward
    weight (reactions whose maximum flux is negative are given forward weights of 0), and
    the minimum possible negative flux is used as its reverse weight.

    For pfba, the resulting flux is used as the weights, with positive values
    being used for forward weights, and negative values being used for reverse weights.
    gfba is the same as pfba, except using geometric instead of parsimonious flux balance
    analysis.
    """
    adj_mat = cast(
        sparse.coo_array,
        create_adjacency_matrix(
            model=model,
            weight=weight,
            directed=directed,
            array_type="coo",
            zero_tolerance=zero_tolerance,
            weight_by_metabolite_stoich=weight_by_metabolite_stoich,
            product_scale_fn=product_scale_fn,
            reactant_scale_fn=reactant_scale_fn,
            **kwargs,
        ),
    )

    if weight_scale_fn is not None:
        adj_mat.data = weight_scale_fn(adj_mat.data)

    if remove_top_metabolites is not None:
        mets_to_remove = set(
            get_top_metabolites(model=model, n=remove_top_metabolites)
        )
    else:
        mets_to_remove = set()

    if nodes_to_remove is not None:
        nodes_to_remove = set(nodes_to_remove) | mets_to_remove
    else:
        nodes_to_remove = mets_to_remove

    # Create the network
    met_network = nx.from_scipy_sparse_array(
        adj_mat, create_using=nx.DiGraph if directed else nx.Graph
    )

    met_network = nx.relabel_nodes(
        met_network,
        {
            idx: node.id
            for idx, node in enumerate(
                itertools.chain(model.reactions, model.metabolites)
            )
        },
    )
    met_network.remove_nodes_from(nodes_to_remove)
    return met_network


def create_reaction_network(
    model: cobra.Model,
    weight: None
    | Literal["stoichiometry", "fva", "pfba", "gfba"]
    | np.typing.ArrayLike
    | pd.Series
    | tuple[np.typing.ArrayLike, np.typing.ArrayLike]
    | tuple[pd.Series, pd.Series] = None,
    directed: bool = True,
    weight_by_metabolite_stoich: bool = True,
    product_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    reactant_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    nodes_to_remove: Iterable[str] | None = None,
    remove_top_metabolites: int | None = None,
    weight_scale_fn: None | Callable[[np.ndarray], np.ndarray] = None,
    projection_weight: str | Callable[[float, float], float] | None = None,
    projection_weight_combine: Callable[[list[float]], float] | None = None,
    zero_tolerance: float = ALMOST_ZERO,
    **kwargs,
):
    """
    Create a reaction connectivity network from the
    metabolic model by projecting the bipartite metabolic network
    onto the reaction nodes

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    weight : {'stoichiometry', "fva", "pfba", "gfba"} or ArrayLike or Series or tuple of ArrayLike or Series, optional
        The reaction weights to use for creating the adjacency matrix. If None the network represented
        by the adjacency matrix will be unweighted (all values will be 0 or 1). If an ArrayLike or Series,
        treated as reaction weights, with positive values being used for forward weights,
        and negative values being used for reverse weights. If a tuple, treated as
        (forward, reverse). For all array arguments, they should be a 1-D array (or
        coercible to a 1-D array), with length equal to the number of reactions in the model.
        Also, all weights (forward and reverse) should be positive.
        See `Notes` for more information.
    directed : bool
        Whether the network should be directed
    weight_by_metabolite_stoich: bool, default=True
        Whether the reaction weights should be multiplied by
        a metabolite's stoichiometric coefficient to find
        the edge weight between a reation and a metabolite
        (or a metabolite and a reaction).
    product_scale_fn, reactant_scale_fn : Callable of coo_array to coo_array, optional
        If provided function will be called on the reactant and product
        edge weight arrays (both with columns for reactions and rows for
        metabolites). The product array is all the weights of edges connecting a
        reaction to a metabolite, and the reactant array represents all of the
        edges connecting a metabolite to a reaction. These functions must return a
        coo_array of the same dimension of the passed array. This allows for rescaling
        or otherwise modifying the edge weights prior to network construction if that is desired.
    nodes_to_remove : Iterable of str, optional
        Iterable of nodes which will be removed from the network before it is returned
    remove_top_metabolites : int, optional
        Number of top most connected metabolites to remove. This can be useful to remove
        common currency metabolites such as ATP, or solvent metabolites like H20.
    weight_scale_fn : callable taking np.ndarray and returning np.ndarray, optional
        Optional function for scaling the weights, called with a 1-D numpy array of all the
        weights in the network, and must return a 1-D numpy array of the same size.
        This could be used to make the weights all fall in a specific range
        (e.g. use a minmax scalar so they are all between 0 and 1),
        or to invert the direction of the weights (so larger weights become smaller) by
        taking the reciprocal of all the weights.
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
    zero_tolerance : float
        Threshold, below which to consider a (absolute value of a) bound/flux
        to be 0
    kwargs
        Passed to COBRApy functions depending on value of `weight`.

            * `flux_variability_analysis <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/index.html#cobra.flux_analysis.flux_variability_analysis>`_ if `weight` is 'fva'
            * `pfba <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/index.html#cobra.flux_analysis.pfba>`_ if `weight` is 'pfba'
            * `geometric_fba <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/geometric/index.html#cobra.flux_analysis.geometric.geometric_fba>`_ if `weight` is 'gfba'
    kwargs
        Keyword arguments are passed to the cobra flux_variability_analysis method
        when weight_by is flux
    """
    # Create the metabolic network
    metabolic_network = create_metabolic_network(
        model=model,
        weight=weight,
        directed=directed,
        nodes_to_remove=nodes_to_remove,
        remove_top_metabolites=remove_top_metabolites,
        weight_scale_fn=weight_scale_fn,
        zero_tolerance=zero_tolerance,
        weight_by_metabolite_stoich=weight_by_metabolite_stoich,
        product_scale_fn=product_scale_fn,
        reactant_scale_fn=reactant_scale_fn,
        **kwargs,
    )
    # Get the reaction nodes
    rxn_nodes = set(metabolic_network.nodes) & {r.id for r in model.reactions}

    # Project onto only reactions
    if weight is not None:
        if projection_weight is None:
            projection_weight = min
        if projection_weight_combine is None:
            projection_weight_combine = max
    return bipartite_project(
        network=metabolic_network,
        node_set=rxn_nodes,
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
    weight: None
    | Literal["stoichiometry", "fva", "pfba", "gfba"]
    | np.typing.ArrayLike
    | pd.Series
    | tuple[np.typing.ArrayLike, np.typing.ArrayLike]
    | tuple[pd.Series, pd.Series] = None,
    directed: bool = True,
    weight_by_metabolite_stoich: bool = True,
    product_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    reactant_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    nodes_to_remove: Iterable[str] | None = None,
    remove_top_metabolites: int | None = None,
    weight_scale_fn: None | Callable[[np.ndarray], np.ndarray] = None,
    projection_weight: str | Callable[[float, float], float] | None = None,
    projection_weight_combine: Callable[[list[float]], float] | None = None,
    zero_tolerance: float = ALMOST_ZERO,
    **kwargs,
):
    """
    Create a metabolite connectivity network from the
    metabolic model by projecting the bipartite metabolic network
    onto the metabolite nodes

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    weight : {'stoichiometry', "fva", "pfba", "gfba"} or ArrayLike or Series or tuple of ArrayLike or Series, optional
        The reaction weights to use for creating the adjacency matrix. If None the network represented
        by the adjacency matrix will be unweighted (all values will be 0 or 1). If an ArrayLike or Series,
        treated as reaction weights, with positive values being used for forward weights,
        and negative values being used for reverse weights. If a tuple, treated as
        (forward, reverse). For all array arguments, they should be a 1-D array (or
        coercible to a 1-D array), with length equal to the number of reactions in the model.
        Also, all weights (forward and reverse) should be positive.
        See `Notes` for more information.
    directed : bool
        Whether the network should be directed
    nodes_to_remove : Iterable of str, optional
        Iterable of nodes which will be removed from the network before it is returned
    remove_top_metabolites : int, optional
        Number of top most connected metabolites to remove. This can be useful to remove
        common currency metabolites such as ATP, or solvent metabolites like H20.
    weight_scale_fn : callable taking np.ndarray and returning np.ndarray, optional
        Optional function for scaling the weights, called with a 1-D numpy array of all the
        weights in the network, and must return a 1-D numpy array of the same size.
        This could be used to make the weights all fall in a specific range
        (e.g. use a minmax scalar so they are all between 0 and 1),
        or to invert the direction of the weights (so larger weights become smaller) by
        taking the reciprocal of all the weights.
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
    zero_tolerance : float
        Threshold, below which to consider a (absolute value of a) bound/flux
        to be 0
    kwargs
        Passed to COBRApy functions depending on value of `weight`.

            * `flux_variability_analysis <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/index.html#cobra.flux_analysis.flux_variability_analysis>`_ if `weight` is 'fva'
            * `pfba <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/index.html#cobra.flux_analysis.pfba>`_ if `weight` is 'pfba'
            * `geometric_fba <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/geometric/index.html#cobra.flux_analysis.geometric.geometric_fba>`_ if `weight` is 'gfba'
    kwargs
        Keyword arguments are passed to the cobra flux_variability_analysis method
        when weight_by is flux
    """
    # Create the metabolic network
    metabolic_network = create_metabolic_network(
        model=model,
        weight=weight,
        directed=directed,
        nodes_to_remove=nodes_to_remove,
        remove_top_metabolites=remove_top_metabolites,
        weight_scale_fn=weight_scale_fn,
        zero_tolerance=zero_tolerance,
        weight_by_metabolite_stoich=weight_by_metabolite_stoich,
        product_scale_fn=product_scale_fn,
        reactant_scale_fn=reactant_scale_fn,
        **kwargs,
    )
    # Get the metabolite nodes
    met_nodes = set(metabolic_network.nodes) & {
        m.id for m in model.metabolites
    }

    # Project onto only reactions
    if weight is not None:
        if projection_weight is None:
            projection_weight = min
        if projection_weight_combine is None:
            projection_weight_combine = max
    return bipartite_project(
        network=metabolic_network,
        node_set=met_nodes,
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
    directed: bool = True,
    nodes_to_remove: list[str] | None = None,
    remove_top_metabolites: int | None = None,
    essential: bool = False,
) -> nx.Graph | nx.DiGraph:
    """
    Create a gene connectivity network from the metabolic model,
    see notes for details

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    directed : bool
        Whether the network should be directed. If True,
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
    remove_top_metabolites : int, optional
        Number of top most connected metabolites to remove. This can be useful to remove
        common currency metabolites such as ATP, or solvent metabolites like H20.
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
        weight=None,
        directed=directed,
        nodes_to_remove=nodes_to_remove,
        remove_top_metabolites=remove_top_metabolites,
    )
    return reaction_to_gene_network(
        model=model,
        reaction_network=rxn_network,
        directed=directed,
        essential=essential,
    )


######################
### Group Networks ###
######################


def create_group_neighborhood_network(
    network: nx.Graph | nx.DiGraph,
    groups: dict[Hashable, Iterable[Hashable]],
    max_distance: int = 1,
    weighted: Literal["count", "proportion", "enrichment"] | None = None,
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
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


def create_group_distance_network(
    network: nx.Graph | nx.DiGraph,
    groups: dict[Hashable, Iterable[Hashable]],
    weight: str | None = None,
    linkage: Literal["mean", "min", "max"] = "mean",
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
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


def create_group_distance_adjacency_matrix(
    network: nx.Graph | nx.DiGraph,
    groups: dict[Hashable, Iterable[Hashable]],
    weight: str | None = None,
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
    distance_dict = dict(nx.shortest_path_length(network, weight=weight))
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
            adj_mat.loc[g2, g1] = adj_mat.loc[g1, g2]  # type: ignore
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


##########################
### Network Conversion ###
##########################


def reaction_to_gene_network(
    model: cobra.Model,
    reaction_network: nx.Graph | nx.DiGraph,
    directed: bool | None = None,
    essential: bool = False,
) -> nx.Graph | nx.DiGraph:
    """
    Create a gene connectivity network from a reaction connectivity
    network, see notes for details

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    reaction_network : nx.Graph or nx.DiGraph
        The reaction network to convert into a gene network
    directed : bool
        Whether the network should be directed. If True,
        the network's edges direction will be decided by the
        directionality of the reaction network, and
        multiple genes associated with a single reaction
        will have two (reciprocal) edges connecting them.
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
    # Create the new gene network
    gene_list = reaction_to_gene_list(
        model=model, reaction_list=reaction_network.nodes, essential=False
    )
    # Create the new network
    if not directed:
        gene_network: nx.Graph | nx.DiGraph = nx.Graph()
    elif directed and reaction_network.is_directed():
        gene_network = nx.DiGraph()
    else:
        gene_network = nx.Graph()
    gene_network.add_nodes_from(gene_list)

    # Add edges
    for rxn in reaction_network.nodes:
        reaction_gene_set = reaction_to_gene_ids(
            model=model, reaction=rxn, essential=essential
        )
        # This won't run at all if there are not at least 2 genes
        for g1, g2 in itertools.combinations(reaction_gene_set, 2):
            gene_network.add_edge(g1, g2)
            gene_network.add_edge(g2, g1)
        # Go through all neighboring reactions (successors for directed)
        # NOTE: For networkx DiGraphs, neighbors and successors are the same
        for g1, g2 in itertools.product(
            reaction_gene_set,
            reaction_to_gene_list(
                model=model,
                reaction_list=reaction_network.neighbors(rxn),
                essential=essential,
            ),
        ):
            gene_network.add_edge(g1, g2)
    return gene_network


########################
### Adjacency Matrix ###
########################
def create_adjacency_matrix(
    model: cobra.Model,
    weight: None
    | Literal["stoichiometry", "fva", "pfba", "gfba"]
    | np.typing.ArrayLike
    | pd.Series
    | tuple[np.typing.ArrayLike, np.typing.ArrayLike]
    | tuple[pd.Series, pd.Series] = None,
    directed: bool = True,
    weight_by_metabolite_stoich: bool = True,
    product_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    reactant_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    array_type: Literal[
        "dense", "frame", "bsr", "coo", "csc", "csr", "dia", "dok", "lil"
    ] = "frame",
    zero_tolerance: float = ALMOST_ZERO,
    **kwargs,
) -> pd.DataFrame | np.ndarray | sparse.sparray:
    """
    Create an adjacency matrix representing the bipartite metabolic network of a provided
    cobra Model, with nodes representing both reactions and metabolites

    Parameters
    ----------
    model : cobra.Model
        Cobra Model to create the network from
    weight : {'stoichiometry', "fva", "pfba", "gfba"} or ArrayLike or Series or tuple of ArrayLike or Series, optional
        The reaction weights to use for creating the adjacency matrix. If None the network represented
        by the adjacency matrix will be unweighted (all values will be 0 or 1). If an ArrayLike or Series,
        treated as reaction weights, with positive values being used for forward weights,
        and negative values being used for reverse weights. If a tuple, treated as
        (forward, reverse). For all array arguments, they should be a 1-D array (or
        coercible to a 1-D array), with length equal to the number of reactions in the model.
        Also, all weights (forward and reverse) should be positive.
        See `Notes` for more information.
    directed : bool
        Whether the network should be directed
    weight_by_metabolite_stoich: bool, default=True
        Whether the reaction weights should be multiplied by
        a metabolite's stoichiometric coefficient to find
        the edge weight between a reation and a metabolite
        (or a metabolite and a reaction).
    product_scale_fn, reactant_scale_fn : Callable of coo_array to coo_array, optional
        If provided function will be called on the reactant and product
        edge weight arrays (both with columns for reactions and rows for
        metabolites). The product array is all the weights of edges connecting a
        reaction to a metabolite, and the reactant array represents all of the
        edges connecting a metabolite to a reaction. These functions must return a
        coo_array of the same dimension of the passed array. This allows for rescaling
        or otherwise modifying the edge weights prior to network construction if that is desired.
    array_type : {'dense', 'frame', 'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil'}, default='frame'
        The type to use for the adjacency matrix. "dense" will return a numpy.ndarray,
        "frame" will return a dataframe (indexed by reaction and metabolite ids). The other
        types will return a `scipy sparse array <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_
        of that type (so "coo" will return a `coo_array <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html#scipy.sparse.coo_array>`_).
    zero_tolerance : float
        Threshold, below which to consider a (absolute value of a) bound/flux
        to be 0
    kwargs
        Passed to COBRApy functions depending on value of `weight`.

            * `flux_variability_analysis <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/index.html#cobra.flux_analysis.flux_variability_analysis>`_ if `weight` is 'fva'
            * `pfba <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/index.html#cobra.flux_analysis.pfba>`_ if `weight` is 'pfba'
            * `geometric_fba <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/flux_analysis/geometric/index.html#cobra.flux_analysis.geometric.geometric_fba>`_ if `weight` is 'gfba'

    Returns
    -------
    pd.DataFrame or np.ndarray or scipy.sparse.sparray
        The adjacency matrix, the index is ordered based on the
        cobra model's order, reactions first, and then metabolites.

    Notes
    -----
    When creating a weighted network, for each (reaction, metabolite) edge the weight
    is the reaction weight  multiplied by the stoichiometric coefficient of the metabolite
    (optionally, depending on `weihght_by_metabolite_stoich`).
    Each reaction is allowed a forward, and a reverse weight. The forward weights
    are used to connect reactions to their products, and the reverse weights are
    used to connect reactions to their reactants.

    As an example, take a reaction named rxn1 with formula 2A + B -> 3C, a forward weight of
    2.5, and a reverse weight of 5.0. The reaction will connect to the A,B and C
    metabolites, and the edges will have weights 10.0, 5.0, and 7.5 respectively.
    Or, if `weight_by_metabolite_stoich` is False, then the reaction connects to
    A, B, and C with weights of 2.5, 2.5, and 5.0 respectively.

    For the weights parameter, these forward and reverse weights can be supplied
    directly as a tuple of (forward, reverse), where forward and reverse can be
    either numpy arrays or pandas series (they should have length equal to the number
    of reactions in the model). Alternatively, they can be supplied as a single
    numpy array or series, where each reaction has only a forward or (exclusive) a
    reverse weight. In this case positive values will be treated as the forward
    weight, and negative values will be treated as reverse weights (but their
    absolute value will be the actual weight value).

    Another option is to use the stoichiometry directly as weights, this is equivalent
    to supplying 1 for all forward weights for reactions which can run in the forward
    direction, and 0 for all reactions that can't. Simmilarly for the reverse weights,
    values of 1 for all reactions which can run in reverse, and 0 for all reactions
    that can't.

    Alternatively, several strategies of using flux to weight to edges can be employed,
    specifically flux variability analysis (fva), parsimonious flux balance analysis (pfba),
    or geometric flux balance analysis (gfba).

    For fva, the maximum possible positive flux through a reaction is used as its forward
    weight (reactions whose maximum flux is negative are given forward weights of 0), and
    the minimum possible negative flux is used as its reverse weight.

    For pfba, the resulting flux is used as the weights, with positive values
    being used for forward weights, and negative values being used for reverse weights.
    gfba is the same as pfba, except using geometric instead of parsimonious flux balance
    analysis.
    """
    # Construct the reaction weights
    weighted = weight is not None
    if weight is None:
        # For unweighted, use the lower and upper bound (clipped at 0)
        # for the weights
        forward = sparse.coo_array(
            np.array(model.reactions.list_attr("upper_bound"))
        )
        forward[forward < 0.0] = 0.0

        reverse = sparse.coo_array(
            np.array(model.reactions.list_attr("lower_bound"))
        )
        reverse[reverse > 0.0] = 0.0
        reverse *= -1
    elif isinstance(weight, str):
        if weight == "stoichiometry":
            # Want 1 for all forward with upper bound greater than 0.0, 0 otherwise
            # and 1 for all reverse with lower bound less than 0.0, 0 otherwise
            forward = sparse.coo_array(
                np.array(model.reactions.list_attr("upper_bound"))
            )
            forward[forward < 0.0] = 0.0
            forward[forward > 0.0] = 1.0
            reverse = sparse.coo_array(
                np.array(model.reactions.list_attr("lower_bound"))
            )
            reverse[reverse > 0.0] = 0.0
            reverse[reverse < 0.0] = 1.0
        elif weight == "fva":
            fva_result = _enforce_threshold(
                cobra.flux_analysis.flux_variability_analysis(
                    model=model, **kwargs
                ),
                zero_tolerance,
            )
            min_series = fva_result["minimum"].clip(upper=0.0).abs()
            max_series = fva_result["maximum"].clip(lower=0.0)
            # Convert into COO
            forward = sparse.coo_array(max_series)
            reverse = sparse.coo_array(min_series)
        elif weight == "pfba":
            pfba_res = _enforce_threshold(
                cobra.flux_analysis.pfba(model=model, **kwargs).fluxes,
                zero_tolerance,
            )
            forward = sparse.coo_array(pfba_res.clip(lower=0.0))
            reverse = sparse.coo_array(pfba_res.clip(upper=0.0).abs())
        elif weight == "gfba":
            gfba_res = _enforce_threshold(
                cobra.flux_analysis.geometric_fba(
                    model=model, **kwargs
                ).fluxes,
                zero_tolerance,
            )
            forward = sparse.coo_array(gfba_res.clip(lower=0.0))
            reverse = sparse.coo_array(gfba_res.clip(upper=0.0).abs())
        else:
            raise ValueError(
                f"Expected weight to be one of 'stoichiometry', 'fva', 'pfba', or 'gfba' but received {weight}"
            )
    elif isinstance(weight, pd.Series):
        forward = sparse.coo_array(weight.clip(lower=0.0))
        reverse = sparse.coo_array(weight.clip(upper=0.0).abs())
    elif isinstance(weight, tuple):
        forward_weight, reverse_weight = weight
        forward = sparse.coo_array(forward_weight)
        reverse = sparse.coo_array(reverse_weight)
    else:
        weight = np.array(weight)
        forward = sparse.coo_array(weight)
        reverse = sparse.coo_array(weight)

        forward[forward < 0.0] = 0.0
        reverse[reverse > 0.0] = 0.0
        reverse *= -1
    forward.eliminate_zeros()
    reverse.eliminate_zeros()
    # Create the sparse adjacency matrix
    adj_mat = _create_sparse_adjacency_matrix(
        model=model,
        forward=forward,
        reverse=reverse,
        directed=directed,
        weighted=weighted,
        weight_by_metabolite_stoich=weight_by_metabolite_stoich,
        product_scale_fn=product_scale_fn,
        reactant_scale_fn=reactant_scale_fn,
        zero_tolerance=zero_tolerance,
    )
    if array_type == "dense":
        return adj_mat.todense()
    elif array_type == "frame":
        adj_index = pd.Index(
            model.reactions.list_attr("id") + model.metabolites.list_attr("id")
        )
        return pd.DataFrame(
            adj_mat.todense(), index=adj_index, columns=adj_index
        )
    elif array_type == "bsr":
        return adj_mat.tobsr()
    elif array_type == "coo":
        return adj_mat.tocoo()
    elif array_type == "csc":
        return adj_mat.tocsc()
    elif array_type == "csr":
        return adj_mat.tocsr()
    elif array_type == "dia":
        return adj_mat.todia()
    elif array_type == "dok":
        return adj_mat.todok()
    elif array_type == "lil":
        return adj_mat.tolil()
    raise ValueError(
        f"Expected array_type to be one of 'dense', 'frame', 'bsr', 'coo', 'csc', 'csr', 'dia', 'dok', or 'lil' but received {array_type}"
    )


###############################
### Sparse Adjacency Matrix ###
###############################
def _create_sparse_adjacency_matrix(
    model: cobra.Model,
    forward: sparse.sparray,
    reverse: sparse.sparray,
    directed: bool = True,
    weighted: bool = True,
    weight_by_metabolite_stoich: bool = True,
    product_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    reactant_scale_fn: None
    | Callable[[sparse.coo_array], sparse.coo_array] = None,
    zero_tolerance: float = ALMOST_ZERO,
) -> sparse.coo_array:
    """
    Creates an Adjacency matrix from a stoichiometric matrix

    Parameters
    ----------
    model : cobra.Model
        The model to construct the adjacency matrix for
    forward : scipy sparse array
        1-D Array representing forward reaction weights
    reverse : scipy sparse array
        1-D Array representing reverse reaction weights
    directed : bool, default=True
        Whether the adjacency matrix should be directed. If False,
        then the directed version is found first, and connections are
        decided by combining the forward and reverse directions for
        each node pair.
    weighted : bool, default=True
        Whether the adjacency matrix should be weighted. If False,
        all weights above `zero_tolerance` are set to 1, and
        all weights below `zero_tolerance` are set to 0.
    weight_by_metabolite_stoich: bool, default=True
        Whether the reaction weights should be multiplied by
        a metabolite's stoichiometric coefficient to find
        the edge weight between a reation and a metabolite
        (or a metabolite and a reaction).
    product_scale_fn, reactant_scale_fn : Callable of coo_array to coo_array, optional
        If provided function will be called on the reactant and product
        edge weight arrays (both with columns for reactions and rows for
        metabolites). The product array is all the weights of edges connecting a
        reaction to a metabolite, and the reactant array represents all of the
        edges connecting a metabolite to a reaction. These functions must return a
        coo_array of the same dimension of the passed array. This allows for rescaling
        or otherwise modifying the edge weights prior to network construction if that is desired.
    zero_tolerance : float, default=1e-15
        Tolerance for values to be considered differently from 0. Weights
        whose absolute values are less than this will be set to 0.

    Returns
    -------
    adjacency_matrix : sparse.coo_array
        The adjacency matrix in the form of a sparse COOrdinate array
    """
    # Get the sparse stoichiometric matrix
    stoichiometric_matrix = _create_stoichiometric_matrix(model=model)
    if not weighted or not weight_by_metabolite_stoich:
        stoichiometric_matrix: sparse.coo_array = stoichiometric_matrix.sign()  # ty: ignore[unresolved-attribute]
    # Get the number of reactions, and metabolites
    n_met, n_rxns = stoichiometric_matrix.shape
    # Convert Forward and reverse to csr
    forward = sparse.csr_array(forward.reshape((-1,)))  # ty: ignore[unresolved-attribute]
    reverse = sparse.csr_array(reverse.reshape((-1,)))  # ty: ignore[unresolved-attribute]

    # Split the stoichiomety into products and reactants
    product_array = stoichiometric_matrix
    reactant_array = stoichiometric_matrix.copy()
    product_array[product_array < 0.0] = 0.0
    reactant_array[reactant_array > 0.0] = 0.0
    reactant_array = reactant_array * -1

    # Convert to csr arrays for the multiplication
    product_array = product_array.tocsr()
    reactant_array = reactant_array.tocsr()

    # Multiply by the stoich matrices by the forward/reverse weightings
    # NOTE: Multiplying by reverse yields the opposite type (product/reactant)
    product_forward: sparse.coo_array = (product_array * forward).tocoo()
    reactant_reverse: sparse.coo_array = (product_array * reverse).tocoo()
    reactant_forward: sparse.coo_array = (reactant_array * forward).tocoo()
    product_reverse: sparse.coo_array = (reactant_array * reverse).tocoo()

    # Create the reaction->metabolite, and the metabolite->reaction
    # matrices, both of which will
    product_array: sparse.coo_array = product_forward.maximum(product_reverse)
    reactant_array: sparse.coo_array = reactant_forward.maximum(
        reactant_reverse
    )
    if product_scale_fn is not None:
        product_array = product_scale_fn(product_array)
    if reactant_scale_fn is not None:
        reactant_array = reactant_scale_fn(reactant_array)

    # Build the blocks of the matrix
    rxn_rxn_block = sparse.coo_array((n_rxns, n_rxns))
    met_met_block = sparse.coo_array((n_met, n_met))
    met_rxn_block: sparse.coo_array = reactant_array
    rxn_met_block: sparse.coo_array = product_array.T

    # Create the adjacency matrix
    adj_mat = sparse.vstack(
        [
            sparse.hstack([rxn_rxn_block, rxn_met_block]),
            sparse.hstack([met_rxn_block, met_met_block]),
        ]
    ).tocoo()

    if not weighted:
        adj_mat = adj_mat.sign()
    # Convert all entries within zero_tolerance of zero to be 0
    adj_mat.data[
        (adj_mat.data < zero_tolerance) & (adj_mat.data > -zero_tolerance)
    ] = 0.0
    adj_mat.eliminate_zeros()
    if not directed:
        # Convert to undirected, using the maximum directed weight
        adj_mat: sparse.coo_array = adj_mat.maximum(adj_mat.T).tocoo()
    return adj_mat


##################################
### Find Connected Metabolites ###
##################################


def get_top_metabolites(
    model: cobra.Model,
    n: int,
    type: Literal["substrate", "reactant", "product"] = "substrate",
) -> list[str]:
    """
    Get a list of the top `n` metabolites involved in the
    most reactions in the `model`

    Parameters
    ----------
    model : cobra.Model
        The model to find the top metabolites for
    n : int
        The number of top metabolites to find

    Returns
    -------
    list of str
        A list of the ids of the top `n` metabolites in the `model`
    """
    # Get a count of the reactions each metabolite is involved in
    stoich_mat = cobra.util.create_stoichiometric_matrix(
        model=model, array_type="DataFrame"
    )
    assert isinstance(stoich_mat, pd.DataFrame), (
        "Cobra returned incorrect stoichiometric matrix type"
    )
    if type == "substrate":
        counts = (stoich_mat.abs() > 0).sum(axis=1)
    elif type == "reactant":
        counts = (stoich_mat.clip(upper=0.0) < 0.0).sum(axis=1)
    elif type == "product":
        counts = (stoich_mat.clip(lower=0.0) > 0.0).sum(axis=1)
    else:
        raise ValueError(
            f"Type must be 'substrate', 'reactant', or 'product', but received {type}"
        )
    return list(counts.sort_values(ascending=False).iloc[:n].index)


def get_top_metabolite_pairs(
    model: cobra.Model,
    n: int,
    ignore_top: int = 0,
) -> list[tuple[str, str]]:
    """
    Get a list including tuples of the most frequent metabolite
    pairs in the model

    Parameters
    ----------
    model : cobra.Model
        The model to find the top metabolite pairs for
    n : int
        The number of top metabolite pairs to find
    ignore_top : int
        Before finding pairwise frequency of metabolites,
        remove the top `ignore_top` number of metabolites

    Returns
    -------
    list of tuples of str,str
        A list of the most common metabolite pairs in the form
        of a list of tuples, each containg a pair of metabolite ids
    """
    # Get a count of the reactions each metabolite is involved in
    stoich_mat = cobra.util.create_stoichiometric_matrix(
        model=model, array_type="DataFrame"
    ).drop(  # type:ignore
        get_top_metabolites(model=model, n=ignore_top, type="substrate"),
        axis=0,
    )
    met_pair_freq = pd.DataFrame(
        0.0, index=stoich_mat.index, columns=stoich_mat.index
    )
    assert isinstance(stoich_mat, pd.DataFrame), (
        "Cobra returned incorrect stoichiometric matrix type"
    )
    for _, met_series in stoich_mat.items():  # noqa: PERF102
        for met1, met2 in itertools.combinations(
            met_series[met_series > 0].index, 2
        ):
            met_pair_freq.loc[met1, met2] += 1.0
            met_pair_freq.loc[met2, met1] += 1.0
    top_met_pair_list = []
    for _ in range(n):
        # Find the metabolite with the highest frquency
        m1 = met_pair_freq.max(axis=0).idxmax()
        m2 = met_pair_freq[m1].idxmax()
        met_pair_freq.drop([m1, m2], axis=1, inplace=True)
        met_pair_freq.drop([m1, m2], axis=0, inplace=True)
        top_met_pair_list.append((m1, m2))
    return top_met_pair_list  # type: ignore


#######################
### Helper Functions###
#######################
def _enforce_threshold(
    data: pd.DataFrame | pd.Series, threshold: float
) -> pd.DataFrame | pd.Series:
    data[(data >= -threshold) & (data <= threshold)] = 0.0
    return data


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


def _create_stoichiometric_matrix(model: cobra.Model) -> sparse.coo_array:
    """
    Replacing COBRApy's version since that uses the old sparse matrix
    instead of sparse array
    """
    n_met, n_rxn = len(model.metabolites), len(model.reactions)
    stoich_array = sparse.coo_array((n_met, n_rxn))

    met_ind = model.metabolites.index
    rxn_ind = model.reactions.index

    for rxn in model.reactions:
        for met, stoich in rxn.metabolites.items():
            stoich_array[met_ind(met), rxn_ind(rxn)] = stoich
    return stoich_array

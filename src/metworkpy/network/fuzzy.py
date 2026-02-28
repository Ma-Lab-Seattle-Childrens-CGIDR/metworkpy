"""
Sub-module for finding fuzzy sets of reactions
"""

# Standard Library Imports
from __future__ import annotations
import functools
import math
from typing import (
    cast,
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Union,
    Literal,
)

# External Imports
import cobra
from joblib import Parallel, delayed
import networkx as nx
import numpy as np
import pandas as pd
from robustrankaggregpy.aggregate_ranks import (
    rank_matrix_from_df,
    rho_scores,
)
from scipy.stats import gmean

# Local Imports
from metworkpy.network.neighborhoods import (
    _graph_gene_neighborhood,
    _graph_neighborhood,
)
from metworkpy.utils.translate import get_reaction_to_gene_translation_dict


class FuzzyMembershipFunction(Protocol):
    """
    Protocol for fuzzy membership functions, which should take a reaction,
    a network, a set of target genes, and a reaction to gene set mapping,
    and return a float between 0 and 1. This method can also take in
    additional parameters as kwargs, which will be passed through form
    the calling functions.
    """

    def __call__(
        self,
        reaction: cobra.Reaction,
        network: nx.Graph,
        gene_set: set[str],
        reaction_to_gene_dict: dict[str, set[str]],
        **kwargs,
    ) -> float: ...


# region Membership Functions


def membership_simple_gene_density(
    reaction: cobra.Reaction,
    network: nx.Graph,
    gene_set: set[str],
    reaction_to_gene_dict: dict[str, set[str]],
    radius: int,
) -> float:
    """
    Membership function which computes the membership based on how
    many genes within distance `radius` are in the target gene set

    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction to find the membership of
    network : nx.Graph
        Connectivity graph of the network
    gene_set : set of str
        The set of gene's to translate into a reaction set
    reaction_to_gene_dict : dict of str to set of str
        A dict for translating from reactions to sets of genes
        associated with each reaction
    radius : int
        The distance used to define network neighborhoods

    Returns
    -------
    membership : float
        The membership of the reaciton in the reaction set
    """
    gene_neighborhood = _graph_gene_neighborhood(
        network=network,
        radius=radius,
        node=reaction.id,
        rxn_to_gene_set_dict=reaction_to_gene_dict,
    )
    if len(gene_neighborhood) == 0:
        return 0.0
    return float(len(gene_neighborhood & gene_set)) / len(gene_neighborhood)


def membership_simple_reaction_density(
    reaction: cobra.Reaction,
    network: nx.Graph,
    gene_set: set[str],
    reaction_to_gene_dict: dict[str, set[str]],
    radius: int,
) -> float:
    """
    Membership function which computes the membership based on how
    many reactions withhin distance `radius` are associated with
    a gene in the target gene set

    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction to find the membership of
    network : nx.Graph
        Connectivity graph of the network
    gene_set : set of str
        The set of gene's to translate into a reaction set
    reaction_to_gene_dict : dict of str to set of str
        A dict for translating from reactions to sets of genes
        associated with each reaction
    radius : int
        The distance used to define network neighborhoods

    Returns
    -------
    membership : float
        The membership of the reaciton in the reaction set
    """
    reaction_neighborhood = _graph_neighborhood(
        network=network, radius=radius, node=reaction.id
    )
    if len(reaction_neighborhood) == 0:
        return 0.0
    target_rxn_count = 0.0
    for rxn in reaction_neighborhood:
        rxn = cast(str, rxn)
        if len(gene_set & reaction_to_gene_dict.get(rxn, set())) > 0:
            target_rxn_count += 1.0
    return target_rxn_count / float(len(reaction_neighborhood))


# region weight functions


def weight_fn_geom(distance: int) -> float:
    """
    Weights layers using geometric series of 1/2^(n+1)
    """
    return 1.0 / (math.pow(2.0, distance + 1))


# endregion weight functions


def membership_distance_weighted_gene_density(
    reaction: cobra.Reaction,
    network: nx.Graph,
    gene_set: set[str],
    reaction_to_gene_dict: dict[str, set[str]],
    max_radius: int = 3,
    weight_fn: Callable[[int], float] = weight_fn_geom,
    allow_repeats: bool = False,
) -> float:
    """
    Membership function which computes the membership based on how
    many genes within distance `radius` are in the target gene set,
    decreasing the weight of each gene as it moves farther from the
    reaction, and still normalizing for the number of genes in each
    layer (i.e. the number of genes which show up at a certain
    distance)

    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction to find the membership of
    network : nx.Graph
        Connectivity graph of the network
    gene_set : set of str
        The set of gene's to translate into a reaction set
    reaction_to_gene_dict : dict of str to set of str
        A dict for translating from reactions to sets of genes
        associated with each reaction
    max_radius : int
        The maximum distance used to include genes from
    weight_fn : Callable taking int returning float
        The function used to compute the weight for genes
        depending on their distance from the reaction. Should
        take in the distance, and return a weight. For this
        to act as a membership function, the sum of the
        weights should be 1.0
    allow_repeats : bool, default=False
        Whether to allow genes to be counted multiple times,
        genes that have been seen before will be removed from the
        gene neighborhood prior to calculating the membership
        contribution for the layer

    Returns
    -------
    membership : float
        The membership of the reaction in the reaction set
    """
    # Iterate through the layers in a bfs search
    membership = 0.0
    # Track previously seen genes
    previous_genes = set()
    for distance, nodes in enumerate(nx.bfs_layers(network, reaction.id)):
        if len(nodes) == 0:
            continue
        if distance > max_radius:
            break
        # Convert the nodes into a gene set
        gene_neighborhood = functools.reduce(
            lambda left, right: left | right,
            map(reaction_to_gene_dict.get, nodes),
        )

        if len(gene_neighborhood - previous_genes) == 0:
            continue
        membership += weight_fn(distance) * (
            len((gene_neighborhood & gene_set) - previous_genes)
            / len(gene_neighborhood - previous_genes)
        )
        if not allow_repeats:
            previous_genes |= gene_neighborhood

    return membership


def membership_distance_weighted_reaction_density(
    reaction: cobra.Reaction,
    network: nx.Graph,
    gene_set: set[str],
    reaction_to_gene_dict: dict[str, set[str]],
    max_radius: int = 3,
    weight_fn: Callable[[int], float] = weight_fn_geom,
) -> float:
    """
    Membership function which computes the membership based on how
    many genes within distance `radius` are in the target gene set

    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction to find the membership of
    network : nx.Graph
        Connectivity graph of the network
    gene_set : set of str
        The set of gene's to translate into a reaction set
    reaction_to_gene_dict : dict of str to set of str
        A dict for translating from reactions to sets of genes
        associated with each reaction
    radius : int
        The distance used to define network neighborhoods
    max_radius : int
        The maximum distance used to include genes from
    weight_fn : Callable taking int returning float
        The function used to compute the weight for genes
        depending on their distance from the reaction. Should
        take in the distance, and return a weight.

    Returns
    -------
    membership : float
        The membership of the reaction in the reaction set
    """
    # Iterate through the layers in a bfs search
    membership = 0.0
    for distance, nodes in enumerate(nx.bfs_layers(network, reaction.id)):
        if distance > max_radius:
            break
        if len(nodes) == 0:
            continue
        target_rxn_count = 0.0
        # Count the number of reactions
        for rxn in nodes:
            rxn = cast(str, rxn)
            if len(gene_set & reaction_to_gene_dict.get(rxn, set())) > 0:
                target_rxn_count += 1.0

        membership += weight_fn(distance) * (
            target_rxn_count / float(len(nodes))
        )

    return membership


def membership_knn_gene_distance(
    reaction: cobra.Reaction,
    network: nx.Graph,
    gene_set: set[str],
    reaction_to_gene_dict: dict[str, set[str]],
    max_radius: int = 5,
    k_neighbors: int = 3,
    allow_repeats: bool = False,
) -> float:
    r"""
    Membership function which computes the membership based on distance
    to the kth neighbor.

    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction to find the membership of
    network : nx.Graph
        Connectivity graph of the network
    gene_set : set of str
        The set of gene's to translate into a reaction set
    reaction_to_gene_dict : dict of str to set of str
        A dict for translating from reactions to sets of genes
        associated with each reaction
    max_radius : int
        The maximum radius to search for neighbors,
        if less than k-neighbors are found in this radius
        the membership will be 0.0
    k_neighbors : int
        The number of neighbors in the gene_set to use for estimating
        density in the graph
    allow_repeats : bool, default=False
        Whether to allow genes to be counted multiple times,
        genes that have been seen before will not be counted
        towards the number of neighbors

    Returns
    -------
    membership : float
        The membership of the reaction in the reaction set

    Notes
    -----
    The membership is calculated as

    .. math:: \frac{1}{knn-distance+1}

    The :math:`knn-distance` is the distance from the reaction to the
    kth neighbor which is in the gene set, so that if k=1, and the reaction
    is directly associated with a gene the knn-distance will be 0. If k=2,
    the distance from the reaction to the second closest node associated
    with a gene in gene_set will be used.
    """
    if k_neighbors == 0:
        raise ValueError("k_neighbors must be greater than 0")
    neighbors_seen = 0
    previous_genes = set()
    for distance, nodes in enumerate(nx.bfs_layers(network, reaction.id)):
        # Get the genes in the neighborhood
        gene_neighborhood = functools.reduce(
            lambda left, right: left | right,
            map(reaction_to_gene_dict.get, nodes),
        )
        neighbors_seen += len((gene_neighborhood & gene_set) - previous_genes)
        if not allow_repeats:
            previous_genes |= gene_neighborhood
        if neighbors_seen >= k_neighbors:
            break
        if distance + 1 > max_radius:
            return 0.0
    return 1 / float(distance + 1)


def membership_knn_reaction_distance(
    reaction: cobra.Reaction,
    network: nx.Graph,
    gene_set: set[str],
    reaction_to_gene_dict: dict[str, set[str]],
    max_radius: int = 5,
    k_neighbors: int = 3,
) -> float:
    r"""
    Membership function which computes the membership based on distance
    to the kth neighbor.

    Parameters
    ----------
    reaction : cobra.Reaction
        The reaction to find the membership of
    network : nx.Graph
        Connectivity graph of the network
    gene_set : set of str
        The set of gene's to translate into a reaction set
    reaction_to_gene_dict : dict of str to set of str
        A dict for translating from reactions to sets of genes
        associated with each reaction
    max_radius : int
        The maximum radius to search for neighbors,
        if less than k-neighbors are found in this radius
        the membership will be 0.0
    k_neighbors : int
        The number of neighbors in the gene_set to use for estimating
        density in the graph
    dimension : int, default=2
        Dimension parameter, used for scaling the density, see notes
        Larger values will result in smaller membership values.
    diameter : int, optional
        The diameter of the network. If not provided it will be calculated from
        `network`

    Returns
    -------
    membership : float
        The membership of the reaction in the reaction set

    Notes
    -----
    The membership is calculated as

    .. math:: \frac{1}{knn-distance+1}

    The :math:`knn-distance` is the distance from the reaction to the
    kth neighbor reaction which is associated with a gene in the gene set,
    so that if k=1, and the reaction is directly associated with a gene
    the knn-distance will be 0. If k=2, the distance from the reaction
    to the second closest node associated with a gene in gene_set will
    be used.
    """
    if k_neighbors == 0:
        raise ValueError("k_neighbors must be greater than 0")
    neighbors_seen = 0
    for distance, nodes in enumerate(nx.bfs_layers(network, reaction.id)):
        # Count the number of reactions in the layer
        # Which are associated with a gene in gene_set
        target_rxn_count = 0.0
        for rxn in nodes:
            rxn = cast(str, rxn)
            if len(gene_set & reaction_to_gene_dict.get(rxn, set())) > 0:
                target_rxn_count += 1.0
        neighbors_seen += target_rxn_count
        if neighbors_seen >= k_neighbors:
            break
        if distance + 1 > max_radius:
            return 0.0
    # The subtraction from distance is due to it incrementing
    # an additional time past when the knn was found, since the check
    # is at the start of the for loop
    return 1 / float(distance + 1)


# endregion Membership Functions

# region Fuzzy Reaction Set

MEMBERSHIP_FUNCTIONS = {
    "simple gene density": membership_simple_gene_density,
    "simple reaction density": membership_simple_reaction_density,
    "weighted gene density": membership_distance_weighted_gene_density,
    "weighted reaction density": membership_distance_weighted_reaction_density,
    "knn gene density": membership_knn_gene_distance,
    "knn reaction density": membership_knn_reaction_distance,
}


def fuzzy_reaction_set(
    metabolic_network: Union[nx.Graph, nx.DiGraph],
    metabolic_model: cobra.Model,
    gene_set: Iterable[str],
    membership_fn: Union[str, FuzzyMembershipFunction] = "simple gene density",
    scale: Optional[Union[bool, float]] = None,
    essential: bool = False,
    processes: Optional[int] = None,
    **kwargs,
) -> pd.Series:
    """
    Convert from a gene set to a fuzzy reaction set

    Parameters
    ----------
    metabolic_network : nx.Graph or nx.DiGraph
        Metabolic reaction network represented by a networkx Graph or DiGraph.
        DiGraphs will be converted to Graphs before processing.
    metabolic_model : cobra.Model
        Metabolic model from which the metabolic network was constructed
        (used for translating reactions to genes)
    gene_set : Iterable of str
        Set of genes to convert into a fuzzy reaction set
    membership_fn : str or `FuzzyMembershipFunction`
        The membership function to use, can be a string giving the
        functions name, or the function itself which must match the
        signature of `FuzzyMembershipFunction`
    scale : bool or float, optional
        Whether to scale the results of the membership values. If
        False or None, no scaling will be applied. If True, will
        be scaled to be between 0 and 1 using a min-max scaler.
        If a float, the scaling will use a min-max scaler, but
        treat `scale` as the max.
    essential : bool
        Whether, when translating from reactions to genes, only
        genes required for a reaction to function should be associated
        with a particular reaction.
    processes : int, optional
        Number of processes to use for parallel processing
    kwargs
        Keyword arguments passed to the membership_fn

    Returns
    -------
    reaction_set : pd.Series
        The fuzzy reaction set, described by a pandas series. The index
        is the reaction id, and the values are the set membership.

    Notes
    -----
    The options for membership functions to be selected by
    name (i.e. str arg to `membership_fn`) are

    * 'simple gene density'
    * 'simple reaction density'
    * 'weighted gene density'
    * 'weighted reaction density'
    * 'knn gene density'
    * 'knn reaction density'

    The difference between the gene and reaction density functions, are
    how multiple genes being associated with a single reaction are counted.
    For the gene type, multiple genes will all count towards the membership,
    whereas with the reaction type reactions are counted only once regardless
    of how many genes associated with them are in the gene set.

    See Also
    --------
    membership_simple_gene_density : Used when 'simple gene density' selected
    membership_simple_reaction_density : Used when 'simple reaction density' selected
    membership_distance_weighted_gene_density : Used when 'weighted gene density' selected
    membership_distance_weighted_reaction_density : Used when 'weighted reaction density' selected
    membership_knn_gene_density : Used when 'knn gene density' selected
    membership_knn_reaction_density : Used when 'knn reaction density' selected
    """
    # Get the correct membership function
    if isinstance(membership_fn, str):
        if membership_fn not in MEMBERSHIP_FUNCTIONS:
            raise ValueError(
                f"Unable to find correct membership function, options are "
                f"{list(MEMBERSHIP_FUNCTIONS.keys())}"
            )

        membership_fn = MEMBERSHIP_FUNCTIONS[membership_fn]  # type: ignore

    if not callable(membership_fn):
        raise ValueError("Received invalid membership function")
    # Convert directed graphs into undirected graphs
    if isinstance(metabolic_network, nx.DiGraph):
        metabolic_network = nx.to_undirected(metabolic_network)
    if not isinstance(metabolic_network, nx.Graph):
        raise ValueError(
            f"metabolic_network must be a networkx Graph or "
            f"DiGraph but received {type(metabolic_network)}"
        )
    # Ensure the gene_set is a set of genes
    gene_set = set(gene_set)
    # Get a mapping from reactions to genes
    rxn_to_gene_dict: dict[str, set[str]] = (
        get_reaction_to_gene_translation_dict(
            model=metabolic_model, essential=essential
        )
    )
    # Create the results series
    rxn_set = pd.Series(
        0.0, index=pd.Index(metabolic_model.reactions.list_attr("id"))
    )
    for rxn, membership in zip(
        rxn_set.index,
        Parallel(n_jobs=processes, return_as="generator")(
            delayed(membership_fn)(
                metabolic_model.reactions.get_by_id(rxn),
                network=metabolic_network,
                gene_set=gene_set,
                reaction_to_gene_dict=rxn_to_gene_dict,
                **kwargs,
            )
            for rxn in rxn_set.index
        ),
    ):
        rxn_set[rxn] = membership

    if scale:
        if isinstance(scale, float):
            max_val = scale
        else:
            max_val = rxn_set.max()
        rxn_set = (rxn_set - rxn_set.min()) / max_val
    return rxn_set


# endregion Fuzzy Reaction Set

# region Fuzzy Reaction Intersection


def fuzzy_reaction_intersection(
    gene_sets: Iterable[Iterable[str]],
    metabolic_network: Union[nx.Graph, nx.DiGraph],
    metabolic_model: cobra.Model,
    intersection_fn: Union[
        Callable[[pd.DataFrame], pd.Series],
        Literal["mean", "min", "max", "geom", "rra"],
    ],
    intersection_fn_kwargs=Optional[dict[str, Any]],
    rank_method: Literal["average", "min", "max", "first", "dense"] = "max",
    **kwargs,
) -> pd.Series:
    """
    Converts `gene_sets` into fuzzy reaction sets, and find their intersection
    using `intersection_fn`

    Parameters
    ----------
    gene_sets : iterable of iterable of str
        Sets of genes to find the fuzzy reaction set intersection for
    metabolic_network : nx.Graph or nx.DiGraph
        Metabolic reaction network represented by a networkx Graph or DiGraph.
        DiGraphs will be converted to Graphs before processing.
    metabolic_model : cobra.Model
        Metabolic model from which the metabolic network was constructed
        (used for translating reactions to genes)
    intersection_fn : {"mean", "min", "max", "geom", "rra"} or Callable[[pd.DataFrame], pd.Series]
        Either a str specifying an intersection function (see notes), or
        a Callable which takes a DataFrame, where each column is a fuzzy reaction
        set and returns a Series which is a new fuzzy reaction set representing
        the intersection of the input fuzzy reaction sets.
    intersection_fn_kwargs : dict of str to Any
        kwargs passed to the intersection function
    rank_method : {"average", "min", "max", "first", "dense"}
        If the `intersection_fn` is 'rra', how are ties in the
        membership values handled when performing ranking
    kwargs
        Keyword arguments are passed to `fuzzy_reaction_set`

    Returns
    -------
    intersection : pd.Series
        A pandas Series representing a fuzzy reaction set constructed
        by intersecting the fuzzy reaction sets derived from the
        `gene_sets`.

    Notes
    -----
    The possible methods for the intersection are:
    * mean: Take the arithmetic mean of the membership values
    * min: Take the minimum of the membership values
    * max: Take the max of the membership values
    * geom: Take the geometric mean of the membership values
    * rra: Perform robust rank aggregation on the membership values,
      and the subtract the resulting rho-score from 1.0
    """
    # Construct the DataFrame from the gene sets
    rxn_set_list = []
    for gene_set in gene_sets:
        rxn_set_list.append(
            fuzzy_reaction_set(
                metabolic_network=metabolic_network,
                metabolic_model=metabolic_model,
                gene_set=gene_set,
                **kwargs,
            )
        )
    rxn_set_df = pd.concat(rxn_set_list, axis=1)
    if callable(intersection_fn):
        if intersection_fn_kwargs is None:
            intersection_fn_kwargs = {}
        rxn_intersect_series = intersection_fn(
            rxn_set_df, **intersection_fn_kwargs
        )
    elif intersection_fn == "mean":
        rxn_intersect_series = rxn_set_df.mean(axis=1)
    elif intersection_fn == "min":
        rxn_intersect_series = rxn_set_df.min(axis=1)
    elif intersection_fn == "max":
        rxn_intersect_series = rxn_set_df.max(axis=1)
    elif intersection_fn == "geom":
        rxn_intersect_series = rxn_set_df.aggregate(gmean, axis=1)
    elif intersection_fn == "rra":
        rank_mat = rank_matrix_from_df(
            rxn_set_df,
            ascending=False,
            rank_method=rank_method,
            **intersection_fn_kwargs,
        )
        rxn_intersect_series = 1 - pd.Series(
            np.apply_along_axis(
                rho_scores, 1, rank_mat.to_numpy(), **intersection_fn_kwargs
            ),
            index=rank_mat.index,
        )
    else:
        raise ValueError(f"Invalid intersection_fn: {intersection_fn}")

    return rxn_intersect_series


# endregion Fuzzy Reaction Intersection

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import NamedTuple, Iterable, Optional

# External Imports
import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import flux_variability_analysis
import networkx as nx
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse import sparray, csr_array, csc_array

# Local Imports
from metworkpy.network._array_utils import (
    _split_arr_col,
    _split_arr_sign,
    _split_arr_row,
    _sparse_max,
    _broadcast_mult_arr_vec,
)
from metworkpy.utils._arguments import _parse_str_args_dict
from metworkpy.information.mutual_information_network import mi_network_adjacency_matrix


# region Main Function
def create_mutual_information_network(
    model: Optional[cobra.Model] = None,
    flux_samples: pd.DataFrame | np.ndarray | None = None,
    reaction_names: Iterable[str] | None = None,
    n_samples: int = 10_000,
    n_neighbors: int = 5,
    truncate: bool = True,
    reciprocal_weights: bool = False,
    processes: Optional[int] = 1,
    progress_bar: bool = False,
) -> nx.Graph:
    """
    Create a mutual information network from the provided metabolic model

    :param model: Metabolic model to construct the mutual information network from. Only required if the flux_samples
        parameter is None
    :type model: Optional[cobra.Model]
    :param flux_samples: Flux samples used to calculate mutual information between reactions. If None,
        the passed model will be sampled to generate these flux samples.
    :type flux_samples: Optional[pd.DataFrame|np.ndarray]
    :param reaction_names: Names for the reactions
    :type reaction_names: Optional[Iterable[str]]
    :param n_samples: Number of samples to take if flux_samples is None (ignored if flux_samples is not None)
    :type n_samples: int
    :param n_neighbors: Number of neighbors to use during the mutual information estimation
    :type n_neighbors: int
    :param truncate: Whether the mutual information values should be truncated at 0. Mutual information should
        always be greater than or equal to 0, but the estimates can be negative. If true, all the mutual
        information values which are less than 0 will be set to 0.
    :type truncate: bool
    :param reciprocal_weights: Whether the non-zero weights in the network should be the reciprocal of mutual
        information.
    :type reciprocal_weights: bool
    :param processes: Number of processes to use during the mutual information calculation
    :type processes: Optional[int]
    :param progress_bar: Whether a progress bar should be shown for the mutual information calculations
    :type progress_bar: bool
    :return: A networkx Graph, which nodes representing different reactions and edge weights corresponding to
        estimated mutual information
    :rtype: nx.Graph
    """
    if model is None and flux_samples is None:
        raise ValueError(
            "Requires either a metabolic model, or flux samples but received neither"
        )
    if flux_samples is None:
        flux_samples = cobra.sampling.sample(
            model=model, n=n_samples, processes=processes
        )
    if isinstance(flux_samples, pd.DataFrame):
        sample_array = flux_samples.to_numpy()
        if not reaction_names:
            reaction_names = flux_samples.columns
    elif isinstance(flux_samples, np.ndarray):
        sample_array = flux_samples
        if not reaction_names:
            if model:
                reaction_names = model.reactions.list_attr("id")
            else:
                reaction_names = [f"rxn_{i}" for i in range(sample_array.shape[1])]
    else:
        raise ValueError(
            f"Invalid type for flux samples, requires pandas DataFrame or numpy ndarray, but"
            f"received {type(flux_samples)}"
        )
    adj_mat = mi_network_adjacency_matrix(
        samples=sample_array,
        n_neighbors=n_neighbors,
        processes=processes,
        progress_bar=progress_bar,
    )
    if truncate:
        adj_mat[adj_mat < 0] = 0
    if reciprocal_weights:
        # Should be all floats, so no issue with integer division
        adj_mat[adj_mat > 0] = np.reciprocal(adj_mat[adj_mat > 0])
    mi_network = nx.from_numpy_array(
        adj_mat, create_using=nx.Graph, parallel_edges=False
    )
    _ = nx.relabel_nodes(
        mi_network,
        {idx: f"{rxn}" for idx, rxn in enumerate(reaction_names)},
        copy=False,
    )
    return mi_network


def create_metabolic_network(
    model: cobra.Model,
    weighted: bool,
    directed: bool,
    weight_by: str = "stoichiometry",
    nodes_to_remove: list[str] | None = None,
    reaction_data: list[str] | None = None,
    metabolite_data: list[str] | None = None,
    reciprocal_weights: bool = False,
    threshold: float = 1e-4,
    loopless: bool = False,
    fva_proportion: float = 1.0,
) -> nx.Graph | nx.DiGraph:
    """
    Create a metabolic network from a cobrapy Model

    :param model: Cobra Model to create the network from
    :type model: cobra.Model
    :param weighted: Whether the network should be weighted
    :type weighted: bool
    :param directed: Whether the network should be directed
    :type directed: bool
    :param weight_by: String indicating if the network should be weighted by
        'stoichiometry', or 'flux' (see notes for more information). Ignored if
        `weighted = False`
    :type weight_by: str
    :param reaction_data: List of additional attributes to include as node attributes
        for each reaction
    :type reaction_data: list[str] | None
    :param nodes_to_remove: List of any metabolites or reactions that should be removed
        from the final network. This can be used to remove metabolites that participate
        in a large number of reactions, but are not desired in downstream analysis
        such as water, or ATP, or pseudo reactions like biomass. Each
        metabolite/reaction should be the string ID associated with them in the cobra
        model.
    :type nodes_to_remove: list[str] | None
    :param metabolite_data: List of additional data to include as node
        attributes for each metabolite. Must be an attribute of the
        metabolites in the cobra Model
    :type metabolite_data: list[str] | None
    :param reciprocal_weights: Whether to use the reciprocal of the weights, useful
        if higher flux should equate with lower weights in the final network
        (for use with graph algorithms)
    :type reciprocal_weights: bool
    :param threshold: Threshold, below which to consider a bound to be 0
    :type threshold: float
    :param loopless: Whether to use loopless flux variability analysis when determining
        minimum and maximum fluxes for weighting the network (ignored if
        `weighted = False`)
    :type loopless: bool
    :param fva_proportion: Proportion of optimal to use for the flux variability
        analysis when determining minimum and maximum fluxes for weighting the
        network (ignored if `weighted = False`). Must be between 0 and 1.
    :type fva_proportion: float
    :return: A network representing the metabolic network from the provided
        cobrapy model
    :rtype: nx.Graph | nx.DiGraph

    .. note:
       When creating a weighted network, the options are to weight the edges based on
       flux, or stoichiometry. If stoichiometry is chosen the edge weight will
       correspond to the stoichiometric coefficient of the metabolite, in a given
       reaction.

       For flux weighting, first flux variability analysis is performed. The edge
       weight is determined by the maximum flux through a reaction in a particular
       direction (forward if the metabolite is a product of the reaction,
       reverse if the metabolite is a substrate) multiplied by the metabolite
       stoichiometry. If the network is unweighted, the maximum of the forward
       and the reverse flux is used instead.
    """
    adjacency_frame, index, index_dict = create_adjacency_matrix(
        model=model,
        weighted=weighted,
        directed=directed,
        weight_by=weight_by,
        threshold=threshold,
        loopless=loopless,
        fva_proportion=fva_proportion,
        out_format="frame",
    )

    if reciprocal_weights:
        adjacency_frame.data = np.reciprocal(adjacency_frame.data)

    # Create the base network
    if directed:
        out_network = nx.from_pandas_adjacency(adjacency_frame, create_using=nx.DiGraph)
    else:
        out_network = nx.from_pandas_adjacency(adjacency_frame, create_using=nx.Graph)

    # Add node information if needed
    if reaction_data:
        # Create information dataframe for the Reactions
        node_info_rxn = pd.DataFrame(
            None,
            index=model.reactions.list_attr("id"),
            columns=["node_type"] + reaction_data,
            dtype="string",
        )
        for data_type in reaction_data:
            node_info_rxn[data_type] = model.reactions.list_attr(data_type)
        node_info_rxn["node_type"] = "reaction"

        out_network.add_nodes_from((n, dict(d)) for n, d in node_info_rxn.iterrows())

    if metabolite_data:
        # Create information dataframe for the Metabolites
        node_info_met = pd.DataFrame(
            None,
            index=model.metabolites.list_attr("id"),
            columns=["node_type"] + metabolite_data,
            dtype="string",
        )
        for data_type in metabolite_data:
            node_info_met[data_type] = model.metabolites.list_attr(data_type)
        node_info_met["node_type"] = "metabolite"
        out_network.add_nodes_from((n, dict(d)) for n, d in node_info_met.iterrows())
    # Remove any metabolites desired
    if nodes_to_remove:
        out_network.remove_nodes_from(nodes_to_remove)
    return out_network


def create_adjacency_matrix(
    model: cobra.Model,
    weighted: bool,
    directed: bool,
    weight_by: str = "stoichiometry",
    threshold: float = 1e-4,
    loopless: bool = False,
    fva_proportion: float = 1.0,
    out_format: str = "Frame",
) -> tuple[ArrayLike | sparray, list[str], dict[str, str]]:
    """
    Create an adjacency matrix representing the metabolic network of a provided
        cobra Model

    :param model: Cobra Model to create the network from
    :type model: cobra.Model
    :param weighted: Whether the network should be weighted
    :type weighted: bool
    :param directed: Whether the network should be directed
    :type directed: bool
    :param weight_by: String indicating if the network should be weighted by
        'stoichiometry', or 'flux' (see notes for more information). Ignored if
        `weighted = False`
    :type weight_by: str
    :param threshold: Threshold, below which to consider a bound to be 0
    :type threshold: float
    :param loopless: Whether to use loopless flux variability analysis when determining
        minimum and maximum fluxes for weighting the network (ignored if
        `weighted = False`)
    :type loopless: bool
    :param fva_proportion: Proportion of optimal to use for the flux variability
        analysis when determining minimum and maximum fluxes for weighting the
        network (ignored if `weighted = False`). Must be between 0 and 1.
    :type fva_proportion: float
    :param out_format: Format for the returned adjacency matrix
    :type out_format: str
    :return: Tuple of

        1. Adjacency matrix
        2.  Index of the matrix: a list of strings with the
            reaction or metabolite id for each node
        3. Index dictionary: a dictionary with keys 'reaction' and
           'metabolite', and values of lists of string ids corresponding to the
           reaction, and metabolite node respectively
    :rtype: tuple[pd.DataFrame | sparray, list[str], dict[str,str]]

    .. note:
       When creating a weighted network, the options are to weight the edges based on
       flux, or stoichiometry. If stoichiometry is chosen the edge weight will
       correspond to the stoichiometric coefficient of the metabolite, in a given
       reaction.

       For flux weighting, first flux variability analysis is performed. The edge
       weight is determined by the maximum flux through a reaction in a particular
       direction (forward if the metabolite is a product of the reaction,
       reverse if the metabolite is a substrate) multiplied by the metabolite
       stoichiometry. If the network is unweighted, the maximum of the forward
       and the reverse flux is used instead.
    """
    if not isinstance(model, cobra.Model):
        raise ValueError(
            f"Model must be a cobra.Model, received a " f"{type(model)} instead"
        )
    try:
        out_format = _parse_str_args_dict(
            out_format,
            {
                "frame": ["dataframe", "frame"],
                "dok": [
                    "dok",
                    "dictionary of keys",
                    "dictionary_of_keys",
                    "dictionary-of-keys",
                ],
                "lil": ["lil", "list of lists", "list-of-lists", "list_of_lists"],
                "csc": [
                    "csc",
                    "condensed sparse columns",
                    "condensed-sparse-columns",
                    "condensed_sparse_columns",
                ],
                "csr": [
                    "csr",
                    "condensed sparse rows",
                    "condensed-sparse-rows",
                    "condensed_sparse_rows",
                ],
            },
        )
    except ValueError as err:
        raise ValueError("Couldn't parse format") from err
    try:
        weight_by = _parse_str_args_dict(
            weight_by,
            {
                "flux": [
                    "flux",
                    "fva",
                    "flux-variability-analysis",
                    "flux variability analysis",
                    "flux_variability_analysis",
                ],
                "stoichiometry": ["stoichiometry"],
            },
        )
    except ValueError as err:
        raise ValueError("Couldn't parse weight_by") from err

    if weighted:
        if weight_by == "flux":
            fva_res = flux_variability_analysis(
                model, loopless=loopless, fraction_of_optimum=fva_proportion
            )
            fva_min = csc_array(fva_res["minimum"].values.reshape(-1, 1))
            fva_max = csc_array(fva_res["maximum"].values.reshape(-1, 1))
            fva_bounds = (fva_min, fva_max)
            if directed:
                adj_mat = _adj_mat_d_w_flux(
                    model=model, rxn_bounds=fva_bounds, threshold=threshold
                )
            else:
                adj_mat = _adj_mat_ud_w_flux(
                    model=model, rxn_bounds=fva_bounds, threshold=threshold
                )
        elif weight_by == "stoichiometry":
            if directed:
                adj_mat = _adj_mat_d_w_stoichiometry(model=model, threshold=threshold)
            else:
                adj_mat = _adj_mat_ud_w_stoichiometry(model=model, threshold=threshold)
        else:
            raise ValueError("Invalid weight_by")
    else:
        if directed:
            adj_mat = _adj_mat_d_uw(model=model, threshold=threshold)
        else:
            adj_mat = _adj_mat_ud_uw(model=model, threshold=threshold)
    index = model.metabolites.list_attr("id") + model.reactions.list_attr("id")
    index_dict = {
        "reactions": model.reactions.list_attr("id"),
        "metabolites": model.metabolites.list_attr("id"),
    }
    if out_format == "frame":
        adj_frame = pd.DataFrame.sparse.from_spmatrix(
            data=adj_mat, index=index, columns=index
        )
        return adj_frame, index, index_dict
    return adj_mat.asformat(out_format), index, index_dict


# endregion Main Function

# region Undirected Unweighted


def _adj_mat_ud_uw(model: cobra.Model, threshold: float = 1e-4) -> csr_array:
    """
    Create an unweighted undirected adjacency matrix from a given model

    :param model: Model to create the adjacency matrix from
    :type model: cobra.Model
    :param threshold: Threshold for a bound to be taken as a 0
    :type threshold: float
    :return: Adjacency Matrix
    :rtype: csr_array

    .. note:
       The index of the adjacency matrix is the metabolites followed by the reactions
       for both the rows and columns.
    """
    const_mat, for_prod, for_sub, rev_prod, rev_sub = _split_model_arrays(model)

    # Get the bounds, and split them

    bounds = const_mat.variable_bounds.tocsr()[:, 1]

    bounds.data[bounds.data <= threshold] = 0.0
    bounds.eliminate_zeros()

    for_bound, rev_bound = _split_arr_row(bounds, into=2)

    adj_block = _sparse_max(
        _broadcast_mult_arr_vec(for_sub.tocsr(), for_bound),
        _broadcast_mult_arr_vec(for_prod.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_sub.tocsr(), rev_bound),
        _broadcast_mult_arr_vec(rev_prod.tocsr(), rev_bound),
    )

    adj_block.data.fill(1)

    nmet, nrxn = adj_block.shape

    zero_block_rxn = csr_array((nrxn, nrxn))
    zero_block_met = csr_array((nmet, nmet))

    adjacency_matrix = sparse.hstack(
        [
            sparse.vstack([zero_block_met, adj_block.T]),
            sparse.vstack([adj_block, zero_block_rxn]),
        ]
    ).tocsr()

    return adjacency_matrix


# endregion Undirected Unweighted

# region Directed Unweighted


def _adj_mat_d_uw(model: cobra.Model, threshold: float = 1e-4) -> csr_array:
    """
    Create an unweighted directed adjacency matrix from a given model

    :param model: Model to create the adjacency matrix from
    :type model: cobra.Model
    :param threshold: Threshold for a bound to be taken as a 0
    :type threshold: float
    :return: Adjacency Matrix
    :rtype: csr_array

    .. note:
       The index of the adjacency matrix is the metabolites followed by the reactions
       for both the rows and columns.
    """
    const_mat, for_prod, for_sub, rev_prod, rev_sub = _split_model_arrays(model)

    # Get the bounds, and split them
    bounds = const_mat.variable_bounds.tocsc()[:, 1]

    bounds.data[bounds.data <= threshold] = 0.0
    bounds.eliminate_zeros()

    for_bound, rev_bound = _split_arr_row(bounds, into=2)

    consume_mat = _sparse_max(
        _broadcast_mult_arr_vec(for_sub.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_sub.tocsr(), rev_bound),
    )
    consume_mat.data.fill(1)

    generate_mat = _sparse_max(
        _broadcast_mult_arr_vec(for_prod.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_prod.tocsr(), rev_bound),
    )
    generate_mat.data.fill(1)

    nmet = len(model.metabolites)
    nrxn = len(model.reactions)

    zero_block_met = csr_array((nmet, nmet))
    zero_block_rxn = csr_array((nrxn, nrxn))

    adj_matrix = sparse.hstack(
        [
            sparse.vstack([zero_block_met, generate_mat.transpose()]),
            sparse.vstack([consume_mat, zero_block_rxn]),
        ]
    ).tocsr()

    return adj_matrix


# endregion Directed Unweighted

# region Undirected Weighted by flux


def _adj_mat_ud_w_flux(
    model: cobra.Model, rxn_bounds: tuple[csc_array, csc_array], threshold: float = 1e-4
) -> csr_array:
    """
    Create a weighted directed adjacency matrix from a given model

    :param model: Model to create the adjacency matrix from
    :type model: cobra.Model
    :param rxn_bounds: Bounds for the reactions, used to determine weights. Should
        be tuple with first element being the minimum, and the second element
        being the maximum.
    :type rxn_bounds: tuple[csr_array, csr_array]
    :param threshold: Threshold for a bound to be taken as a 0
    :type threshold: float
    :return: Adjacency Matrix, weighted using the bounds (higher bound translates
        to higher weight)
    :rtype: csr_array

    .. note:
       The index of the adjacency matrix is the metabolites followed by the reactions
       for both the rows and columns.

       The reaction bounds must have the same order as the reactions in the cobra
       model.
    """
    const_mat, for_prod, for_sub, rev_prod, rev_sub = _split_model_arrays(model)

    # Get the bounds, and split them
    rxn_min, rxn_max = rxn_bounds

    # Convert reaction bounds into forward and reverse bounds
    for_bound, _ = _split_arr_sign(rxn_max)
    _, rev_bound = _split_arr_sign(rxn_min)
    rev_bound *= -1

    # Eliminate any values below threshold
    for_bound.data[for_bound.data <= threshold] = 0.0
    for_bound.eliminate_zeros()

    rev_bound.data[rev_bound.data <= threshold] = 0.0
    rev_bound.eliminate_zeros()

    adj_block = _sparse_max(
        _broadcast_mult_arr_vec(for_sub.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_sub.tocsr(), rev_bound),
        _broadcast_mult_arr_vec(for_prod.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_prod.tocsr(), rev_bound),
    )

    nmet = len(model.metabolites)
    nrxn = len(model.reactions)

    zero_block_met = csr_array((nmet, nmet))
    zero_block_rxn = csr_array((nrxn, nrxn))

    adj_matrix = sparse.hstack(
        [
            sparse.vstack([zero_block_met, adj_block.transpose()]),
            sparse.vstack([adj_block, zero_block_rxn]),
        ]
    ).tocsr()

    return adj_matrix


# endregion Undirected Weighted by flux

# region Undirected Weighted by stoichiometry


def _adj_mat_ud_w_stoichiometry(
    model: cobra.Model, threshold: float = 1e-4
) -> csr_array:
    """
    Create an undirected adjacency matrix from a given model, with edge weights
    corresponding to stoichiometry

    :param model: Model to create the adjacency matrix from
    :type model: cobra.Model
    :param threshold: Threshold for a bound to be taken as a 0
    :type threshold: float
    :return: Adjacency Matrix
    :rtype: csr_array

    .. note:
       The index of the adjacency matrix is the metabolites followed by the reactions
       for both the rows and columns.
    """
    const_mat, for_prod, for_sub, rev_prod, rev_sub = _split_model_arrays(model)

    # Get the bounds, and split them

    bounds = const_mat.variable_bounds.tocsr()[:, 1]

    bounds.data[bounds.data <= threshold] = 0.0
    bounds.eliminate_zeros()

    # Change all the non-zero bounds to 1.
    bounds.data.fill(1)

    for_bound, rev_bound = _split_arr_row(bounds, into=2)

    adj_block = _sparse_max(
        _broadcast_mult_arr_vec(for_sub.tocsr(), for_bound),
        _broadcast_mult_arr_vec(for_prod.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_sub.tocsr(), rev_bound),
        _broadcast_mult_arr_vec(rev_prod.tocsr(), rev_bound),
    )

    nmet, nrxn = adj_block.shape

    zero_block_rxn = csr_array((nrxn, nrxn))
    zero_block_met = csr_array((nmet, nmet))

    adjacency_matrix = sparse.hstack(
        [
            sparse.vstack([zero_block_met, adj_block.T]),
            sparse.vstack([adj_block, zero_block_rxn]),
        ]
    ).tocsr()

    return adjacency_matrix


# endregion Undirected Weighted by stoichiometry

# region Directed Weighted by flux


def _adj_mat_d_w_flux(
    model: cobra.Model, rxn_bounds: tuple[csc_array, csc_array], threshold: float = 1e-4
) -> csr_array:
    """
    Create a weighted directed adjacency matrix from a given model

    :param model: Model to create the adjacency matrix from
    :type model: cobra.Model
    :param rxn_bounds: Bounds for the reactions, used to determine weights. Should
        be tuple with first element being the minimum, and the second element
        being the maximum.
    :type rxn_bounds: tuple[csr_array, csr_array]
    :param threshold: Threshold for a bound to be taken as a 0
    :type threshold: float
    :return: Adjacency Matrix, weighted using the bounds (higher bound translates
        to higher weight)
    :rtype: csr_array

    .. note:
       The index of the adjacency matrix is the metabolites followed by the reactions
       for both the rows and columns.

       The reaction bounds must have the same order as the reactions in the cobra
       model.
    """
    const_mat, for_prod, for_sub, rev_prod, rev_sub = _split_model_arrays(model)

    # Get the bounds, and split them
    rxn_min, rxn_max = rxn_bounds

    # Convert reaction bounds into forward and reverse bounds
    for_bound, _ = _split_arr_sign(rxn_max)
    _, rev_bound = _split_arr_sign(rxn_min)
    rev_bound *= -1

    # Eliminate any values below threshold
    for_bound.data[for_bound.data <= threshold] = 0.0
    for_bound.eliminate_zeros()

    rev_bound.data[rev_bound.data <= threshold] = 0.0
    rev_bound.eliminate_zeros()

    consume_mat = _sparse_max(
        _broadcast_mult_arr_vec(for_sub.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_sub.tocsr(), rev_bound),
    )

    generate_mat = _sparse_max(
        _broadcast_mult_arr_vec(for_prod.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_prod.tocsr(), rev_bound),
    )

    nmet = len(model.metabolites)
    nrxn = len(model.reactions)

    zero_block_met = csr_array((nmet, nmet))
    zero_block_rxn = csr_array((nrxn, nrxn))

    adj_matrix = sparse.hstack(
        [
            sparse.vstack([zero_block_met, generate_mat.transpose()]),
            sparse.vstack([consume_mat, zero_block_rxn]),
        ]
    ).tocsr()

    return adj_matrix


# endregion Directed Weighted by flux

# region Directed Weighted by stoichiometry


def _adj_mat_d_w_stoichiometry(
    model: cobra.Model, threshold: float = 1e-4
) -> csr_array:
    """
    Create a directed adjacency matrix from a given model, with edge weights
    corresponding to stoichiometry

    :param model: Model to create the adjacency matrix from
    :type model: cobra.Model
    :param threshold: Threshold for a bound to be taken as a 0
    :type threshold: float
    :return: Adjacency Matrix
    :rtype: csr_array

    .. note:
       The index of the adjacency matrix is the metabolites followed by the reactions
       for both the rows and columns.
    """
    const_mat, for_prod, for_sub, rev_prod, rev_sub = _split_model_arrays(model)

    # Get the bounds, and split them
    bounds = const_mat.variable_bounds.tocsc()[:, 1]

    bounds.data[bounds.data <= threshold] = 0.0
    bounds.eliminate_zeros()

    # Change all the non-zero bounds to 1
    bounds.data.fill(1.0)

    for_bound, rev_bound = _split_arr_row(bounds, into=2)

    consume_mat = _sparse_max(
        _broadcast_mult_arr_vec(for_sub.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_sub.tocsr(), rev_bound),
    )

    generate_mat = _sparse_max(
        _broadcast_mult_arr_vec(for_prod.tocsr(), for_bound),
        _broadcast_mult_arr_vec(rev_prod.tocsr(), rev_bound),
    )

    nmet = len(model.metabolites)
    nrxn = len(model.reactions)

    zero_block_met = csr_array((nmet, nmet))
    zero_block_rxn = csr_array((nrxn, nrxn))

    adj_matrix = sparse.hstack(
        [
            sparse.vstack([zero_block_met, generate_mat.transpose()]),
            sparse.vstack([consume_mat, zero_block_rxn]),
        ]
    ).tocsr()

    return adj_matrix


# endregion Directed Weighted by stoichiometry

# region Helper Functions


def _split_model_arrays(
    model: cobra.Model,
) -> tuple[NamedTuple, csc_array, csc_array, csc_array, csc_array]:
    const_mat = cobra.util.array.constraint_matrices(
        model,
        array_type="lil",
    )
    # Get the stoichiometric matrix
    equalities = const_mat.equalities.tocsc()

    # Split the stoichiometric matrix into forward and reverse variables
    for_arr, rev_arr = _split_arr_col(equalities, into=2)

    # Split the array into the products and the substrates, reversing substrate sign
    for_prod, for_sub = _split_arr_sign(for_arr)
    for_sub *= -1

    # Split the array into the products and the substrates, reversing substrate sign
    rev_prod, rev_sub = _split_arr_sign(rev_arr)
    rev_sub *= -1

    return const_mat, for_prod, for_sub, rev_prod, rev_sub


# endregion Helper Functions

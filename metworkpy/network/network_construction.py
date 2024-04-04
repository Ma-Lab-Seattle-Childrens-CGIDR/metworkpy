# Imports
# Standard Library Imports
from __future__ import annotations

# External Imports
import cobra
import networkx as nx
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse import sparray, csr_array

# Local Imports
from metworkpy.network._array_utils import (_split_arr_col, _split_arr_sign,
                                          _split_arr_row, _sparse_max,
                                          _broadcast_mult_arr_vec)
from metworkpy.utils._arguments import _parse_str_args_dict


# region Main Function
def create_graph(model: cobra,
                 weighted: bool,
                 directed: bool) -> nx.Graph | nx.DiGraph:
    pass


def create_adjacency_matrix(model: cobra,
                            weighted: bool,
                            directed: bool,
                            out_format: str = "Frame") -> ArrayLike | sparray:
    try:
        out_format = _parse_str_args_dict(out_format, {
            "frame":["dataframe", "frame"],
            "dok": ["dok", "dictionary of keys", "dictionary_of_keys",
                    "dictionary-of-keys"],
            "lil": ["lil", "list of lists", "list-of-lists", "list_of_lists"],
            "csc": ["csc", "condensed sparse columns", "condensed-sparse-columns",
                    "condensed_sparse_columns"],
            "csr": ["csr", "condensed sparse rows", "condensed-sparse-rows",
                    "condensed_sparse_rows"]
        })
    except ValueError as err:
        raise ValueError("Couldn't parse format") from err
    pass


# endregion Main Function

# region Undirected Unweighted

def _adj_mat_ud_uw(model: cobra.Model) -> csr_array:
    """
    Create an unweighted undirected adjacency matrix from a given model

    :param model: Model to create the adjacency matrix from
    :type model: cobra.Model
    :return: Adjacency Matrix
    :rtype: csr_array

    .. note:
       The index of the adjacency matrix is the metabolites, then the reactions
       for both the rows and columns.
    """
    const_mat = cobra.util.array.constraint_matrices(model,
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

    # Get the bounds, and split them
    bounds = const_mat.variable_bounds.tocsc()[:, 1]
    for_bound, rev_bound = _split_arr_row(bounds, into=2)

    adj_block = _sparse_max(_broadcast_mult_arr_vec(for_sub.tocsr(), for_bound),
                            _broadcast_mult_arr_vec(for_prod.tocsr(), for_bound),
                            _broadcast_mult_arr_vec(rev_sub.tocsr(), rev_bound),
                            _broadcast_mult_arr_vec(rev_prod.tocsr(), rev_bound))

    adj_block[adj_block.nonzero()] = 1

    nmet, nrxn = adj_block.shape

    zero_block_rxn = csr_array((nrxn, nrxn))
    zero_block_met = csr_array((nmet, nmet))

    adjacency_matrix = sparse.hstack([sparse.vstack([zero_block_met, adj_block.T]),
                                      sparse.vstack([adj_block, zero_block_rxn])])

    return adjacency_matrix

# endregion Undirected Unweighted

# region Directed Unweighted

# endregion Directed Unweighted

# region Undirected Weighted

# endregion Undirected Weighted

# region Directed Weighted

# endregion Directed Weighted

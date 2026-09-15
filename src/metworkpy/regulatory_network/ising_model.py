"""
Methods for investigating gene regulatory networks
using the Ising formalism
"""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from metworkpy.utils._scipy_compat import _check_scipy_version_greater

DEFAULT_ISING_STATES = np.array([0, 1], dtype=np.int16)


class IsingGRN:
    """
    Represents a Gene Regulatory Network with edges that can be either -1 or 1

    Parameters
    ----------
    network : pd.DataFrame or nx.DiGraph or scipy Sparse Array
        The gene regulatory network to represent, can be a

        * DataFrame: A dataframe with 3 columns, 'source', 'target', and 'weight'.
          Each row represents a regulatory relationship from 'source', to 'target'.
          The 'weight' column should contain -1, 0, and 1. A value of -1 represents
          repression, a value of 0 represents no interaction (all relationships not
          explicity provided will be given this value), a value of 1 represents
          activation. Default column names can be overridden with the `source`, `target`,
          and `weight` parameters.
        * DiGraph: A directed graph with the nodes representing the genes in the
          network. Each node represents a gene, and edges represent a regulatory
          relationship. The weight of the edge indicates the type of the
          relationship (1 for activating, -1 for repressing, 0 for no relationship).
          The edge weight by default is taken to be the 'weight' edge attribute,
          but this can be overridden with the `weight` parameter.
        * NDArray: An array representing the regulatory relationships,
          should be a square array with entry i,j representing a regulatory relationship
          from j to i. The values must be only -1, 0, or 1 with -1 representing
          repression, 1 representing activation, and 0 representing no relationship.
        * Sparse Array: An array representing the regulatory relationships,
          should be a square array with entry i,j representing a regulatory relationship
          from j to i. The values must be only -1, 0, or 1 with -1 representing
          repression, 1 representing activation, and 0 representing no relationship.

    source : str,optional
       Optional string to specify the column of the network DataFrame to find the source
       genes for the regulatory relationships
    target : str,optional
       Optional string to specify the column of the network DataFrame to find the target
       genes for the regulatory relationships
    weight : str,optional
        Optional string to specify the column of the network DataFrame or the edge
        attribute of the network DiGraph to find the weight (or type) of the regulatory
        relationship.
    """

    def __init__(
        self,
        network: pd.DataFrame | nx.DiGraph | sparse.sparray | np.ndarray,
        source: str | None = None,
        target: str | None = None,
        weight: str | None = None,
    ):
        # Create the _array and _index representing the network
        match network:
            case pd.DataFrame():
                self._index, array = self._df_init(
                    network, source, target, weight
                )
            case nx.DiGraph():
                self._index, array = self._graph_init(network)  # ty: ignore[invalid-argument-type]
            case sparse.sparray():
                self._index, array = self._sp_array_init(network)
            case np.ndarray():
                self._index, array = self._np_array_init(network)
            case t:
                raise TypeError(
                    f"Expected a DataFrame, DiGraph, or sparse array, received {t}"
                )
        self._array = array.tocsr()
        self._array.eliminate_zeros()
        # Check that the weights are all -1, 0, or 1
        vals = np.unique(self._array.data)
        if (
            (len(vals) > 2)
            or (len(vals) == 1 and (vals[0] != -1 and vals[0] != 1))
            or (len(vals) == 2 and (vals[0] != -1 or vals[1] != 1))
        ):
            raise ValueError(
                f"Weights should only be -1, 0, or 1 but weights includes incorrect values: {vals}"
            )

    def _df_init(
        self,
        network: pd.DataFrame,
        source: str | None,
        target: str | None,
        weight: str | None,
    ) -> tuple[pd.Index, sparse.dok_array]:
        source = source if source is not None else "source"
        target = target if target is not None else "target"
        weight = weight if weight is not None else "weight"
        # Get the genes in the regulatory network
        idx = pd.Index(
            set(network[source].unique()) | set(network[target].unique())
        )
        array = sparse.dok_array((len(idx), len(idx)), dtype=np.int16)
        for _, (s, t, w) in network[[source, target, weight]].iterrows():
            array[idx.get_loc(t), idx.get_loc(s)] = np.int16(w)
        return idx, array

    def _graph_init(
        self, network: nx.DiGraph, weight: str | None = None
    ) -> tuple[pd.Index, sparse.dok_array]:
        idx = pd.Index(network.nodes)
        array = sparse.dok_array((len(idx), len(idx)), dtype=np.int16)
        weight = weight if weight is not None else "weight"
        for u, v, d in network.edges(data=True):
            array[idx.get_loc(v), idx.get_loc(u)] = np.int16(d[weight])
        return idx, array

    def _sp_array_init(
        self, network: sparse.sparray
    ) -> tuple[pd.Index, sparse.dok_array]:
        # NOTE: The type ignores are due to how scipy creates its sparse arrays,
        # the shape and todok is available for all the implementations,
        # just not on the base class directly
        if network.shape[0] != network.shape[1]:  # ty: ignore[unresolved-attribute]
            raise ValueError("Network must be a square matrix")
        idx = pd.RangeIndex(network.shape[0])  # ty: ignore[unresolved-attribute]
        array = network.todok()  # ty: ignore[unresolved-attribute]
        return idx, array

    def _np_array_init(
        self, network: np.ndarray
    ) -> tuple[pd.Index, sparse.dok_array]:
        if network.shape[0] != network.shape[1]:
            raise ValueError("Network must be a square matrix")
        idx = pd.RangeIndex(network.shape[0])
        array = sparse.dok_array(network, dtype=np.int16)
        return idx, array


def _find_ising_steady_states(
    regulatory_matrix: sparse.csr_array,
    initial_states: sparse.csr_array,
    max_steps: int = 1000,
    states: np.ndarray = DEFAULT_ISING_STATES,
):
    """
    Find the steady states for the gene-regulatory network specified in
    the regulatory matrix for all the initial_states

    Parameters
    ----------
    regulatory_matrix : csc_array
        Sparse array describing the regulatory relationships in the
        gene regulatory network. Each (i,j) entry represents the
        regulatory relationship from gene j to i, with 1 indicating
        that gene j activates gene i, -1 indicating gene j represses gene i,
        and 0 representing no regulatory relationship. This matrix must
        be square.
    initial_states : csr_array
        Sparse array describing the initial states of the genes,
        entries can be 0/1 or -1/1 depending on the value of
        `states`. The columns are the genes, and each row is a
        different initial state.
    max_steps : int
        The maximum number of steps to use to try and find steady state,
        if the iteration fails to converge, the last state found will be
        returned and a warning will be issued
    states : np.ndarray, default=[0,1]
        The alternative states to use, the first value indicates
        inactive, the second indicates active. Default is 0 for
        inactive, 1 for active.

    Returns
    -------
    dok_array
        The steady states, with the rows corresponding those in the initial
        states matrix, and the columns representing genes in the regulatory
        network
    """
    steady_states = sparse.dok_array(initial_states.shape, dtype=np.int16)
    for idx in range(initial_states.shape[0]):
        steady_states[idx] = _find_ising_steady_state(
            regulatory_matrix,
            initial_state=initial_states[idx],
            max_steps=max_steps,
            states=states,
        )
    return steady_states


def _find_ising_steady_state(
    regulatory_matrix: sparse.csr_array,
    initial_state: sparse.coo_array,
    max_steps: int = 1000,
    states: np.ndarray = DEFAULT_ISING_STATES,
) -> sparse.coo_array:
    """
    Run an Ising model to equillibrium from an initial state,
    based on a regulatory matrix

    Parameters
    ----------
    regulatory_matrix : csc_array
        Sparse array describing the regulatory relationships in the
        gene regulatory network. Each (i,j) entry represents the
        regulatory relationship from gene j to i, with 1 indicating
        that gene j activates gene i, -1 indicating gene j represses gene i,
        and 0 representing no regulatory relationship. This matrix must
        be square.
    initial_state : coo_array
        Sparse array describing the initial state of the genes,
        entries can be 0/1 or -1/1 depending on the value of
        `states`.
    max_steps : int
        The maximum number of steps to use to try and find steady state,
        if the iteration fails to converge, the last state found will be
        returned and a warning will be issued
    states : np.ndarray, default=[0,1]
        The alternative states to use, the first value indicates
        inactive, the second indicates active. Default is 0 for
        inactive, 1 for active.

    Returns
    -------
    sparse.coo_array
        The steady state, or the last state found if
        convergence fails
    """
    if not _check_scipy_version_greater(1, 17, 0):
        return _ising_iteration_compat(
            regulatory_matrix=regulatory_matrix,
            initial_state=initial_state,
            max_steps=max_steps,
            states=states,
        )
    state_vec = initial_state.reshape(-1, 1).copy()
    prev_state_vec = None

    # Iterate until equillibrium reached
    for _ in range(max_steps):
        update_vec = regulatory_matrix @ state_vec
        state_vec[update_vec < 0] = states[0]
        state_vec[update_vec > 0] = states[1]
        if (
            prev_state_vec is not None
            and (state_vec != prev_state_vec).max() > 0
        ):
            break
        prev_state_vec = state_vec.copy()
    else:
        warnings.warn(f"Failed to converge in max_steps ({max_steps} steps)")
    return state_vec


def _ising_iteration_compat(
    regulatory_matrix: sparse.csr_array,
    initial_state: sparse.coo_array,
    max_steps: int = 1000,
    states: np.ndarray = DEFAULT_ISING_STATES,
):
    state_vec = initial_state.copy().reshape((-1, 1)).todok()
    prev_state_vec = None

    for _ in range(max_steps):
        update_vec = regulatory_matrix @ state_vec.tocsc()
        state_vec[update_vec < 0] = states[0]
        state_vec[update_vec > 0] = states[1]
        if (
            prev_state_vec is not None
            and (state_vec != prev_state_vec).max() > 0
        ):
            return state_vec
        prev_state_vec = state_vec.copy()
    return state_vec


def _generate_random_initial_states(
    num_genes: int,
    num_states: int,
    seed: np.random.Generator | int | None,
    states: np.ndarray = DEFAULT_ISING_STATES,
) -> sparse.csr_array:
    if isinstance(seed, (None, int)):
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    assert isinstance(rng, np.random.Generator), (
        f"Failed to convert seed into RNG, seed: {seed}"
    )
    initial_states = sparse.dok_array((num_states, num_genes), dtype=np.int16)
    idx_options = np.arange(initial_states.shape[1])
    for idx, num_selected in enumerate(
        rng.integers(
            1, initial_states.shape[1] + 1, size=initial_states.shape[0]
        )
    ):
        selected_on = rng.choice(idx_options, num_selected, replace=False)
        initial_states[idx, selected_on] = states[1]
        if states[0] != 0:
            selected_off = np.ones((initial_states.shape[1]), dtype=bool)
            selected_off[selected_on] = True
            initial_states[idx, selected_off] = states[0]
    return initial_states.tocsr()

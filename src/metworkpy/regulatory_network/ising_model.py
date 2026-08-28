"""
Methods for investigating gene regulatory networks
using the Ising formalism
"""

import warnings

import numpy as np
from scipy import sparse

from metworkpy.utils._scipy_compat import _check_scipy_version_greater

DEFAULT_ISING_STATES = np.array([0, 1], dtype=np.int16)


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

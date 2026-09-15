"""
Methods for investigating gene regulatory networks
using the Ising formalism
"""

import warnings
from collections.abc import Iterable
from typing import Literal

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
        states: np.ndarray[
            tuple[int], np.dtype[np.int16]
        ] = DEFAULT_ISING_STATES,
        *,
        source: str | None = None,
        target: str | None = None,
        weight: str | None = None,
    ):
        self.source = source
        self.target = target
        self.weight = weight
        # Create the _array and _index representing the network
        match network:
            case pd.DataFrame():
                self._index, array = self._df_init(network)
            case nx.DiGraph():
                self._index, array = self._graph_init(network)  # ty: ignore[invalid-argument-type]
            case sparse.sparray():
                self._index, array = self._sparray_init(network)
            case np.ndarray():
                self._index, array = self._nparray_init(network)
            case t:
                raise TypeError(
                    f"Expected a DataFrame, DiGraph, or sparse array, received {type(t)}"
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
        # Save the states
        self._states = states
        # Create caches for various return types
        self._digraph: nx.DiGraph | None = None
        self._dataframe: pd.DataFrame | None = None
        self._nparray: (
            np.ndarray[tuple[int, int], np.dtype[np.int16]] | None
        ) = None

    def _reset_cache(self):
        self._digraph = None
        self._dataframe = None
        self._sparray = None
        self._nparray = None

    def _df_init(
        self,
        network: pd.DataFrame,
    ) -> tuple[pd.Index, sparse.dok_array]:
        # Get the genes in the regulatory network
        idx = pd.Index(
            set(network[self._source].unique())
            | set(network[self._target].unique())
        )
        array = sparse.dok_array((len(idx), len(idx)), dtype=np.int16)
        for _, (s, t, w) in network[
            [self.source, self.target, self.weight]
        ].iterrows():
            array[idx.get_loc(t), idx.get_loc(s)] = np.int16(w)
        return idx, array

    def _graph_init(
        self, network: nx.DiGraph
    ) -> tuple[pd.Index, sparse.dok_array]:
        idx = pd.Index(network.nodes)
        array = sparse.dok_array((len(idx), len(idx)), dtype=np.int16)
        for u, v, d in network.edges(data=True):
            array[idx.get_loc(v), idx.get_loc(u)] = np.int16(d[self.weight])
        return idx, array

    def _sparray_init(
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

    def _nparray_init(
        self, network: np.ndarray
    ) -> tuple[pd.Index, sparse.dok_array]:
        if network.shape[0] != network.shape[1]:
            raise ValueError("Network must be a square matrix")
        idx = pd.RangeIndex(network.shape[0])
        array = sparse.dok_array(network, dtype=np.int16)
        return idx, array

    @property
    def states(self):
        """
        The alternate states genes can be in. Default is 0 for inactive, 1 for active.
        """
        return self._states

    @states.setter
    def states(self, states=np.ndarray[tuple[int], np.dtype[np.int16]]):
        self._states = states

    @property
    def source(self):
        """
        The column of the long-form DataFrame used to specify
        the regulator in regulatory relationships (i.e. which
        gene is activating/repressing the gene in the 'target'
        column).
        """
        return self._source

    @source.setter
    def source(self, source: str | None):
        self._source = source if source is not None else "source"

    @property
    def target(self):
        """
        The column of the long-form DataFrame used to specify
        the regulatory target in regulatory relationships (i.e. which
        gene which is activated/repressed by the gene in the 'source'
        column).
        """
        return self._target

    @target.setter
    def target(self, target: str | None):
        self._target = target if target is not None else "target"

    @property
    def weight(self):
        """
        The column in the long-form DataFrame or edge attribute in the DiGraph
        which is used to specify the regulatory relationship direction.
        The column/attribute should only include -1, 0, and 1, with
        -1 indicating repression, 1 indicating activation, and 0
        indicating no regulation. This property is also used as the
        edge attribute for weight when returning a DiGraph from the
        `graph` property.
        """
        return self._weight

    @weight.setter
    def weight(self, weight: str | None):
        self._weight = weight if weight is not None else "weight"

    @property
    def graph(self):
        """
        The gene regulatory network in the form of a DiGraph. Nodes represent
        genes in the network, and edges represent regulatory relationships.
        Each edge has an attribute (name specified by the `weight` property)
        specifying direction of regulation, -1 for repression, 1 for activation.
        """
        if self._digraph is not None:
            return self._digraph
        self._digraph = self._create_graph()
        return self._digraph

    @graph.setter
    def graph(self, network: nx.DiGraph, weight: str | None = None):
        self.weight = weight
        self._reset_cache()
        self._idx, array = self._graph_init(network=network)
        self._array: sparse.csr_array = array.tocsr()

    def _create_graph(self):
        assert isinstance(self._array, sparse.csr_array)
        coo = self._array.tocoo()
        row_idx, col_idx = coo.coords
        vals = coo.data
        idx = self.index
        g = nx.DiGraph()
        g.add_edges_from(
            (idx[j], idx[i], {"weight": v})
            for i, j, v in zip(row_idx, col_idx, vals)
        )
        return g

    @property
    def sparray(self):
        """
        The gene regulatory network in the form of a sparse array.
        Each i,j entry represents a regulatory relationship, with
        gene j regulating gene i. An entry of -1 represents
        gene j repressing gene i, 1 represents gene j activating
        gene i, and 0 indicates no regulatory relationship
        from gene j to gene i.
        """
        return self._array

    @sparray.setter
    def sparray(self, network: sparse.sparray):
        self._reset_cache()
        self._idx, array = self._sparray_init(network=network)
        self._array: sparse.csr_array = array.tocsr()

    @property
    def array(self):
        """
        The gene regulatory network in the form of a numpy array.
        Each i,j entry represents a regulatory relationship, with
        gene j regulating gene i. An entry of -1 represents
        gene j repressing gene i, 1 represents gene j activating
        gene i, and 0 indicates no regulatory relationship
        from gene j to gene i.
        """
        if self._nparray is not None:
            return self._nparray
        self._nparray = self._array.todense()
        return self._nparray

    @array.setter
    def array(self, network: np.ndarray[tuple[int, int], np.dtype[np.int16]]):
        self._reset_cache()
        self._idx, array = self._nparray_init(network=network)
        self._array: sparse.csr_array = array.tocsr()

    @property
    def df(self):
        """
        The gene regulatory network in the form of a pandas DataFrame.
        Row and column indexes are the gene labels.
        Each i,j entry represents a regulatory relationship, with
        gene j regulating gene i. An entry of -1 represents
        gene j repressing gene i, 1 represents gene j activating
        gene i, and 0 indicates no regulatory relationship
        from gene j to gene i.
        """
        if self._dataframe is not None:
            return self._dataframe
        self._dataframe = pd.DataFrame(self._array.todense(), index=self.index)
        return self._dataframe

    @df.setter
    def df(
        self,
        network: pd.DataFrame,
        source: str | None = None,
        target: str | None = None,
        weight: str | None = None,
    ):
        self._reset_cache()
        self.source = source
        self.target = target
        self.weight = weight
        self._idx, array = self._df_init(network=network)
        self._array: sparse.csr_array = array.tocsr()

    @property
    def index(self):
        """
        The labels for the genes in the regulatory network, in the
        order they appear in the underlying sparse array representation
        of the network (and thus also the arrays from the `sparray`
        and `array` properties). Used for naming nodes when for the
        `graph` property, and as the index and columns for the
        `df` property.
        """
        return self._index

    @index.setter
    def index(self, labels: Iterable[str]):
        self._reset_cache()
        idx = pd.Index(labels)
        if len(idx) != self._array.shape[0]:
            raise ValueError(
                f"Index must be the same length as the number of genes in the"
                f" regulatory network, but network has {self._array.shape[0]}"
                f" genes and there are {len(idx)} labels in the provided index"
            )
        self._index = idx

    def find_steady_state(
        self,
        initial_state: dict[str, int]
        | pd.Series
        | sparse.sparray
        | np.ndarray,
        max_steps: int = 1000,
    ) -> dict[str, int] | pd.Series | sparse.sparray | np.ndarray:
        """
        Find the steady state for the gene-regulatory network resulting
        from an `initial_state`

        Parameters
        ----------
        initial_state : dict of str to int or pd.Series or sparse array or NDArray
            The initial state to start the Ising iteration from
        max_steps : int,default=1000
            The maximum number of steps to perform, if the steady state
            is not reached in this number of steps a warning will be
            issued and the final state vector will be returned.

        Returns
        -------
        steady_state : dict of str to int or pandas Series or sparse array or NDArray
            The steady state the initial state converges to, the type
            will match the type of `initial_state`.

        Notes
        -----
        To find the steady state of the gene regulatory network,
        at each step each gene is evaluated for the state of its regulators.
        The state of each regulator, multiplied with the weight of the relationship
        between that regulator and the gene of interest is summed. If this
        sum is greater than 0 the gene is activated, if the sum
        is less than 0 the gene is disactivated, and if the sum is 0
        the gene retains its previous state. Once two sequential states
        are identical, the iteration ends since the steady state has been
        found.
        """
        match initial_state:
            case dict():
                istate = sparse.coo_array(
                    pd.Series(initial_state)[self.index]
                    .to_numpy()
                    .reshape((-1, 1))
                )
            case pd.Series():
                istate = sparse.coo_array(
                    initial_state[self.index].to_numpy().reshape((-1, 1))
                )
            case sparse.sparray():
                istate = initial_state.reshape((-1, 1)).tocoo()  # ty: ignore[unresolved-attribute]
            case np.ndarray():
                istate = sparse.coo_array(initial_state.reshape((-1, 1)))
            case t:
                raise TypeError(
                    f"Expected initial state to be Series, sparse array, or NDArray, but received {type(t)}"
                )
        steady_state = _find_ising_steady_state(
            regulatory_matrix=self._array,
            initial_state=istate,
            max_steps=max_steps,
            states=self.states,
        )
        match initial_state:
            case dict():
                return {
                    idx: val
                    for idx, val in zip(self.index, steady_state.todense())
                }
            case pd.Series():
                return pd.Series(steady_state.todense(), index=self.index)
            case sparse.sparray():
                return steady_state
            case np.ndarray():
                return steady_state.todense()
            case t:
                raise TypeError(
                    f"Expected initial state to be Series, sparse array, or NDArray, but received {type(t)}"
                )

    def find_steady_states(
        self,
        initial_states: pd.DataFrame
        | sparse.sparray
        | np.ndarray
        | None = None,
        n_initial_states: int = 1000,
        max_steps: int = 1000,
        return_type: Literal[
            "dense", "frame", "bsr", "coo", "csc", "csr", "dia", "dok", "lil"
        ]
        | None = None,
        seed: np.random.Generator | int | None = None,
    ) -> pd.DataFrame | sparse.sparray | np.ndarray:
        """
        Find the steady states for the gene-regulatory network resulting from
        the provided `initial_states`, or random initial states if none
        are provided.

        Parameters
        ----------
        initial_states : pd.DataFrame or sparse.sparray or np.ndarray or None,optional
            The initial states to iterate from to find the gene regulatory networks
            steady states. If provided (as a DataFrame, numpy NDArray, or SciPy sparse
            array) the columns should represent the genes in the regulatory network,
            and the rows should represent the different initial states. If not specified,
            `n_initial_states` will be randomly generated instead. For these random
            states, a random proportion in the range (0, 1) will first be selected,
            this will determe how many genes will be active in that initial state.
            Then, that number of genes will be randomly selected and set to be
            active, while all other genes will be set to be inactive. This will be
            repeated for each random initial state.
        n_initial_states : int,default=1000
             The number of random initial states to generate.
             Used if `initial_states` is None, ignored otherwise.
        max_steps : int,default=1000
            The maximum number of steps to perform for EACH initial state,
            if the steady state is not reached in this number of steps a
            warning will be issued and the final state vector will be returned.
        return_type : {"dense", "frame", "bsr", "coo", "csc", "csr", "dia", "dok", "lil"} or None
            The desired return type. If not specified, the type of the provided `initial_states`
            will be matched (or a DataFrame will be returned if no initial states were provided).
            Otherwise 'dense' will return a numpy NDArray, 'frame' will return a pandas DataFrame,
            and the other types will return the SciPy sparse array of that type
            (so 'coo' will return a `coo_array <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html#scipy.sparse.coo_array>`_).
        seed : int or numpy random Generator, optional
            Seed to use for generating the random initial states. Only used
            if `initial_states` is None, otherwise ignored.

        Returns
        -------
        steady_states : pandas DataFrame or sparse array or NDArray
            The steady states resulting from each of the initial states. Each
            column represents a gene in the regulatory network, and each row
            represents a different steady state. If the `initial_states` were
            provided, the rows of the returned steady states and `initial_states`
            will correspond. If `return_type` is specified, that is the type
            that will be returned. If `return_type` is None, then the return
            type will match the type of the provided `initial_states`, unless
            `initial_states` was None, and then the return will be a DataFrame.
        """
        match initial_states:
            case pd.DataFrame():
                istates = sparse.csr_array(
                    initial_states[self._index].to_numpy(), dtype=np.int16
                )
            case sparse.sparray():
                istates = initial_states.tocsr()  # ty: ignore[unresolved-attribute]
            case np.ndarray():
                istates = sparse.csr_array(initial_states)
            case None:
                istates = _generate_random_initial_states(
                    num_genes=len(self.index),
                    num_states=n_initial_states,
                    seed=seed,
                    states=self.states,
                )
            case t:
                raise TypeError(
                    f"Expected initial state to be Series, sparse array, or NDArray, but received {type(t)}"
                )
        steady_states = _find_ising_steady_states(
            self._array,
            initial_states=istates,
            max_steps=max_steps,
            states=self.states,
        )
        match return_type:
            case None:
                match initial_states:
                    case pd.DataFrame() | None:
                        return pd.DataFrame(
                            steady_states.todense(),
                            columns=self.index,
                            index=initial_states.index
                            if initial_states is not None
                            else None,
                        )
                    case sparse.sparray():
                        return steady_states
                    case np.ndarray():
                        return steady_states.todense()
            case "bsr":
                return steady_states.tobsr()
            case "coo":
                return steady_states.tocoo()
            case "csc":
                return steady_states.tocsc()
            case "csr":
                return steady_states.tocsr()
            case "dia":
                return steady_states.todia()
            case "dok":
                return steady_states.todok()
            case "lil":
                return steady_states.tolil()
            case "frame":
                return pd.DataFrame(
                    steady_states.todense(), columns=self.index
                )
            case "dense":
                return steady_states.todense()


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

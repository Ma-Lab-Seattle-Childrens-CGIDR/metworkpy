"""
Functions for analyzing the variable components of the
optimal growth solutions
"""

from typing import Hashable, Optional, Union

import cobra
import networkx as nx

from metworkpy.network.network_construction import create_metabolic_network


def find_variable_components(
    model: cobra.Model,
    network: Optional[Union[nx.Graph, nx.DiGraph]] = None,
    tolerance: float = 1e-7,
    directed: bool = False,
    strongly_connected: bool = False,
    **kwargs,
) -> list[set[Hashable]]:
    """
    Identify the variable components in the metabolic network,
    that is the components of the network which can vary under at the
    optimum solution

    Parameters
    ----------
    model : cobra.Model
        Model to find the variable components in
    network : nx.Graph or nx.DiGraph, optional
        A metabolic network graph constructed from `model`,
        used to find the connected components after removing reactions
        which can't vary under the optimal solution
    tolerance : float, default=1e-7
        The tolerance, reactions which have minimum and maximum fluxes less
        than this value will be considered constant
    directed : bool, default=False
        If network is not passed, this decides if the constructed network is
        directed or not
    strongly_connected : bool, default=False
        Whether to find the strongly connected components of the
        graph (only used if the provided network is directed)
    kwargs
        Keyword arguments are passed to `cobra.flux_analysis.flux_variability_analysis`

    Returns
    -------
    list of set of nodes
        List of sets of nodes in the metabolic network, each node
        represents a variable component of the model at optimum

    Notes
    -----
    Uses the cobra Model to find the reactions which are constant across
    optimal solutions, and then identifies the connected groups of variable
    reactions and associated metabolites
    """
    # If a network isn't passed, construct a simple one to use
    if network is None:
        network = create_metabolic_network(
            model=model, weighted=False, directed=False
        )
    # Perform FVA
    fva_solution = cobra.flux_analysis.flux_variability_analysis(
        model=model, **kwargs
    )
    variable_reactions = [
        r
        for r in fva_solution[
            (fva_solution["maximum"] - fva_solution["minimum"]).abs()
            >= tolerance
        ].index
        if r in network.nodes
    ]
    # Get the induced subgraph
    subgraph = network.subgraph(variable_reactions)
    if subgraph.is_directed():
        if strongly_connected:
            return list(nx.strongly_connected_components(subgraph))
        else:
            return list(nx.weakly_connected_components(subgraph))
    return nx.connected_components(subgraph)

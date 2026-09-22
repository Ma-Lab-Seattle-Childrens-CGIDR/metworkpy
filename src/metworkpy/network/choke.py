"""
Function for finding the choke points of a network
"""

from collections.abc import Iterable
from typing import cast

import cobra
import networkx as nx


def find_choke_points(
    metabolic_network: nx.DiGraph,
    model: cobra.Model | None = None,
    metabolite_nodes: Iterable[str] | None = None,
):
    """
    Find choke points in the metabolite network, that is reactions which
    uniquely produce or consume a metabolite

    Parameters
    ----------
    metabolic_network : nx.DiGraph
        The metabolic network as a directed bipartite graph, with nodes
        representing reactions and metabolites. The network is not checked
        for being bipartite.
    model : cobra.Model, optional
        Cobra Model, used to identify which nodes represent metabolites. Must provide
        either this model or `metabolite_nodes`. If both are provided, `metabolite_nodes`
        takes precedence.
    metabolite_nodes : iterable of str, optional
        Iterable of metabolite ids, used to identify which nodes represent metabolites.
        Must provide either this or `model`. If both are provided, `metabolite_nodes`
        takes precedence.

    Returns
    -------
    set of str:
        The ids of the choke point reactions

    Notes
    -----
    Choke points are reactions which uniquely produce or consume a metabolite [1].

    References
    ----------
    .. [1] Rahman, S. A.; Schomburg, D. Observing Local and Global
       Properties of Metabolic Pathways: ‘Load Points’ and ‘Choke Points’
       in the Metabolic Networks. Bioinformatics 2006, 22 (14), 1767–1774.
       https://doi.org/10.1093/bioinformatics/btl181
    """
    if not metabolic_network.is_directed():
        raise ValueError(
            "Metabolic network must be directed, but received undirected network"
        )
    if metabolite_nodes is not None:
        metabolite_set = set(metabolite_nodes)
    else:
        metabolite_set = None
        if model is None:
            raise ValueError(
                "Model parameter must be provided if metabolites_nodes is None, but both are None"
            )
    choke_points = set()
    for node in metabolic_network:
        node = cast(str, node)
        # Check if the node represents a metabolite
        if metabolite_set is not None and node not in metabolite_set:
            continue
        else:
            if model is None:
                raise ValueError(
                    "Model parameter must be provided if metabolites_nodes is None, but both are None"
                )
            try:
                _ = model.metabolites.get_by_id(node)
            except KeyError:
                continue
        # If it does, find chokepoints if they exist
        pred = list(metabolic_network.predecessors(node))
        suc = list(metabolic_network.successors(node))
        # Only reaction which produces this metabolite
        if len(pred) == 1:
            choke_points.add(pred[0])
        # Only reaction which consumes this metabolite
        if len(suc) == 1:
            choke_points.add(suc[0])
    return choke_points

from importlib.metadata import version

__author__ = "Braden Griebel"
__version__ = version("metworkpy")
__all__ = [  # noqa: RUF022
    "divergence",
    "gpr",
    "hyper",
    "imat",
    "information",
    "metabolites",
    "network",
    "sampling",
    "synleth",
    "utils",
    # Divergence
    "kl_divergence",
    "js_divergence",
    "get_example_model",
    # Mutual information
    "mutual_information",
    "variation_of_information",
    "mi_network_adjacency_matrix",
    # Hyper
    "HyperGraph",
    # Metabolites
    "metchange",
    "find_metabolite_synthesis_network_genes",
    "find_metabolite_synthesis_network_reactions",
    # Networks
    "bipartite_project",
    "fuzzy_reaction_set",
    "fuzzy_reaction_intersection",
    "create_metabolic_network",
    "create_reaction_network",
    "create_metabolite_network",
    "create_gene_network",
    "create_group_distance_adjacency_matrix",
    "create_group_distance_network",
    "create_group_neighborhood_network",
    "create_mutual_information_network",
    "find_dense_clusters",
    # Utils
    "eval_gpr",
    "gene_to_rxn_weights",
    "fisher_enrichment",
    "extended_mannwhitneyu_test",
    "reaction_to_gene_ids",
    "gene_to_reaction_ids",
    "gene_to_reaction_list",
    "reaction_to_gene_list",
    "get_gene_to_reaction_translation_dict",
    "get_reaction_to_gene_translation_dict",
    "read_model",
    "write_model",
    "model_eq",
    "model_bounds_eq",
]

from metworkpy import (
    divergence,
    gpr,
    hyper,
    imat,
    information,
    metabolites,
    network,
    sampling,
    synleth,
    utils,
)
from metworkpy.divergence import js_divergence, kl_divergence
from metworkpy.examples import get_example_model
from metworkpy.gpr import eval_gpr, gene_to_rxn_weights
from metworkpy.hyper import HyperGraph
from metworkpy.information import (
    mi_network_adjacency_matrix,
    mutual_information,
    variation_of_information,
)
from metworkpy.metabolites import (
    find_metabolite_synthesis_network_genes,
    find_metabolite_synthesis_network_reactions,
    metchange,
)

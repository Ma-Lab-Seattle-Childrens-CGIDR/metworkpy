from importlib.metadata import version

__author__ = "Braden Griebel"
__version__ = version("metworkpy")
__all__ = [
    "utils",
    "imat",
    "gpr",
    "information",
    "divergence",
    "network",
    "synleth",
    "read_model",
    "write_model",
    "model_eq",
    "model_bounds_eq",
    "mutual_information",
    "mi_network_adjacency_matrix",
    "kl_divergence",
    "js_divergence",
    "get_example_model",
    "create_metabolic_network",
    "create_reaction_network",
    "create_metabolite_network",
    "create_mutual_information_network",
    "create_adjacency_matrix",
    "reaction_target_density",
    "find_dense_clusters",
    "reaction_to_gene_ids",
    "gene_to_reaction_ids",
    "gene_to_reaction_list",
    "reaction_to_gene_list",
    "get_gene_to_reaction_translation_dict",
    "get_reaction_to_gene_translation_dict",
    "bipartite_project",
    "fuzzy_reaction_set",
    "fuzzy_reaction_intersection",
    "find_metabolite_synthesis_network_genes",
    "find_metabolite_synthesis_network_reactions",
    "metchange",
    "metabolites",
    "eval_gpr",
    "gene_to_rxn_weights",
    "race_gene_set_entropy",
    "infer_gene_set_entropy",
    "crane_gene_set_entropy",
    "dirac_gene_set_entropy",
    "dirac_gene_set_classification",
    "crane_gene_set_classification",
    "DiracClassifier",
    "CraneClassifier",
]

from metworkpy import (
    utils,
    imat,
    gpr,
    information,
    divergence,
    metabolites,
    network,
    synleth,
)

from metworkpy.utils import (
    read_model,
    write_model,
    model_eq,
    model_bounds_eq,
    gene_to_reaction_list,
    reaction_to_gene_list,
)

from metworkpy.information import (
    mutual_information,
    mi_network_adjacency_matrix,
)

from metworkpy.divergence import kl_divergence, js_divergence

from metworkpy.examples import get_example_model

from metworkpy.network import (
    create_metabolic_network,
    create_mutual_information_network,
    create_adjacency_matrix,
    reaction_target_density,
    find_dense_clusters,
    bipartite_project,
    fuzzy_reaction_set,
    fuzzy_reaction_intersection,
)

from metworkpy.metabolites import (
    find_metabolite_synthesis_network_reactions,
    find_metabolite_synthesis_network_genes,
    metchange,
)

from metworkpy.gpr import eval_gpr, gene_to_rxn_weights

from metworkpy.rank_entropy import (
    race_gene_set_entropy,
    infer_gene_set_entropy,
    crane_gene_set_entropy,
    dirac_gene_set_entropy,
    dirac_gene_set_classification,
    crane_gene_set_classification,
    DiracClassifier,
    CraneClassifier,
)

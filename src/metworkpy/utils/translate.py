"""Module for translating between genes and reactions"""

# region Imports
# Standard Library Imports
from __future__ import annotations
import functools
from typing import cast, Iterable

# External Imports
import cobra

# Local Imports


# endregion Imports


def reaction_to_gene_ids(
    model: cobra.Model, reaction: str, essential: bool = False
) -> set[str]:
    """
    Convert a reaction id into associated gene ids

    Parameters
    ----------
    model : cobra.Model
        Model to use for perfoming the translation
    reaction : str
        id of reaction to translate into gene ids
    essential : bool, default=False
        Whether the genes should be only those required for
        the reaction to function

    Returns
    -------
    gene_set : set[str]
        Set of genes associated with a particular reaction
    """
    rxn_obj = cast(cobra.Reaction, model.reactions.get_by_id(reaction))
    gene_set = rxn_obj.genes
    if len(gene_set) == 0:
        return set()
    if not essential or len(gene_set) == 1:
        return {g.id for g in gene_set}
    associated_gene_set = set()
    gpr = rxn_obj.gpr
    for gene in gene_set:
        if not gpr.eval(gene.id):
            associated_gene_set.add(gene.id)
    return associated_gene_set


def gene_to_reaction_ids(
    model: cobra.Model, gene: str, essential: bool = False
) -> set[str]:
    """
    Convert a gene id into associated reaction ids

    Parameters
    ----------
    model : cobra.Model
        Model to use for performing the translation
    gene : str
        id for gene to translate into reaciton ids
    essential : bool, default=False
        Whether the reactions should only be those for
        which the gene is required

    Returns
    -------
    reaction_set : set[str]
        Set of reactions associated with a particular gene
    """
    gene_obj = cast(cobra.Gene, model.genes.get_by_id(gene))
    rxn_set = gene_obj.reactions
    if len(rxn_set) == 0:
        return set()
    if not essential:
        return {r.id for r in rxn_set}
    associated_reaction_set = set()
    for rxn in rxn_set:
        if not rxn.gpr.eval(gene):
            associated_reaction_set.add(rxn.id)
    return associated_reaction_set


# region Translation Dict


def get_gene_to_reaction_translation_dict(
    model: cobra.Model, essential: bool = False
) -> dict[str, set[str]]:
    """
    Get a dictionary to translate from genes to associated reactions

    Parameters
    ----------
    model : cobra.Model
        Model to construct the translation dict for
    essential : bool, default=False
        Whether the reactions should only be those for which
        the gene is required

    Returns
    -------
    translation_dict : dict[str, set[str]]
        Dict keyed by gene ids within the model, with
        values that are sets of reactions associated with
        the gene
    """
    return {
        g.id: gene_to_reaction_ids(model, g.id, essential) for g in model.genes
    }


def get_reaction_to_gene_translation_dict(
    model: cobra.Model, essential: bool = False
) -> dict[str, set[str]]:
    """
    Get a dictionary to translate from reactions to associated genes

    Parameters
    ----------
    model : cobra.Model
        Model to construct the translation dict for
    essential : bool, default=False
        Whether the genes should only be those which
        are required by the reaction

    Returns
    -------
    translation_dict : dict[str, set[str]]
        Dict keyed by reaction ids within the model, with
        values that are sets of genes associated with
        each reaction
    """
    return {
        r.id: reaction_to_gene_ids(model, r.id, essential)
        for r in model.reactions
    }


# endregion Translation Dict


# region Translate List


def gene_to_reaction_list(
    model: cobra.Model, gene_list: Iterable[str], essential: bool = False
):
    """
    Convert a list (or other Iterable) of gene ids into a list of associated reaction ids

    Parameters
    ----------
    model : cobra.Model
        Model to use for performing the translation
    gene_list : list of str
        list of gene ids to translate
    essential : bool, default=False
        Whether the reactions should only be those for
        which the genes are required

    Returns
    -------
    reaction_list : list[str]
        list of reactions associated with the genes in `gene_list`

    Note
    ----
    The order of the genes is not perseved in the order of the reactions
    """
    return list(
        functools.reduce(
            lambda x, y: x | y,
            map(
                lambda g: gene_to_reaction_ids(
                    model=model, gene=g, essential=essential
                ),
                gene_list,
            ),
        )
    )


def reaction_to_gene_list(
    model: cobra.Model, reaction_list: Iterable[str], essential: bool = False
):
    """
    Convert a list (or other Iterable) of reaction ids into a list of associated gene ids

    Parameters
    ----------
    model : cobra.Model
        Model to use for performing the translation
    reaction_list : list of str
        list of reaction ids to translate
    essential : bool, default=False
        Whether the genes should only be those
        which the reactions require to function

    Returns
    -------
    gene_list : list[str]
        list of genes associated with the reactions in `reaction_list`

    Note
    ----
    The order of the reactions is not perseved in the order of the genes
    """
    return list(
        functools.reduce(
            lambda x, y: x | y,
            map(
                lambda r: reaction_to_gene_ids(
                    model=model, reaction=r, essential=essential
                ),
                reaction_list,
            ),
        )
    )


# endregion Translate List

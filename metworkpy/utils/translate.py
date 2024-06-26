"""
Module for translating between genes and reactions
"""
# region Imports
# Standard Library Imports
from __future__ import annotations
from collections import defaultdict
from typing import Literal

# External Imports
import cobra
import pandas as pd


# Local Imports

# endregion Imports


# region Translate List
def gene_to_reaction_list(model: cobra.Model, gene_list: list[str]) -> list[str]:
    """
    Translate gene symbols to reactions which are associated with them

    :param model: Cobra model containing the genes and reactions in order to translate
    :type model: cobra.Model
    :param gene_list: List of genes to translate
    :type gene_list: list[str]
    :return: List of reactions associated with the provided list of genes
    :rtype:
    """
    rxn_list = []
    for gene in gene_list:
        for rxn in model.genes.get_by_id(gene).reactions:
            rxn_list += [rxn.id]
    return rxn_list


def reaction_to_gene_list(model: cobra.Model, reaction_list: list[str]) -> list[str]:
    """
    Translate reaction ids to genes which are associated with them

    :param model: Cobra model containing the genes and reactions in order to translate
    :type model: cobra.Model
    :param reaction_list: List of reactions to translate
    :type reaction_list: list[str]
    :return: List of genes associated with the provided list of reactions
    :rtype: list[str]
    """
    gene_list = []
    for rxn in reaction_list:
        for gene in model.reactions.get_by_id(rxn).genes:
            gene_list += [gene.id]
    return gene_list


# endregion Translate List


# region Translate to Dict
def gene_to_reaction_dict(model: cobra.Model, gene_list: list[str]):
    """
    Translate gene symbols to a dict of gene symbol: reaction list

    :param model: Cobra model containing the genes and reactions in order to translate
    :type model: cobra.Model
    :param gene_list: List of genes to translate
    :type gene_list: list[str]
    :return: Dictionary with gene symbols as keys, and lists of reactions as
        values
    :rtype: dict[str, list[str]]
    """
    gene_rxn_dict = defaultdict(list)
    for gene in gene_list:
        for rxn in model.genes.get_by_id(gene).reactions:
            gene_rxn_dict[gene] += [rxn.id]
    return gene_rxn_dict


def reaction_to_gene_dict(model: cobra.Model, reaction_list: list[str]):
    """
    Translate reaction IDs to a dict of reaction: gene list

    :param model: Cobra model containing the genes and reactions in order to translate
    :type model: cobra.Model
    :param reaction_list: List of reactions to translate
    :type reaction_list: list[str]
    :return: Dictionary with reaction ids as keys, and lists of genes as values
    :rtype: dict[str, list[str]]
    """
    rxn_gene_dict = defaultdict(list)
    for rxn in reaction_list:
        for gene in model.reactions.get_by_id(rxn).genes:
            rxn_gene_dict[rxn] += [gene.id]
    return rxn_gene_dict


# endregion Translate to Dict

# region Translate DataFrame


def gene_to_reaction_df(
    model: cobra.Model,
    gene_df: pd.DataFrame,
    how: Literal["left", "right", "outer", "inner", "cross"] = "left",
) -> pd.DataFrame:
    """
    Translate from a dataframe indexed by gene symbols to one indexed by reaction ids

    :param model: Cobra model to use for translating
    :type model: cobra.Model
    :param gene_df: DataFrame to translate, should be indexed by genes
    :type gene_df: pd.DataFrame
    :param how: When the reaction-indexed dataframe is joined to the
        gene-indexed dataframe, what type of join should be used
        (see Pandas `Merge`_ documentation)
    :type how: Literal["left", "right", "outer", "inner", "cross"]
    :return: Dataframe indexed by gene
    :rtype: pd.DataFrame

    .. _Merge: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
    """
    gene_list = gene_df.index.to_list()
    gene_reaction_dict = {"genes": [], "reactions": []}
    for gene in gene_list:
        for rxn in model.genes.get_by_id(gene).reactions:
            gene_reaction_dict["genes"] += [gene]
            gene_reaction_dict["reactions"] += [rxn.id]
    gene_reaction_df = pd.DataFrame(gene_reaction_dict).set_index("reactions")
    return gene_reaction_df.merge(gene_df, left_on="genes", right_index=True, how=how)


def reaction_to_gene_df(
    model: cobra.Model,
    reaction_df: pd.DataFrame,
    how: Literal["left", "right", "outer", "inner", "cross"] = "left",
) -> pd.DataFrame:
    """
    Translate from a dataframe indexed by reaction ids to one indexed by gene symbols

    :param model: Cobra model to use for translating
    :type model: cobra.Model
    :param reaction_df: DataFrame to translate, should be indexed by reactions
    :type reaction_df: pd.DataFrame
    :param how: When the gene-indexed dataframe is joined to the
        reaction-indexed dataframe, what type of join should be used
        (see Pandas `Merge`_ documentation)
    :type how: Literal["left", "right", "outer", "inner", "cross"]
    :return: Dataframe indexed by gene
    :rtype: pd.DataFrame

    .. _Merge: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
    """
    rxn_list = reaction_df.index.to_list()
    rxn_gene_dict = {"genes": [], "reactions": []}
    for rxn in rxn_list:
        for gene in model.reactions.get_by_id(rxn).genes:
            rxn_gene_dict["reactions"] += [rxn]
            rxn_gene_dict["genes"] += [gene.id]
    rxn_gene_dict = pd.DataFrame(rxn_gene_dict).set_index("genes")
    return rxn_gene_dict.merge(
        reaction_df, left_on="reactions", right_index=True, how=how
    )


# endregion Translate DataFrame

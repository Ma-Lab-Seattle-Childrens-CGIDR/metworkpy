"""
Module containing utility functions for working with gene expression
data, and converting it into qualitative weights
"""

from __future__ import annotations

# Standard library imports
from typing import Callable, Union, Iterable, Optional
from warnings import warn

# External imports
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd


# Local imports


# region IMAT weights function
def expr_to_imat_gene_weights(
    expression: Union[pd.Series, pd.DataFrame],
    quantile: Union[float, tuple[float, float]] = 0.15,
    aggregator: Callable[[ArrayLike], float] = np.median,
    subset: Optional[Iterable] = None,
    sample_axis: Union[int, str] = 0,
) -> pd.Series:
    """
    Convert gene expression data to qualitative gene weights

    :param expression: Normalized gene expression data. If it is a DataFrame
        representing multiple samples, those samples will be aggregated
        using the aggregator function (default median).
    :type expression: pd.Series | pd.DataFrame
    :param quantile: Quantile or quantiles to use for binning expression data.
        Should be between 0 and 1. If single value the bottom quantile
        will be converted to -1, the top quantile converted to 1, and
        all expression values between to 0. If a tuple is provided,
        the first is treated as the low quantile cutoff, and the second
        is treated as the high quantile cutoff.
    :type quantile: float | tuple[float, float]
    :param aggregator: Function used to aggregated gene expression data
        across samples, only used if expression is a DataFrame (default
        median).
    :type aggregator: Callable[[np.ArrayLike], float]
    :param subset: Subset of genes to perform calculations on. `expression` is
        filtered to only include these genes before quantiles are calculated. If
        any genes are present in the subset, but not in expression, they will be
        assigned a value of 0 following the trinarization.
    :type subset: Optional[Iterable]
    :param sample_axis: Which axis represents samples in the expression
        data (only used if expression is DataFrame). "index" or 0 if rows
        represent different samples, "column" or 1 if columns represent
        different samples (default is rows).
    :type sample_axis: int | str
    :return: Series of qualitative weights, -1 for low expression, 1 for
        high expression, and 0 otherwise.
    :rtype: pd.Series

    .. note::
        The expression data should only represent biological replicates
        as it will be aggregated. If multiple different conditions are
        represented in your expression data, they should be seperated
        before this function is used.

        For the quantile, if a tuple like `(0.15, 0.90)` is provided,
        the bottom 15% of genes in terms of expression will have weights
        of -1, while the top 10% will have weights of 1, and everything
        in between will have weights of 0.
    """
    # Convert float to tuple if necessary
    if isinstance(quantile, float):
        quantile = (quantile, 1 - quantile)
    if isinstance(expression, pd.DataFrame):
        expression = expression.apply(aggregator, axis=sample_axis)
    if not subset:
        low, high = np.quantile(expression, quantile)
        return expression.map(lambda x: -1 if x <= low else (1 if x >= high else 0))
    # Only use subset genes which are in expression data
    subset_genes = [gene for gene in subset if gene in expression.index]
    expression = expression[subset_genes]
    result_series = pd.Series(0, index=subset)
    low, high = np.quantile(expression, quantile)
    result_series[subset_genes] = expression.map(
        lambda x: -1 if x <= low else (1 if x >= high else 0)
    )[subset_genes]
    return result_series


# endregion IMAT weights function


# region Metchange Weights Functions
def expr_to_metchange_gene_weights(
    expression: pd.Series | pd.DataFrame,
    quantile_cutoff: float,
    subset: Iterable[str] | None = None,
    aggregator: Callable[[ArrayLike[float]], float] = np.median,
    sample_axis: str | int = 0,
) -> pd.Series:
    """
    Convert gene expression values into metchange gene weights

    :param expression: Gene expression values. Either a series with genes as the index,
        or a Dataframe with genes as one axis, and samples as the other. In the case
        of a dataframe, the expression values are aggregated before the weights
        are calculated.
    :type expression: pd.Series | pd.DataFrame
    :param quantile_cutoff: Cutoff used for defining the weights. The expression value
        corresponding to this quantile is used as the threshold. Everything above
        the threshold is weighted 0, and everything below is weighted in proportion
        to distance from the threshold. The weight will be between 0 and 1, with
        values near the threshold being near 0, and values near 0 being weighted 1.
    :type quantile_cutoff: float
    :param subset: Subset of genes to use in weighting. Default of None will use
        all genes in `expression`. If not none, expression will be filtered down
        to this subset of genes before the quantile threshold is calculated,
        and the returned series will only include this subset of genes.
    :type subset: Iterable[str] | None
    :param aggregator: Aggregation function to use for aggregating expression
        data across multiple samples. Should accept a single Arraylike argument,
        and return a float.  Default is median.
    :type aggregator: Callable[[Arraylike[float]], float]
    :param sample_axis: Which axis in `expression` dataframe represents samples.
        Can be 'index', 'columns', 0 or 1. A value of 0 or 'index' means rows
        represent different samples, while a value of 1 or 'columns' means that
        columns represent different samples.
    :type sample_axis: str | int
    :return: Series of gene weights (floats between 0 and 1, representing the
        probability that a gene product is absent), indexed by gene ids.
    :rtype: pd.Series

    .. note:
       This does not convert the expression values into reaction weights, to do so
       `metworkpy.parse.gpr.gene_to_rxn_weights` can be used. The function dict will need to
       be altered from the default, with `{'AND':max, 'OR':min}` due to the metchange
       weights being probability of absense rather than presence.

    .. seealso:
       :func: `metworkpy.parse.gpr.gene_to_rxn_weights`
    """
    if isinstance(expression, pd.DataFrame):
        expression = expression.apply(aggregator, axis=sample_axis)
    if subset is None:
        subset = expression.index
    subset = [gene for gene in subset if gene in expression.index]
    expression = expression[subset]
    return _expr_to_metchange_gene_weight_series(
        expression=expression, quantile_cutoff=quantile_cutoff
    )


def _expr_to_metchange_gene_weight_series(
    expression: pd.Series, quantile_cutoff: float
) -> pd.Series:
    cutoff = np.quantile(expression, quantile_cutoff)
    weights = pd.Series(np.nan, index=expression.index)
    weights[expression < cutoff] = (
        expression[expression < cutoff] * (-1) + cutoff
    ) / cutoff
    weights[expression >= cutoff] = 0.0
    return weights


# endregion Metchange Weights Functions


# region Conversion functions
def count_to_rpkm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Normalize raw count data using RPKM

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: RPKM normalized counts
    :rtype: pd.DataFrame
    """
    # Ensure that the count data frame and feature length series have the same genes
    count_genes = set(count.columns)
    fl_genes = set(feature_length.index)
    # Get the library size before dropping genes
    per_million = count.sum(axis=1) / 1e6
    if not (count_genes == fl_genes):
        warn(
            "Different genes in count dataframe and feature length series, dropping any not in common"
        )
        genes = sorted(list(count_genes.intersection(fl_genes)))
        count = count[genes]
        feature_length = feature_length[genes]
    rpm = count.divide(per_million, axis=0)
    rpkm = rpm.divide(feature_length / 1000, axis=1)
    return rpkm


def count_to_fpkm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Converts count data to FPKM normalized expression

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows. Specifically,
        the count data represents the number of fragments, where a fragment corresponds to a single cDNA molecule, which
        can be represented by a pair of reads from each end.
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: FPKM normalized counts
    :rtype: pd.DataFrame
    """
    return count_to_rpkm(count, feature_length)


def count_to_tpm(count: pd.DataFrame, feature_length: pd.Series) -> pd.DataFrame:
    """
    Converts count data to TPM normalized expression

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :param feature_length: Series containing the feature length for all the genes
    :type feature_length: pd.Series
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    rpkm = count_to_rpkm(count, feature_length)
    return rpkm_to_tpm(rpkm)


def count_to_cpm(count: pd.DataFrame) -> pd.DataFrame:
    """
    Converts count data to counts per million

    :param count: Dataframe containing gene count data, with genes as the columns and samples as the rows
    :type count: pd.DataFrame
    :return: CPM normalized counts
    :rtype: pd.DataFrame
    """
    total_reads = count.sum(axis=1)
    per_mil_scale = total_reads / 1e6
    return count.divide(per_mil_scale, axis=0)


def rpkm_to_tpm(rpkm: pd.DataFrame):
    """
    Convert RPKM normalized counts to TPM normalized counts

    :param rpkm: RPKM normalized count data, with genes as columns and samples as rows
    :type rpkm: pd.DataFrame
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    return rpkm.divide(rpkm.sum(axis=1), axis=0) * 1.0e6


def fpkm_to_tpm(fpkm: pd.DataFrame):
    """
    Convert FPKM normalized counts to TPM normalized counts

    :param fpkm: RPKM normalized count data, with genes as columns and samples as rows
    :type fpkm: pd.DataFrame
    :return: TPM normalized counts
    :rtype: pd.DataFrame
    """
    return rpkm_to_tpm(fpkm)


# endregion Conversion functions

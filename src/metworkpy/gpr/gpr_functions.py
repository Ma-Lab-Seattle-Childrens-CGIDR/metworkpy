# Standard Library Imports
from __future__ import annotations
import ast
from typing import Any, Optional, Union
import warnings

# External Imports
import cobra
import pandas as pd

# Global function dictionary declarations
METCHANGE_FUNC_DICT = {"AND": max, "OR": min}
IMAT_FUNC_DICT = {"AND": min, "OR": max}


def gene_to_rxn_weights(
    model: cobra.Model,
    gene_weights: pd.Series,
    fn_dict: Optional[dict] = None,
    fill_val: Any = 0,
) -> pd.Series:
    """Convert a gene weights series to a reaction weights series using the
    provided function dictionary.

    Parameters
    ----------
    model : cobra.Model
        cobra.Model: A cobra model
    gene_weights : pd.Series or dict of str to Value
        pd.Series: A series of gene weights
    fn_dict : dict
        dict: A dictionary of functions to use for each operator
    fill_val : Any
        Any: The value to fill missing values with

    Returns
    -------
    pd.Series
        A series of reaction weights

    Notes
    -----
    The fill value is applied to fill NaN values after the GPR rules have been
    applied.
    If there are genes missing from the expression data, they will silently be
    assigned a value of 0 before the GPR processing is performed.
    """
    # Check that all genes in the model are in the gene expression data,
    # and if not add them with a weight of 0
    model_genes = set(model.genes.list_attr("id"))
    expr_genes = set(gene_weights.index)
    missing_genes = list(model_genes - expr_genes)
    if missing_genes:
        warnings.warn(
            f"Genes {missing_genes} are in model but not in gene weights, "
            f"setting their weight to {fill_val}."
        )
        missing_genes_series = pd.Series(0, index=missing_genes)
        gene_weights = pd.concat([gene_weights, missing_genes_series])

    # Convert the fill_val into the same type as the gene_weights
    fill_val = pd.Series(fill_val, dtype=gene_weights.dtype).iloc[0]

    # Create the rxn_weight series, filled with all 0
    rxn_weights = pd.Series(
        fill_val, index=pd.Index(model.reactions.list_attr("id"))
    )

    if fn_dict is None:
        fn_dict = IMAT_FUNC_DICT
    # For each reaction, trinarize it based on the expression data
    for rxn in model.reactions:
        rxn_weights[rxn.id] = eval_gpr(
            rxn.gpr, gene_weights, fn_dict, fill_val
        )
    return rxn_weights


def eval_gpr(
    gpr: Optional[
        Union[cobra.core.GPR, ast.Expression, list, ast.BoolOp, ast.Name]
    ],
    gene_weights: Union[pd.Series, dict],
    fn_dict: dict,
    fill_val: Any = 0,
) -> Any:
    """
    Evaluate a cobra GPR with specified functions for the Boolean operations

    Parameters
    ----------
    gpr : GPR or Expression or list or BoolOp or Name optional
        The GPR to evaluate
    gene_weights: pd.Series or dict
        Weights to assign to each gene
    fn_dict : dict
        Dict of 'AND' and 'OR' (strings) to functions which can
        take two gene weights and return a single value
    fill_val : Any
        Value to replace any missing weights with
    """
    if isinstance(gpr, (ast.Expression, cobra.core.GPR)):
        if not gpr.body:
            return fill_val
        return eval_gpr(
            gpr=gpr.body,  # type:ignore
            gene_weights=gene_weights,
            fn_dict=fn_dict,
            fill_val=fill_val,
        )
    elif isinstance(gpr, ast.Name):
        return gene_weights.get(gpr.id, fill_val)
    elif isinstance(gpr, ast.BoolOp):
        op = gpr.op
        if isinstance(op, ast.Or):
            return fn_dict["OR"](  # type: ignore
                *[
                    eval_gpr(e, gene_weights, fn_dict, fill_val)  # type: ignore
                    for e in gpr.values
                ]
            )
        elif isinstance(op, ast.And):
            return fn_dict["AND"](  # type: ignore
                *[
                    eval_gpr(e, gene_weights, fn_dict, fill_val)  # type: ignore
                    for e in gpr.values
                ]
            )
        else:
            raise TypeError(
                f"Unsupported Boolean Operation: {op.__class__.__name__}"
            )
    elif gpr is None:
        return fill_val
    raise TypeError(f"Unsupported GPR type: {type(gpr)}")

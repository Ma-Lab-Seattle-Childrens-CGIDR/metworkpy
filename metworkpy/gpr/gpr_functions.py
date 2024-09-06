# Standard Library Imports
from __future__ import annotations
from collections import deque
import re
from typing import Optional, Any
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
    """
    Convert a gene weights series to a reaction weights series using the
    provided function dictionary.

    :param model: cobra.Model: A cobra model
    :type model: cobra.Model
    :param gene_weights: pd.Series: A series of gene weights
    :type gene_weights: pd.Series
    :param fn_dict: dict: A dictionary of functions to use for each operator
    :type fn_dict: dict
    :param fill_val: Any: The value to fill missing values with
    :type fill_val: Any
    :return: A series of reaction weights
    :rtype: pd.Series

    .. note:
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

    # Create the rxn_weight series, filled with all 0
    rxn_weights = pd.Series(0, index=[rxn.id for rxn in model.reactions])

    # For each reaction, trinarize it based on the expression data
    for rxn in model.reactions:
        gpr = rxn.gene_reaction_rule
        rxn_weights[rxn.id] = eval_gpr(gpr, gene_weights, fn_dict)
    rxn_weights.fillna(fill_val, inplace=True)
    return rxn_weights


def eval_gpr(
    gpr: str, gene_weights: pd.Series, fn_dict: Optional[dict] = None
) -> Any | None:
    """
    Evaluate a single GPR string using the provided gene weights and
    function dictionary.

    :param gpr: str: A single GPR string
    :type gpr: str
    :param gene_weights: pd.Series: A series of gene weights
    :type gene_weights: pd.Series
    :param fn_dict: dict: A dictionary of functions to use for each operator,
        in GPR the operators are normally AND and OR, by default this is
        {"AND":min, "OR":max}
    :type fn_dict: dict
    :return: The GPR score
    :rtype: float | None
    """
    if not gpr:  # If GPR is empty string, return None
        return None
    if fn_dict is None:
        fn_dict = {"AND": min, "OR": max}
    gpr_expr = _str_to_deque(gpr)
    gpr_expr = _to_postfix(gpr_expr)
    return _eval_gpr_deque(
        gpr_expr=gpr_expr, gene_weights=gene_weights, fn_dict=fn_dict
    )


def _eval_gpr_deque(gpr_expr: deque, gene_weights: pd.Series, fn_dict: dict):
    eval_stack = []
    for token in gpr_expr:
        if token not in fn_dict:
            eval_stack.append(gene_weights[token])
            continue
        val1 = eval_stack.pop()
        val2 = eval_stack.pop()
        eval_stack.append(fn_dict[token](val1, val2))
    if len(eval_stack) != 1:
        raise ValueError(f"Failed to parse GPR Expression: {gpr_expr}")
    return eval_stack.pop()


def _str_to_deque(in_string: str, replacements: Optional[dict] = None) -> deque[str]:
    """
    Convert a string to a list of strings, splitting on whitespace and
    parentheses.

    :param in_string: str: Specify the input string
    :type in_string: str
    :param replacements: dict: Replace certain strings with other strings
        before splitting, uses regex
    :type replacements: dict
    :return: A deque of strings
    :rtype: deque[str]
    """
    if not replacements:
        replacements = {
            "\\b[Aa][Nn][Dd]\\b": "AND",
            "\\b[Oo][Rr]\\b": "OR",
            "&&?": " AND ",
            r"\|\|?": " OR ",
        }
    in_string = in_string.replace("(", " ( ").replace(")", " ) ")
    for key, value in replacements.items():
        in_string = re.sub(key, value, in_string)
    return deque(in_string.split())


def _process_token(
    token: str, postfix: deque[str], operator_stack: deque[str], precedence
):
    """
    The process_token function takes in a token, the postfix list, the
    operator stack and precedence dictionary. It performs the shunting
    yard algorithm for a single provided token.

    :param token: Current token
    :type token: str
    :param postfix: Current state of output
    :type postfix: list[str]
    :param operator_stack: Current operator stack
    :type operator_stack: list[str]
    :param precedence: Determines the operators precedence
    :type precedence: dict[str:int]
    :return: Nothing
    :rtype: None
    """
    # If token is not operator, add it to postfix
    if (token not in precedence) and (token != "(") and (token != ")"):
        postfix.append(token)
        return
    # If token is operator, move higher priority operators from stack to
    # output, then add the operator itself to the postfix expression
    if token in precedence:
        while (
            (len(operator_stack) > 0)
            and (operator_stack[-1] != "(")
            and (precedence[operator_stack[-1]] >= precedence[token])
        ):
            op = operator_stack.pop()
            postfix.append(op)
        operator_stack.append(token)
        return
    # For left parenthesis add to operator stack
    if token == "(":
        operator_stack.append(token)
        return
    # For right parenthesis pop operator stack until reach
    # matching left parenthesis
    if token == ")":
        if len(operator_stack) == 0:  # Check for mismatch in parentheses
            raise ValueError("Mismatched Parenthesis in Expression")
        while len(operator_stack) > 0 and operator_stack[-1] != "(":
            op = operator_stack.pop()
            postfix.append(op)
        if (
            len(operator_stack) == 0 or operator_stack[-1] != "("
        ):  # Check for mismatch in parentheses
            raise ValueError("Mismatched Parenthesis in Expression")
        _ = operator_stack.pop()  # Remove left paren from stack
        return None


def _to_postfix(infix: deque[str], precedence: Optional[dict] = None) -> deque[str]:
    """
    Convert an infix expression to postfix notation.
    :param infix: deque[str]: A deque of strings representing an infix expression
    :type infix: deque[str]
    :param precedence: Dictionary of operators determining precedence
    :type precedence: dict[str:int]
    :return: A list of strings representing the postfix expression
    :rtype: list[str]
    """
    # Set default precedence
    if precedence is None:
        precedence = {"AND": 1, "OR": 1}
    postfix = deque()
    operator_stack = deque()
    # For each token, use shunting yard algorithm to process it
    for token in infix:
        _process_token(token, postfix, operator_stack, precedence)
    # Empty the operator stack
    while len(operator_stack) > 0:
        op = operator_stack.pop()
        if op == "(":
            raise ValueError("Mismatched Parenthesis in Expression")
        postfix.append(op)
    return postfix

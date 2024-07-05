"""
Script for running Metchange from the command line
"""

# Imports
# Standard Library Imports
from __future__ import annotations
import argparse
import itertools
import pathlib
import re

# External Imports
import cobra
import numpy as np
import pandas as pd

# Local Imports
import metworkpy
from ._script_utils import (
    _parse_samples,
    _parse_aggregation_method,
    _parse_sample_groups_and_names,
)


def parse_args(arg_list: list[str] | None) -> argparse.Namespace:
    """
    Parse Command line arguments

    :param arg_list: List of command line strings (defaults to reading from stdin)
    :type arg_list: list[str]|None
    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="metchange",
        description="Run the metchange algorithm using gene expression data",
    )
    parser.add_argument(
        "-M",
        "--model",
        dest="model_file",
        default=None,
        help="Path to cobra model file (json, sbml, yaml)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gene-expression",
        dest="gene_expression_file",
        default=None,
        help="Path to normalized gene expression file (csv format). Columns should represent genes "
        "with the first row being gene id, matching the gene ids in the cobra model. "
        "Rows should represent samples, with the first column being the sample name. "
        "The --transpose argument can be used to specify the orientation if other than the "
        "default. Data should be read depth and length normalized such as TPM, or RPKM",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-w",
        "--wildtype",
        dest="wildtype",
        default=None,
        help="A list of samples which represent the wildtype, or negative control condition. "
        "Represented by a set of numbers, seperated by commas (no spaces) that represent the "
        "0-indexed rows (or columns if --transpose flag is used) for the wildtype samples. "
        "Colons can be used to specify an inclusive range of values. For example 1,2:5,7 will "
        "specify rows 1,2,3,4,5,7. (THESE VALUES ARE 0-indexed, so 1 represents the 2nd sample).",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--sample-groups",
        dest="sample_groups",
        default=None,
        help="Groups of samples to aggregate. Should be lists of comma seperated row numbers "
        "(0-indexed), seperated by parentheses (no spaces, optional commas) wrapped in quotes. "
        "Colons can be used "
        'to describe inclusive ranges. For example, "(1:3,4)(5,6)" would represent two groups '
        "of "
        "samples '1,2,3,4' and '5,6'. These groups of samples should represent replicates, "
        "as they will be aggregated by the --aggregate method. If no aggregation is "
        "required, leave this argument blank. Defaults to every sample being its own "
        "group (so no aggregation).",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        default="metchange_results.csv",
        help="Path to output the results of metchange algorithm. Will be a csv of normalized "
        "inconsistency scores, with the first column being metabolite ids from the model, "
        "the next columns will be , and remaining columns being information"
        "about the metabolites if --extra-info flag is used.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-f",
        "--model-format",
        dest="model_format",
        default=None,
        help="The format of the input model file ("
        "can be json, yaml, or sbml). If not provided, "
        "it will be inferred from the models file extension.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--metabolites",
        dest="metabolites_list",
        default=None,
        help="List of metabolites to include in metchange algorithm, defaults to using all metabolites"
        "in model. Should be id strings matching the metabolite ids in the cobra model. "
        "If both --metabolites and --metabolites-file are passed, will only include "
        "the metabolites passed to the --metabolites argument.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--metabolites-file",
        dest="metabolites_file",
        default=None,
        help="A file containing a list of metabolites to include in the metchange algorithm. Should"
        "be a text file, with each metabolite on a newline. Metabolites must be "
        "id strings matching"
        " the metabolite ids in the cobra model.If both --metabolites and "
        "--metabolites-file are passed, the metabolites from the --metabolites argument will"
        "be the only ones included.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--sample-group-names",
        dest="sample_group_names",
        default=None,
        help="Names for the sample groups identified in --sample-groups argument. Should "
        "be comma seperated list of names (no spaces between, or in any of the names). "
        "Must have one name for each sample group if provided. ",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-q",
        "--quantile-cutoff",
        dest="quantile_cutoff",
        default=0.15,
        type=float,
        help="Quantile cutoff for converting gene expression values into gene weights. "
        "Genes with expression values above this cutoff will have a weight of 0, "
        "while genes below this cutoff will have a weight proportional to how "
        "far below the cutoff their expression (in terms of their quantile). "
        "This weight will be scaled to be between 0 and 1. For example, with a "
        "cutoff of 0.15, a gene whose expression is at the 0.15 quantile will have "
        "a weight of 0, a gene whose expression is at the 0.0 quantile will have a "
        "weight of 1.0, and a gene whose expression is at the 0.12 quantile will have"
        "a weight of 0.2.",
        required=False,
    )
    parser.add_argument(
        "--objective-tolerance",
        dest="objective_tolerance",
        default=5e-2,
        type=float,
        help="The tolerance for the objective value. The objective value for "
        "each metabolite will be constrained to be within "
        "objective-tolerance*objective-value of the unconstrained "
        "objective value. Defaults to 0.05.",
    )
    parser.add_argument(
        "--aggregation-method",
        dest="aggregation_method",
        default="median",
        help="Method used to aggregate sample groups into a single value for each gene. "
        "Can be 'median', 'mean', 'min', or 'max'. Defaults to median.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--transpose",
        dest="transpose",
        action="store_true",
        help="Specify that the gene expression input data is transposed from the default (i.e. "
        "this flag indicates the rows represent genes, and the columns samples)",
        required=False,
    )
    parser.add_argument(
        "--subset",
        dest="subset",
        action="store_true",
        help="Specify that the gene expression to gene weight conversion should only include the "
        "subset of genes found in the input model. This will calculate the quantiles based only "
        "on genes found in the model, rather than all genes present in the gene expression data.",
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Specify that a verbose output is desired",
        required=False,
    )
    parser.add_argument(
        "--extra-info",
        dest="extra_info",
        action="store_true",
        help="Specify that additional information from the model should be included in the "
        "metchange output file. Additional information will include the metabolite name, "
        "formula, compartment, and annotations.",
    )
    parser.add_argument(
        "--solver",
        dest="solver",
        default="glpk",
        help="Which solver to use for solving the metabolite optimizations in metchange. Can be "
        "'glpk', 'gurobi', or 'cplex'. Defaults to 'glpk'",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--seperator",
        dest="sep",
        default=",",
        help="Which seperator is used in the gene expression file (such as ',' for "
        "comma seperated files, or '\\t' for tab seperated files. Defaults to ','",
        required=False,
        type=str,
    )
    return parser.parse_args(arg_list)


def run(arg_list: list[str] | None = None) -> None:
    """
    Function to run the command line interface
    :param arg_list: List of command line arguments, defaults to those passed to stdin
    :type arg_list: list[str]|None
    :return: No return
    :rtype: None
    """
    # Parse args
    args = parse_args(arg_list)
    out_file = pathlib.Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # GPR function dict for metchange
    METCHANGE_FUNC_DICT = {"AND": max, "OR": min}
    METABOLITE_SEPERATOR = re.compile(r"[\s,;]+")
    if args.verbose:
        print("Reading input Model")
    # Set the default cobra solver
    if args.solver:
        cobra.core.Configuration().solver = args.solver
    # Read in the model
    in_model = metworkpy.read_model(args.model_file, file_type=args.model_format)
    if args.solver:
        in_model.solver = args.solver

    # parse metabolites
    if (not args.metabolites_list) and (not args.metabolites_file):
        metabolites = in_model.metabolites.list_attr("id")
    elif args.metabolites_list:
        metabolites = re.split(METABOLITE_SEPERATOR, args.metabolites_list)
    elif args.metabolites_file:
        with open(args.metabolites_file, "r") as f:
            metabolites = [
                i.strip() for i in re.split(METABOLITE_SEPERATOR, f.read()) if i.strip()
            ]
    else:
        raise ValueError(
            "Couldn't understand --metabolites, or --metabolites-file. "
            "For the --metabolites argument pass a list of metabolite ids "
            "separated by commas, for the --metabolites-file argument, "
            "should be the path to a text file containing a list of metabolites "
            "seperated by newlines, semicolons, or commas."
        )
    # Read in the gene expression data
    if args.verbose:
        print("Reading in gene expression data")
    gene_expression = pd.read_csv(
        args.gene_expression_file, index_col=0, header=0, sep=args.sep
    )
    # Transpose if needed
    if args.transpose:
        gene_expression = gene_expression.transpose
    if args.subset:
        subset = in_model.genes.list_attr("id")
    else:
        subset = None
    if args.aggregation_method:
        aggregator = _parse_aggregation_method(args.aggregation_method)
    else:
        aggregator = np.median
    # Process the wildtype gene expression
    if args.verbose:
        print("Converting Wildtype gene expression to reaction weights")
    wt_samples = _parse_samples(args.wildtype)
    wt_rxn_weights = []
    for s in wt_samples:
        wt_gene_weights = metworkpy.utils.expr_to_metchange_gene_weights(
            expression=gene_expression.iloc[s],
            quantile_cutoff=args.quantile_cutoff,
            subset=subset,
            aggregator=aggregator,
            sample_axis=0,
        )
        wt_rxn_weights.append(
            metworkpy.gpr.gene_to_rxn_weights(
                model=in_model,
                gene_weights=wt_gene_weights,
                fn_dict=METCHANGE_FUNC_DICT,
            )
        )
    # Process the other samples
    if args.verbose:
        print("Converting remaining samples gene expression to reaction weights")
    if args.sample_groups:
        sample_groups, sample_names = _parse_sample_groups_and_names(
            args.sample_groups, args.sample_group_names
        )
    else:
        sample_groups = [
            [i] for i in range(len(gene_expression.index)) if i not in wt_samples
        ]
        sample_names = list(
            gene_expression.index[list(itertools.chain.from_iterable(sample_groups))]
        )
    sample_rxn_weights_list = []
    for sample_group in sample_groups:
        expr = gene_expression.iloc[sample_group]
        weights = metworkpy.utils.expr_to_metchange_gene_weights(
            expression=expr,
            quantile_cutoff=args.quantile_cutoff,
            subset=subset,
            aggregator=aggregator,
            sample_axis=0,
        )
        sample_rxn_weights_list.append(
            metworkpy.gene_to_rxn_weights(
                model=in_model, gene_weights=weights, fn_dict=METCHANGE_FUNC_DICT
            )
        )
    # Run the Metchange algorithm
    # For the wildtype
    if args.verbose:
        print("Running Metchange algorithm for wildtype samples")
    wt_metchange = pd.DataFrame(
        0.0, index=range(len(wt_rxn_weights)), columns=metabolites
    )
    for idx, wt_weights in enumerate(wt_rxn_weights):
        res = metworkpy.metabolites.metchange(
            model=in_model.copy(),
            reaction_weights=wt_weights,
            metabolites=metabolites,
            objective_tolerance=args.objective_tolerance,
            progress_bar=args.verbose,
        )
        wt_metchange.loc[idx, res.index] = res
    # For the remaining samples
    wt_mean = wt_metchange.mean(axis=0)
    wt_std = wt_metchange.std(axis=0)
    metchange_res = pd.DataFrame(0.0, index=sample_names, columns=metabolites)
    for name, weights in zip(sample_names, sample_rxn_weights_list):
        res = (
            metworkpy.metchange(
                model=in_model,
                reaction_weights=weights,
                metabolites=metabolites,
                objective_tolerance=args.objective_tolerance,
                progress_bar=args.verbose,
            )
            - wt_mean
        ) / wt_std
        metchange_res.loc[name, res.index] = res
    metchange_res = metchange_res.transpose()
    if not args.extra_info:
        metchange_res.to_csv(out_file)
        return None
    # Add additional information
    name = []
    formula = []
    compartment = []
    charge = []
    for met_id in metchange_res.index:
        metabolite = in_model.metabolites.get_by_id(met_id)
        name.append(metabolite.name)
        formula.append(metabolite.formula)
        compartment.append(metabolite.compartment)
        charge.append(metabolite.charge)
    metchange_res["name"] = name
    metchange_res["formula"] = formula
    metchange_res["compartment"] = compartment
    metchange_res["charge"] = charge
    metchange_res.to_csv(out_file)
    return None


if __name__ == "__main__":
    run()

"""
Script for generating IMAT models from the command line
"""

# Imports
# Standard Library Imports
from __future__ import annotations
import argparse
import os.path
import pathlib
import sys

# External Imports
import cobra.core
import numpy as np
import pandas as pd

# Local Imports
import metworkpy
from metworkpy.scripts._script_utils import (
    _parse_samples,
    _parse_quantile,
    _parse_aggregation_method,
    _parse_sample_groups_and_names,
    _parse_format_to_extension,
)


# region Subcommand Functions
def run_single(args: argparse.Namespace):
    if args.verbose:
        print("Reading input Model")
    config = cobra.core.Configuration()
    config.solver = args.solver
    if args.processes is not None:
        config.processes = args.processes
    # Read in the model
    in_model = metworkpy.read_model(args.model_file, file_type=args.model_format)
    # Set solver to desired solver (cplex, gurobi, glpk)
    # Gurobi and cplex are much faster than glpk, but require licences.
    # GLPK is the default since it is installed automatically alongside cobra,
    # so should always be present
    if args.solver:
        in_model.solver = args.solver
    # Read gene expression data
    if args.verbose:
        print("Reading gene expression data")
    gene_expression = pd.read_csv(
        args.gene_expression_file, index_col=0, header=0, sep=args.sep
    )
    # Transpose if needed
    if args.transpose:
        gene_expression = gene_expression.transpose()
    # Filter for only the samples of interest
    if args.samples:
        gene_expression = gene_expression.iloc[_parse_samples(args.samples)]
    # Convert gene expression to qualitative weights (i.e. -1, 0, 1)
    if args.verbose:
        print("Converting gene expression into gene weights")
    if args.subset:
        subset = in_model.genes.list_attr("id")
    else:
        subset = None
    gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
        expression=gene_expression,
        quantile=_parse_quantile(args.quantile),
        aggregator=_parse_aggregation_method(args.aggregation_method),
        subset=subset,
        sample_axis=0,
    )
    # Convert Gene Weights to reaction weights
    if args.verbose:
        print("Converting gene weights into reaction weights")
    rxn_weights = metworkpy.gpr.gene_to_rxn_weights(
        model=in_model,
        gene_weights=gene_weights,
        fn_dict={"AND": min, "OR": max},
        fill_val=0,
    )
    # Generate IMAT model
    if args.verbose:
        print("Generating IMAT model")
    method = metworkpy.imat.model_creation._parse_method(args.method)
    if method == "imat_constraint":
        out_model = metworkpy.imat.model_creation.imat_constraint_model(
            in_model,
            rxn_weights,
            args.epsilon,
            args.threshold,
            args.objective_tolerance,
        )
    elif method == "simple_bounds":
        out_model = metworkpy.imat.model_creation.simple_bounds_model(
            in_model, rxn_weights, args.epsilon, args.threshold
        )
    elif method == "subset":
        out_model = metworkpy.imat.model_creation.subset_model(
            in_model, rxn_weights, args.epsilon, args.threshold
        )
    elif method == "fva":
        out_model = metworkpy.imat.model_creation.fva_model(
            in_model,
            rxn_weights,
            args.epsilon,
            args.threshold,
            args.objective_tolerance,
            args.loopless,
        )
    elif method == "milp":
        out_model = metworkpy.imat.model_creation.milp_model(
            in_model, rxn_weights, args.epsilon, args.threshold
        )
    else:
        raise ValueError(
            f"Invalid method: {method}. Valid methods are: 'simple_bounds', \
            'imat_restrictions', "
            f"'subset', 'fva', 'milp'."
        )
    if args.verbose:
        print("Writing IMAT model to file")
    metworkpy.write_model(
        model=out_model, model_path=args.output_file, file_type=args.output_format
    )


def run_multi(args: argparse.Namespace):
    if args.fold_change and not args.wildtype:
        print(
            "Wildtype argument must be provided to use the fold change method for gene weights. Exiting."
        )
        sys.exit()
    if args.wildtype:
        wildtype = _parse_samples(args.wildtype)
    else:
        wildtype = []
    # Set cobra processes argument if needed
    if args.processes:
        cobra.core.configuration.Configuration().processes = args.processes
    # Read in the model and expression data
    if args.verbose:
        print("Reading in Model")
    in_model = metworkpy.read_model(model_path=args.model_file)
    if args.verbose:
        print("Reading in gene expression data")
    gene_expression = pd.read_csv(args.gene_expression_file, index_col=0, header=0)
    if args.transpose:
        gene_expression = gene_expression.transpose()
    # Deal with sample groups and sample group names
    if args.sample_groups:
        sample_groups, sample_names = _parse_sample_groups_and_names(
            args.sample_groups, args.sample_group_names
        )
    else:
        sample_groups = [
            [idx] for idx, _ in enumerate(gene_expression.index) if idx not in wildtype
        ]
        sample_names = [f"s{i}" for i in range(1, len(sample_groups) + 1)]
    # Converting gene expression to reaction weights
    if args.aggregation_method:
        aggregator = _parse_aggregation_method(args.aggregation_method)
    else:
        aggregator = np.median
    if args.subset:
        subset = in_model.genes.list_attr("id")
    else:
        subset = None
    rxn_weights = {}
    if args.verbose:
        print("Converting gene expression values to reaction weights")
    if not args.fold_change:
        # Use the quantile method to create the reaction weights for each sample group
        for group, name in zip(sample_groups, sample_names):
            expr = gene_expression.iloc[group, :]
            g_weights = metworkpy.utils.expr_to_imat_gene_weights(
                expression=expr,
                quantile=_parse_quantile(args.quantile),
                aggregator=aggregator,
                subset=subset,
                sample_axis=0,
            )
            rxn_weights[name] = metworkpy.gpr.gene_to_rxn_weights(
                model=in_model,
                gene_weights=g_weights,
                fn_dict={"AND": min, "OR": max},
                fill_val=0,
            )
    else:
        fc = args.fold_change
        wt_expr = gene_expression.iloc[wildtype, :].apply(aggregator, axis=0)
        for group, name in zip(sample_groups, sample_names):
            log2fc = np.log2(
                gene_expression.iloc[group, :].apply(aggregator, axis=0).div(wt_expr)
                + 1e-10
            )
            g_weights = log2fc.apply(
                lambda x: -1 if x <= -fc else (1 if x >= fc else 0)
            )
            rxn_weights[name] = metworkpy.gpr.gene_to_rxn_weights(
                model=in_model,
                gene_weights=g_weights,
                fn_dict={"AND": min, "OR": max},
                fill_val=0,
            )
    if args.verbose:
        print("Generating IMAT Models")
    # Handle output
    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir)
    else:
        out_dir = pathlib.Path(args.output_dir)
    if not os.path.exists(out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
    if args.prefix:
        prefix = args.prefix + "_"
    else:
        prefix = ""
    for name, r_weights in rxn_weights.items():
        if args.verbose:
            print(f"Generating IMAT model for {name}")
        filename = f"{prefix}{name}.{_parse_format_to_extension(args.output_format)}"
        output_path = out_dir / filename
        method = metworkpy.imat.model_creation._parse_method(args.method)
        if method == "imat_constraint":
            out_model = metworkpy.imat.model_creation.imat_constraint_model(
                model=in_model,
                rxn_weights=r_weights,
                epsilon=args.epsilon,
                threshold=args.threshold,
                objective_tolerance=args.objective_tolerance,
            )
        elif method == "simple_bounds":
            out_model = metworkpy.imat.model_creation.simple_bounds_model(
                model=in_model,
                rxn_weights=r_weights,
                epsilon=args.epsilon,
                threshold=args.threshold,
            )
        elif method == "subset":
            out_model = metworkpy.imat.model_creation.subset_model(
                model=in_model,
                rxn_weights=r_weights,
                epsilon=args.epsilon,
                threshold=args.threshold,
            )
        elif method == "fva":
            out_model = metworkpy.imat.model_creation.fva_model(
                model=in_model,
                rxn_weights=r_weights,
                epsilon=args.epsilon,
                threshold=args.threshold,
                objective_tolerance=args.objective_tolerance,
                loopless=args.loopless,
            )
        elif method == "milp":
            out_model = metworkpy.imat.model_creation.milp_model(
                model=in_model,
                rxn_weights=r_weights,
                epsilon=args.epsilon,
                threshold=args.threshold,
            )
        else:
            raise ValueError(
                f"Invalid method: {args.method}. Valid methods are 'simple_bounds', 'imat_restrictions', "
                f"'subset', 'fva', or 'milp'"
            )
        if args.verbose:
            print(f"Writing IMAT model generated for {name} to output file")
        metworkpy.write_model(
            model=out_model, model_path=output_path, file_type=args.output_format
        )


# endregion Subcommand Functions

# region Parse Arguments


def parse_args(arg_list: list[str] | None) -> argparse.Namespace:
    """
    Parse Command Line Arguments

    :param arg_list: List of command line strings (defaults to reading from stdin)
    :type arg_list: list[str]|None
    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    # Top level parser
    toplevel_parser = argparse.ArgumentParser(
        prog="imatgen",
        description="Generate IMAT models from gene expression data",
    )

    # Create subparsers
    subparsers = toplevel_parser.add_subparsers(
        required=True,
        title="Modes",
        description="Choose a mode for imatgen",
        help="Choose between single (used for generating a single IMAT "
        "model), and multi (used for generating more than one IMAT model, "
        "or generating a single model based on fold-changes in gene "
        "expression).",
    )
    # Single parser
    single_parser = subparsers.add_parser(
        "single",
        help="Generate a single IMAT model from gene expression data "
        "using quantile cutoffs.",
    )
    single_parser.set_defaults(func=run_single)

    _add_common_args(single_parser)

    # region Single Args
    single_parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        default="imat_model.json",
        help="Path to output the generated IMAT model file. Will output to"
        "imat_model.json in current directory if not specified",
        required=False,
        type=str,
    )
    single_parser.add_argument(
        "-s",
        "--samples",
        dest="samples",  ### UNIQUE
        default=None,
        help="Which samples from the gene expression data "
        "should be used to generate the IMAT model. These"
        "samples will be aggregated (aggregation method can be selected "
        "using the --aggregate-method argument), and then used to "
        "compute the gene expression weights used in IMAT. Should be "
        "a set of numbers, seperated by commas (no spaces) that represent "
        "the 0-indexed rows (or columns if --transpose flag is used) for the "
        "samples of interest. Colons can be used to specify an "
        "inclusive range "
        "of values. For example '1,2:5,7' will specify rows '1,2,3,4,5,7'.",
        type=str,
    )

    # endregion Single Args

    # Multi parser
    multi_parser = subparsers.add_parser(
        "multi",
        help="Generate multiple IMAT models, or use fold-change "
        "gene expression to generate a single model.",
    )
    multi_parser.set_defaults(func=run_multi)

    _add_common_args(multi_parser)

    # region Multi Args
    multi_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        required=False,
        default="imatgen_results",
        help="Path to output directory for results models (defaults to "
        "'imatgen_results' in working directory. In this directory will "
        "be the output model files with names <prefix->sample-group-name.<ext> "
        "if sample group names are provided, or s1.<ext>, s2.<ext>,... "
        "will be used if no sample names provided.",
        type=str,
    )
    multi_parser.add_argument(
        "-p",
        "--prefix",
        dest="prefix",
        required=False,
        default=None,
        help="Prefix for the output model files. Output model file names "
        "will include the prefix before the group names.",
    )
    multi_parser.add_argument(
        "-s",
        "--sample-groups",
        type=str,
        dest="sample_groups",
        default=None,
        help="Groups of samples representing biological replicates. Samples within a group "
        "will be aggregated using 'aggregation-method'. Should be lists of comma seperated "
        "row numbers (0-indexed), seperated by parentheses (no spaces, optional commas), "
        "wrapped in quotes. "
        'Colons can be used to describe inclusive ranges. For example "(1:3,4)(5,6)" '
        "would "
        "represent two groups of samples, '1,2,3,4' and '5,6'. Defaults to every sample "
        "being its own group.",
        required=False,
    )
    multi_parser.add_argument(
        "--fold-change",
        type=float,
        dest="fold_change",
        default=None,
        help="Use fold change from wild-type condition rather than the quantile cutoff method "
        "for determining gene weights. This value represents the log2(fold-change) beyond "
        "which a change in gene expression will be giving a weight. For example, for "
        "a value of 1, all genes whose expression is 2-fold above or below wildtype "
        "(so a doubling, or a halving) will be given weights. All genes with expression "
        "doubled relative to wildtype will have gene weights of 1, all genes with "
        "expression levels below half of wildtype will have gene weights of -1.",
        required=False,
    )
    multi_parser.add_argument(
        "-w",
        "--wildtype",
        type=str,
        dest="wildtype",
        default=None,
        help="When using the fold change method to determine gene weights, which samples "
        "are wildtype (or negative control). Should be a comma seperated list of "
        "row numbers (0-indexed). Colons can be used to describe inclusive ranges, so"
        "0:2,4,7 will indicate that rows 0,1,2,4,7 are all wildtype samples.",
        required=False,
    )
    multi_parser.add_argument(
        "--sample-names",
        type=str,
        dest="sample_groups_names",
        default=None,
        help="Names to give the sample groups. If provided, should include one name for "
        "each sample group in a comma seperated list (no spaces in the list, or in the "
        "names). Defaults to using s followed by integer indices to refer to the sample "
        "groups starting from 1 (so three sample groups would be referred to as s1,s2,s3). "
        "Names will be included in the output model file name.",
        required=False,
    )

    # endregion Multi Args
    return toplevel_parser.parse_args(arg_list)


# endregion Parse Arguments


# region Main function
def main_run(arg_list: list[str] | None = None) -> None:
    """
    Function to run the command line interface

    :param arg_list: A list of arguments (defaults to reading the stdin)
    :type arg_list: list[str]|None
    :return: None
    :rtype: None
    """
    args = parse_args(arg_list)
    args.func(args)


if __name__ == "__main__":
    main_run()


# endregion Main Function


# region Helper Functions


def _add_common_args(parser: argparse.ArgumentParser):
    # region Common Args
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
        help="Path to normalized gene expression file (csv format). Columns should "
        "represent genes "
        "with the first row being the gene id, matching the gene ids in the "
        "cobra model. Rows should represent samples, with the first column being the "
        "sample name. The --transpose argument can be used to specify the orientation "
        "if other than the default. Data should be read depth and length normalized "
        "such as TPM, or RPKM.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        default="subset",
        help="Method used to generate the IMAT model, can be "
        "one of the following: subset, fva, milp, imat-restrictions, simple. "
        "Defaults to subset.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        dest="epsilon",
        default=1.0,
        help="Cutoff, above which a reaction is considered active",
        required=False,
        type=float,
    )
    parser.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        default=0.001,
        help="Cutoff, below which a reaction is considered inactive",
        required=False,
        type=float,
    )
    parser.add_argument(
        "-T",
        "--objective-tolerance",
        dest="objective_tolerance",
        default=5e-2,
        help="The tolerance for the objective value, "
        "(used for imat-restrictions and fva methods). The objective "
        "value will be constrained to be within objective-tolerance*objective-value of "
        "the unconstrained objective value. Defaults to 0.05.",
        type=float,
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
        "--output-format",
        dest="output_format",
        default="json",
        help="The format of the output model file ("
        "can be json, yaml, or sbml). If not provided, "
        "it will default to json.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--aggregation-method",
        dest="aggregation_method",
        default="median",
        help="Method used to aggregate multiple samples from "
        "biological replicates into a single value for each gene. "
        "Can be median, mean, min, max. Defaults to median. ",
        type=str,
    )
    parser.add_argument(
        "--quantile",
        dest="quantile",  ### UNIQUE
        default="0.15",
        help="Quantile for determining which genes are highly expressed, and lowly "
        "expressed. Can either be a single number such as 0.15, or two numbers"
        "seperated by a comma (no spaces). If a single number, represents the "
        "quantile cutoff where genes in the bottom quantile will be considered "
        "lowly expressed, and genes in the top quantile will be considered highly "
        "expressed. So a value of 0.15, will indicate the bottom 15 percent will "
        "be "
        "considered lowly expressed, and the top 15 percent of genes will be "
        "considered highly expressed. If two numbers, represent the bottom and "
        "top quantiles desired, so 0.15,0.90 will indicate that the bottom 15 "
        "percent of "
        "genes will be considered lowly expressed, and the top 10 percent of "
        "genes will "
        "be considered highly expressed. Defaults to 0.15",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--transpose",
        dest="transpose",
        action="store_true",
        help="Specify that the gene expression "
        "input data is transposed from the "
        "default (i.e. this flag indicates that the rows represent "
        "genes, and the columns represent "
        "samples)",
    )
    parser.add_argument(
        "--subset",
        dest="subset",
        action="store_true",
        help="Specify that the gene expression to gene weight "
        "conversion should only include the subset of genes "
        "found in the input model. This will calculate the "
        "quantiles "
        "based only on the genes found in the model, rather than "
        "all "
        "genes present in the gene expression data.",
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Specify that verbose output is desired",
        required=False,
    )
    parser.add_argument(
        "--solver",
        dest="solver",
        default="glpk",
        help="Which solver to use for solving the IMAT optimazation problem. "
        "Can be 'glpk', 'cplex', or 'gurobi'. Defaults to glpk.",
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
    )
    parser.add_argument(
        "--loopless",
        dest="loopless",
        action="store_true",
        help="Whether the FVA method should perform loopless FVA. "
        "Ignored if method is not fva. Takes significantly longer "
        "than non-loopless fva.",
    )
    parser.add_argument(
        "--processes",
        dest="processes",
        default=None,
        help="How many processes should be used for performing the calculations associated "
        "with model generation. Defaults to cobrapy default, currently all hyperthreads "
        "minus 1."
        " Only impacts FVA currently.",
        required=False,
        type=int,
    )
    # endregion Common Args


# endregion Helper Functions

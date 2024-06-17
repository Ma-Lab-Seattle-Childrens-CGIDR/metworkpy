"""
Script for generating IMAT models from the command line
"""
# Imports
# Standard Library Imports
from __future__ import annotations
import argparse

import cobra.core
# External Imports
import pandas as pd

# Local Imports
import metworkpy
from metworkpy.scripts._script_utils import _parse_samples, _parse_quantile, _parse_aggregation_method


# region Parse Arguments
def parse_args(arg_list:list[str]|None) -> argparse.Namespace:
    """
    Parse command line arguments

    :param arg_list: List of command line strings (defaults to reading from stin)
    :type arg_list: list[str]|None
    :return: parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="imatgen",
        description="Generate an IMAT model from gene expression data"
    )
    parser.add_argument("-M", "--model",
                        dest="model_file", default=None,
                        help="Path to cobra model file (json, sbml, yaml)",
                        required=True)
    parser.add_argument("-g", "--gene-expression",
                        dest="gene_expression_file", default=None,
                        help="Path to normalized gene expression file (csv format). Columns should represent genes "
                             "with the first row being the gene id, matching the gene ids in the "
                             "cobra model. Rows should represent samples, with the first column being the "
                             "sample name. The --transpose argument can be used to specify the orientation "
                             "if other than the default. Multiple samples will be aggregated, so if there "
                             "are samples from different biological replicates, specify which samples should "
                             "be used using the --samples argument. Data should be read depth and length normalized "
                             "such as TPM, or RPKM.",
                        required=True)
    parser.add_argument("-o", "--output",
                        dest="output_file", default="imat_model.json",
                        help="Path to output the generated IMAT model file. Will output to"
                             "imat_model.json in current directory if not specified",
                        required=False)
    parser.add_argument("-m", "--method",
                        dest="method", default="subset",
                        help="Method used to generate the IMAT model, can be "
                             "one of the following: subset, fva, milp, imat-restrictions, simple. "
                             "Defaults to subset.",
                        required=False)
    parser.add_argument("-e", "--epsilon",
                        dest="epsilon", default=1.,
                        help="Cutoff, above which a reaction is considered active",
                        required=False)
    parser.add_argument("-t", "--threshold",
                        dest="threshold", default=0.001,
                        help="Cutoff, below which a reaction is considered inactive",
                        required=False, type=float)
    parser.add_argument("-T", "--objective-tolerance",
                        dest="objective_tolerance", default=5e-2,
                        help="The tolerance for the objective value, "
                             "(used for imat-restrictions and fva methods). The objective "
                             "value will be constrained to be within objective-tolerance*objective-value of the "
                             "unconstrained objective value. Defaults to 0.05.", type=float)
    parser.add_argument("-f", "--model-format",
                        dest="model_format", default=None,
                        help="The format of the input model file ("
                             "can be json, yaml, or sbml). If not provided, "
                             "it will be inferred from the models file extension.",
                        required=False)
    parser.add_argument("--output-format", dest="output_format",
                        default="json", help="The format of the output model file ("
                                             "can be json, yaml, or sbml). If not provided, "
                                             "it will default to json.",
                        required=False)
    parser.add_argument("-s", "--samples", dest="samples",
                        default=None, help="Which samples from the gene expression data "
                                           "should be used to generate the IMAT model. These"
                                           "samples will be aggregated (aggregation method can be selected "
                                           "using the --aggregate-method argument), and then used to "
                                           "compute the gene expression weights used in IMAT. Should be "
                                           "a set of numbers, seperated by commas (no spaces) that represent "
                                           "the 0-indexed rows (or columns if --transpose flag is used) for the "
                                           "samples of interest. Colons can be used to specify an inclusive range "
                                           "of values. For example '1,2:5,7' will specify rows '1,2,3,4,5,7'.",
                        type=str)
    parser.add_argument("--aggregation-method", dest="aggregation_method",
                        default="median", help="Method used to aggregate multiple samples from "
                                               "biological replicates into a single value for each gene. "
                                               "Can be median, mean, min, max. Defaults to median. ")
    parser.add_argument("--transpose", dest="transpose",
                        action="store_true", help="Specify that the gene expression "
                                                  "input data is transposed from the "
                                                  "default (i.e. this flag indicates that the rows represent "
                                                  "genes, and the columns represent "
                                                  "samples)")
    parser.add_argument("--quantile", dest="quantile",
                        default="0.15", help="Quantile for determining which genes are highly expressed, and lowly "
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
                        required=False, type=str)
    parser.add_argument("--subset", dest="subset",
                        action="store_true", help="Specify that the gene expression to gene weight "
                                                  "conversion should only include the subset of genes "
                                                  "found in the input model. This will calculate the quantiles "
                                                  "based only on the genes found in the model, rather than all "
                                                  "genes present in the gene expression data.",
                        required=False)
    parser.add_argument("-v", "--verbose", dest="verbose",
                        action="store_true", help="Specify that verbose output is desired",
                        required=False)
    parser.add_argument("--solver", dest="solver",
                        default="glpk", help="Which solver to use for solving the IMAT optimazation problem. "
                                             "Can be 'glpk', 'cplex', or 'gurobi'. Defaults to glpk.",
                        required=False)
    parser.add_argument("--seperator", dest="sep",
                        default=",", help="Which seperator is used in the gene expression file (such as ',' for "
                                          "comma seperated files, or '\\t' for tab seperated files. Defaults to ','",
                        required=False)
    parser.add_argument("--loopless", dest="loopless",
                        action="store_true", help="Whether the FVA method should perform loopless FVA. "
                                                  "Ignored if method is not fva. Takes significantly longer "
                                                  "than non-loopless fva.")
    parser.add_argument("--processes", dest="processes", default=None,
                        help="How many processes should be used for performing the calculations associated "
                             "with model generation. Defaults to cobrapy default, currently all hyperthreads minus 1."
                             " Only impacts FVA currently.", required=False)
    return parser.parse_args(arg_list)


# endregion Parse Arguments

# region Main Function
def run(arg_list: list[str]|None=None) -> None:
    """
    Function to run the command line interface
    """
    args = parse_args(arg_list)
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
    gene_expression = pd.read_csv(args.gene_expression_file, index_col=0, header=0, sep=args.sep)
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
    gene_weights = metworkpy.utils.expr_to_imat_gene_weights(expression=gene_expression,
                                                             quantile=_parse_quantile(args.quantile),
                                                             aggregator=_parse_aggregation_method(args.aggregation_method),
                                                             subset=subset,
                                                             sample_axis=0)
    # Convert Gene Weights to reaction weights
    if args.verbose:
        print("Converting gene weights into reaction weights")
    rxn_weights = metworkpy.gpr.gene_to_rxn_weights(model=in_model,
                                                    gene_weights=gene_weights,
                                                    fn_dict={"AND": min, "OR": max},
                                                    fill_val=0)
    # Generate IMAT model
    if args.verbose:
        print("Generating IMAT model")
    method = metworkpy.imat.model_creation._parse_method(args.method)
    if method == "imat_constraint":
        out_model = metworkpy.imat.model_creation.imat_constraint_model(
            in_model, rxn_weights, args.epsilon, args.threshold, args.objective_tolerance
        )
    elif method == "simple_bounds":
        out_model = metworkpy.imat.model_creation.simple_bounds_model(in_model, rxn_weights, args.epsilon,
                                                                      args.threshold)
    elif method == "subset":
        out_model = metworkpy.imat.model_creation.subset_model(in_model, rxn_weights, args.epsilon, args.threshold)
    elif method == "fva":
        out_model = metworkpy.imat.model_creation.fva_model(
            in_model, rxn_weights, args.epsilon, args.threshold, args.objective_tolerance
        )
    elif method == "milp":
        out_model = metworkpy.imat.model_creation.milp_model(in_model, rxn_weights, args.epsilon, args.threshold)
    else:
        raise ValueError(
            f"Invalid method: {method}. Valid methods are: 'simple_bounds', \
            'imat_restrictions', "
            f"'subset', 'fva', 'milp'."
        )
    if args.verbose:
        print("Writing IMAT model to file")
    metworkpy.write_model(model=out_model,
                          model_path=args.output_file,
                          file_type=args.output_format)


if __name__ == "__main__":
    run()


# endregion Main Function

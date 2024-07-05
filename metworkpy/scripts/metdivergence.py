"""
Script for calculating divergence between flux distributions
"""

# Imports
# Standard Library Imports
from __future__ import annotations
import argparse
import importlib.util
import sys
import warnings

# External imports
import pandas as pd
from tqdm import tqdm

# Local Imports
import metworkpy
from metworkpy.utils._arguments import _parse_metric, _parse_str_args_dict


# region Parse Arguments
def parse_args(arg_list: list[str] | None) -> argparse.Namespace:
    """
    Parse command line arguments

    :param arg_list: List of command line strings (defaults to reading from stdin)
    :type arg_list: list[str] | None
    :return: Parsed Arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="metdivergence", description="Calculate divergence between flux samples"
    )
    parser.add_argument(
        "-d",
        "--treatment-distribution",
        dest="treatment_distribution_file",
        required=True,
        help="Path to file containing flux sample results. Should have columns representing "
        "reactions, and rows representing samples. Should NOT include an index column.",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--wildtype-distribution",
        dest="wildtype_distribution_file",
        required=True,
        help="Path to file containing flux sample results for wildtype. Should have columns "
        "representing reactions, and rows representing samples. Should NOT include an index "
        "column.",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        dest="output_file",
        required=False,
        default="divergence_results.csv",
        type=str,
        help="Path to desired output file. Will be formatted as a csv. Defaults to a file in the "
        "current working directory named divergence_results.csv",
    )
    parser.add_argument(
        "-f",
        "--input-format",
        dest="input_format",
        required=False,
        default="csv",
        help="Format of the input flux distribution file. Can be csv, parquet, feather, json or excel. "
        "If excel, the sheet-name argument must also be provided.",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--neighbors",
        dest="neighbors",
        required=False,
        default=5,
        help="Number of neighbors to use for computing divergence. Should be an integer greater than "
        "1, default is 5.",
        type=int,
    )
    parser.add_argument(
        "-t",
        "--divergence-type",
        dest="divergence_type",
        required=False,
        default="js",
        help="Type of divergence to compute, either KL/Kullback-Leibler or JS/Jensen-Shannon",
        type=str,
    )
    parser.add_argument(
        "--metric",
        dest="metric",
        required=False,
        default="euclidean",
        help="Metric to use for computing distance during the divergence calculations, "
        "can be Euclidean, Manhattan, or Chebyshev. Defaults to euclidean",
        type=str,
    )
    parser.add_argument(
        "-M",
        "--model",
        dest="model_file",
        required=False,
        default=None,
        help="Path to cobra model used to add additional information to the results, "
        "including reaction name, "
        "gene association, subsystem, etc. If not provided, no additional information will be "
        "added to the divergence results. If this additional information is desired, the input "
        "distributions column names must reflect the reaction ids in the models.",
        type=str,
    )
    parser.add_argument(
        "--model-format",
        dest="model_format",
        required=False,
        default=None,
        help="Format of the optional cobra model",
        type=str,
    )
    parser.add_argument(
        "--sheet-name",
        dest="sheet_name",
        required=False,
        default="Flux Samples",
        help="Sheet name of the flux samples if Excel is the input format.",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Flag for verbose output",
    )
    return parser.parse_args(arg_list)


# endregion Parse Arguments


# region Main Function
def main_run(arg_list: list[str] | None = None):
    args = parse_args(arg_list)
    metric = _parse_metric(args.metric)
    neighbors = args.neighbors
    div_type = _parse_div_type(args.divergence_type)
    if args.verbose:
        print("Reading in flux distributions")
    wt_dist, treatment_dist = _read_distributions(args)
    rxn_list = [rxn for rxn in treatment_dist.columns if rxn in wt_dist.columns]
    if (len(rxn_list) != len(wt_dist.columns)) or (
        len(rxn_list) != len(treatment_dist.columns)
    ):
        warnings.warn(
            "Some reactions are only present in one of the input distributions, "
            "using the intersection of the sets of reactions."
        )
    div_res = pd.DataFrame(0.0, index=rxn_list, columns=[f"{div_type}_divergence"])
    if args.verbose:
        print("Calculating divergence")
    if div_type == "js":
        div_func = metworkpy.divergence.js_divergence
    elif div_type == "kl":
        div_func = metworkpy.divergence.kl_divergence
    else:
        print(
            f"Couldn't parse Divergence type, must be kl, Kullback-Leibler, js, or Jensen-Shannon, but "
            f"received {div_type}",
            file=sys.stderr,
        )
        sys.exit()
    for rxn in tqdm(rxn_list, disable=(not args.verbose)):
        div_res.loc[rxn, f"{div_type}_divergence"] = div_func(
            p=wt_dist[rxn],
            q=treatment_dist[rxn],
            n_neighbors=neighbors,
            discrete=False,
            jitter=1e-10,
            jitter_seed=None,
            distance_metric=metric,
        )
    if args.model_file:
        if args.verbose:
            print("Reading in model file")
        model = metworkpy.read_model(args.model_file, file_type=args.model_format)
        rxn_names = []
        genes = []
        subsystems = []
        rxn_str = []
        if args.verbose:
            print("Adding information to the divergence")
        for rxn in div_res.index:
            reaction = model.reactions.get_by_id(rxn)
            rxn_names.append(reaction.name)
            genes.append(reaction.gene_name_reaction_rule)
            subsystems.append(reaction.subsystem)
            rxn_str.append(reaction.build_reaction_string(use_metabolite_names=True))
        div_res["Reaction Name"] = rxn_names
        div_res["Reaction Expression"] = rxn_str
        div_res["Subsystem"] = subsystems
        div_res["Genes"] = genes
    if args.verbose:
        print("Writing results to output file")
    div_res.to_csv(args.output_file)
    return None


def _read_distributions(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_format = _parse_format(args.input_format)
    if input_format == "csv":
        treatment_dist = pd.read_csv(
            args.treatment_distribution_file, index_col=None, header=0
        )
        wildtype_dist = pd.read_csv(
            args.wildtype_distribution_file, index_col=None, header=0
        )
    elif input_format == "parquet":
        treatment_dist = pd.read_parquet(args.treatment_distribution_file)
        wildtype_dist = pd.read_parquet(args.wildtype_distribution_file)
    elif input_format == "feather":
        treatment_dist = pd.read_feather(args.treatment_distribution_file)
        wildtype_dist = pd.read_feather(args.wildtype_distribution_file)
    elif input_format == "excel":
        if not importlib.util.find_spec("openpyxl"):
            print(
                "Input format is excel, but openpyxl is not installed so can't read the excel file. Exiting",
                file=sys.stderr,
            )
            sys.exit()
        treatment_dist = pd.read_excel(
            args.treatment_distribution_file,
            sheet_name=args.sheet_name,
            index_col=None,
            header=0,
        )
        wildtype_dist = pd.read_excel(
            args.wildtype_distribution_file,
            sheet_name=args.sheet_name,
            index_col=None,
            header=0,
        )
    elif input_format == "json":
        treatment_dist = pd.read_json(args.treatment_distribution_file)
        wildtype_dist = pd.read_json(args.wildtype_distribution_file)
    else:
        raise ValueError(
            f"Couldn't parse input file type, should be csv, parquet, feather, excel, or json, "
            f"but received {args.input_format}."
        )
    return wildtype_dist, treatment_dist


if __name__ == "__main__":
    main_run()


# endregion Main Function


# region Helper Functions
def _parse_div_type(metric_str: str) -> str:
    return _parse_str_args_dict(
        metric_str,
        {
            "kl": ["kullback-leibler", "kullback leibler", "kullback_leibler", "kl"],
            "js": ["jensen-shannon", "jensen shannon", "jensen_shannon", "js"],
        },
    )


def _parse_format(format_str: str) -> str:
    return _parse_str_args_dict(
        format_str,
        {
            "csv": [
                "csv",
                "comma-seperated-values",
                "comma_seperated_values",
                "comma seperated values",
            ],
            "parquet": ["parquet"],
            "feather": ["feather"],
            "excel": ["excel", "xlsx"],
            "json": ["json"],
        },
    )


# endregion Helper Functions

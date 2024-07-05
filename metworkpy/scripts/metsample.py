"""
Script for sampling cobra models
"""

# Imports
# Standard Library Imports
from __future__ import annotations
import argparse
import importlib.util
import math
import os.path
import pathlib
import sys

# External Imports
import cobra

# Local Imports
import metworkpy
from metworkpy.utils._arguments import _parse_str_args_dict


# region Parse Arguments
def parse_args(arg_list: list[str] | None) -> argparse.Namespace:
    """
    Parse command line arguments

    :param args_list: List of command line strings (defaults to reading from stdin)
    :type args_list: list[str] | None
    :return: Parsed Arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="metsample", description="Flux sample a Cobra metabolic model"
    )
    parser.add_argument(
        "-M",
        "--model",
        type=str,
        dest="model_file",
        required=True,
        help="Path to the input cobra model file (json, sbml, yaml)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        dest="num_samples",
        default=10_000,
        help="Number of samples to generate. Defaults to 10,000.",
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        default=pathlib.Path("./flux_sampling_results.csv"),
        type=str,
    )
    parser.add_argument(
        "-f",
        "--output-format",
        dest="output_format",
        default="csv",
        help="Format for output file, can be csv, parquet, feather, excel, or json. "
        "No index column will be included with any of the methods. Writing to "
        "excel requires openpyxl or xlxswriter to be installed.",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        default="optgp",
        help="Desired sampling method, can be achr or optgp (not case sensitive). "
        "Defaults to optgp",
        type=str,
    )
    parser.add_argument(
        "--model-format",
        dest="model_format",
        default=None,
        help="Optional input model format (json, yaml, sbml, or mat), indicating what "
        "format the model is in.",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--processes",
        default=1,
        dest="processes",
        help="Number of processes to use for sampling (only applied to optgp method). Defaults to "
        "using a single process.",
        type=int,
    )
    parser.add_argument(
        "-t",
        "--thinning",
        default=100,
        dest="thinning",
        help="Thinning factor of the sampling chain",
        type=int,
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        default=None,
        help="Desired random number seed (cobra initializes this to current time stamp if "
        "not provided)",
        type=int,
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        help="Whether to validate that the samples represent feasible solutions to the model. "
        "Will only record valid samples if this flag is set. ",
    )
    parser.add_argument(
        "-b",
        "--batches",
        default=None,
        dest="batches",
        help="Number of batches to break up the sampling into. The default behavior is to "
        "sample all the points, and then write that out to the output file. This can take "
        "up a lot of ram, so this option allows for batches of samples to be taken, "
        "to reduce this issue. Batches are only allowed with csv and parquet file types. "
        "If samples is not a multiple of batches (so it can't be cleanly broken up into batches) "
        "the number of samples taken will be increased to the closest multiple of batches. "
        "Batch sampling with a parquet file requires fastparquet.",
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Whether a verbose output is desired.",
    )
    return parser.parse_args(arg_list)


# endregion Parse Arguments


# region Main Function
def main_run(arg_list: list[str] | None = None) -> None:
    """
    Function to run the command line interface

    :param arg_list: List of command line arguments, defaults to those passed to stdin
    :type arg_list: list[str]|None
    :return: No return
    :rtype: None
    """
    # Parse args
    args = parse_args(arg_list)
    method = _parse_method(args.method)
    verbose = args.verbose
    # Handle out_file path
    out_file = pathlib.Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # Read in the model
    if verbose:
        print("Reading in model")
    model = metworkpy.read_model(
        model_path=args.model_file, file_type=args.model_format
    )
    # Parse the output format
    out_format = _parse_format(args.output_format)
    # Check if batch and out format are compatible
    if args.batches and not (out_format == "csv" or out_format == "parquet"):
        print(
            f"If batch sampling desired, output format must be csv or parquet, but "
            f"{out_format} was selected instead",
            file=sys.stderr,
        )
        sys.exit()
    if args.batches and out_format == "parquet":
        if importlib.util.find_spec("fastparquet") is None:
            print(
                "Batch sampling with parquet output requires fastparquet, but it is not installed/n",
                file=sys.stderr,
            )
            sys.exit()
    if verbose:
        print("Creating Sampler")
    if method == "achr":
        sampler = cobra.sampling.achr.ACHRSampler(
            model=model, thinning=args.thinning, seed=args.seed
        )
    elif method == "optgp":
        sampler = cobra.sampling.optgp.OptGPSampler(
            model=model,
            thinning=args.thinning,
            processes=args.processes,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Method must be achr or optgp, but {method} was provided")
    if args.batches:
        if verbose:
            print("Starting Batch Sampling")
        _batch_sample(
            sampler=sampler,
            out_file=out_file,
            out_format=out_format,
            batches=args.batches,
            samples=args.num_samples,
            validate=args.validate,
        )
        if verbose:
            print("Finished sampling")
    else:
        if verbose:
            print("Starting Sampling")
        samples = sampler.sample(args.num_samples)
        if verbose:
            print("Finished Sampling, writing output")
        if args.validate:
            valid = sampler.validate(samples)  # Samples is a DataFrame, not a matrix
            samples = samples.loc[valid == "v", :]
        if out_format == "csv":
            samples.to_csv(out_file, index=False)
        elif out_format == "parquet":
            samples.to_parquet(out_file, index=False)
        elif out_format == "feather":
            samples.to_feather(out_file)
        elif out_format == "excel":
            samples.to_excel(out_file, index=False, sheet_name="Flux Samples")
        elif out_format == "json":
            samples.to_json(out_file, index=False)
        if verbose:
            print("Finished writing output")


if __name__ == "__main__":
    main_run()

# endregion Main Function


# region Helper Functions
def _batch_sample(
    sampler: cobra.sampling.HRSampler,
    out_file: str | pathlib.Path,
    out_format: str,
    batches: int = 10,
    samples: int = 10_000,
    validate: bool = False,
) -> None:
    """
    Perform batch sampling

    :param sampler: Sampler object (either OptGpSampler, or ACHRSampler)
    :type sampler: cobra.sampling.HRSampler
    :param out_file: Path to output file
    :type out_file: str | pathlib.Path
    :param out_format: Format for output file, must be csv or parquet
    :type out_format: str
    :param batches: Number of batches to break the sampling up into
    :type batches: int
    :param samples: Total number of samples to generate, if not a multiple of batches, will be increased to the
        closest multiple of batches
    :type samples:
    :param validate:
    :type validate:
    :return:
    :rtype:
    """
    # Clear file, so that if you are writing to the same file as previous run it clears it rather than append
    if os.path.exists(out_file):
        os.remove(out_file)
    samples_per_batch = math.ceil(samples / batches)
    for sample in sampler.batch(samples_per_batch, batches):
        if validate:
            valid = sampler.validate(sample)
            # Filter for only valid samples
            sample = sample.loc[valid == "v", :]
        if out_format == "csv":
            if not os.path.exists(out_file):
                sample.to_csv(out_file, index=False, header=True)
            else:
                with open(out_file, "a") as f:
                    sample.to_csv(f, header=False, index=False)
        elif out_format == "parquet":
            if not os.path.exists(out_file):
                sample.to_parquet(out_file, index=False, engine="fastparquet")
            else:
                sample.to_parquet(
                    out_file, index=False, engine="fastparquet", append=True
                )
        else:
            raise ValueError(f"Invalid out_format: {out_format}")


def _parse_method(method_str: str) -> str:
    """
    Parse a method specification string to canonical method str

    :param method_str: String specifying desired method
    :type method_str: str
    :return: Method string, either optgp or achr
    :rtype: str
    """
    return _parse_str_args_dict(
        method_str,
        {
            "achr": [
                "achr",
                "automatic-centering-hit-and-run",
                "automatic_centering_hit_and_run",
                "automatic centering hit and run",
            ],
            "optgp": ["optgp"],
        },
    )


def _parse_format(format_str: str) -> str:
    """
    Parse format specification to a file format

    :param format_str: String specifying desired format
    :type format_str: str
    :return: Format string, either csv, parquet, feather, excel, or json
    :rtype: str
    """
    return _parse_str_args_dict(
        format_str,
        {
            "csv": [
                "csv",
                "comma seperated values",
                "comma-seperated-values",
                "comma_seperated_values",
            ],
            "parquet": ["parquet"],
            "feather": ["feather"],
            "excel": ["excel", "xlsx"],
            "json": ["json"],
        },
    )


# endregion Helper Functions

"""Determine the divergence in the network caused by a gene knock out"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Iterable, Optional, Union, cast
import warnings

# External Imports
import cobra  # type: ignore
from cobra.manipulation import knock_out_model_genes  # type: ignore
import numpy as np
import pandas as pd
import tqdm  # type: ignore

# Local Imports
from metworkpy.divergence.js_divergence_functions import js_divergence
from metworkpy.divergence.kl_divergence_functions import kl_divergence
from metworkpy.utils._arguments import _parse_str_args_dict


# region Main Function


# This is going to be a very slow function since it needs to do repeated flux sampling,
# to make it faster, it would be ideal to not have to repeatedly get warm up points
# But that would likely require modifying the cobra optgp sampling code
# ...which is probably not actually possible since a single gene change might influence many other limits
# there might be ways to adjust the sampling distribution but I'm not really sure...
def ko_divergence(
    model: cobra.Model,
    target_networks: list[str] | dict[str, list[str]],
    genes_to_ko: Optional[Iterable[str]] = None,
    divergence_metric: str = "Jensen-Shannon",
    n_neighbors: int = 5,
    sample_count: int = 1000,
    jitter: Optional[float] = None,
    jitter_seed: Optional[int] = None,
    distance_metric: Union[float, str] = "euclidean",
    progress_bar: bool = False,
    use_unperturbed_as_true: bool = True,
    sampler_seed: Optional[int | np.random.Generator] = None,
    **kwargs,
) -> pd.DataFrame:
    """Determine the impacts of gene knock-outs on different target reaction or gene networks

    Parameters
    ----------
    model : cobra.Model
        Base cobra model to test effects of gene knockouts on
    target_networks : list[str] | dict[str, list[str]]
        Target networks to investigate the impact of the gene knock-outs
        on. Can be a list or a dict of lists. If a dict, the keys will
        be used to name the network and the lists will specify the
        networks. If a list should be a single network. Entries in the
        lists can be either reaction or gene ids. Gene ids will be
        translated into reaction ids using the model. If a list is
        passed the name of the target network in the returned dataframe
        will be target_network, if a dict is passed the keys are used as
        the column names.
    genes_to_ko : Iterable[str], optional
        List of genes to investigate impact of their knock-out,
        defaults to all genes in the model
    divergence_metric : str
        Which metric to use for divergence, can be Jensen-Shannon, or
        Kullback-Leibler
    n_neighbors : int
        Number of neighbors to use when estimating divergence
    sample_count : int
        Number of samples to take when performing flux sampling (will be
        repeated for each gene knocked out)
    jitter : Union[None, float, tuple[float,float]]
        Amount of noise to add to avoid ties. If None no noise is added.
        If a float, that is the standard deviation of the random noise
        added to the continuous samples. If a tuple, the first element
        is the standard deviation of the noise added to the x array, the
        second element is the standard deviation added to the y array.
    jitter_seed : Union[None, int]
        Seed for the random number generator used for adding noise
    distance_metric : Union[str, float]
        Metric to use for computing distance between points in p and q,
        can be \"Euclidean\", \"Manhattan\", or \"Chebyshev\". Can also
        be a float representing the Minkowski p-norm.
    progress_bar : bool
        Whether a progress bar is desired
    use_unperturbed_as_true : bool, default=True
        Which distribution to use as the "True" distribution (the P distribution)
        when estimating divergence between the perturbed (that is the model with a gene knock-out)
        and the unperturbed (model prior to the gene knock-out) flux samples.
        Doesn't impact Jensen-Shannon as that is symetric, but will modify the
        Kullback-Leibler divergence.
    sampler_seed : None or int or np.Generator, optional
        Seed used for sampling in order to create reproducible results,
        can be a numpy generator (in which cae it is used directly),
        or an integer (in which case it is used to seed a numpy generator).
    **kwargs
        Arguments passed to the sample method of COBRApy, see `COBRApy
        Documentation <https://cobrapy.readthedocs.io/en/latest/autoapi/
        cobra/sampling/index.html#cobra.sampling.sample>`_

    Returns
    -------
    pd.DataFrame
        Dataframe with index of genes, and columns representing the
        different target networks. Values represent the divergence of a
        particular target network between the unperturbed model and the
        model following the gene knock-out.
    """
    if genes_to_ko is None:
        genes_to_ko = model.genes.list_attr("id")
    # Setup Random seeding for the sampling
    if isinstance(sampler_seed, int) or sampler_seed is None:
        rng = np.random.default_rng(sampler_seed)
    elif isinstance(sampler_seed, np.random.Generator):
        rng = sampler_seed
    else:
        raise ValueError(
            f"Seed must be int, numpy Generator, or None but received {type(sampler_seed)}"
        )
    divergence_metric = _parse_divergence_method(divergence_metric)
    if divergence_metric == "js":
        divergence_function = js_divergence
    elif divergence_metric == "kl":
        divergence_function = kl_divergence
    else:
        raise ValueError(
            f"Invalid specification for divergence metric, must be js or kl, but received {divergence_metric}"
        )
    ko_res_list = []
    unperturbed_sample = cobra.sampling.sample(
        model=model,
        n=sample_count,
        seed=rng.integers(low=0, high=np.iinfo(np.intp).max),
        **kwargs,
    )
    # If needed, convert the gene network into a dict
    if isinstance(target_networks, list):
        target_networks = {"target_network": target_networks}
    elif isinstance(target_networks, dict):
        pass
    else:
        raise ValueError(
            f"target_gene_network must be a list or a dict, but received a {type(target_networks)}"
        )

    for key, target_list in target_networks.items():
        target_networks[key] = _convert_target_network(model, target_list)
    for gene_to_ko in tqdm.tqdm(genes_to_ko, disable=not progress_bar):
        with model as ko_model:
            try:
                _ = knock_out_model_genes(ko_model, gene_list=[gene_to_ko])
                perturbed_sample = cobra.sampling.sample(
                    model=ko_model,
                    n=sample_count,
                    seed=rng.integers(low=0, high=np.iinfo(np.intp).max),
                    **kwargs,
                )
            except ValueError:
                # This can happen if the gene knock out causes all reactions to be 0. (or very close)
                # So continue, leaving that part of the results dataframe as all np.nan
                res_series = pd.Series(
                    np.nan, index=list(target_networks.keys())
                )
                res_series.name = gene_to_ko
                ko_res_list.append(res_series)
                continue
        res_series = pd.Series(np.nan, index=list(target_networks.keys()))
        for network, rxn_list in tqdm.tqdm(
            target_networks.items(), disable=not progress_bar, leave=False
        ):
            # NOTE: The Kullback-Leibler divergence is not symmetrical so the ordering here
            # can matter. Which distribution is assigned to P vs Q can be controlled with
            # the use_unperturbed_as_true flag, which defaults to True. If that flag is true,
            # the unperturbed_sample is used as the P distribution, and the perturbed sample is
            # used as the Q distrbution. Otherwise these are reversed.
            res_series[network] = divergence_function(
                p=(
                    unperturbed_sample[rxn_list]
                    if use_unperturbed_as_true
                    else perturbed_sample[rxn_list]
                ),
                q=(
                    unperturbed_sample[rxn_list]
                    if not use_unperturbed_as_true
                    else perturbed_sample[rxn_list]
                ),
                n_neighbors=n_neighbors,
                discrete=False,
                jitter=jitter,
                jitter_seed=jitter_seed,
                distance_metric=distance_metric,
            )
        res_series.name = gene_to_ko
        ko_res_list.append(res_series)
    return pd.concat(ko_res_list, axis=1).T


# endregion Main Function


# region helper functions
def _convert_target_network(
    model: cobra.Model, network: list[str]
) -> list[str]:
    """Converts gene/rxn networks into rxn networks only"""
    res_list = []
    reactions = set(model.reactions.list_attr("id"))
    for val in network:
        if val in reactions:
            res_list.append(val)
        else:
            try:
                gene: cobra.Gene = cast(cobra.Gene, model.genes.get_by_id(val))
                res_list += [r.id for r in gene.reactions]
            except KeyError:
                warnings.warn(
                    f"Couldn't find {val} in model genes or reactions, skipping"
                )
    return res_list


def _parse_divergence_method(method: str) -> str:
    return _parse_str_args_dict(
        method,
        {
            "js": [
                "js",
                "jensen-shannon-divergence",
                "jensen_shannon_divergence",
                "jensen shannon divergence",
            ],
            "kl": [
                "kl",
                "kullback–leibler-divergence",
                "kullback_leibler_divergence",
                "kullback leibler divergence",
            ],
        },
    )


# endregion helper functions

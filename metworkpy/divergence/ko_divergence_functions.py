"""
Determine the divergence in the network caused by a gene knock out
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Iterable, Optional, Union
import warnings

# External Imports
import cobra
from cobra.manipulation import knock_out_model_genes
import numpy as np
import pandas as pd
import tqdm

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
    genes_to_ko: Iterable[str],
    target_networks: list[str] | dict[str, list[str]],
    divergence_metric: str = "Jensen-Shannon",
    n_neighbors: int = 5,
    sample_count: int = 1000,
    jitter: Optional[float] = None,
    jitter_seed: Optional[int] = None,
    distance_metric: Union[float, str] = "euclidean",
    progress_bar: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Determine the impacts of gene knock-outs on different target reaction or gene networks

    :param model: Base cobra model to test effects of gene knockouts on
    :type model: cobra.Model
    :param genes_to_ko: List of genes to investigate impact of their knock-out
    :type genes_to_ko: Iterable[str]
    :param target_networks: Target networks to investigate the impact of the gene knock-outs on. Can be a list or a dict
        of lists. If a dict, the keys will be used to name the network and the lists will specify the networks. If a
        list should be a single network. Entries in the lists can be either reaction or gene ids. Gene ids will be
        translated into reaction ids using the model. If a list is passed
        the name of the target network in the returned dataframe will be target_network, if a dict is passed the
        keys are used as the column names.
    :type target_networks: list[str] | dict[str, list[str]]
    :param divergence_metric: Which metric to use for divergence, can be Jensen-Shannon, or Kullback-Leibler
    :type divergence_metric: str
    :param n_neighbors: Number of neighbors to use when estimating divergence
    :type n_neighbors: int
    :param sample_count: Number of samples to take when performing flux sampling (will be repeated for each
        gene knocked out)
    :type sample_count: int
    :param jitter: Amount of noise to add to avoid ties. If None no noise is added. If a float, that is the standard
        deviation of the random noise added to the continuous samples. If a tuple, the first element is the standard
        deviation of the noise added to the x array, the second element is the standard deviation added to the y array.
    :type jitter: Union[None, float, tuple[float,float]]
    :param jitter_seed: Seed for the random number generator used for adding noise
    :type jitter_seed: Union[None, int]
    :param distance_metric: Metric to use for computing distance between points in p and q, can be \"Euclidean\",
        \"Manhattan\", or \"Chebyshev\". Can also be a float representing the Minkowski p-norm.
    :type distance_metric: Union[str, float]
    :param progress_bar: Whether a progress bar is desired
    :type progress_bar: bool
    :param kwargs: Arguments passed to the sample method of COBRApy, see `COBRApy Documentation <https://cobrapy.readthedocs.io/en/latest/autoapi/cobra/sampling/index.html#cobra.sampling.sample>`_
    :return: Dataframe with index of genes, and columns representing the different target networks. Values
        represent the divergence of a particular target network between the unperturbed model and the model
        following the gene knock-out.
    :rtype: pd.DataFrame
    """
    divergence_metric = _parse_divergence_method(divergence_metric)
    ko_res_list = []
    unperturbed_sample = cobra.sampling.sample(model, sample_count, **kwargs)
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
                    ko_model, sample_count, **kwargs
                )
            except ValueError:
                # This can happen if the gene knock out causes all reactions to be 0. (or very close)
                # So continue, leaving that part of the results dataframe as all np.nan
                res_series = pd.Series(np.nan, index=list(target_networks.keys()))
                res_series.name = gene_to_ko
                ko_res_list.append(res_series)
                continue
        res_series = pd.Series(np.nan, index=list(target_networks.keys()))
        for network, rxn_list in tqdm.tqdm(
            target_networks.items(), disable=not progress_bar, leave=False
        ):
            if divergence_metric == "js":
                res_series[network] = js_divergence(
                    p=perturbed_sample[rxn_list],
                    q=unperturbed_sample[rxn_list],
                    n_neighbors=n_neighbors,
                    discrete=False,
                    jitter=jitter,
                    jitter_seed=jitter_seed,
                    distance_metric=distance_metric,
                )
            elif divergence_metric == "kl":
                res_series[network] = kl_divergence(
                    p=perturbed_sample[rxn_list],
                    q=unperturbed_sample[rxn_list],
                    n_neighbors=n_neighbors,
                    discrete=False,
                    jitter=jitter,
                    jitter_seed=jitter_seed,
                    distance_metric=distance_metric,
                )
            else:
                raise ValueError(
                    f"Invalid specification for divergence metric, must be js or kl, but received {divergence_metric}"
                )
        res_series.name = gene_to_ko
        ko_res_list.append(res_series)
    return pd.concat(ko_res_list, axis=1).T


# endregion Main Function


# region helper functions
def _convert_target_network(model: cobra.Model, network: list[str]) -> list[str]:
    """Converts gene/rxn networks into rxn networks only"""
    res_list = []
    reactions = set(model.reactions.list_attr("id"))
    for val in network:
        if val in reactions:
            res_list.append(val)
        else:
            try:
                gene = model.genes.get_by_id(val)
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

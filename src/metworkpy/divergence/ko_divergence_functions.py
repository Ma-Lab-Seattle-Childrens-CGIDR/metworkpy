"""Determine the divergence in the network caused by a gene knock out"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import cast, Any, Iterable, Literal, Optional, Union, Tuple
import warnings

# External Imports
import cobra  # type: ignore
from cobra.manipulation import knock_out_model_genes  # type: ignore
import numpy as np
import pandas as pd
import tqdm  # type: ignore

# Local Imports
from metworkpy.divergence.group_divergence import calculate_divergence_grouped


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
    divergence_type: Literal["js", "kl"] = "kl",
    calculate_pvalue: bool = False,
    sample_count: int = 1000,
    progress_bar: bool = False,
    use_unperturbed_as_true: bool = True,
    sampler_seed: Optional[int | np.random.Generator] = None,
    sampler_kwargs: Optional[dict[str, Any]] = None,
    processes: int = 1,
    **kwargs,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
    divergence_type : 'kl' or 'js', default='kl'
        Which metric to use for divergence, can be 'kl' for Kullback-Leibler (default)
        or 'js' for Jensen-Shannon,
    calculate_pvalue : bool, default=False
        Whether to calculate the significance value for the divergence
    sample_count : int
        The number of samples to take in order to estimate the divergence
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
    sampler_kwargs : dict of str to Any
        Arguments passed to the sample method of COBRApy, see `COBRApy
        Documentation <https://cobrapy.readthedocs.io/en/latest/autoapi/
        cobra/sampling/index.html#cobra.sampling.sample>`_
    processes : int, default=1
        Number of processes to use for this function, passed to the
        sampler and also used as the number of processes for calculating
        the divergence for the different groups. Note that if you want a different
        number of processes for the sampler, you can use the sampler_kwargs dictionary.
    **kwargs
        Keyword arguments passed to the divergence method

    Returns
    -------
    pd.DataFrame
        Dataframe with index of genes, and columns representing the
        different target networks. Values represent the divergence of a
        particular target network between the unperturbed model and the
        model following the gene knock-out.
    """
    if sampler_kwargs is None:
        sampler_kwargs = {"processes": processes}
    else:
        if "processes" not in sampler_kwargs:
            sampler_kwargs["processes"] = processes
    if genes_to_ko is None:
        genes_to_ko = model.genes.list_attr("id")
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
    # Setup Random seeding for the sampling
    if isinstance(sampler_seed, int) or sampler_seed is None:
        rng = np.random.default_rng(sampler_seed)
    elif isinstance(sampler_seed, np.random.Generator):
        rng = sampler_seed
    else:
        raise ValueError(
            f"Seed must be int, numpy Generator, or None but received {type(sampler_seed)}"
        )
    ko_div_df = pd.DataFrame(
        np.nan,
        index=pd.Index(genes_to_ko),
        columns=pd.Index(target_networks.keys()),
    )
    if calculate_pvalue:
        ko_pvalue_df = pd.DataFrame(
            np.nan,
            index=pd.Index(genes_to_ko),
            columns=pd.Index(target_networks.keys()),
        )
    # Add in any keyword arguments to the calculate divergence grouped function
    kwargs["calculate_pvalue"] = calculate_pvalue
    kwargs["processes"] = processes
    kwargs["divergence_type"] = divergence_type
    # Generate the sample for the base model
    unperturbed_sample = cobra.sampling.sample(
        model=model,
        n=sample_count,
        seed=rng.integers(low=0, high=np.iinfo(np.intp).max),
        **sampler_kwargs,
    )

    for gene_to_ko in tqdm.tqdm(genes_to_ko, disable=not progress_bar):
        with model as ko_model:
            try:
                _ = knock_out_model_genes(ko_model, gene_list=[gene_to_ko])
                perturbed_sample = cobra.sampling.sample(
                    model=ko_model,
                    n=sample_count,
                    seed=rng.integers(low=0, high=np.iinfo(np.intp).max),
                    **sampler_kwargs,
                )
            except ValueError:
                # This can happen if the gene knock out causes all reactions to be 0. (or very close)
                # So continue, leaving that part of the results/pvalue dataframe as all np.nan
                continue
        grouped_div_res = calculate_divergence_grouped(
            dataset1=(
                unperturbed_sample
                if use_unperturbed_as_true
                else perturbed_sample
            ),
            dataset2=(
                unperturbed_sample
                if not use_unperturbed_as_true
                else perturbed_sample
            ),
            divergence_groups=target_networks,  # type: ignore   # str is hashable
            **kwargs,
        )
        if not calculate_pvalue:
            ko_div_df.loc[gene_to_ko] = grouped_div_res
        else:
            div, pvalue = grouped_div_res
            ko_div_df.loc[gene_to_ko] = div
            ko_pvalue_df.loc[gene_to_ko] = pvalue
    if not calculate_pvalue:
        return ko_div_df
    return ko_div_df, ko_pvalue_df


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


# endregion helper functions

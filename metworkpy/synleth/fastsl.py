"""
Module for finding Synthetic Lethality Gene Groups in
"""
# Imports
# Standard Library Imports
from __future__ import annotations
import concurrent.futures
import multiprocessing
import multiprocessing.queues
from multiprocessing.managers import ListProxy
from typing import Iterable

# External Imports
import cobra
from cobra.manipulation import knock_out_model_genes
import numpy as np


# Local Imports


# region Synethetic Lethal Genes
def find_synthetic_lethal_genes(
    model: cobra.Model,
    max_depth: int = 3,
    active_cutoff: float = cobra.core.configuration.Configuration().tolerance,
    pfba_fraction_of_optimum: float = 0.95,
    essential_proportion: float = 0.01,
    processes: int | None = None,
    show_queue_size: bool = False,
) -> list[set[str]]:
    """
    Find groups of genes whose combined knockout leads to growth inhibition

    :param model: Cobra model to find synthetic lethal groups of genes for
    :type model: cobra.Model
    :param max_depth: Maximum number of genes in a synthetic lethal group
    :type max_depth: int
    :param active_cutoff: Minimum (absolute value of) flux through a reaction to be considered active. All
        reactions found to have a flux below this during pFBA will be ignored when finding essential genes.
    :type active_cutoff: float
    :param pfba_fraction_of_optimum: Proportion of maximum objective maximum required to be maintained during
        pFBA. The original objective function is constrained to be greater than the maximum objective multiplied by
        pfba_fraction_of_optimum
    :type pfba_fraction_of_optimum: float
    :param essential_proportion: Proportion of maximum objective value, below which a gene knock out is considered
        growth inhibitory. A value of 0.01 indicates that if the maximum objective value after a gene knock out
        is less than 1% of the value before the knockout, the gene is essential.
    :type essential_proportion: float
    :param processes: Number of processes to use during calculations, if None will use all available
    :type processes: int | None
    :param show_queue_size: If True will print the approximate current queue size each time a job is taken from
        the queue during calculations. Default False.
    :type show_queue_size: bool
    :return: List of synethetically lethal groups of genes, recorded as sets of gene ids
    :rtype: list[set[str]]
    """
    # Find the essential cutoff
    essential_cutoff = essential_proportion * model.slim_optimize()
    # Create manager
    with multiprocessing.Manager() as manager:
        gene_queue = manager.Queue()
        results_list = manager.list()
        # Add initially active gene set to queue
        for g in _get_potentially_active_genes(
            model=model,
            pfba_fraction_of_optimum=pfba_fraction_of_optimum,
            active_cutoff=active_cutoff,
        ):
            gene_queue.put({g})
        with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
            futures = [
                executor.submit(
                    _process_gene_set_worker,
                    gene_queue=gene_queue,
                    results_list=results_list,
                    model=model,
                    max_depth=max_depth,
                    pfba_fraction_of_optimum=pfba_fraction_of_optimum,
                    active_cutoff=active_cutoff,
                    essential_cutoff=essential_cutoff,
                    show_queue_size=show_queue_size,
                )
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
            synleth_list = list(results_list)
        return synleth_list


# endregion Synthetic Lethal Genes


# region Helper Functions
def _process_gene_set_worker(
    gene_queue: multiprocessing.queues.Queue,
    results_list: ListProxy,
    model: cobra.Model,
    max_depth: int,
    pfba_fraction_of_optimum: float,
    active_cutoff: float,
    essential_cutoff: float,
    show_queue_size: bool = False,
):
    while True:
        try:
            gene_set = gene_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break
        with model as m:
            if show_queue_size:
                print(gene_queue.qsize())
            knock_out_model_genes(m, list(gene_set))
            # Case where the gene set is currently essential
            # Should only happen genes, since they are added without being checked

            objective_value = m.slim_optimize(error_value=np.nan)
            if np.isnan(objective_value) or (objective_value <= essential_cutoff):
                results_list.append(gene_set)
            else:
                potentially_active_genes = _get_potentially_active_genes(
                    model=m,
                    pfba_fraction_of_optimum=pfba_fraction_of_optimum,
                    active_cutoff=active_cutoff,
                )
                for gene in potentially_active_genes:
                    new_set = gene_set.union({gene})
                    if (new_set != gene_set) and (len(new_set) <= max_depth):
                        if _is_essential(
                            model=m, gene=gene, essential_cutoff=essential_cutoff
                        ):
                            results_list.append(new_set)
                        else:
                            gene_queue.put(new_set)


def _is_essential(model: cobra.Model, gene: str, essential_cutoff: float) -> bool:
    with model as m:
        m.genes.get_by_id(gene).knock_out()
        objective_value = m.slim_optimize(error_value=np.nan)
        if (objective_value <= essential_cutoff) or np.isnan(objective_value):
            return True
        elif objective_value > essential_cutoff:
            return False
        else:
            raise RuntimeError("Error in finding essential gene function")


def _rxns_to_genes(model: cobra.Model, rxns: Iterable[str]) -> set[str]:
    res_set = set()
    for r in rxns:
        res_set = res_set.union(model.reactions.get_by_id(r).genes)
    return {g.id for g in res_set}


def _get_potentially_active_genes(
    model: cobra.Model, pfba_fraction_of_optimum: float, active_cutoff: float
) -> set(str):
    pfba_res = cobra.flux_analysis.pfba(
        model=model, fraction_of_optimum=pfba_fraction_of_optimum
    ).fluxes
    active_reactions = pfba_res[np.abs(pfba_res) > active_cutoff].index
    return _rxns_to_genes(model=model, rxns=active_reactions)


# endregion Helper Functions

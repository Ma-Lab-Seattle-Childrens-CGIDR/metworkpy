"""
Module for finding Synthetic Lethality Gene Groups in Cobra Models
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from collections import deque
import concurrent.futures
import multiprocessing
import multiprocessing.queues
from multiprocessing.managers import ListProxy, DictProxy
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
    genes_of_interest: None | Iterable = None,
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
    :param genes_of_interest: Set of genes of interest, this method will only find synthetic lethal groups
        which include at least one of these genes. Should be an iterable (list, set, etc.) of
        gene id strings, which match the gene ids in the model. If None, will default to all
        genes in the model. This filtering is done after the synthetic lethal sets are already found
        (see note for reason) and so is for convenience rather than speed.
    :type genes_of_interest: None | Iterable
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
    :return: List of synthetically lethal groups of genes, recorded as sets of gene ids
    :rtype: list[set[str]]

    .. note:
       For parallel operation, 3 processes is a recommended minimum. Currently, the parallel implementation
       requires a manager process (which handles the queue of gene sets to be processed), and so if processes=2
       this will end up using only 1 core to do most of the calculation, with overhead from parallel processing
       causing this to likely be slower than serial operation. Currently, setting processes=2 will result in
       this function using 2 processes, but in future this might be changed so that it just uses the
       faster serial implementation in this case. Further, due to the relatively high overhead in the parallel
       implementation, the actual speedup granted by the parallel implementation may be relatively small
       (especially on smaller models).

       For the genes_of_interest argument, this function still needs to check all synthetic lethal groups
       exhaustively, so it will actually take longer than if this is not provided. The reason all groups
       have to be checked exhaustively is to ensure that no subsets of the group are already synthetically
       lethal. Take for example a set of genes A, B, C, D, E, F. If A and B together are essential,
       and you are interested in the gene set D,E,F, then the knock out of A,B,D would be essential.
       Even though you are interested in D, the essentiality of this gene set is caused by A and B alone.
       To ensure that all synthetic lethal gene sets are actually synthetically lethal (rather than a
       superset including irrelevant genes), the gene sets are filtered for any sets that are supersets of other
       sets. This means that it is impossible to only check sets that include genes of interest, since (like
       in the example), the genes causing the essentiality maybe outside the genes of interest.
    """
    # Find the essential cutoff
    essential_cutoff = essential_proportion * model.slim_optimize()
    # Get genes of interest, create set
    if genes_of_interest:
        genes_of_interest = {g for g in genes_of_interest}
    if processes > 1:
        synleth_list = _fastsl_parallel(
            model=model,
            max_depth=max_depth,
            pfba_fraction_of_optimum=pfba_fraction_of_optimum,
            active_cutoff=active_cutoff,
            essential_cutoff=essential_cutoff,
            processes=processes,
            show_queue_size=show_queue_size,
        )
    elif processes == 1:
        synleth_list = _fastsl_serial(
            model=model,
            max_depth=max_depth,
            pfba_fraction_of_optimum=pfba_fraction_of_optimum,
            active_cutoff=active_cutoff,
            essential_cutoff=essential_cutoff,
            show_queue_size=show_queue_size,
        )
    # Now need to filter sets
    synleth_list = _filter_supersets(synleth_list)
    if not genes_of_interest:
        return synleth_list
    return [s for s in synleth_list if len(s & genes_of_interest) > 0]


# endregion Synthetic Lethal Genes


# region Parallel Helper Functions
def _fastsl_parallel(
    model: cobra.Model,
    max_depth: int,
    pfba_fraction_of_optimum: float,
    active_cutoff: float,
    essential_cutoff: float,
    processes,
    show_queue_size: bool,
) -> list[set[str]]:
    # Create manager
    with multiprocessing.Manager() as manager:
        gene_queue = manager.Queue()
        results_list = manager.list()
        # Need to create a way to make sure repeats are not processed
        processed_dict = manager.dict()
        # Add initially active gene set to queue
        for g in _get_potentially_active_genes(
            model=model,
            pfba_fraction_of_optimum=pfba_fraction_of_optimum,
            active_cutoff=active_cutoff,
        ):
            gene_queue.put({g})
        # The processes-1 is because the manager process consumes one processes
        # but needs to have at least 1 process
        max_workers = max(1, processes - 1)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(
                    _process_gene_set_worker,
                    gene_queue=gene_queue,
                    results_list=results_list,
                    processed_dict=processed_dict,
                    model=model,
                    max_depth=max_depth,
                    pfba_fraction_of_optimum=pfba_fraction_of_optimum,
                    active_cutoff=active_cutoff,
                    essential_cutoff=essential_cutoff,
                    show_queue_size=show_queue_size,
                )
                for _ in range(processes)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
            synleth_list = list(results_list)
    return synleth_list


def _process_gene_set_worker(
    gene_queue: multiprocessing.queues.Queue,
    results_list: ListProxy,
    processed_dict: DictProxy,
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
        frozen_gene_set = frozenset(gene_set)
        if frozen_gene_set in processed_dict:
            pass
        else:
            processed_dict[frozen_gene_set] = True
            with model as m:
                if show_queue_size:
                    print(gene_queue.qsize())
                knock_out_model_genes(m, list(gene_set))
                # Case where the gene set is currently essential
                objective_value = m.slim_optimize(error_value=np.nan)
                if np.isnan(objective_value) or (objective_value <= essential_cutoff):
                    results_list.append(gene_set)
                else:
                    if len(gene_set) >= max_depth:
                        pass
                    else:
                        potentially_active_genes = _get_potentially_active_genes(
                            model=m,
                            pfba_fraction_of_optimum=pfba_fraction_of_optimum,
                            active_cutoff=active_cutoff,
                        )
                        for gene in potentially_active_genes:
                            new_set = gene_set.union({gene})
                            if (new_set != gene_set) and (len(new_set) <= max_depth):
                                gene_queue.put(new_set)


# endregion Parallel Helper Functions


# region Serial Helper Functions
def _fastsl_serial(
    model: cobra.Model,
    max_depth: int,
    pfba_fraction_of_optimum: float,
    active_cutoff: float,
    essential_cutoff: float,
    show_queue_size: bool,
) -> list[set[str]]:
    gene_queue = deque()
    results_queue = deque()
    processed_set = set()
    # Add potentially active genes to the queue
    for g in _get_potentially_active_genes(
        model=model,
        pfba_fraction_of_optimum=pfba_fraction_of_optimum,
        active_cutoff=active_cutoff,
    ):
        gene_queue.append({g})
    # While the gene queue isn't empty, run the process function
    while gene_queue:
        _process_gene_set_serial(
            model=model,
            gene_queue=gene_queue,
            results_queue=results_queue,
            processed_set=processed_set,
            max_depth=max_depth,
            active_cutoff=active_cutoff,
            essential_cutoff=essential_cutoff,
            pfba_fraction_of_optimum=pfba_fraction_of_optimum,
            show_queue_size=show_queue_size,
        )
    synleth_list = list(results_queue)
    return synleth_list


def _process_gene_set_serial(
    model: cobra.Model,
    gene_queue: deque[set],
    results_queue: deque[set],
    processed_set: set,
    max_depth: int,
    active_cutoff: float,
    essential_cutoff: float,
    pfba_fraction_of_optimum: float,
    show_queue_size: bool,
):
    gene_set = gene_queue.popleft()
    if show_queue_size:
        print(len(gene_queue))
    # This checking for already processed sets is still necessary since alternate orderings
    # could produce the same gene sets
    frozen_gene_set = frozenset(gene_set)
    if frozen_gene_set in processed_set:
        return None
    processed_set.add(frozen_gene_set)
    with model as m:
        knock_out_model_genes(m, list(gene_set))
        objective_value = m.slim_optimize(error_value=np.nan)
        if np.isnan(objective_value) or (objective_value <= essential_cutoff):
            results_queue.append(gene_set)
        else:
            if len(gene_set) >= max_depth:
                return None
            potentially_active_genes = _get_potentially_active_genes(
                model=m,
                pfba_fraction_of_optimum=pfba_fraction_of_optimum,
                active_cutoff=active_cutoff,
            )
            for gene in potentially_active_genes:
                new_set = gene_set.union({gene})
                if (new_set != gene_set) and (len(new_set) <= max_depth):
                    gene_queue.append(new_set)


# endregion Serial Helper Functions


# region Helper Functions
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


def _filter_supersets(sets: list[set[str]]) -> list[set[str]]:
    sets.sort(key=len)  # Sort sets by length
    filtered_sets = []
    while len(sets) > 0:
        elem = sets.pop(0)
        filtered_sets.append(elem)
        if sets:
            sets = [s for s in sets if not elem.issubset(s)]  # Filter out all supersets
    return filtered_sets


# endregion Helper Functions

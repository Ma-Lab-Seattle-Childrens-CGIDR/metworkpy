"""
Bootstrap p-values for the various rank entropy methods
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from functools import partial
from multiprocessing import Pool, cpu_count, shared_memory
from typing import Callable, Tuple, Optional, Union

# External Imports
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import gaussian_kde, ecdf

# Local Imports
from metworkpy.utils._parallel import _create_shared_memory_numpy_array


# region Main Function
def _bootstrap_rank_entropy_p_value(
    samples_array: NDArray[float | int] | pd.DataFrame,
    sample_group1,
    sample_group2,
    gene_network,
    rank_entropy_fun: Callable[[NDArray[float | int], NDArray[float | int]], float],
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str | float | Callable[[gaussian_kde], float]]] = None,
    iterations: int = 1_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> Tuple[float, float]:
    """
    Generate a rank entropy value from the rank_entropy_fun function, and bootstrap a p-value for it

    :param samples_array: Gene expression data, either a numpy array or a pandas DataFrame, with rows representing
        different samples, and columns representing different genes
    :type samples_array: NDArray[int|float] | pd.DataFrame
    :param sample_group1: Which samples belong to group1. If expression_data is a numpy array, this should be
        a list/array/iterable of ints. If expression_data is a pandas dataframe, this can be anything that can index
        a dataframe inside a .loc (see pandas documentation for details)
    :param sample_group2: Which samples belong to group2, see sample_group1 information for more details.
    :param gene_network: List of indices for genes in the gene network
    :param rank_entropy_fun: Function used to calculate the rank entropy difference between two sample groups,
        should take two np.ndarrays as arguments and return a float
    :type rank_entropy_fun: Callable[[NDArray[float | int], NDArray[float | int]], float]
    :param kernel_density_estimate: Whether to use a kernel density estimate for calculating the p-value. If True,
        will use a Gaussian Kernel Density Estimate, if False will use an empirical CDF
    :type kernel_density_estimate: bool
    :param bw_method: Bandwidth method, see `scipy.stats.gaussian_kde <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`_ for details
    :type bw_method: Optional[Union[str|float|Callable[[gaussian_kde], float]]]
    :param iterations: Number of iterations to perform during bootstrapping the null distribution
    :type iterations: int
    :param replace: Whether to sample with replacement when randomly sampling from the sample groups
        during bootstrapping
    :type replace: bool
    :param seed: Seed to use for the random number generation during bootstrapping
    :type seed: int
    :param processes: Number of processes to use during the bootstrapping, default 1
    :type processes: int
    :return: Tuple of the return value from rank_entropy_fun(sample_group1 array, sample_group2 array), and the
        p-value found by bootstrapping
    :rtype: Tuple[float, float]
    """
    # Begin by converting the expression data into the proper form
    # Convert dataframe into numpy array
    if isinstance(samples_array, pd.DataFrame):
        sg1 = samples_array.loc[sample_group1, gene_network]
        sg2 = samples_array.loc[sample_group2, gene_network]
        sg1_size, gn_size = sg1.shape
        sg2_size, _ = sg2.shape
        gene_network = list(range(gn_size))
        sample_group1 = list(range(sg1_size))
        sample_group2 = list(range(sg1_size, sg1_size + sg2_size))
        samples_array = np.vstack((sg1.to_numpy(), sg2.to_numpy()))
    elif isinstance(samples_array, np.ndarray):
        sg1 = samples_array[sample_group1][:, gene_network]
        sg2 = samples_array[sample_group2][:, gene_network]
        sg1_size, gn_size = sg1.shape
        sg2_size, _ = sg2.shape
        gene_network = list(range(gn_size))
        sample_group1 = list(range(sg1_size))
        sample_group2 = list(range(sg1_size, sg1_size + sg2_size))
        samples_array = np.vstack((sg1, sg2))
    sample_group1 = list(sample_group1)
    sample_group2 = list(sample_group2)
    gene_network = list(gene_network)
    # start by putting the numpy samples array into shared memory
    (
        shared_nrows,
        shared_ncols,
        shared_dtype,
        shared_mem_name,
    ) = _create_shared_memory_numpy_array(samples_array)
    # Make sure to unlink memory even if something fails
    try:
        # Calculate the value for the unshuffled array
        rank_entropy = rank_entropy_fun(
            samples_array[sample_group1][:, gene_network],
            samples_array[sample_group2][:, gene_network],
        )
        # Create a numpy random number generator with provided seed
        rng_gen = np.random.default_rng(seed=seed)
        # Get the sequence of seeds for the subprocesses
        # The high value for this seed array is at most the maximum of the int64 dtype
        seed_array = rng_gen.integers(2**63 - 1, size=iterations)
        # Get the combined samples array
        samples = np.array(sample_group1 + sample_group2)
        sample_group1_size = len(sample_group1)
        sample_group2_size = len(sample_group2)
        # Get the number of processes to use
        processes = min(processes, cpu_count())
        # Set up the pool
        with Pool(processes) as pool:
            rank_entropy_samples = np.array(
                [
                    val
                    for val in pool.imap_unordered(
                        partial(
                            _bootstrap_rank_entropy_p_values_worker,
                            rank_entropy_fun=rank_entropy_fun,
                            samples=samples,
                            sample_group1_size=sample_group1_size,
                            sample_group2_size=sample_group2_size,
                            shared_nrows=shared_nrows,
                            shared_ncols=shared_ncols,
                            shared_dtype=shared_dtype,
                            shared_mem_name=shared_mem_name,
                            replace=replace,
                        ),
                        seed_array,
                        chunksize=iterations // processes,
                    )
                ]
            )
        if not kernel_density_estimate:
            empirical_cdf = ecdf(rank_entropy_samples)
            pvalue = empirical_cdf.sf.evaluate(rank_entropy)[()]
        else:
            kde = gaussian_kde(rank_entropy_samples, bw_method=bw_method)
            pvalue = kde.integrate_box_1d(rank_entropy, np.inf)
    finally:
        shm = shared_memory.SharedMemory(name=shared_mem_name)
        shm.unlink()
    return rank_entropy, pvalue


# endregion Main Function

# region Worker Functions


def _bootstrap_rank_entropy_p_values_worker(
    seed: int,
    rank_entropy_fun: Callable[[NDArray[float | int], NDArray[float | int]], float],
    samples: NDArray[int],
    sample_group1_size: int,
    sample_group2_size: int,
    shared_nrows: int,
    shared_ncols: int,
    shared_dtype: np.dtype,
    shared_mem_name: str,
    replace: bool,
) -> float:
    # Get access to the shared numpy array
    shm = shared_memory.SharedMemory(name=shared_mem_name)
    shared_array = np.ndarray(
        (shared_nrows, shared_ncols), dtype=shared_dtype, buffer=shm.buf
    )
    # Create numpy random number generator
    rng_gen = np.random.default_rng(seed=seed)
    # Sample from the sample groups
    if replace:
        sg1 = rng_gen.choice(samples, size=sample_group1_size, replace=replace)
        sg2 = rng_gen.choice(samples, size=sample_group2_size, replace=replace)
    else:
        shuffled_samples = rng_gen.permuted(
            samples
        )  # This creates a copy, but want to be safe for now
        sg1 = shuffled_samples[:sample_group1_size]
        sg2 = shuffled_samples[sample_group1_size:]
    return rank_entropy_fun(shared_array[sg1, :], shared_array[sg2, :])


# endregion Worker Functions

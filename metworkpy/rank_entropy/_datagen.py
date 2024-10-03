"""
Generate data for testing the rank entropy functions
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Union, Optional, Tuple

import numpy as np

# External Imports
from scipy.stats import rv_continuous, rv_discrete

# Local Imports

# Typing information
Distribution = Union[rv_continuous, rv_discrete]


# region Main Function


def _generate_rank_entropy_data(
    n_ordered_samples: int,
    n_unordered_samples: int,
    n_genes_ordered: int,
    n_genes_unordered: int,
    dist: Distribution,
    shuffle_genes: bool = True,
    shuffle_samples: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data with ordered and disordered genes/samples
    :param n_ordered_samples: Number of ordered samples
    :type n_ordered_samples: int
    :param n_unordered_samples: Number of unordered samples
    :type n_unordered_samples: int
    :param n_genes_ordered: Number of ordered genes
    :type n_genes_ordered: int
    :param n_genes_unordered: Number of unordered genes
    :type n_genes_unordered: int
    :param dist: Distribution to use for sampling, should be a scipy rv_continuous, or rv_discrete (or at least
        have a rvs method which takes a single argument size and returns a random sample as a np array of length size)
    :type dist: Distribution
    :param shuffle_genes: Whether the order of the genes should be shuffled
    :type shuffle_genes: bool
    :param shuffle_samples: Whether the order of the samples should be shuffled
    :type shuffle_samples: bool
    :param seed: Seed to use for the random number generator used whe shuffling (doesn't change the sampling from
        the provided dist)
    :type seed: Optional[int]
    :return: Tuple of np.ndarrays, representing:
        1. the generated expression data, with rows representing samples, and columns representing genes.
        2. the indices of the ordered samples
        3. the indices of the unordered samples
        4. the indices of the ordered genes
        5. the indices of the unordered genes
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    rng_generator = np.random.default_rng(seed=seed)
    ordered_array = _ordered_array(
        nrow=n_ordered_samples,
        ncol=n_genes_ordered,
        dist=dist,
        col_shuffle=shuffle_genes,
        rng_generator=rng_generator,
    )
    unordered_array = _unordered_array(
        nrow=n_unordered_samples,
        ncol=n_genes_ordered,
        dist=dist,
        rng_generator=rng_generator,
    )
    ordered_genes_array = np.vstack((ordered_array, unordered_array))
    if shuffle_samples:
        samples_shuffled = rng_generator.permuted(
            list(range(n_ordered_samples + n_unordered_samples))
        )
        ordered_samples = samples_shuffled[:n_ordered_samples]
        unordered_samples = samples_shuffled[n_ordered_samples:]
        ordered_genes_array[ordered_samples, :] = ordered_array
        ordered_genes_array[unordered_samples, :] = unordered_array
    else:
        ordered_samples = np.array(range(n_ordered_samples))
        unordered_samples = np.array(
            range(n_ordered_samples, n_unordered_samples + n_ordered_samples)
        )
    unordered_genes_array = _unordered_array(
        nrow=n_ordered_samples + n_unordered_samples,
        ncol=n_genes_unordered,
        dist=dist,
        rng_generator=rng_generator,
    )
    res_array = np.hstack((ordered_genes_array, unordered_genes_array))
    if shuffle_genes:
        genes_shuffled = rng_generator.permuted(
            list(range(n_genes_ordered + n_genes_unordered))
        )
        ordered_genes = genes_shuffled[:n_genes_ordered]
        unordered_genes = genes_shuffled[n_genes_ordered:]
        res_array[:, ordered_genes] = ordered_genes_array
        res_array[:, unordered_genes] = unordered_genes_array
    else:
        ordered_genes = np.array(range(n_genes_ordered))
        unordered_genes = np.array(
            range(n_genes_ordered, n_genes_unordered + n_genes_ordered)
        )
    return res_array, ordered_samples, unordered_samples, ordered_genes, unordered_genes


# endregion Main Function


# region Unordered
def _unordered_vector(
    size: int,
    dist: Distribution,
    rng_generator: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    return dist.rvs(size, random_state=rng_generator)


def _unordered_array(
    nrow: int,
    ncol: int,
    dist: Distribution,
    rng_generator: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    res_array = np.zeros((nrow, ncol), dtype=dist.rvs(0).dtype)
    for row in range(nrow):
        res_array[row, :] = _unordered_vector(
            size=ncol, dist=dist, rng_generator=rng_generator
        )
    return res_array


# endregion Unordered


# region Ordered
def _ordered_vector(
    size: int,
    dist: Distribution,
    rng_generator: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    return np.sort(_unordered_vector(size, dist, rng_generator=rng_generator))


def _ordered_array(
    nrow: int,
    ncol: int,
    dist: Distribution,
    col_shuffle: bool = True,
    rng_generator: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    res_array = np.zeros((nrow, ncol), dtype=dist.rvs(0).dtype)
    for row in range(nrow):
        res_array[row, :] = _ordered_vector(
            size=ncol, dist=dist, rng_generator=rng_generator
        )
    if col_shuffle and ncol != 0:
        new_col_order = rng_generator.permuted(list(range(ncol)))
        res_array = res_array[:, new_col_order]
    return res_array


# endregion Ordered

"""A wrapper function around methods which can calculate divergences for continuous and discrete distributions"""

# Imports
# Standard Library Imports
from functools import partial
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Union,
    NamedTuple,
)

# External Imports
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

# Local Imports
from metworkpy.utils._arguments import _parse_metric
from metworkpy.divergence._data_validation import (
    _validate_samples,
    _validate_discrete,
)
from metworkpy.utils._jitter import _jitter_single


class ContinuousDivergenceFunction(Protocol):
    def __call__(
        self,
        p: NDArray[float],
        q: NDArray[float],
        n_neighbors: int,
        metric: float,
    ) -> float: ...


class DivergenceResult(NamedTuple):
    divergence: float
    pvalue: float


def _wrap_divergence_functions(
    p: ArrayLike,
    q: ArrayLike,
    discrete_method: Callable[[NDArray[Any], NDArray[Any]], float],
    continuous_method: ContinuousDivergenceFunction,
    calculate_pvalue: bool = False,
    alternative: Literal["less", "greater", "two-sided"] = "greater",
    permutations: int = 9999,
    n_neighbors: int = 5,
    discrete: bool = False,
    jitter: Optional[float] = None,
    jitter_seed: Optional[int] = None,
    distance_metric: Union[float, str] = "euclidean",
) -> Union[float, DivergenceResult]:
    """Calculate the divergence between two distributions represented by samples p and q

    Parameters
    ----------
    p : ArrayLike
        Array representing sample from a distribution, should have shape
        (n_samples, n_dimensions). If `p` is one dimensional, it will be
        reshaped to (n_samples,1). If it is not a np.ndarray, this
        function will attempt to coerce it into one.
    q : ArrayLike
        Array representing sample from a distribution, should have shape
        (n_samples, n_dimensions). If `q` is one dimensional, it will be
        reshaped to (n_samples,1). If it is not a np.ndarray, this
        function will attempt to coerce it into one.
    discrete_method
        Method to use to calculate the divergence between two discrete
        distributions, should take two positional arguments for p and q.
    continuous_method
        Method to use to calculate the divergence between two continuous
        distributions, should take two positional arguments for p and q,
        as well as keyword arguments for n_neighbors, and metric (which
        will be a float representing a Minkowski p-norm).
    calculate_pvalue : bool, default=False
        Whether the p-value should be calculated using a permutation test
    alternative : 'less', 'greater', or 'two-sided', default='greater'
        The alternative to use, passed to SciPy's `permutation_test<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html>`_
    permutations : int, default=9999
         The number of permuatations to use when calculating the p-value
    n_neighbors : int
        Number of neighbors to use for computing mutual information.
        Will attempt to coerce into an integer. Must be at least 1.
        Default 5.
    discrete : bool
        Whether the samples are from discrete distributions
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
        can be "Euclidean", "Manhattan", or "Chebyshev". Can also be a
        float representing the Minkowski p-norm.

    Returns
    -------
    float
        The divergence between p and q, or if calculate_pvalue is True,
        returns a named tuple of divergence and p-value.
    """
    try:
        n_neighbors = int(n_neighbors)
    except ValueError as err:
        raise ValueError(
            f"n_neighbors must be able to be converted to an integer, but a {type(n_neighbors)} was"
            f"given instead."
        ) from err
    distance_metric = _parse_metric(distance_metric)
    p, q = _validate_samples(p, q)
    if jitter and not discrete:
        generator = np.random.default_rng(jitter_seed)
        p = _jitter_single(p, jitter=jitter, generator=generator)
        q = _jitter_single(q, jitter=jitter, generator=generator)
    permutation_test_kwargs = {
        "permutation_type": "independent",
        "n_resamples": permutations,
        "alternative": alternative,
        "axis": 0,
    }
    if discrete:
        try:
            p = _validate_discrete(p)
            q = _validate_discrete(q)
        except ValueError as err:
            raise ValueError(
                f"p and q must represent single dimensional samples, and so have shape (n_samples, 1)"
                f"but p has dimension {p.shape[1]}, and q has dimension {q.shape[1]}."
            ) from err
        if not calculate_pvalue:
            return discrete_method(p, q)
        else:
            perm_res = stats.permutation_test(
                [p, q], discrete_method, **permutation_test_kwargs
            )
            return DivergenceResult(
                divergence=perm_res.statistic, pvalue=perm_res.pvalue
            )
    if not calculate_pvalue:
        return continuous_method(
            p, q, n_neighbors=n_neighbors, metric=distance_metric
        )
    perm_res = stats.permutation_test(
        [p, q],
        partial(
            continuous_method, n_neighbors=n_neighbors, metric=distance_metric
        ),
        **permutation_test_kwargs,
    )
    return DivergenceResult(
        divergence=perm_res.statistic, pvalue=perm_res.pvalue
    )

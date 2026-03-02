"""
Some helpful statistics methods
"""

from typing import Hashable, Literal, NamedTuple

import numpy as np
from scipy import stats


class SignificanceResult(NamedTuple):
    """
    Class for return values from significance tests
    """

    statistic: float
    pvalue: float


class MannWhitneyUResult(NamedTuple):
    """
    Class for the results of the extended Mann-Whitney U-test,
    which includes the U1,U2, AUC ROC, and p-value
    """

    u1: float
    u2: float
    auc_roc: float
    pvalue: float


def fisher_enrichment(
    group1: set[Hashable],
    group2: set[Hashable],
    total_count: int,
    alternative: Literal["two-sided", "less", "greater"],
) -> SignificanceResult:
    """
    Perform enrichment analysis using the Fisher Exact Test

    Parameters
    ----------
    group1, group2 : set of Hashable
        The groups to evaluate the significance of the overlap for
    total_count : int
        The size of the set from which group1 and group2 are subsets
    alternative : {'two-sided', 'less', 'greater'}
        The alternative hypothesis, see
        `SciPy's fisher_exact<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html>`_
        for details.

    Results
    -------
    SignificanceResult
        Named tuple of statistic and p-value for the result of the
        Fisher's exact test
    """
    fisher_res = stats.fisher_exact(
        [
            [len(group1 & group2), len(group1 - group2)],
            [len(group2 - group1), total_count - len(group1 | group2)],
        ],
        alternative=alternative,
    )
    return SignificanceResult(
        statistic=fisher_res.statistic, pvalue=fisher_res.statistic
    )


def extended_mannwhitneyu_test(
    x: np.typing.ArrayLike,
    y: np.typing.ArrayLike,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    axis: int = 0,
    **kwargs,
) -> MannWhitneyUResult:
    """
    Perform a Mann-Whitney U-test and calculate additional information
    about the result, specifically U2 and AUC ROC

    Parameters
    ----------
    x,y : np.ArrayLike
        N-d array of samples. Arrays must be broadcastable
        except along the dimension given by axis
    alternative : {'two-sided', 'less', 'greater'}
        The alternative hypothesis to evaluate
    axis : int, default=0
        The axis of the input along which to compute the statistic
    kwargs
        Keyword arguments to pass to the scipy.stats.mannwhitneyu function

    Returns
    -------
    MannWhitneyUResult
        The results of the Mann-Whitney U-test with the addition of u2 and AUC ROC
    """
    x = np.array(x)
    y = np.array(y)
    mannu_res = stats.mannwhitneyu(
        x, y, alternative=alternative, axis=axis, **kwargs
    )
    n1n2 = x.shape[axis] * y.shape[axis]
    u1 = mannu_res.statistic
    u2 = n1n2 - u1
    auc_roc = u1 / n1n2
    pvalue = mannu_res.pvalue
    return MannWhitneyUResult(u1=u1, u2=u2, auc_roc=auc_roc, pvalue=pvalue)

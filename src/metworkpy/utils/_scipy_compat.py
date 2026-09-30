"""
Functions for maintaining SciPy compatibility
"""

import scipy


def _check_scipy_version_greater(maj, min, bug):
    scipy_maj, scipy_min, scipy_bug = scipy.__version__.split(".")
    scipy_maj, scipy_min, scipy_bug = (
        int(scipy_maj),
        int(scipy_min),
        int(scipy_bug),
    )
    if scipy_maj < maj:
        return False
    if scipy_maj > maj:
        return True
    if scipy_min < min:
        return False
    if scipy_min > min:
        return True
    if scipy_bug < bug:
        return False
    if scipy_bug > bug:
        return True
    return True

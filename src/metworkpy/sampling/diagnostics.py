"""
Submodule containing some diagnostics for convergence of sampling
"""

import numpy as np
import pandas as pd


def geweke(
    samples: pd.DataFrame,
    first: float = 0.1,
    last: float = 0.5,
) -> pd.Series:
    """
    Compute the Geweke diagnostic for a set of samples

    Parameters
    ----------
    samples : pd.DataFrame
        A DataFrame of samples, the columns should be the
        variables being sampled, and the rows should represent
        the samples (in the order they were acquired).
    first : float,default=0.1
        First portion of the sampling chain
    last : float, default=0.5
        Last portion of the sampling chain

    Returns
    pd.Series
        The Geweke diagnostic for each column of the DataFrame,
        the index of the Series will match the columns of the
        input DataFrame
    """
    # Get the number of samples in each proportion
    n_samples_first = int(first * samples.shape[0])
    n_samples_last = int(last * samples.shape[0])
    # Select these sample sets from the dataframe
    samples_first = samples.iloc[:n_samples_first]
    samples_last = samples.iloc[-n_samples_last:]
    # Calculate the Geweke diagnostic for each reaction
    return (samples_first.mean(axis=0) - samples_last.mean(axis=0)) / np.sqrt(
        samples_first.var(axis=0) + samples_last.var(axis=0)
    )

# Standard Library Imports

# External Imports
import numpy as np

# Local Imports


def univariate_normal_kl_divergence(
    mu0: float, mu1: float, sigma0: float, sigma1: float
):
    """
    Exact value for the Kullback-Leibler divergence between two Gaussian normal distributions

    Parameters
    ----------
    mu0 : float
        Mean of the first distribution
    mu1 : float
        Mean of the second distribution
    sigma0 : float
        Standard Deviation of the first distribution
    sigma1 : float
        Standard Deviation of the second distribution

    Returns
    -------
    float
        The Kullback-Leibler divergence between the two distributions
    """
    log_sigma_ratio = np.log(sigma1 / sigma0)
    mean_diff_square = pow(mu0 - mu1, 2)
    return (
        log_sigma_ratio
        + (pow(sigma0, 2) + mean_diff_square) / (2 * pow(sigma1, 2))
        - 0.5
    )


def multivariate_normal_kl_divergence(
    mu0: np.ndarray, mu1: np.ndarray, sigma0: np.ndarray, sigma1: np.ndarray
) -> float:
    """
    Calculate the Kullback-Leibler divergence between two multivariate normal distributions

    Parameters
    ----------
    mu0 : np.ndarray
        The mean of the first distribution (a square diagonal matrix)
    mu1 : np.ndarray
        The mean of the second distribution (a square diagonal matrix)
    sigma0 : np.ndarray
        The covariance matrix of the first distribution (non-singular square matrix)
    sigma1 : np.ndarray
        The covariance matrix of the second distribution (non-singular square matrix)

    Returns
    -------

    """
    dim = mu0.shape[0]
    log_det_ratio: float = np.log(np.linalg.det(sigma1) / np.linalg.det(sigma0))
    mean_diff: np.ndarray = mu1 - mu0
    sigma1_inv = np.linalg.inv(sigma1)
    mean_diff_sigma_mult = mean_diff.T @ sigma1_inv @ mean_diff
    sigma_trace = np.linalg.trace(sigma1_inv @ sigma0)
    return 0.5 * (sigma_trace - dim + mean_diff_sigma_mult + log_det_ratio)

"""
Functions for computing differential rank conservation (DIRAC)
"""

# Imports
# Standard Library Imports
from __future__ import annotations
from typing import Union, Optional, Callable, Tuple

# External Imports
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import gaussian_kde

# Local Imports
from metworkpy.rank_entropy._bootstrap_pvalue import _bootstrap_rank_entropy_p_value
from metworkpy.rank_entropy.rank_entropy_exceptions import NotFitError


# region Main Functions


def dirac_gene_set_classification(
    expression_data: NDArray[float | int] | pd.DataFrame,
    sample_group1,
    sample_group2,
    gene_network,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str | float | Callable[[gaussian_kde], float]]] = None,
    iterations: int = 10_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> Tuple[float, float]:
    """
    Calculate the classification rate using DIRAC rank difference scores for a given network and its significance

    :param expression_data: Gene expression data, either a numpy array or a pandas DataFrame, with rows representing
        different samples, and columns representing different genes
    :type expression_data: np.ndarray | pd.DataFrame
    :param sample_group1: Which samples belong to group1. If expression_data is a numpy array, this should be
        a something able to index the rows of the array. If expression_data is a pandas dataframe, this should be
        something that can index rows of a dataframe inside a .loc (see pandas documentation for details)
    :param sample_group2: Which samples belong to group2, see sample_group1 information for more details.
    :param gene_network: Which genes belong to the gene network. If expression_data is a numpy array, this
        should be something able to index the columns of the array. If expression_data is a pandas dataframe, this
        should be something be anything that can index columns of a dataframe inside a .loc (see pandas documentation
        for details)
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
    :return: Tuple of the classification rate, and the significance level found via bootstrapping
    :rtype: Tuple[float,float]
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_group1=sample_group1,
        sample_group2=sample_group2,
        gene_network=gene_network,
        rank_entropy_fun=_dirac_classification_rate,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


def dirac_gene_set_entropy(
    expression_data: NDArray[float | int] | pd.DataFrame,
    sample_group1,
    sample_group2,
    gene_network,
    kernel_density_estimate: bool = True,
    bw_method: Optional[Union[str | float | Callable[[gaussian_kde], float]]] = None,
    iterations: int = 1_000,
    replace: bool = True,
    seed: Optional[int] = None,
    processes=1,
) -> Tuple[float, float]:
    """
    Calculate the difference in rank conservation indices, and its significance

    :param expression_data: Gene expression data, either a numpy array or a pandas DataFrame, with rows representing
        different samples, and columns representing different genes
    :type expression_data: np.ndarray | pd.DataFrame
    :param sample_group1: Which samples belong to group1. If expression_data is a numpy array, this should be
        a something able to index the rows of the array. If expression_data is a pandas dataframe, this should be
        something that can index rows of a dataframe inside a .loc (see pandas documentation for details)
    :param sample_group2: Which samples belong to group2, see sample_group1 information for more details.
    :param gene_network: Which genes belong to the gene network. If expression_data is a numpy array, this
        should be something able to index the columns of the array. If expression_data is a pandas dataframe, this
        should be something be anything that can index columns of a dataframe inside a .loc (see pandas documentation
        for details)
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
    :return: Tuple of the difference in rank conservation index, and the significance level found via bootstrapping
    :rtype: Tuple[float,float]
    """
    return _bootstrap_rank_entropy_p_value(
        samples_array=expression_data,
        sample_group1=sample_group1,
        sample_group2=sample_group2,
        gene_network=gene_network,
        rank_entropy_fun=_dirac_differential_entropy,
        kernel_density_estimate=kernel_density_estimate,
        bw_method=bw_method,
        iterations=iterations,
        replace=replace,
        seed=seed,
        processes=processes,
    )


# endregion Main Functions

# region Dirac Classifier


class DiracClassifier:
    """
    Class for using DIRAC to perform classification
    """

    def __init__(self):
        self.rank_templates = None
        self.classes = None
        self.num_labels = None

    def fit(
        self,
        X: NDArray[float | int] | pd.DataFrame,
        y: NDArray[float | int] | pd.DataFrame | pd.Series,
    ) -> DiracClassifier:
        """
        Fit the classifier

        :param X: Features array, should be a pandas DataFrame or numpy ndarray with columns representing genes
            in a gene network, and rows representing different samples, and values corresponding to expression level
        :type X: NDArray[float|int]|pd.DataFrame
        :param y: Target array, should be a pandas Series or numpy ndarray, with length equal to the number of rows in
            X. Each entry should represent the class of the corresponding sample in X. The order should correspond
            between X and y, and the indexes will not be aligned between them.
        :type y:  NDArray[float|int]|pd.DataFrame|pd.Series
        :return: Fitted DIRAC classifier object
        :rtype: DiracClassifier

        .. note:
           This updates the classifier in place, and also returns itself.
        """
        rank_templates = []
        classes = np.unique(y)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError(
                "Invalid feature array type, must be pandas DataFrame or numpy ndarray"
            )

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise ValueError(
                "Invalid feature array type, must be pandas DataFrame or numpy ndarray"
            )
        # Reshape y to be 1D for easier indexing
        y = y.reshape(-1)

        for c in classes:
            # get all the rows corresponding to this class
            c_X = X[y == c, :]
            rank_templates.append(_rank_template(c_X))
        self.rank_templates = rank_templates
        self.classes = classes
        self.num_labels = len(classes)
        return self

    def classify(
        self, X: NDArray[float | int] | pd.DataFrame
    ) -> Union[pd.Series, NDArray]:
        """
        Use the fitted classifier to classify samples

        :param X: Features array, should be a pandas DataFrame or numpy ndarray with columns representing genes
            in a gene network, and rows representing different samples, and values corresponding to expression level
        :type X: NDArray[float|int]|pd.DataFrame
        :return: Predicted classes for all the samples. If X is a DataFrame, this will be a pandas Series;
            if X is a ndarray, this will be a 1-dimensional numpy array
        :rtype: pd.Series | NDArray
        """
        if self.rank_templates is None:
            raise NotFitError(
                "DIRAC Classifier must be fit before use (try calling the fit method)"
            )
        if isinstance(X, pd.DataFrame):
            return pd.Series(self._classify_arr(X.to_numpy()), index=X.index)
        elif isinstance(X, np.ndarray):
            return self._classify_arr(X)
        else:
            raise ValueError(
                f"X must be either a pandas DataFrame or a numpy ndarray, received {type(X)}"
            )

    def _classify_arr(self, X: NDArray[float | int]) -> NDArray:
        class_array = np.zeros((X.shape[0], self.num_labels), dtype=float)
        rank_array = _rank_array(X)
        for idx, template in enumerate(self.rank_templates):
            class_array[:, idx] = np.equal(rank_array, template).mean(axis=1)
        return self.classes[np.argmax(class_array, axis=1)]


# endregion Dirac Classifier

# region Rank Vector


def _rank_vector(in_vector: NDArray[int | float]) -> NDArray[int]:
    rank_array = np.repeat(in_vector.reshape(1, -1), len(in_vector), axis=0)
    diff_array = rank_array - rank_array.T
    return (diff_array[np.triu_indices(len(in_vector), k=1)] > 0).astype(int)


def _rank_array(in_array: NDArray[int | float]) -> NDArray[int]:
    return np.apply_along_axis(_rank_vector, axis=1, arr=in_array)


def _rank_template(in_array: NDArray[int | float]) -> NDArray[int]:
    return (
        np.greater(_rank_array(in_array).mean(axis=0), 0.5).astype(int).reshape(1, -1)
    )


def _rank_matching_scores(in_array: NDArray[int | float]) -> NDArray[float]:
    rank_array = _rank_array(in_array)
    rank_template = np.greater(rank_array.mean(axis=0), 0.5).astype(int).reshape(1, -1)
    return np.equal(rank_array, rank_template).mean(axis=1)


def _rank_conservation_index(in_array: NDArray[int]) -> float:
    return _rank_matching_scores(in_array).mean()


def _dirac_differential_entropy(
    a: NDArray[float | int], b: NDArray[float | int]
) -> float:
    return np.abs(_rank_conservation_index(a) - _rank_conservation_index(b))


# endregion Rank Vector

# region classification


def _dirac_classification_rate(
    a: NDArray[float | int], b: NDArray[float | int]
) -> float:
    # Find the rank Templates
    rank_array_a = _rank_array(a)
    rank_array_b = _rank_array(b)

    rank_template_a = (rank_array_a.mean(axis=0) > 0.5).astype(int).reshape(1, -1)
    rank_template_b = (rank_array_b.mean(axis=0) > 0.5).astype(int).reshape(1, -1)

    # Compute the Rank matching score for each array, for each phenotype
    rank_matching_score_array_a_phenotype_a = (
        np.equal(rank_array_a, rank_template_a)
    ).mean(axis=1)
    rank_matching_score_array_a_phenotype_b = (
        np.equal(rank_array_a, rank_template_b)
    ).mean(axis=1)

    rank_matching_score_array_b_phenotype_a = (
        np.equal(rank_array_b, rank_template_a)
    ).mean(axis=1)
    rank_matching_score_array_b_phenotype_b = (
        np.equal(rank_array_b, rank_template_b)
    ).mean(axis=1)

    # Calculate Rank Difference Scores
    rank_difference_a = (
        rank_matching_score_array_a_phenotype_a
        - rank_matching_score_array_a_phenotype_b
    )
    rank_difference_b = (
        rank_matching_score_array_b_phenotype_a
        - rank_matching_score_array_b_phenotype_b
    )

    # Calculate the accuracy
    total_samples = a.shape[0] + b.shape[0]
    correct_samples = (rank_difference_a > 0.0).sum() + (rank_difference_b <= 0.0).sum()

    return correct_samples / total_samples


# endregion classification

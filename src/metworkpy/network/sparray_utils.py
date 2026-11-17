import numpy as np
from scipy import sparse


def normalize_array(array: sparse.sparray, axis: int) -> sparse.coo_array:
    if axis == 1:
        array: sparse.csr_array = array.tocsr()  # ty: ignore[unresolved-attribute]
    elif axis == 0:
        array: sparse.csc_array = array.tocsc()  # ty: ignore[unresolved-attribute]
    else:
        raise ValueError(f"Axis must be 0 or 1, received {axis}")
    norm = array.sum(axis=axis)
    norm[norm > 0.0] = np.reciprocal(norm[norm > 0.0])
    if axis == 1:
        array = array.multiply(norm.reshape(-1, 1))  # ty: ignore[conflicting-declarations]
        return array.tocoo()
    elif axis == 0:
        array = array.multiply(norm.reshape(1, -1))  # ty: ignore[conflicting-declarations]
        return array.tocoo()


def allclose_array(
    a: sparse.sparray,
    b: sparse.sparray,
    atol: float = 1e-8,
    rtol: float = 1e-5,
):
    """
    Check if two sparse arrays are within tolerance of each other

    Parameters
    ----------
    a,b : sparse.sparray
        The two arrays to compare
    atol : float, default=1e-8
        The absolute tolerance
    rtol : float, default=1e-5
        The relative tolerance

    Returns
    -------
    bool
        True if all elements of the arrays are within tolerance, and False otherwise
    """
    a = a.tocsc()  # ty: ignore[unresolved-attribute]
    b = b.tocsc()  # ty: ignore[unresolved-attribute]
    c = abs(a - b) - rtol * abs(b)
    return c.max() <= atol


def prune_array(
    array: sparse.sparray,
    threshold: float = 1e-5,
) -> sparse.csc_array:
    """
    Set any values in `array` less than `threshold` to 0

    Parameters
    ----------
    array : sparse array
        Array to prune
    threshold : float,default=1e-5
        Threshold below which values will be set to zero

    Returns
    -------
    pruned_array : sparse.csc_array
        The array with the values less than threshold set to 0
    """
    array = array.tocoo()  # ty: ignore[unresolved-attribute]
    array.data[array.data <= threshold] = 0
    array.eliminate_zeros()

    return array.tocsc()

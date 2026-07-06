"""
Some type definitions used across MetworkPy
"""

# Standard Library
from typing import Union

# External
import numpy as np

# Array Types
Array2D = np.ndarray[tuple[int, int], np.dtype[Union[np.float32, np.float64]]]
Array1D = np.ndarray[tuple[int], np.dtype[Union[np.float32, np.float64]]]

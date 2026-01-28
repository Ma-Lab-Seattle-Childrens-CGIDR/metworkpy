"""
Submodule for reading in an example model (mostly for use in documentation)
"""

from __future__ import annotations

# Standard Library Imports
from importlib import resources

# External Imports
import cobra  # type: ignore

# Local Imports
from metworkpy.utils import read_model
from metworkpy import examples


def get_example_model() -> cobra.Model:
    """
    Get a simple example cobra Model object

    Returns
    -------
    cobra.Model
        The model object
    """
    with resources.as_file(
        resources.files(examples) / "example_model.json"
    ) as f:
        # with importlib.resources.path(examples, "example_model.json") as f:
        model = read_model(f, "json")
    return model

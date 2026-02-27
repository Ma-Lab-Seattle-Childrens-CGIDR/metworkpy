# Standard Library Imports
import unittest

# External Imports

# Local Imports
from metworkpy.examples import get_example_model
from metworkpy.network import create_reaction_network


class TestFuzzyReactionSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_example_model()
        # Convert model into a reaction network
        cls.network = create_reaction_network(
            model=cls.model,
            weighted=False,
            directed=False,
            nodes_to_remove=None,
        )

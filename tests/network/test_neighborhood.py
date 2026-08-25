# Imports
from __future__ import annotations

# Standard library imports
import unittest

# External Imports
import networkx as nx

from metworkpy.network.neighborhoods import neighborhood_map


class TestNeighborhoodMap(unittest.TestCase):
    def test_neighborhood_size_r0(self):
        test_graph = nx.karate_club_graph()
        neighborhood_size = neighborhood_map(
            len, test_graph, radius=0, include_node=True, processes=1
        )
        for size in neighborhood_size.values():
            self.assertEqual(size, 1)

    def test_neighborhood_size_r1(self):
        test_graph = nx.karate_club_graph()
        neighborhood_size = neighborhood_map(
            len, test_graph, radius=1, include_node=True, processes=1
        )
        expected_neighborhood_size = {
            node: int(c * (len(test_graph) - 1)) + 1
            for node, c in nx.degree_centrality(test_graph).items()
        }
        self.assertDictEqual(neighborhood_size, expected_neighborhood_size)

    def test_callable_class(self):
        class TestFn:
            def __init__(self):
                self.counter = 0

            def __call__(self, neighborhood: set) -> int:
                self.counter += 1
                return len(neighborhood)

        test_graph = nx.karate_club_graph()
        test_fn = TestFn()
        neighborhood_size = neighborhood_map(
            test_fn, test_graph, radius=1, include_node=True, processes=1
        )
        expected_neighborhood_size = {
            node: int(c * (len(test_graph) - 1)) + 1
            for node, c in nx.degree_centrality(test_graph).items()
        }
        self.assertDictEqual(neighborhood_size, expected_neighborhood_size)
        # Should have been called exactly once per node
        self.assertEqual(test_fn.counter, len(test_graph))


if __name__ == "__main__":
    unittest.main()

"""KDTree data testing.
"""

import copy
import unittest
import numpy

from .network_objects import network_lattice_1x1_geomelem
from .network_objects import network_empirical_simplified


class TestKDTreeLattice1x1(unittest.TestCase):
    def setUp(self):
        network = copy.deepcopy(network_lattice_1x1_geomelem)

        # build kd tree
        self.net_nodes_kdtree = network.nodes_kdtree()

        # build kd tree -- from coordinates only
        self.net_nodes_kdtree_coords_only = network.nodes_kdtree(only_coords=True)

    def test_net_nodes_kdtree_data(self):
        known_kdtree_data = numpy.array(
            [[4.5, 0.0], [4.5, 4.5], [4.5, 9.0], [0.0, 4.5], [9.0, 4.5]]
        )
        observed_kdtree_data = self.net_nodes_kdtree.data
        numpy.testing.assert_array_equal(observed_kdtree_data, known_kdtree_data)

    def test_net_nodes_kdtree_query_dists(self):
        known_kdtree_query_dists = numpy.array([0.0, 4.5, 6.36396103])
        observed_kdtree_query_dists = numpy.array(
            self.net_nodes_kdtree.query([4.5, 0.0], k=3)
        )[0, :]
        numpy.testing.assert_array_almost_equal(
            observed_kdtree_query_dists, known_kdtree_query_dists
        )

    def test_net_nodes_kdtree_query_indices(self):
        known_kdtree_query_indices = numpy.array([0.0, 1.0, 3.0])
        observed_kdtree_query_indices = numpy.array(
            self.net_nodes_kdtree.query([4.5, 0.0], k=3)
        )[1, :]
        numpy.testing.assert_array_equal(
            observed_kdtree_query_indices, known_kdtree_query_indices
        )

    def test_net_nodes_kdtree_data_only_coords(self):
        known_kdtree_data = numpy.array(
            [[4.5, 0.0], [4.5, 4.5], [4.5, 9.0], [0.0, 4.5], [9.0, 4.5]]
        )
        observed_kdtree_data = self.net_nodes_kdtree.data
        numpy.testing.assert_array_equal(observed_kdtree_data, known_kdtree_data)

    def test_net_nodes_kdtree_query_dists_only_coords(self):
        known_kdtree_query_dists = numpy.array([0.0, 4.5, 6.36396103])
        observed_kdtree_query_dists = numpy.array(
            self.net_nodes_kdtree.query([4.5, 0.0], k=3)
        )[0, :]
        numpy.testing.assert_array_almost_equal(
            observed_kdtree_query_dists, known_kdtree_query_dists
        )

    def test_net_nodes_kdtree_query_indices_only_coords(self):
        known_kdtree_query_indices = numpy.array([0.0, 1.0, 3.0])
        observed_kdtree_query_indices = numpy.array(
            self.net_nodes_kdtree.query([4.5, 0.0], k=3)
        )[1, :]
        numpy.testing.assert_array_equal(
            observed_kdtree_query_indices, known_kdtree_query_indices
        )


class TestKDTreeEmpirical(unittest.TestCase):
    def setUp(self):
        network = copy.deepcopy(network_empirical_simplified)

        # build kd tree
        self.net_nodes_kdtree = network.nodes_kdtree()

    def test_net_nodes_kdtree_data(self):
        known_kdtree_data = numpy.array(
            [
                [624484.3321678, 165959.33602943],
                [621472.15969302, 164456.34801777],
                [623580.79853327, 164531.84479675],
                [623531.61385791, 164455.13730139],
                [622213.77398256, 166384.29556895],
            ]
        )
        observed_kdtree_data = self.net_nodes_kdtree.data[-5:, :]
        numpy.testing.assert_array_almost_equal(observed_kdtree_data, known_kdtree_data)

    def test_net_nodes_kdtree_query_dists(self):
        known_kdtree_query_dists = numpy.array([0.0, 138.50270336, 369.19792667])
        observed_kdtree_query_dists = numpy.array(
            self.net_nodes_kdtree.query(self.net_nodes_kdtree.data[-1, :], k=3)
        )[0, :]
        numpy.testing.assert_array_almost_equal(
            observed_kdtree_query_dists, known_kdtree_query_dists
        )

    def test_net_nodes_kdtree_query_indices(self):
        known_kdtree_query_indices = numpy.array([285.0, 195.0, 196.0])
        observed_kdtree_query_indices = numpy.array(
            self.net_nodes_kdtree.query(self.net_nodes_kdtree.data[-1, :], k=3)
        )[1, :]
        numpy.testing.assert_array_equal(
            observed_kdtree_query_indices, known_kdtree_query_indices
        )


if __name__ == "__main__":
    unittest.main()

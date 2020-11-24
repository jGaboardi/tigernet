"""KDTree data testing.
"""

import tigernet
import unittest
import numpy


class TestKDTreeLattice1x1(unittest.TestCase):
    def setUp(self):
        # instantiate network
        lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        network = tigernet.Network(s_data=lattice, record_geom=True)

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
        pass

        """
        
        self.network
        
        # build kd tree
        self.net_nodes_kdtree = self.network.nodes_kdtree()
        """

    """
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
    """


if __name__ == "__main__":
    unittest.main()

"""Cost matrix and shorted path tree testing.
"""


import copy
import unittest
import numpy

from .network_objects import network_lattice_1x1_wcm_attr
from .network_objects import network_lattice_1x1_wpaths_attr
from .network_objects import network_lattice_1x1_wcm_var
from .network_objects import network_lattice_1x1_wpaths_var

from .network_objects import network_lattice_2x1x1_wcm_attr
from .network_objects import network_lattice_2x1x1_wpaths_attr
from .network_objects import network_lattice_2x1x1_wcm_var
from .network_objects import network_lattice_2x1x1_wpaths_var

from .network_objects import graph_barb_wcm_copy_attr
from .network_objects import graph_barb_wpaths_copy_attr
from .network_objects import graph_barb_wcm_copy_var
from .network_objects import graph_barb_wpaths_copy_var

from .network_objects import network_barb_wcm_inplace_attr
from .network_objects import network_barb_wpaths_inplace_attr
from .network_objects import network_barb_wcm_inplace_var
from .network_objects import network_barb_wpaths_inplace_var

from .network_objects import network_empirical_simplified_wcm


inf = numpy.inf


class TestNetworkCostMatrixLattice1x1_1(unittest.TestCase):
    def setUp(self):
        # generate cost matrix without paths as an attribute
        self.network_mtx = copy.deepcopy(network_lattice_1x1_wcm_attr)

        # generate cost matrix with paths as an attribute
        self.network_mtx_paths = copy.deepcopy(network_lattice_1x1_wpaths_attr)

        # generate cost matrix without paths as an attribute
        self.matrix = copy.deepcopy(network_lattice_1x1_wcm_var)

        # generate cost matrix with paths as an attribute
        self.paths = copy.deepcopy(network_lattice_1x1_wpaths_var)

    def test_network_cost_matrix_attr(self):
        known_cost_matrix = numpy.array(
            [
                [0.0, 4.5, 9.0, 9.0, 9.0],
                [4.5, 0.0, 4.5, 4.5, 4.5],
                [9.0, 4.5, 0.0, 9.0, 9.0],
                [9.0, 4.5, 9.0, 0.0, 9.0],
                [9.0, 4.5, 9.0, 9.0, 0.0],
            ]
        )
        observed_cost_matrix = self.network_mtx.n2n_matrix
        numpy.testing.assert_array_equal(observed_cost_matrix, known_cost_matrix)

    def test_network_paths_attr(self):
        known_paths = {
            0: {0: [0], 1: [0], 2: [1, 0], 3: [1, 0], 4: [1, 0]},
            1: {0: [1], 1: [1], 2: [1], 3: [1], 4: [1]},
            2: {0: [1, 2], 1: [2], 2: [2], 3: [1, 2], 4: [1, 2]},
            3: {0: [1, 3], 1: [3], 2: [1, 3], 3: [3], 4: [1, 3]},
            4: {0: [1, 4], 1: [4], 2: [1, 4], 3: [1, 4], 4: [4]},
        }
        observed_paths = self.network_mtx_paths.n2n_paths
        self.assertEqual(observed_paths, known_paths)

    def test_network_cost_matrix_return(self):
        known_cost_matrix = numpy.array(
            [
                [0.0, 4.5, 9.0, 9.0, 9.0],
                [4.5, 0.0, 4.5, 4.5, 4.5],
                [9.0, 4.5, 0.0, 9.0, 9.0],
                [9.0, 4.5, 9.0, 0.0, 9.0],
                [9.0, 4.5, 9.0, 9.0, 0.0],
            ]
        )
        observed_cost_matrix = self.matrix
        numpy.testing.assert_array_equal(observed_cost_matrix, known_cost_matrix)

    def test_network_paths_return(self):
        known_paths = {
            0: {0: [0], 1: [0], 2: [1, 0], 3: [1, 0], 4: [1, 0]},
            1: {0: [1], 1: [1], 2: [1], 3: [1], 4: [1]},
            2: {0: [1, 2], 1: [2], 2: [2], 3: [1, 2], 4: [1, 2]},
            3: {0: [1, 3], 1: [3], 2: [1, 3], 3: [3], 4: [1, 3]},
            4: {0: [1, 4], 1: [4], 2: [1, 4], 3: [1, 4], 4: [4]},
        }
        observed_paths = self.paths
        self.assertEqual(observed_paths, known_paths)


class TestNetworkCostMatrixLattice1x1_2(unittest.TestCase):
    def setUp(self):
        # generate cost matrix without paths as an attribute
        self.network_mtx = copy.deepcopy(network_lattice_2x1x1_wcm_attr)

        # generate cost matrix with paths as an attribute
        self.network_mtx_paths = copy.deepcopy(network_lattice_2x1x1_wpaths_attr)

        # generate cost matrix without paths as an attribute
        self.matrix = copy.deepcopy(network_lattice_2x1x1_wcm_var)

        # generate cost matrix with paths as an attribute
        self.paths = copy.deepcopy(network_lattice_2x1x1_wpaths_var)

    def test_network_cost_matrix_attr(self):
        known_cost_matrix = numpy.array(
            [
                [0.0, 4.5, 9.0, 9.0, 9.0, inf, inf, inf, inf, inf],
                [4.5, 0.0, 4.5, 4.5, 4.5, inf, inf, inf, inf, inf],
                [9.0, 4.5, 0.0, 9.0, 9.0, inf, inf, inf, inf, inf],
                [9.0, 4.5, 9.0, 0.0, 9.0, inf, inf, inf, inf, inf],
                [9.0, 4.5, 9.0, 9.0, 0.0, inf, inf, inf, inf, inf],
                [inf, inf, inf, inf, inf, 0.0, 1.0, 2.0, 2.0, 2.0],
                [inf, inf, inf, inf, inf, 1.0, 0.0, 1.0, 1.0, 1.0],
                [inf, inf, inf, inf, inf, 2.0, 1.0, 0.0, 2.0, 2.0],
                [inf, inf, inf, inf, inf, 2.0, 1.0, 2.0, 0.0, 2.0],
                [inf, inf, inf, inf, inf, 2.0, 1.0, 2.0, 2.0, 0.0],
            ]
        )
        observed_cost_matrix = self.network_mtx.n2n_matrix
        numpy.testing.assert_array_equal(observed_cost_matrix, known_cost_matrix)

    def test_network_paths_attr(self):
        known_paths = {
            0: {
                0: [0],
                1: [0],
                2: [1, 0],
                3: [1, 0],
                4: [1, 0],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            1: {
                0: [1],
                1: [1],
                2: [1],
                3: [1],
                4: [1],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            2: {
                0: [1, 2],
                1: [2],
                2: [2],
                3: [1, 2],
                4: [1, 2],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            3: {
                0: [1, 3],
                1: [3],
                2: [1, 3],
                3: [3],
                4: [1, 3],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            4: {
                0: [1, 4],
                1: [4],
                2: [1, 4],
                3: [1, 4],
                4: [4],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            5: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [5],
                6: [5],
                7: [6, 5],
                8: [6, 5],
                9: [6, 5],
            },
            6: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [6],
                6: [6],
                7: [6],
                8: [6],
                9: [6],
            },
            7: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [6, 7],
                6: [7],
                7: [7],
                8: [6, 7],
                9: [6, 7],
            },
            8: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [6, 8],
                6: [8],
                7: [6, 8],
                8: [8],
                9: [6, 8],
            },
            9: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [6, 9],
                6: [9],
                7: [6, 9],
                8: [6, 9],
                9: [9],
            },
        }
        observed_paths = self.network_mtx_paths.n2n_paths
        self.assertEqual(observed_paths, known_paths)

    def test_network_cost_matrix_return(self):
        known_cost_matrix = numpy.array(
            [
                [0.0, 4.5, 9.0, 9.0, 9.0, inf, inf, inf, inf, inf],
                [4.5, 0.0, 4.5, 4.5, 4.5, inf, inf, inf, inf, inf],
                [9.0, 4.5, 0.0, 9.0, 9.0, inf, inf, inf, inf, inf],
                [9.0, 4.5, 9.0, 0.0, 9.0, inf, inf, inf, inf, inf],
                [9.0, 4.5, 9.0, 9.0, 0.0, inf, inf, inf, inf, inf],
                [inf, inf, inf, inf, inf, 0.0, 1.0, 2.0, 2.0, 2.0],
                [inf, inf, inf, inf, inf, 1.0, 0.0, 1.0, 1.0, 1.0],
                [inf, inf, inf, inf, inf, 2.0, 1.0, 0.0, 2.0, 2.0],
                [inf, inf, inf, inf, inf, 2.0, 1.0, 2.0, 0.0, 2.0],
                [inf, inf, inf, inf, inf, 2.0, 1.0, 2.0, 2.0, 0.0],
            ]
        )
        observed_cost_matrix = self.matrix
        numpy.testing.assert_array_equal(observed_cost_matrix, known_cost_matrix)

    def test_network_paths_return(self):
        known_paths = {
            0: {
                0: [0],
                1: [0],
                2: [1, 0],
                3: [1, 0],
                4: [1, 0],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            1: {
                0: [1],
                1: [1],
                2: [1],
                3: [1],
                4: [1],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            2: {
                0: [1, 2],
                1: [2],
                2: [2],
                3: [1, 2],
                4: [1, 2],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            3: {
                0: [1, 3],
                1: [3],
                2: [1, 3],
                3: [3],
                4: [1, 3],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            4: {
                0: [1, 4],
                1: [4],
                2: [1, 4],
                3: [1, 4],
                4: [4],
                5: [5],
                6: [6],
                7: [7],
                8: [8],
                9: [9],
            },
            5: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [5],
                6: [5],
                7: [6, 5],
                8: [6, 5],
                9: [6, 5],
            },
            6: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [6],
                6: [6],
                7: [6],
                8: [6],
                9: [6],
            },
            7: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [6, 7],
                6: [7],
                7: [7],
                8: [6, 7],
                9: [6, 7],
            },
            8: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [6, 8],
                6: [8],
                7: [6, 8],
                8: [8],
                9: [6, 8],
            },
            9: {
                0: [0],
                1: [1],
                2: [2],
                3: [3],
                4: [4],
                5: [6, 9],
                6: [9],
                7: [6, 9],
                8: [6, 9],
                9: [9],
            },
        }
        observed_paths = self.paths
        self.assertEqual(observed_paths, known_paths)


class TestNetworkCostMatrixSimplifyBarb(unittest.TestCase):
    def setUp(self):
        # ---------------------------------------------------- copy
        self.graph_mtx_copy = graph_barb_wcm_copy_attr
        self.graph_mtx_paths_copy = graph_barb_wpaths_copy_attr
        self.graph_matrix_copy = graph_barb_wcm_copy_var
        self.graph_paths_copy = graph_barb_wpaths_copy_var

        # ---------------------------------------------------- inplace
        self.network_mtx_inplace = network_barb_wcm_inplace_attr
        self.network_mtx_paths_inplace = network_barb_wpaths_inplace_attr
        self.network_matrix_inplace = network_barb_wcm_inplace_var
        self.network_paths_inplace = network_barb_wpaths_inplace_var

    def test_graph_copy_cost_matrix_attr(self):
        known_cost_matrix = numpy.array(
            [[0.0, 4.5, 9.0], [4.5, 0.0, 4.5], [9.0, 4.5, 0.0]]
        )
        observed_cost_matrix = self.graph_mtx_copy.n2n_matrix
        numpy.testing.assert_array_equal(observed_cost_matrix, known_cost_matrix)

    def test_graph_copy_paths_attr(self):
        known_paths = {
            0: {0: [0], 1: [0], 2: [1, 0]},
            1: {0: [1], 1: [1], 2: [1]},
            2: {0: [1, 2], 1: [2], 2: [2]},
        }
        observed_paths = self.graph_mtx_paths_copy.n2n_paths
        self.assertEqual(observed_paths, known_paths)

    def test_graph_copy_cost_matrix_return(self):
        known_cost_matrix = numpy.array(
            [[0.0, 4.5, 9.0], [4.5, 0.0, 4.5], [9.0, 4.5, 0.0]]
        )
        observed_cost_matrix = self.graph_matrix_copy
        numpy.testing.assert_array_equal(observed_cost_matrix, known_cost_matrix)

    def test_graph_copy_paths_return(self):
        known_paths = {
            0: {0: [0], 1: [0], 2: [1, 0]},
            1: {0: [1], 1: [1], 2: [1]},
            2: {0: [1, 2], 1: [2], 2: [2]},
        }
        observed_paths = self.graph_paths_copy
        self.assertEqual(observed_paths, known_paths)

    def test_graph_inplace_cost_matrix_attr(self):
        known_cost_matrix = numpy.array(
            [[0.0, 4.5, 9.0], [4.5, 0.0, 4.5], [9.0, 4.5, 0.0]]
        )
        observed_cost_matrix = self.network_mtx_inplace.n2n_matrix
        numpy.testing.assert_array_equal(observed_cost_matrix, known_cost_matrix)

    def test_graph_inplace_paths_attr(self):
        known_paths = {
            0: {0: [0], 1: [0], 2: [1, 0]},
            1: {0: [1], 1: [1], 2: [1]},
            2: {0: [1, 2], 1: [2], 2: [2]},
        }
        observed_paths = self.network_mtx_paths_inplace.n2n_paths
        self.assertEqual(observed_paths, known_paths)

    def test_graph_inplace_cost_matrix_return(self):
        known_cost_matrix = numpy.array(
            [[0.0, 4.5, 9.0], [4.5, 0.0, 4.5], [9.0, 4.5, 0.0]]
        )
        observed_cost_matrix = self.network_matrix_inplace
        numpy.testing.assert_array_equal(observed_cost_matrix, known_cost_matrix)

    def test_graph_inplace_paths_return(self):
        known_paths = {
            0: {0: [0], 1: [0], 2: [1, 0]},
            1: {0: [1], 1: [1], 2: [1]},
            2: {0: [1, 2], 1: [2], 2: [2]},
        }
        observed_paths = self.network_paths_inplace
        self.assertEqual(observed_paths, known_paths)


class TestNetworkCostMatrixEmpircalGDF(unittest.TestCase):
    def setUp(self):
        # cost matrix
        self.matrix = network_empirical_simplified_wcm.n2n_matrix

        # shortest path trees
        self.paths = network_empirical_simplified_wcm.n2n_paths

    def test_network_cost_matrix(self):
        known_cost_matrix = numpy.array(
            [
                [0.0, 212.3835341, 167.23911279, 343.31388701],
                [212.3835341, 0.0, 120.53015797, 296.60493218],
                [167.23911279, 120.53015797, 0.0, 176.07477422],
                [343.31388701, 296.60493218, 176.07477422, 0.0],
            ]
        )
        observed_cost_matrix = self.matrix[:4, :4]
        numpy.testing.assert_array_almost_equal(observed_cost_matrix, known_cost_matrix)

    def test_network_paths_1(self):
        known_src, known_dest = 0, 0
        known_path = [0]
        observed_path = self.paths[known_src][known_dest]
        self.assertEqual(observed_path, known_path)

    def test_network_paths_2(self):
        known_src, known_dest = 19, 99
        known_path = [98, 18, 19]
        observed_path = self.paths[known_src][known_dest]
        self.assertEqual(observed_path, known_path)

    def test_network_paths_3(self):
        known_src, known_dest = 20, 17
        known_path = [
            16,
            15,
            2,
            1,
            57,
            56,
            73,
            188,
            101,
            199,
            216,
            148,
            154,
            110,
            21,
            20,
        ]
        observed_path = self.paths[known_src][known_dest]
        self.assertEqual(observed_path, known_path)


if __name__ == "__main__":
    unittest.main()

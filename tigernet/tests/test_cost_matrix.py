"""Cost matrix and shorted path tree testing.
"""

import tigernet
import unittest
import numpy

inf = numpy.inf


class TestNetworkCostMatrixLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)

        # generate cost matrix without paths as an attribute
        self.network_mtx = tigernet.Network(s_data=self.lattice.copy())
        self.network_mtx.cost_matrix()

        # generate cost matrix with paths as an attribute
        self.network_mtx_paths = tigernet.Network(s_data=self.lattice.copy())
        self.network_mtx_paths.cost_matrix(wpaths=True)

        # generate cost matrix without paths as an attribute
        self.matrix = self.network_mtx.cost_matrix(asattr=False)

        # generate cost matrix with paths as an attribute
        _, self.paths = self.network_mtx_paths.cost_matrix(wpaths=True, asattr=False)

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
        lat1 = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        lat2 = tigernet.generate_lattice(
            n_hori_lines=1, n_vert_lines=1, bounds=[6, 6, 8, 8]
        )
        self.lattices = lat1.append(lat2)
        self.lattices.reset_index(drop=True, inplace=True)

        # generate cost matrix without paths as an attribute
        self.network_mtx = tigernet.Network(
            s_data=self.lattices.copy(), record_components=True
        )
        self.network_mtx.cost_matrix()

        # generate cost matrix with paths as an attribute
        self.network_mtx_paths = tigernet.Network(
            s_data=self.lattices.copy(), record_components=True
        )
        self.network_mtx_paths.cost_matrix(wpaths=True)

        # generate cost matrix without paths as an attribute
        self.matrix = self.network_mtx.cost_matrix(asattr=False)

        # generate cost matrix with paths as an attribute
        _, self.paths = self.network_mtx_paths.cost_matrix(wpaths=True, asattr=False)

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
        lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1, wbox=True)
        self.barb = lattice[~lattice["SegID"].isin([1, 2, 5, 7, 9, 10])]
        kws = {"record_components": True, "record_geom": True, "def_graph_elems": True}

        # ---------------------------------------------------- copy
        # generate cost matrix without paths as an attribute
        self.network_mtx_copy = tigernet.Network(s_data=self.barb.copy())
        self.graph_mtx_copy = self.network_mtx_copy.simplify_network(**kws)
        self.graph_mtx_copy.cost_matrix()

        # generate cost matrix with paths as an attribute
        self.network_mtx_paths_copy = tigernet.Network(s_data=self.barb.copy())
        self.graph_mtx_paths_copy = self.network_mtx_paths_copy.simplify_network(**kws)
        self.graph_mtx_paths_copy.cost_matrix(wpaths=True)

        # generate cost matrix without paths as an attribute
        self.graph_matrix_copy = self.graph_mtx_copy.cost_matrix(asattr=False)

        # generate cost matrix with paths as an attribute
        _, self.graph_paths_copy = self.graph_mtx_copy.cost_matrix(
            wpaths=True, asattr=False
        )

        # ---------------------------------------------------- inplace
        # generate cost matrix without paths as an attribute
        self.network_mtx_inplace = tigernet.Network(s_data=self.barb.copy())
        self.network_mtx_inplace.simplify_network(inplace=True, **kws)
        self.network_mtx_inplace.cost_matrix()

        # generate cost matrix with paths as an attribute
        self.network_mtx_paths_inplace = tigernet.Network(s_data=self.barb.copy())
        self.network_mtx_paths_inplace.simplify_network(inplace=True, **kws)
        self.network_mtx_paths_inplace.cost_matrix(wpaths=True)

        # generate cost matrix without paths as an attribute
        self.network_matrix_inplace = self.network_mtx_inplace.cost_matrix(asattr=False)

        # generate cost matrix with paths as an attribute
        _, self.network_paths_inplace = self.network_mtx_inplace.cost_matrix(
            wpaths=True, asattr=False
        )

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
        pass


class TestGraphCostMatrixEmpircalGDF(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == "__main__":
    unittest.main()

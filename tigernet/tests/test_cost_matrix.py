"""Cost matrix and shorted path tree testing.
"""

import tigernet
import unittest
import numpy
import geopandas

inf = numpy.inf


# get the roads shapefile as a GeoDataFrame
bbox = (-84.279, 30.480, -84.245, 30.505)
f = "zip://test_data/Edges_Leon_FL_2010.zip!Edges_Leon_FL_2010.shp"
gdf = geopandas.read_file(f, bbox=bbox)
gdf = gdf.to_crs("epsg:2779")

# filter out only roads
yes_roads = gdf["ROADFLG"] == "Y"
roads = gdf[yes_roads].copy()

# Tiger attributes primary and secondary
ATTR1, ATTR2 = "MTFCC", "TLID"

# segment welding and splitting stipulations --------------------------------------------
INTRST = "S1100"  # interstates mtfcc code
RAMP = "S1630"  # ramp mtfcc code
SERV_DR = "S1640"  # service drive mtfcc code
SPLIT_GRP = "FULLNAME"  # grouped by this variable
SPLIT_BY = [RAMP, SERV_DR]  # split interstates by ramps & service
SKIP_RESTR = True  # no weld retry if still MLS


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

        # set up the network instantiation parameters
        discard_segs = None
        kwargs = {"s_data": roads.copy(), "from_raw": True}
        attr_kws = {"attr1": ATTR1, "attr2": ATTR2}
        kwargs.update(attr_kws)
        comp_kws = {"record_components": True, "largest_component": True}
        kwargs.update(comp_kws)
        geom_kws = {"record_geom": True, "calc_len": True}
        kwargs.update(geom_kws)
        mtfcc_kws = {"discard_segs": discard_segs, "skip_restr": SKIP_RESTR}
        mtfcc_kws.update({"mtfcc_split": INTRST, "mtfcc_intrst": INTRST})
        mtfcc_kws.update({"mtfcc_split_grp": SPLIT_GRP, "mtfcc_ramp": RAMP})
        mtfcc_kws.update({"mtfcc_split_by": SPLIT_BY, "mtfcc_serv": SERV_DR})
        kwargs.update(mtfcc_kws)

        # create a network instance
        self.network = tigernet.Network(**kwargs)

        # simplify network
        kws = {"record_components": True, "record_geom": True, "def_graph_elems": True}
        self.network.simplify_network(inplace=True, **kws)

        # cost matrix
        self.matrix = self.network.cost_matrix(asattr=False)

        # shortest path trees
        _, self.paths = self.network.cost_matrix(wpaths=True, asattr=False)

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

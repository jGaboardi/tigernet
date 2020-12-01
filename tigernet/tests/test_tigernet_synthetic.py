"""Synthetic data testing.
"""

import copy
import unittest
from .network_objects import network_lattice_1x1_no_args
from .network_objects import network_lattice_2x1x1_all
from .network_objects import network_lattice_2x1x1_largest
from .network_objects import network_lattice_1x1_geomelem
from .network_objects import graph_barb, network_barb


class TestNetworkBuildLattice1x1(unittest.TestCase):
    def setUp(self):
        self.network = network_lattice_1x1_no_args

    def test_lattice_network_sdata(self):
        known_segments = 4
        observed_segments = self.network.s_data.shape[0]
        self.assertEqual(observed_segments, known_segments)

        known_length = 18.0
        observed_length = self.network.s_data.length.sum()
        self.assertEqual(observed_length, known_length)

    def test_lattice_network_ndata(self):
        known_nodes = 5
        observed_nodes = self.network.n_data.shape[0]
        self.assertEqual(observed_nodes, known_nodes)

        known_bounds = [0.0, 0.0, 9.0, 9.0]
        observed_bounds = list(self.network.n_data.total_bounds)
        self.assertEqual(observed_bounds, known_bounds)

    def test_lattice_network_sdata_ids(self):
        known_ids = [0, 1, 2, 3]
        observed_ids = list(self.network.s_data["SegID"])
        self.assertEqual(observed_ids, known_ids)

    def test_lattice_network_ndata_ids(self):
        known_ids = [0, 1, 2, 3, 4]
        observed_ids = list(self.network.n_data["NodeID"])
        self.assertEqual(observed_ids, known_ids)

    def test_lattice_network_segm2xyid(self):
        known_xyid = {3: ["x4.5y4.5", "x9.0y4.5"]}
        observed_xyid = self.network.segm2xyid[3]
        self.assertEqual(observed_xyid, known_xyid[3])

    def test_lattice_network_node2xyid(self):
        known_xyid = {4: ["x9.0y4.5"]}
        observed_xyid = self.network.node2xyid[4]
        self.assertEqual(observed_xyid, known_xyid[4])

    def test_lattice_network_s_ids(self):
        known_ids = [0, 1, 2, 3]
        observed_ids = self.network.s_ids
        self.assertEqual(observed_ids, known_ids)

    def test_lattice_network_n_ids(self):
        known_ids = [0, 1, 2, 3, 4]
        observed_ids = self.network.n_ids
        self.assertEqual(observed_ids, known_ids)

    def test_lattice_network_n_segm(self):
        known_segm_count, observed_segm_count = 4, self.network.n_segm
        self.assertEqual(observed_segm_count, known_segm_count)

    def test_lattice_network_n_node(self):
        known_node_count, observed_node_count = 5, self.network.n_node
        self.assertEqual(observed_node_count, known_node_count)


class TestNetworkTopologyLattice1x1(unittest.TestCase):
    def setUp(self):
        self.network = network_lattice_1x1_no_args

    def test_lattice_network_segm2node(self):
        known_segm2node = {0: [0, 1], 1: [1, 2], 2: [1, 3], 3: [1, 4]}
        observed_segm2node = self.network.segm2node
        self.assertEqual(observed_segm2node, known_segm2node)

    def test_lattice_network_node2segm(self):
        known_node2segm = {0: [0], 1: [0, 1, 2, 3], 2: [1], 3: [2], 4: [3]}
        observed_node2segm = self.network.node2segm
        self.assertEqual(observed_node2segm, known_node2segm)

    def test_lattice_network_segm2segm(self):
        known_segm2segm = {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [0, 1, 2],
        }
        observed_segm2segm = self.network.segm2segm
        self.assertEqual(observed_segm2segm, known_segm2segm)

    def test_lattice_network_node2node(self):
        known_node2node = {0: [1], 1: [0, 2, 3, 4], 2: [1], 3: [1], 4: [1]}
        observed_node2node = self.network.node2node
        self.assertEqual(observed_node2node, known_node2node)


class TestNetworkComponentsLattice1x1(unittest.TestCase):
    def setUp(self):
        # full network
        self.network = network_lattice_2x1x1_all

        # largest component network
        self.network_largest_cc = network_lattice_2x1x1_largest

    def test_lattice_network_segm_components(self):
        known_ccs = {1: [0, 1, 2, 3], 5: [4, 5, 6, 7]}
        observed_ccs = self.network.segm_cc
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = {1: 18.0, 5: 4.0}
        observed_cc_lens = self.network.cc_lens
        self.assertEqual(observed_cc_lens, known_cc_lens)

        known_ccs = 2
        observed_ccs = self.network.n_ccs
        self.assertEqual(observed_ccs, known_ccs)

        known_segms_in_ccs = 8
        observed_segms_in_ccs = self.network.n_segm
        self.assertEqual(observed_segms_in_ccs, known_segms_in_ccs)

    def test_lattice_network_sdata_components(self):
        known_ccs = [1, 1, 1, 1, 5, 5, 5, 5]
        observed_ccs = list(self.network.s_data["CC"])
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = [18.0, 18.0, 18.0, 18.0, 4.0, 4.0, 4.0, 4.0]
        observed_cc_lens = list(self.network.s_data["ccLength"])
        self.assertEqual(observed_cc_lens, known_cc_lens)

    def test_lattice_network_node_components(self):
        known_ccs = {1: [0, 1, 2, 3, 4], 6: [5, 6, 7, 8, 9]}
        observed_ccs = self.network.node_cc
        self.assertEqual(observed_ccs, known_ccs)

    def test_lattice_network_ndata_components(self):
        known_ccs = [1, 1, 1, 1, 1, 6, 6, 6, 6, 6]
        observed_ccs = list(self.network.n_data["CC"])
        self.assertEqual(observed_ccs, known_ccs)

    def test_lattice_network_segm_components_largest(self):
        known_ccs = {1: [0, 1, 2, 3]}
        observed_ccs = self.network_largest_cc.segm_cc
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = {1: 18.0}
        observed_cc_lens = self.network_largest_cc.cc_lens
        self.assertEqual(observed_cc_lens, known_cc_lens)

        known_ccs = 1
        observed_ccs = self.network_largest_cc.n_ccs
        self.assertEqual(observed_ccs, known_ccs)

        known_segms_in_ccs = 4
        observed_segms_in_ccs = self.network_largest_cc.n_segm
        self.assertEqual(observed_segms_in_ccs, known_segms_in_ccs)

    def test_lattice_network_sdata_components_largest(self):
        known_ccs = [1, 1, 1, 1]
        observed_ccs = list(self.network_largest_cc.s_data["CC"])
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = [18.0, 18.0, 18.0, 18.0]
        observed_cc_lens = list(self.network_largest_cc.s_data["ccLength"])
        self.assertEqual(observed_cc_lens, known_cc_lens)

    def test_lattice_network_node_components_largest(self):
        known_ccs = {1: [0, 1, 2, 3, 4]}
        observed_ccs = self.network_largest_cc.node_cc
        self.assertEqual(observed_ccs, known_ccs)

    def test_lattice_network_ndata_components_largest(self):
        known_ccs = [1, 1, 1, 1, 1]
        observed_ccs = list(self.network_largest_cc.n_data["CC"])


class TestNetworkAssociationsLattice1x1(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_lattice_1x1_geomelem)

    def test_lattice_network_segm2geom(self):
        known_type = "LineString"
        observed_type = self.network.segm2geom[0].geom_type
        self.assertEqual(observed_type, known_type)

        known_wkt = "LINESTRING (4.5 0, 4.5 4.5)"
        observed_wkt = self.network.segm2geom[0].wkt
        self.assertEqual(observed_wkt, known_wkt)

    def test_lattice_network_segm2coords(self):
        known_lookup = {
            0: [(4.5, 0.0), (4.5, 4.5)],
            1: [(4.5, 4.5), (4.5, 9.0)],
            2: [(0.0, 4.5), (4.5, 4.5)],
            3: [(4.5, 4.5), (9.0, 4.5)],
        }
        observed_lookup = self.network.segm2coords
        self.assertEqual(observed_lookup, known_lookup)

    def test_lattice_network_node2geom(self):
        known_type = "Point"
        observed_type = self.network.node2geom[0].geom_type
        self.assertEqual(observed_type, known_type)

        known_wkt = "POINT (4.5 0)"
        observed_wkt = self.network.node2geom[0].wkt
        self.assertEqual(observed_wkt, known_wkt)

    def test_lattice_network_node2coords(self):
        known_lookup = {
            0: [(4.5, 0.0)],
            1: [(4.5, 4.5)],
            2: [(4.5, 9.0)],
            3: [(0.0, 4.5)],
            4: [(9.0, 4.5)],
        }
        observed_lookup = self.network.node2coords
        self.assertEqual(observed_lookup, known_lookup)

    def test_lattice_network_length(self):
        known_length, observed_length = 18.0, self.network.network_length
        self.assertEqual(observed_length, known_length)

    def test_lattice_node2degree(self):
        known_node2degree = {0: 1, 1: 4, 2: 1, 3: 1, 4: 1}
        observed_node2degree = self.network.node2degree
        self.assertEqual(observed_node2degree, known_node2degree)

    def test_lattice_ndata_degree(self):
        known_degree = [1, 4, 1, 1, 1]
        observed_degree = list(self.network.n_data["degree"])
        self.assertEqual(observed_degree, known_degree)


class TestNetworkDefineGraphElementsLattice1x1(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_lattice_1x1_geomelem)

    def test_lattice_network_segm2elem(self):
        known_elements = {0: "leaf", 1: "leaf", 2: "leaf", 3: "leaf"}
        observed_elements = self.network.segm2elem
        self.assertEqual(observed_elements, known_elements)

    def test_lattice_network_sdata_segm2elem(self):
        known_elements = ["leaf", "leaf", "leaf", "leaf"]
        observed_elements = list(self.network.s_data["graph_elem"])
        self.assertEqual(observed_elements, known_elements)

    def test_lattice_network_node2elem(self):
        known_elements = {
            0: "leaf",
            1: "branch",
            2: "leaf",
            3: "leaf",
            4: "leaf",
        }
        observed_elements = self.network.node2elem
        self.assertEqual(observed_elements, known_elements)

    def test_lattice_network_ndata_node2elem(self):
        known_elements = ["leaf", "branch", "leaf", "leaf", "leaf"]
        observed_elements = list(self.network.n_data["graph_elem"])
        self.assertEqual(observed_elements, known_elements)


class TestNetworkSimplifyBarb(unittest.TestCase):
    def setUp(self):
        # copy testing
        self.graph = graph_barb
        # inplace
        self.network = network_barb

    def test_simplify_copy_segm2xyid(self):
        known_xyids = {
            0: ["x4.5y0.0", "x4.5y4.5"],
            1: ["x4.5y4.5", "x9.0y4.5", "x9.0y9.0", "x4.5y9.0", "x4.5y4.5"],
            2: ["x0.0y4.5", "x4.5y4.5"],
        }
        observed_xyids = self.graph.segm2xyid
        self.assertEqual(observed_xyids, known_xyids)

    def test_simplify_copy_segm2coords(self):
        known_coords = {
            0: [(4.5, 0.0), (4.5, 4.5)],
            1: [(4.5, 4.5), (9.0, 4.5), (9.0, 9.0), (4.5, 9.0), (4.5, 4.5)],
            2: [(0.0, 4.5), (4.5, 4.5)],
        }
        observed_coords = self.graph.segm2coords
        self.assertEqual(observed_coords, known_coords)

    def test_simplify_copy_segm2elem(self):
        known_elements = {0: "leaf", 1: "branch", 2: "leaf"}
        observed_elements = self.graph.segm2elem
        self.assertEqual(observed_elements, known_elements)

    def test_simplify_copy_segm_cc(self):
        known_ccs = {1: [0, 1, 2]}
        observed_ccs = self.graph.segm_cc
        self.assertEqual(observed_ccs, known_ccs)

    def test_simplify_copy_segm2len(self):
        known_lens = {0: 4.5, 1: 18.0, 2: 4.5}
        observed_lens = self.graph.segm2len
        self.assertEqual(observed_lens, known_lens)

    def test_simplify_copy_node2xyid(self):
        known_xyids = {0: ["x4.5y0.0"], 1: ["x4.5y4.5"], 2: ["x0.0y4.5"]}
        observed_xyids = self.graph.node2xyid
        self.assertEqual(observed_xyids, known_xyids)

    def test_simplify_copy_node2coords(self):
        known_coords = {0: [(4.5, 0.0)], 1: [(4.5, 4.5)], 2: [(0.0, 4.5)]}
        observed_coords = self.graph.node2coords
        self.assertEqual(observed_coords, known_coords)

    def test_simplify_copy_node2elem(self):
        known_elements = {0: "leaf", 1: "branch", 2: "leaf"}
        observed_elements = self.graph.node2elem
        self.assertEqual(observed_elements, known_elements)

    def test_simplify_copy_node_cc(self):
        known_ccs = {1: [0, 1, 2]}
        observed_ccs = self.graph.node_cc
        self.assertEqual(observed_ccs, known_ccs)

    def test_simplify_copy_node2degree(self):
        known_degree = {0: 1, 1: 4, 2: 1}
        observed_degree = self.graph.node2degree
        self.assertEqual(observed_degree, known_degree)

    def test_simplify_inplace_segm2xyid(self):
        known_xyids = {
            0: ["x4.5y0.0", "x4.5y4.5"],
            1: ["x4.5y4.5", "x9.0y4.5", "x9.0y9.0", "x4.5y9.0", "x4.5y4.5"],
            2: ["x0.0y4.5", "x4.5y4.5"],
        }
        observed_xyids = self.network.segm2xyid
        self.assertEqual(observed_xyids, known_xyids)

    def test_simplify_inplace_segm2coords(self):
        known_coords = {
            0: [(4.5, 0.0), (4.5, 4.5)],
            1: [(4.5, 4.5), (9.0, 4.5), (9.0, 9.0), (4.5, 9.0), (4.5, 4.5)],
            2: [(0.0, 4.5), (4.5, 4.5)],
        }
        observed_coords = self.network.segm2coords
        self.assertEqual(observed_coords, known_coords)

    def test_simplify_inplace_segm2elem(self):
        known_elements = {0: "leaf", 1: "branch", 2: "leaf"}
        observed_elements = self.network.segm2elem
        self.assertEqual(observed_elements, known_elements)

    def test_simplify_inplace_segm_cc(self):
        known_ccs = {1: [0, 1, 2]}
        observed_ccs = self.network.segm_cc
        self.assertEqual(observed_ccs, known_ccs)

    def test_simplify_inplace_segm2len(self):
        known_lens = {0: 4.5, 1: 18.0, 2: 4.5}
        observed_lens = self.network.segm2len
        self.assertEqual(observed_lens, known_lens)

    def test_simplify_inplace_node2xyid(self):
        known_xyids = {0: ["x4.5y0.0"], 1: ["x4.5y4.5"], 2: ["x0.0y4.5"]}
        observed_xyids = self.network.node2xyid
        self.assertEqual(observed_xyids, known_xyids)

    def test_simplify_inplace_node2coords(self):
        known_coords = {0: [(4.5, 0.0)], 1: [(4.5, 4.5)], 2: [(0.0, 4.5)]}
        observed_coords = self.network.node2coords
        self.assertEqual(observed_coords, known_coords)

    def test_simplify_inplace_node2elem(self):
        known_elements = {0: "leaf", 1: "branch", 2: "leaf"}
        observed_elements = self.network.node2elem
        self.assertEqual(observed_elements, known_elements)

    def test_simplify_inplace_node_cc(self):
        known_ccs = {1: [0, 1, 2]}
        observed_ccs = self.network.node_cc
        self.assertEqual(observed_ccs, known_ccs)

    def test_simplify_inplace_node2degree(self):
        known_degree = {0: 1, 1: 4, 2: 1}
        observed_degree = self.network.node2degree
        self.assertEqual(observed_degree, known_degree)


if __name__ == "__main__":
    unittest.main()

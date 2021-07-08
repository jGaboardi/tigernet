"""Empirical data testing from a geopandas.GeoDataFrame.
"""

import copy
import unittest
import numpy

from .network_objects import network_empirical_lcc
from .network_objects import network_empirical_full
from .network_objects import graph_empirical_simplified
from .network_objects import network_empirical_simplified

import platform

os = platform.platform()[:7].lower()
if os == "windows":
    WINDOWS = True
    DECIMAL = -1
else:
    WINDOWS = False
    DECIMAL = 1
# WINDOWS = False
# DECIMAL = 1


class TestNetworkBuildEmpiricalGDF(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_lcc)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_sdata(self):
        known_segments = 408
        observed_segments = self.network.s_data.shape[0]
        self.assertEqual(observed_segments, known_segments)

        known_length = 75155.6197099787
        observed_length = self.network.s_data.length.sum()
        numpy.testing.assert_almost_equal(
            observed_length, known_length, decimal=DECIMAL
        )

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_ndata(self):
        known_nodes = 349
        observed_nodes = self.network.n_data.shape[0]
        self.assertEqual(observed_nodes, known_nodes)

        known_bounds = numpy.array(
            [
                620989.3023002351,
                163937.37839259504,
                624605.9723871874,
                167048.6931314568,
            ]
        )
        observed_bounds = numpy.array(self.network.n_data.total_bounds)
        numpy.testing.assert_array_almost_equal(
            observed_bounds, known_bounds, decimal=DECIMAL
        )

    def test_network_sdata_ids(self):
        known_ids = [412, 414, 415, 416, 417]
        observed_ids = list(self.network.s_data["SegID"])[-5:]
        self.assertEqual(observed_ids, known_ids)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_ndata_ids(self):
        known_ids = [357, 360, 361, 362, 363]
        observed_ids = list(self.network.n_data["NodeID"])[-5:]
        self.assertEqual(observed_ids, known_ids)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_segm2xyid(self):
        known_id_xyid = {
            417: [
                "x622213.7739825583y166384.2955689532",
                "x622195.0607060504y166381.4862537613",
                "x622184.5013186845y166381.46491087825",
                "x622181.715678818y166382.34615142326",
                "x622179.9855276581y166383.45124237766",
                "x622167.096623471y166396.0630971515",
                "x622153.7300268554y166407.56541679866",
                "x622145.5590545179y166413.20272313987",
                "x622143.8271135575y166415.1946855663",
                "x622142.3804582829y166418.51752763707",
                "x622140.6237954168y166432.70389406924",
                "x622140.7789228398y166450.88502889618",
                "x622139.3965777882y166469.83907459615",
                "x622137.935323471y166480.3677023711",
                "x622135.714887791y166486.57131120094",
                "x622131.5685194829y166495.76422229825",
            ],
        }
        known_id = list(known_id_xyid.keys())[0]
        known_xyid = list(known_id_xyid.values())[0]
        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_id = list(self.network.segm2xyid.keys())[-1]
        observed_xyid = list(self.network.segm2xyid.values())[-1]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(
            observed_xyid, known_xyid, decimal=DECIMAL
        )

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_node2xyid(self):
        known_id_xyid = {363: ["x622213.7739825583y166384.29556895603"]}
        known_id = list(known_id_xyid.keys())[0]
        known_xyid = list(known_id_xyid.values())[0]

        observed_id = list(self.network.node2xyid.keys())[-1]
        observed_xyid = list(self.network.node2xyid.values())[-1]

        self.assertEqual(observed_id, known_id)
        self.assertEqual(observed_xyid, known_xyid)

    def test_network_s_ids(self):
        known_ids = [412, 414, 415, 416, 417]
        observed_ids = self.network.s_ids[-5:]
        self.assertEqual(observed_ids, known_ids)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_n_ids(self):
        known_ids = [357, 360, 361, 362, 363]
        observed_ids = self.network.n_ids[-5:]
        self.assertEqual(observed_ids, known_ids)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_n_segm(self):
        known_segm_count, observed_segm_count = 408, self.network.n_segm
        self.assertEqual(observed_segm_count, known_segm_count)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_n_node(self):
        known_node_count, observed_node_count = 349, self.network.n_node
        self.assertEqual(observed_node_count, known_node_count)


class TestNeworkTopologyEmpiricalGDF(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_lcc)

    def test_network_segm2node(self):
        known_segm2node = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [5, 6],
            4: [7, 8],
        }
        observed_segm2node = self.network.segm2node
        for known_k, known_v in known_segm2node.items():
            self.assertEqual(observed_segm2node[known_k], known_v)

    def test_network_node2segm(self):
        known_node2segm = {
            0: [0, 5, 9],
            1: [0, 82, 284],
            2: [1, 5, 135, 139, 284],
            3: [1, 7, 8],
            4: [2, 12, 13],
        }
        observed_node2segm = self.network.node2segm
        for known_k, known_v in known_node2segm.items():
            self.assertEqual(observed_node2segm[known_k], known_v)

    def test_network_segm2segm(self):
        known_segm2segm = {
            0: [5, 9, 82, 284],
            1: [5, 7, 8, 135, 139, 284],
            2: [3, 12, 13, 237, 286],
            3: [2, 15, 18, 237, 286],
            4: [10, 139, 162, 295, 348],
        }
        observed_segm2segm = self.network.segm2segm
        for known_k, known_v in known_segm2segm.items():
            self.assertEqual(observed_segm2segm[known_k], known_v)

    def test_network_node2node(self):
        known_node2node = {
            0: [1, 2, 10],
            1: [0, 2, 63],
            2: [0, 1, 3, 7, 195],
            3: [2, 10, 11],
            4: [5, 13, 14],
        }
        observed_node2node = self.network.node2node
        for known_k, known_v in known_node2node.items():
            self.assertEqual(observed_node2node[known_k], known_v)


class TestNeworkComponentsEmpiricalGDF(unittest.TestCase):
    def setUp(self):
        # full network
        self.network = copy.deepcopy(network_empirical_full)
        # largest component network
        self.network_largest_cc = copy.deepcopy(network_empirical_lcc)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_segm_components(self):
        known_ccs = {106: [105, 106, 108, 110, 263, 264]}
        observed_ccs = self.network.segm_cc
        for known_k, known_v in known_ccs.items():
            self.assertEqual(observed_ccs[known_k], known_v)

        known_cc_lens = {
            74: 75155.6197099787,
            106: 771.0198272877047,
            166: 245.73023272124595,
            172: 22.726003852369274,
            312: 159.71061674935376,
            413: 41.69245093932556,
        }
        observed_cc_lens = self.network.cc_lens
        for k, v in known_cc_lens.items():
            numpy.testing.assert_almost_equal(observed_cc_lens[k], v)

        known_ccs = 6
        observed_ccs = self.network.n_ccs
        self.assertEqual(observed_ccs, known_ccs)

        known_segms_in_ccs = 418
        observed_segms_in_ccs = self.network.n_segm
        self.assertEqual(observed_segms_in_ccs, known_segms_in_ccs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_sdata_components(self):
        known_ccs = [74, 74, 74, 74, 74, 106, 106, 74, 106, 74]
        observed_ccs = list(self.network.s_data["CC"])[100:110]
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = [771.0198272868483, 75155.6197099787]
        observed_cc_lens = list(self.network.s_data["ccLength"])[108:110]
        for k, o in zip(known_cc_lens, observed_cc_lens):
            numpy.testing.assert_almost_equal(o, k)

    def test_network_node_components(self):
        known_ccs = {159: [158, 159, 160, 163, 164, 298, 299]}
        observed_ccs = self.network.node_cc
        for known_k, known_v in known_ccs.items():
            self.assertEqual(observed_ccs[known_k], known_v)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_ndata_components(self):
        known_ccs = [359, 30, 30, 30, 30]
        observed_ccs = list(self.network.n_data["CC"])[359:]
        self.assertEqual(observed_ccs, known_ccs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_segm_components_largest(self):
        known_ccs = {74: [0, 1, 2, 3, 4]}
        observed_ccs = self.network_largest_cc.segm_cc
        for known_k, known_v in known_ccs.items():
            self.assertEqual(observed_ccs[known_k][:5], known_v)

        known_ccs_k, known_cc_lens = 74, 75155.61970997874
        observed_cc_lens = self.network_largest_cc.cc_lens
        numpy.testing.assert_almost_equal(observed_cc_lens[known_ccs_k], known_cc_lens)

        known_ccs = 1
        observed_ccs = self.network_largest_cc.n_ccs
        self.assertEqual(observed_ccs, known_ccs)

        known_segms_in_ccs = 408
        observed_segms_in_ccs = self.network_largest_cc.n_segm
        self.assertEqual(observed_segms_in_ccs, known_segms_in_ccs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_sdata_components_largest(self):
        known_ccs = [74, 74, 74, 74, 74]
        observed_ccs = list(self.network_largest_cc.s_data["CC"])[:5]
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = [75155.6197099787, 75155.6197099787]
        observed_cc_lens = list(self.network_largest_cc.s_data["ccLength"])[:2]
        for k, o in zip(known_cc_lens, observed_cc_lens):
            numpy.testing.assert_almost_equal(o, k)

    def test_network_node_components_largest(self):
        known_ccs = {30: [0, 1, 2, 3, 4]}
        observed_ccs = self.network_largest_cc.node_cc
        for known_k, known_v in known_ccs.items():
            self.assertEqual(observed_ccs[known_k][:5], known_v)

    def test_network_ndata_components_largest(self):
        known_ccs = [30, 30, 30, 30, 30]
        observed_ccs = list(self.network_largest_cc.n_data["CC"])[:5]


class TestNetworkAssociationsEmpiricalGDF(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_lcc)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_segm2geom(self):
        known_type = "LineString"
        observed_type = self.network.segm2geom[0].geom_type
        self.assertEqual(observed_type, known_type)

        known_wkt = "LINESTRING (623605.9583105363 166441.9652821319, 623642.2579218673 166435.6136040619, 623661.2704885595 166432.9939635286, 623683.5540714423 166427.0556520971, 623703.2557014348 166416.5666028635, 623719.0358090349 166399.5284506257, 623728.2024647847 166377.8199573702, 623732.1909850998 166353.5505257068, 623732.1809754729 166313.9739520327)"
        observed_wkt = self.network.segm2geom[0].wkt[:20]
        self.assertEqual(observed_wkt, known_wkt[:20])

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_segm2coords(self):
        known_coords = numpy.array(
            [
                (622213.7739825583, 166384.2955689532),
                (622195.0607060504, 166381.4862537613),
                (622184.5013186845, 166381.46491087825),
                (622181.715678818, 166382.34615142326),
                (622179.9855276581, 166383.45124237766),
                (622167.096623471, 166396.0630971515),
                (622153.7300268554, 166407.56541679866),
                (622145.5590545179, 166413.20272313987),
                (622143.8271135575, 166415.1946855663),
                (622142.3804582829, 166418.51752763707),
                (622140.6237954168, 166432.70389406924),
                (622140.7789228398, 166450.88502889618),
                (622139.3965777882, 166469.83907459615),
                (622137.935323471, 166480.3677023711),
                (622135.714887791, 166486.57131120094),
                (622131.5685194829, 166495.76422229825),
            ],
        )

        observed_coords = numpy.array(self.network.segm2coords[417])
        numpy.testing.assert_array_almost_equal(
            observed_coords, known_coords, decimal=DECIMAL
        )

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_node2geom(self):
        known_type = "Point"
        observed_type = self.network.node2geom[0].geom_type
        self.assertEqual(observed_type, known_type)

        known_wkt = "POINT (623605.9583105363 166441.9652821319)"
        observed_wkt = self.network.node2geom[0].wkt[:20]
        self.assertEqual(observed_wkt, known_wkt[:20])

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_node2coords(self):
        known_coords = numpy.array([(623513.4343699274, 166702.7281705553)])
        observed_coords = numpy.array(self.network.node2coords[286])
        numpy.testing.assert_array_almost_equal(
            observed_coords, known_coords, decimal=DECIMAL
        )

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_length(self):
        known_length, observed_length = 75155.61970997874, self.network.network_length
        numpy.testing.assert_almost_equal(
            observed_length, known_length, decimal=DECIMAL
        )

    def test_node2degree(self):
        known_node2degree = [(100, 3), (101, 3), (102, 2), (103, 3), (104, 3)]
        observed_node2degree = list(self.network.node2degree.items())[100:105]
        self.assertEqual(observed_node2degree, known_node2degree)

    def test_ndata_degree(self):
        known_degree = [3, 3, 5, 3, 3, 4, 3, 3, 4, 1, 3, 2, 1, 3, 1]
        observed_degree = list(self.network.n_data["degree"])[:15]
        self.assertEqual(observed_degree, known_degree)


class TestNetworkDefineGraphElementsEmpiricalGDF(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_lcc)

    def test_network_segm2elem(self):
        known_element_keys = [414, 415, 416, 417]
        known_element_values = ["leaf", "leaf", "leaf", "leaf"]
        observed_element_keys = list(self.network.segm2elem.keys())[-4:]
        observed_element_values = list(self.network.segm2elem.values())[-4:]
        self.assertEqual(observed_element_keys, known_element_keys)
        self.assertEqual(observed_element_values, known_element_values)

    def test_network_sdata_segm2elem(self):
        known_elements = ["leaf", "leaf", "leaf", "leaf"]
        observed_elements = list(self.network.s_data["graph_elem"])[-4:]
        self.assertEqual(observed_elements, known_elements)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_network_node2elem(self):
        known_element_keys = [360, 361, 362, 363]
        known_element_values = ["leaf", "leaf", "leaf", "leaf"]
        observed_element_keys = list(self.network.node2elem.keys())[-4:]
        observed_element_values = list(self.network.node2elem.values())[-4:]
        self.assertEqual(observed_element_keys, known_element_keys)
        self.assertEqual(observed_element_values, known_element_values)

    def test_network_ndata_node2elem(self):
        known_elements = ["leaf", "leaf", "leaf", "leaf"]
        observed_elements = list(self.network.n_data["graph_elem"])[-4:]
        self.assertEqual(observed_elements, known_elements)


class TestNetworkSimplifyEmpiricalGDF(unittest.TestCase):
    def setUp(self):
        # copy testing
        self.graph = copy.deepcopy(graph_empirical_simplified)

        # inplace
        self.network = copy.deepcopy(network_empirical_simplified)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_copy_segm2xyid(self):
        known_id = 344
        known_xyid = [
            "x623470.3003462269y164482.16590880448",
            "x623531.6138579083y164455.13730138965",
        ]
        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_xyid = self.graph.segm2xyid[known_id]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        numpy.testing.assert_array_almost_equal(observed_xyid, known_xyid)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_copy_segm2coords(self):
        known_id = 344
        known_coords = [
            (623470.3003462269, 164482.16590880448),
            (623531.6138579083, 164455.13730138965),
        ]
        observed_coords = self.graph.segm2coords[known_id]
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_copy_segm2elem(self):
        known_elements = {342: "leaf", 343: "leaf", 344: "leaf"}
        observed_elements = self.graph.segm2elem
        for k, v in known_elements.items():
            self.assertEqual(observed_elements[k], v)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_copy_segm_cc(self):
        known_root, known_ccs = 64, [341, 342, 343, 344, 345]
        observed_ccs = self.graph.segm_cc[known_root][-5:]
        self.assertEqual(observed_ccs, known_ccs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_copy_segm2len(self):
        known_id, known_len = 344, 67.00665887424545
        observed_len = self.graph.segm2len[known_id]
        self.assertAlmostEqual(observed_len, known_len)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_copy_node2xyid(self):
        known_id, known_xyid = 285, ["x623531.6138579083y164455.13730138965"]
        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_xyid = self.graph.node2xyid[known_id]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        numpy.testing.assert_array_almost_equal(observed_xyid, known_xyid)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_copy_node2coords(self):
        known_id, known_coords = 285, [(623531.6138579083, 164455.13730138965)]
        known_xyid = numpy.array(known_coords)
        observed_coords = numpy.array(self.graph.node2coords[known_id])
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_simplify_copy_node2elem(self):
        known_elements = {283: "leaf", 284: "leaf", 285: "leaf"}
        observed_elements = self.graph.node2elem
        for k, v in known_elements.items():
            self.assertEqual(observed_elements[k], v)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_copy_node_cc(self):
        known_root, known_ccs = 29, [282, 283, 284, 285, 286]
        observed_ccs = self.graph.node_cc[known_root]
        self.assertEqual(observed_ccs[-5:], known_ccs)

    def test_simplify_copy_node2degree(self):
        known_degree = [(0, 3), (1, 3), (2, 5), (3, 3)]
        observed_degree = list(self.graph.node2degree.items())[:4]
        self.assertEqual(observed_degree, known_degree)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_inplace_segm2xyid(self):
        known_id = 344
        known_xyid = [
            "x623470.3003462269y164482.16590880448",
            "x623531.6138579083y164455.13730138965",
        ]

        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_xyid = self.network.segm2xyid[known_id]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        numpy.testing.assert_array_almost_equal(observed_xyid, known_xyid)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_inplace_segm2coords(self):
        known_id = 344
        known_coords = [
            (623470.3003462269, 164482.16590880448),
            (623531.6138579083, 164455.13730138965),
        ]
        observed_coords = self.network.segm2coords[known_id]
        numpy.testing.assert_array_almost_equal(
            observed_coords, known_coords, decimal=DECIMAL
        )

    def test_simplify_inplace_segm2elem(self):
        known_elements = {342: "leaf", 343: "leaf", 344: "leaf"}
        observed_elements = self.network.segm2elem
        for k, v in known_elements.items():
            self.assertEqual(observed_elements[k], v)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_inplace_segm_cc(self):
        known_root, known_ccs = 64, [341, 342, 343, 344, 345]
        observed_ccs = self.network.segm_cc[known_root][-5:]
        self.assertEqual(observed_ccs, known_ccs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_inplace_segm2len(self):
        known_id, known_len = 344, 67.00665887424545
        observed_len = self.network.segm2len[known_id]
        self.assertAlmostEqual(observed_len, known_len)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_inplace_node2xyid(self):
        known_id, known_xyid = 285, ["x623531.6138579083y164455.13730138965"]
        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_xyid = self.network.node2xyid[known_id]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        numpy.testing.assert_array_almost_equal(observed_xyid, known_xyid)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_inplace_node2coords(self):
        known_id, known_coords = 285, [(623531.6138579083, 164455.13730138965)]
        known_xyid = numpy.array(known_coords)
        observed_coords = numpy.array(self.network.node2coords[known_id])
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_inplace_node2elem(self):
        known_elements = {283: "leaf", 284: "leaf", 285: "leaf"}
        observed_elements = self.network.node2elem
        for k, v in known_elements.items():
            self.assertEqual(observed_elements[k], v)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_simplify_inplace_node_cc(self):
        known_root, known_ccs = 29, [282, 283, 284, 285, 286]
        observed_ccs = self.network.node_cc[known_root]
        self.assertEqual(observed_ccs[-5:], known_ccs)

    def test_simplify_inplace_node2degree(self):
        known_degree = [(0, 3), (1, 3), (2, 5), (3, 3)]
        observed_degree = list(self.network.node2degree.items())[:4]
        self.assertEqual(observed_degree, known_degree)


if __name__ == "__main__":
    unittest.main()

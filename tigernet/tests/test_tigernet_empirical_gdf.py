"""Empirical data testing from a geopandas.GeoDataFrame.
"""

import tigernet
import unittest
import geopandas
import numpy

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


class TestNetworkBuildEmpiricalGDF(unittest.TestCase):
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

        # create a network isntance
        self.network = tigernet.Network(**kwargs)

    def test_network_sdata(self):
        known_segments = 407
        observed_segments = self.network.s_data.shape[0]
        self.assertEqual(observed_segments, known_segments)

        known_length = 74866.5821646358
        observed_length = self.network.s_data.length.sum()
        self.assertAlmostEqual(observed_length, known_length)

    def test_network_ndata(self):
        known_nodes = 348
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
        numpy.testing.assert_array_almost_equal(observed_bounds, known_bounds)

    def test_network_sdata_ids(self):
        known_ids = [412, 414, 415, 416, 417]
        observed_ids = list(self.network.s_data["SegID"])[-5:]
        self.assertEqual(observed_ids, known_ids)

    def test_network_ndata_ids(self):
        known_ids = [358, 361, 362, 363, 364]
        observed_ids = list(self.network.n_data["NodeID"])[-5:]
        self.assertEqual(observed_ids, known_ids)

    def test_network_segm2xyid(self):
        known_id_xyid = [
            417,
            [
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
        ]
        known_id = known_id_xyid[0]
        known_xyid = known_id_xyid[1]
        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_id_xyid = self.network.segm2xyid[-1]
        observed_id = observed_id_xyid[0]
        observed_xyid = observed_id_xyid[1]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(observed_xyid, known_xyid)

    def test_network_node2xyid(self):
        known_id_xyid = [364, ["x622213.7739825583y166384.2955689532"]]
        known_id = known_id_xyid[0]
        known_xyid = known_id_xyid[1]

        observed_id_xyid = self.network.node2xyid[-1]
        observed_id = observed_id_xyid[0]
        observed_xyid = observed_id_xyid[1]

        self.assertEqual(observed_id, known_id)
        self.assertEqual(observed_xyid, known_xyid)


class TestNeworkTopologyEmpiricalGDF(unittest.TestCase):
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

        # create a network isntance
        self.network = tigernet.Network(**kwargs)

    def test_network_segm2node(self):
        known_segm2node = [
            [0, [0, 1]],
            [1, [2, 3]],
            [2, [4, 5]],
            [3, [5, 6]],
            [4, [7, 8]],
        ]
        observed_segm2node = self.network.segm2node[:5]
        self.assertEqual(observed_segm2node, known_segm2node)

    def test_network_node2segm(self):
        known_node2segm = [
            [0, [0, 11, 12]],
            [1, [0, 82, 284]],
            [2, [1, 11, 135, 139, 284]],
            [3, [1, 9, 10]],
            [4, [2, 5, 6]],
        ]
        observed_node2segm = self.network.node2segm[:5]
        self.assertEqual(observed_node2segm, known_node2segm)

    def test_network_segm2segm(self):
        known_segm2segm = [
            [0, [11, 12, 82, 284]],
            [1, [135, 9, 10, 11, 139, 284]],
            [2, [3, 5, 6, 237, 286]],
            [3, [2, 237, 15, 18, 286]],
            [4, [17, 5]],
        ]
        observed_segm2segm = self.network.segm2segm[:5]
        self.assertEqual(observed_segm2segm, known_segm2segm)

    def test_network_node2node(self):
        known_node2node = [
            [0, [1, 2, 13]],
            [1, [0, 2, 63]],
            [2, [0, 1, 3, 195, 10]],
            [3, [2, 13, 14]],
            [4, [8, 9, 5]],
        ]
        observed_node2node = self.network.node2node[:5]
        self.assertEqual(observed_node2node, known_node2node)


class TestNeworkComponentsEmpiricalGDF(unittest.TestCase):
    def setUp(self):

        # set up the network instantiation parameters
        discard_segs = None
        kwargs = {"s_data": roads.copy(), "from_raw": True}
        attr_kws = {"attr1": ATTR1, "attr2": ATTR2}
        kwargs.update(attr_kws)
        comp_kws = {"record_components": True}
        kwargs.update(comp_kws)
        geom_kws = {"record_geom": True, "calc_len": True}
        kwargs.update(geom_kws)
        mtfcc_kws = {"discard_segs": discard_segs, "skip_restr": SKIP_RESTR}
        mtfcc_kws.update({"mtfcc_split": INTRST, "mtfcc_intrst": INTRST})
        mtfcc_kws.update({"mtfcc_split_grp": SPLIT_GRP, "mtfcc_ramp": RAMP})
        mtfcc_kws.update({"mtfcc_split_by": SPLIT_BY, "mtfcc_serv": SERV_DR})
        kwargs.update(mtfcc_kws)

        # full network
        self.network = tigernet.Network(**kwargs)

        # largest component network
        kwargs.update({"largest_component": True})
        self.network_largest_cc = tigernet.Network(**kwargs)

    def test_network_segm_components(self):
        known_ccs = [106, [105, 106, 108, 110, 263, 264]]
        observed_ccs = self.network.segm_cc[0]
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = {
            106: 771.0198272868483,
            131: 74866.5821646358,
            166: 245.73023272135015,
            172: 22.726003850972294,
            312: 159.71061675074486,
            394: 289.03754534132827,
            413: 41.69245093932556,
        }
        observed_cc_lens = self.network.cc_lens
        for k, v in known_cc_lens.items():
            self.assertAlmostEqual(observed_cc_lens[k], v)

        known_ccs = 7
        observed_ccs = self.network.n_ccs
        self.assertEqual(observed_ccs, known_ccs)

        known_segms_in_ccs = 418
        observed_segms_in_ccs = self.network.n_segm
        self.assertEqual(observed_segms_in_ccs, known_segms_in_ccs)

    def test_network_sdata_components(self):
        known_ccs = [131, 131, 131, 131, 131, 106, 106, 131, 106, 131]
        observed_ccs = list(self.network.s_data["CC"])[100:110]
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = [771.0198272868483, 74866.5821646358]
        observed_cc_lens = list(self.network.s_data["ccLength"])[108:110]
        for k, o in zip(known_cc_lens, observed_cc_lens):
            self.assertAlmostEqual(o, k)

    def test_network_node_components(self):
        known_ccs = [159, [158, 159, 160, 163, 164, 298, 299]]
        observed_ccs = self.network.node_cc[0]
        self.assertEqual(observed_ccs, known_ccs)

    def test_network_ndata_components(self):
        known_ccs = [360, 281, 281, 281, 281]
        observed_ccs = list(self.network.n_data["CC"])[360:]
        self.assertEqual(observed_ccs, known_ccs)

    def test_network_segm_components_largest(self):
        known_ccs_kv = [131, [0, 1, 2, 3, 4]]
        known_ccs_k = known_ccs_kv[0]
        known_ccs_v = known_ccs_kv[1]
        observed_ccs_kv = self.network_largest_cc.segm_cc
        observed_ccs_k = observed_ccs_kv[0]
        observed_ccs_v = observed_ccs_kv[1][:5]
        self.assertEqual(observed_ccs_k, known_ccs_k)
        self.assertEqual(observed_ccs_v, known_ccs_v)

        known_cc_lens = 74866.5821646358
        observed_cc_lens = self.network_largest_cc.cc_lens
        self.assertAlmostEqual(observed_cc_lens[known_ccs_k], known_cc_lens)

        known_ccs = 1
        observed_ccs = self.network_largest_cc.n_ccs
        self.assertEqual(observed_ccs, known_ccs)

        known_segms_in_ccs = 407
        observed_segms_in_ccs = self.network_largest_cc.n_segm
        self.assertEqual(observed_segms_in_ccs, known_segms_in_ccs)

    def test_network_sdata_components_largest(self):
        known_ccs = [131, 131, 131, 131, 131]
        observed_ccs = list(self.network_largest_cc.s_data["CC"])[:5]
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = [74866.5821646358, 74866.5821646358]
        observed_cc_lens = list(self.network_largest_cc.s_data["ccLength"])[:2]
        for k, o in zip(known_cc_lens, observed_cc_lens):
            self.assertAlmostEqual(o, k)

    def test_network_node_components_largest(self):
        known_ccs_kv = [281, [0, 1, 2, 3, 4]]
        known_ccs_k = known_ccs_kv[0]
        known_ccs_v = known_ccs_kv[1]
        observed_ccs_kv = self.network_largest_cc.node_cc
        observed_ccs_k = observed_ccs_kv[0]
        observed_ccs_v = observed_ccs_kv[1][:5]
        self.assertEqual(observed_ccs_k, known_ccs_k)
        self.assertEqual(observed_ccs_v, known_ccs_v)

    def test_network_ndata_components_largest(self):
        known_ccs = [281, 281, 281, 281, 281]
        observed_ccs = list(self.network_largest_cc.n_data["CC"])[:5]


class TestNetworkAssociationsEmpiricalGDF(unittest.TestCase):
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

        # network
        self.network = tigernet.Network(**kwargs)

    def test_network_segm2geom(self):
        known_type = "LineString"
        observed_type = self.network.segm2geom[0][1].geom_type
        self.assertEqual(observed_type, known_type)

        known_wkt = "LINESTRING (623605.9583105363 166441.9652821319, 623642.2579218673 166435.6136040619, 623661.2704885595 166432.9939635286, 623683.5540714423 166427.0556520971, 623703.2557014348 166416.5666028635, 623719.0358090349 166399.5284506257, 623728.2024647847 166377.8199573702, 623732.1909850998 166353.5505257068, 623732.1809754729 166313.9739520327)"
        observed_wkt = self.network.segm2geom[0][1].wkt[:25]
        self.assertEqual(observed_wkt, known_wkt[:25])

    def test_network_segm2coords(self):
        known_lookup = [
            417,
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
        ]
        observed_lookup = self.network.segm2coords[-1]
        self.assertEqual(observed_lookup[0], known_lookup[0])
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_lookup[1]), numpy.array(known_lookup[1])
        )

    def test_network_node2geom(self):
        known_type = "Point"
        observed_type = self.network.node2geom[0][1].geom_type
        self.assertEqual(observed_type, known_type)

        known_wkt = "POINT (623605.9583105363 166441.9652821319)"
        observed_wkt = self.network.node2geom[0][1].wkt[:25]
        self.assertEqual(observed_wkt, known_wkt[:25])

    def test_network_node2coords(self):
        known_lookup = [364, [(622213.7739825583, 166384.2955689532)]]
        observed_lookup = self.network.node2coords[-1]
        self.assertEqual(observed_lookup[0], known_lookup[0])
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_lookup[1]), numpy.array(known_lookup[1])
        )

    def test_network_s_ids(self):
        known_ids = [412, 414, 415, 416, 417]
        observed_ids = self.network.s_ids[-5:]
        self.assertEqual(observed_ids, known_ids)

    def test_network_n_ids(self):
        known_ids = [358, 361, 362, 363, 364]
        observed_ids = self.network.n_ids[-5:]
        self.assertEqual(observed_ids, known_ids)

    def test_network_n_segm(self):
        known_segm_count, observed_segm_count = 407, self.network.n_segm
        self.assertEqual(observed_segm_count, known_segm_count)

    def test_network_n_node(self):
        known_node_count, observed_node_count = 348, self.network.n_node
        self.assertEqual(observed_node_count, known_node_count)

    def test_network_length(self):
        known_length, observed_length = 74866.58216463577, self.network.network_length
        self.assertAlmostEqual(observed_length, known_length)

    def test_node2degree(self):
        known_node2degree = [(100, 3), (101, 3), (102, 2), (103, 3), (104, 3)]
        observed_node2degree = list(self.network.node2degree.items())[100:105]
        self.assertEqual(observed_node2degree, known_node2degree)

    def test_ndata_degree(self):
        known_degree = [3, 3, 5, 3, 3, 4, 3, 1, 3, 1, 3, 4, 1, 3, 2]
        observed_degree = list(self.network.n_data["degree"])[:15]
        self.assertEqual(observed_degree, known_degree)


class TestNetworkDefineGraphElementsEmpiricalGDF(unittest.TestCase):
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
        graph_elems_kws = {"def_graph_elems": True}
        kwargs.update(graph_elems_kws)
        mtfcc_kws = {"discard_segs": discard_segs, "skip_restr": SKIP_RESTR}
        mtfcc_kws.update({"mtfcc_split": INTRST, "mtfcc_intrst": INTRST})
        mtfcc_kws.update({"mtfcc_split_grp": SPLIT_GRP, "mtfcc_ramp": RAMP})
        mtfcc_kws.update({"mtfcc_split_by": SPLIT_BY, "mtfcc_serv": SERV_DR})
        kwargs.update(mtfcc_kws)

        # create a network isntance
        self.network = tigernet.Network(**kwargs)

    def test_network_segm2elem(self):
        known_elements = [[414, "leaf"], [415, "leaf"], [416, "leaf"], [417, "leaf"]]
        observed_elements = self.network.segm2elem[-4:]
        self.assertEqual(observed_elements, known_elements)

    def test_network_sdata_segm2elem(self):
        known_elements = ["leaf", "leaf", "leaf", "leaf"]
        observed_elements = list(self.network.s_data["graph_elem"])[-4:]
        self.assertEqual(observed_elements, known_elements)

    def test_network_node2elem(self):
        known_elements = [[361, "leaf"], [362, "leaf"], [363, "leaf"], [364, "leaf"]]
        observed_elements = self.network.node2elem[-4:]
        self.assertEqual(observed_elements, known_elements)

    def test_network_ndata_node2elem(self):
        known_elements = ["leaf", "leaf", "leaf", "leaf"]
        observed_elements = list(self.network.n_data["graph_elem"])[-4:]
        self.assertEqual(observed_elements, known_elements)


class TestNetworkSimplifyEmpiricalGDF(unittest.TestCase):
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

        # network
        self.network = tigernet.Network(**kwargs)

        # copy testing
        kws = {"record_components": True, "record_geom": True, "def_graph_elems": True}
        self.graph = self.network.simplify_network(**kws)

        # inplace
        self.network.simplify_network(inplace=True, **kws)

    def test_simplify_copy_segm2xyid(self):
        known_id_xyid = [
            344,
            [
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
        ]
        known_id = known_id_xyid[0]
        known_xyid = known_id_xyid[1]
        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_id_xyid = self.graph.segm2xyid[-1]
        observed_id = observed_id_xyid[0]
        observed_xyid = observed_id_xyid[1]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(observed_xyid, known_xyid)

    def test_simplify_copy_segm2coords(self):
        known_id_coords = [
            344,
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
        ]
        known_id = known_id_coords[0]
        known_coords = numpy.array(known_id_coords[1])
        observed_id_coords = self.graph.segm2coords[-1]
        observed_id = observed_id_coords[0]
        observed_coords = numpy.array(observed_id_coords[1])
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_simplify_copy_segm2elem(self):
        known_elements = [[342, "leaf"], [343, "leaf"], [344, "leaf"]]
        observed_elements = self.graph.segm2elem[-3:]
        self.assertEqual(observed_elements, known_elements)

    def test_simplify_copy_segm_cc(self):
        known_root_ccs = [[72, [340, 341, 342, 343, 344]]]
        known_root = known_root_ccs[0][0]
        known_ccs = known_root_ccs[0][1]
        observed_root_ccs = self.graph.segm_cc
        observed_root = observed_root_ccs[0][0]
        observed_ccs = observed_root_ccs[0][1][-5:]
        self.assertEqual(observed_root, known_root)
        self.assertEqual(observed_ccs, known_ccs)

    def test_simplify_copy_segm2len(self):
        known_id_len = [344, 165.09903569556914]
        known_id, known_len = known_id_len[0], known_id_len[1]
        observed_id_lens = self.graph.segm2len[-1]
        observed_id, observed_len = observed_id_lens[0], observed_id_lens[1]
        self.assertEqual(observed_id, known_id)
        self.assertAlmostEqual(observed_len, known_len)

    def test_simplify_copy_node2xyid(self):
        known_id_xyid = [285, ["x622213.7739825583y166384.2955689532"]]
        known_id = known_id_xyid[0]
        known_xyid = known_id_xyid[1]
        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_id_xyid = self.graph.node2xyid[-1]
        observed_id = observed_id_xyid[0]
        observed_xyid = observed_id_xyid[1]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(observed_xyid, known_xyid)

    def test_simplify_copy_node2coords(self):
        known_id_coords = [285, [(622213.7739825583, 166384.2955689532)]]
        known_id = known_id_coords[0]
        known_coords = numpy.array(known_id_coords[1])
        observed_id_coords = self.graph.node2coords[-1]
        observed_id = observed_id_coords[0]
        observed_coords = numpy.array(observed_id_coords[1])
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_simplify_copy_node2elem(self):
        known_elements = [[283, "leaf"], [284, "leaf"], [285, "leaf"]]
        observed_elements = self.graph.node2elem[-3:]
        self.assertEqual(observed_elements, known_elements)

    def test_simplify_copy_node_cc(self):
        known_root_ccs = [[89, [281, 282, 283, 284, 285]]]
        known_root = known_root_ccs[0][0]
        known_ccs = known_root_ccs[0][1]
        observed_root_ccs = self.graph.node_cc
        observed_root = observed_root_ccs[0][0]
        observed_ccs = observed_root_ccs[0][1][-5:]
        self.assertEqual(observed_root, known_root)
        self.assertEqual(observed_ccs, known_ccs)

    def test_simplify_copy_node2degree(self):
        known_degree = [(0, 3), (1, 3), (2, 5), (3, 3)]
        observed_degree = list(self.graph.node2degree.items())[:4]
        self.assertEqual(observed_degree, known_degree)

    def test_simplify_inplace_segm2xyid(self):
        known_id_xyid = [
            344,
            [
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
        ]
        known_id = known_id_xyid[0]
        known_xyid = known_id_xyid[1]
        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_id_xyid = self.network.segm2xyid[-1]
        observed_id = observed_id_xyid[0]
        observed_xyid = observed_id_xyid[1]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(observed_xyid, known_xyid)

    def test_simplify_inplace_segm2coords(self):
        known_id_coords = [
            344,
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
        ]
        known_id = known_id_coords[0]
        known_coords = numpy.array(known_id_coords[1])
        observed_id_coords = self.network.segm2coords[-1]
        observed_id = observed_id_coords[0]
        observed_coords = numpy.array(observed_id_coords[1])
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_simplify_inplace_segm2elem(self):
        known_elements = [[342, "leaf"], [343, "leaf"], [344, "leaf"]]
        observed_elements = self.network.segm2elem[-3:]
        self.assertEqual(observed_elements, known_elements)

    def test_simplify_inplace_segm_cc(self):
        known_root_ccs = [[72, [340, 341, 342, 343, 344]]]
        known_root = known_root_ccs[0][0]
        known_ccs = known_root_ccs[0][1]
        observed_root_ccs = self.network.segm_cc
        observed_root = observed_root_ccs[0][0]
        observed_ccs = observed_root_ccs[0][1][-5:]
        self.assertEqual(observed_root, known_root)
        self.assertEqual(observed_ccs, known_ccs)

    def test_simplify_inplace_segm2len(self):
        known_id_len = [344, 165.09903569556914]
        known_id, known_len = known_id_len[0], known_id_len[1]
        observed_id_lens = self.network.segm2len[-1]
        observed_id, observed_len = observed_id_lens[0], observed_id_lens[1]
        self.assertEqual(observed_id, known_id)
        self.assertAlmostEqual(observed_len, known_len)

    def test_simplify_inplace_node2xyid(self):
        known_id_xyid = [285, ["x622213.7739825583y166384.2955689532"]]
        known_id = known_id_xyid[0]
        known_xyid = known_id_xyid[1]
        known_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in known_xyid]
        )
        observed_id_xyid = self.network.node2xyid[-1]
        observed_id = observed_id_xyid[0]
        observed_xyid = observed_id_xyid[1]
        observed_xyid = numpy.array(
            [[float(c) for c in xy[1:].split("y")] for xy in observed_xyid]
        )
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(observed_xyid, known_xyid)

    def test_simplify_inplace_node2coords(self):
        known_id_coords = [285, [(622213.7739825583, 166384.2955689532)]]
        known_id = known_id_coords[0]
        known_coords = numpy.array(known_id_coords[1])
        observed_id_coords = self.network.node2coords[-1]
        observed_id = observed_id_coords[0]
        observed_coords = numpy.array(observed_id_coords[1])
        self.assertEqual(observed_id, known_id)
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_simplify_inplace_node2elem(self):
        known_elements = [[283, "leaf"], [284, "leaf"], [285, "leaf"]]
        observed_elements = self.network.node2elem[-3:]
        self.assertEqual(observed_elements, known_elements)

    def test_simplify_inplace_node_cc(self):
        known_root_ccs = [[89, [281, 282, 283, 284, 285]]]
        known_root = known_root_ccs[0][0]
        known_ccs = known_root_ccs[0][1]
        observed_root_ccs = self.network.node_cc
        observed_root = observed_root_ccs[0][0]
        observed_ccs = observed_root_ccs[0][1][-5:]
        self.assertEqual(observed_root, known_root)
        self.assertEqual(observed_ccs, known_ccs)

    def test_simplify_inplace_node2degree(self):
        known_degree = [(0, 3), (1, 3), (2, 5), (3, 3)]
        observed_degree = list(self.network.node2degree.items())[:4]
        self.assertEqual(observed_degree, known_degree)


if __name__ == "__main__":
    unittest.main()

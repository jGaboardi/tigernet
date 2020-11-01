"""Empirical data testing from a geopandas.GeoDataFrame.
"""

import tigernet
import unittest
import geopandas


##########################################################################################
# Synthetic testing
##########################################################################################


class TestNetworkBuildEmpirical(unittest.TestCase):
    def setUp(self):
        bbox = (-84.279, 30.480, -84.245, 30.505)
        f = "zip://test_data/Edges_Leon_FL_2010.zip!Edges_Leon_FL_2010.shp"
        gdf = geopandas.read_file(f, bbox=bbox)
        gdf = gdf.to_crs("epsg:2779")

        # Tiger attributes primary and secondary
        ATTR1, ATTR2 = "MTFCC", "TLID"
        XVAL, YVAL = "CentX", "CentY"  # individual x and y columns
        # segment welding and splitting
        INTRST = "S1100"  # interstates mtfcc code
        RAMP = "S1630"  # ramp mtfcc code
        SERV_DR = "S1640"  # service drive mtfcc code
        SPLIT_GRP = "FULLNAME"  # grouped by this variable
        SPLIT_BY = [RAMP, SERV_DR]  # split interstates by ramps & service
        SKIP_RESTR = True  # no weld retry if still MLS

        # filter out only roads
        yes_roads = gdf["ROADFLG"] == "Y"
        roads = gdf[yes_roads].copy()

        # discarded segments for "Waverly Hills" subset
        discard_segs = None

        kwargs = {"s_data": roads, "from_raw": True}
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
        self.network = tigernet.Network(**kwargs)

    def test_network_sdata(self):
        known_segments = 407
        observed_segments = self.network.s_data.shape[0]
        self.assertEqual(observed_segments, known_segments)

        known_length = 74866.5821646358
        observed_length = self.network.s_data.length.sum()
        self.assertEqual(observed_length, known_length)

    def test_network_ndata(self):
        known_nodes = 348
        observed_nodes = self.network.n_data.shape[0]
        self.assertEqual(observed_nodes, known_nodes)

        known_bounds = [
            620989.3023002351,
            163937.37839259504,
            624605.9723871874,
            167048.6931314568,
        ]
        observed_bounds = list(self.network.n_data.total_bounds)
        self.assertEqual(observed_bounds, known_bounds)

    def test_network_sdata_ids(self):
        known_ids = [412, 414, 415, 416, 417]
        observed_ids = list(self.network.s_data["SegID"])[-5:]
        self.assertEqual(observed_ids, known_ids)

    def test_network_ndata_ids(self):
        known_ids = [358, 361, 362, 363, 364]
        observed_ids = list(self.network.n_data["NodeID"])[-5:]
        self.assertEqual(observed_ids, known_ids)

    def test_network_segm2xyid(self):
        known_xyid = [
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
        observed_xyid = self.network.segm2xyid[-1]
        self.assertEqual(observed_xyid, known_xyid)

    def test_network_node2xyid(self):
        known_xyid = [364, ["x622213.7739825583y166384.2955689532"]]
        observed_xyid = self.network.node2xyid[-1]
        self.assertEqual(observed_xyid, known_xyid)


"""
class TestNeworkTopologyLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.Network(s_data=self.lattice)

    def test_lattice_network_segm2node(self):
        known_segm2node = [[0, [0, 1]], [1, [1, 2]], [2, [1, 3]], [3, [1, 4]]]
        observed_segm2node = self.lattice_network.segm2node
        self.assertEqual(observed_segm2node, known_segm2node)

    def test_lattice_network_node2segm(self):
        known_node2segm = [[0, [0]], [1, [0, 1, 2, 3]], [2, [1]], [3, [2]], [4, [3]]]
        observed_node2segm = self.lattice_network.node2segm
        self.assertEqual(observed_node2segm, known_node2segm)

    def test_lattice_network_segm2segm(self):
        known_segm2segm = [
            [0, [1, 2, 3]],
            [1, [0, 2, 3]],
            [2, [0, 1, 3]],
            [3, [0, 1, 2]],
        ]
        observed_segm2segm = self.lattice_network.segm2segm
        self.assertEqual(observed_segm2segm, known_segm2segm)

    def test_lattice_network_node2node(self):
        known_node2node = [[0, [1]], [1, [0, 2, 3, 4]], [2, [1]], [3, [1]], [4, [1]]]
        observed_node2node = self.lattice_network.node2node
        self.assertEqual(observed_node2node, known_node2node)


class TestNeworkComponentsLattice1x1(unittest.TestCase):
    def setUp(self):
        lat1 = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        lat2 = tigernet.generate_lattice(
            n_hori_lines=1, n_vert_lines=1, bounds=[6, 6, 8, 8]
        )
        self.lattice = lat1.append(lat2)
        self.lattice.reset_index(drop=True, inplace=True)

        # full network
        self.lattice_network = tigernet.Network(
            s_data=self.lattice, record_components=True
        )
        # largest component network
        self.lattice_network_largest_cc = tigernet.Network(
            s_data=self.lattice, record_components=True, largest_component=True
        )

    def test_lattice_network_segm_components(self):
        known_ccs = [[1, [0, 1, 2, 3]], [5, [4, 5, 6, 7]]]
        observed_ccs = self.lattice_network.segm_cc
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = {1: [18.0, [0, 1, 2, 3]], 5: [4.0, [4, 5, 6, 7]]}
        observed_cc_lens = self.lattice_network.cc_lens
        self.assertEqual(observed_cc_lens, known_cc_lens)

        known_ccs = 2
        observed_ccs = self.lattice_network.n_ccs
        self.assertEqual(observed_ccs, known_ccs)

        known_segms_in_ccs = 8
        observed_segms_in_ccs = self.lattice_network.n_segm
        self.assertEqual(observed_segms_in_ccs, known_segms_in_ccs)

    def test_lattice_network_sdata_components(self):
        known_ccs = [1, 1, 1, 1, 5, 5, 5, 5]
        observed_ccs = list(self.lattice_network.s_data["CC"])
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = [18.0, 18.0, 18.0, 18.0, 4.0, 4.0, 4.0, 4.0]
        observed_cc_lens = list(self.lattice_network.s_data["ccLength"])
        self.assertEqual(observed_cc_lens, known_cc_lens)

    def test_lattice_network_node_components(self):
        known_ccs = [[1, [0, 1, 2, 3, 4]], [6, [5, 6, 7, 8, 9]]]
        observed_ccs = self.lattice_network.node_cc
        self.assertEqual(observed_ccs, known_ccs)

    def test_lattice_network_ndata_components(self):
        known_ccs = [1, 1, 1, 1, 1, 6, 6, 6, 6, 6]
        observed_ccs = list(self.lattice_network.n_data["CC"])
        self.assertEqual(observed_ccs, known_ccs)

    def test_lattice_network_segm_components_largest(self):
        known_ccs = [1, [0, 1, 2, 3]]
        observed_ccs = self.lattice_network_largest_cc.segm_cc
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = {1: [18.0, [0, 1, 2, 3]]}
        observed_cc_lens = self.lattice_network_largest_cc.cc_lens
        self.assertEqual(observed_cc_lens, known_cc_lens)

        known_ccs = 2
        observed_ccs = self.lattice_network_largest_cc.n_ccs
        self.assertEqual(observed_ccs, known_ccs)

        known_segms_in_ccs = 4
        observed_segms_in_ccs = self.lattice_network_largest_cc.n_segm
        self.assertEqual(observed_segms_in_ccs, known_segms_in_ccs)

    def test_lattice_network_sdata_components_largest(self):
        known_ccs = [1, 1, 1, 1]
        observed_ccs = list(self.lattice_network_largest_cc.s_data["CC"])
        self.assertEqual(observed_ccs, known_ccs)

        known_cc_lens = [18.0, 18.0, 18.0, 18.0]
        observed_cc_lens = list(self.lattice_network_largest_cc.s_data["ccLength"])
        self.assertEqual(observed_cc_lens, known_cc_lens)

    def test_lattice_network_node_components_largest(self):
        known_ccs = [1, [0, 1, 2, 3, 4]]
        observed_ccs = self.lattice_network_largest_cc.node_cc
        self.assertEqual(observed_ccs, known_ccs)

    def test_lattice_network_ndata_components_largest(self):
        known_ccs = [1, 1, 1, 1, 1]
        observed_ccs = list(self.lattice_network_largest_cc.n_data["CC"])


class TestNetworkAssociationsLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.Network(s_data=self.lattice, record_geom=True)

    def test_lattice_network_segm2geom(self):
        known_type = "LineString"
        observed_type = self.lattice_network.segm2geom[0][1].geom_type
        self.assertEqual(observed_type, known_type)

        known_wkt = "LINESTRING (4.5 0, 4.5 4.5)"
        observed_wkt = self.lattice_network.segm2geom[0][1].wkt
        self.assertEqual(observed_wkt, known_wkt)

    def test_lattice_network_segm2coords(self):
        known_lookup = [
            [0, [(4.5, 0.0), (4.5, 4.5)]],
            [1, [(4.5, 4.5), (4.5, 9.0)]],
            [2, [(0.0, 4.5), (4.5, 4.5)]],
            [3, [(4.5, 4.5), (9.0, 4.5)]],
        ]
        observed_lookup = self.lattice_network.segm2coords
        self.assertEqual(observed_lookup, known_lookup)

    def test_lattice_network_node2geom(self):
        known_type = "Point"
        observed_type = self.lattice_network.node2geom[0][1].geom_type
        self.assertEqual(observed_type, known_type)

        known_wkt = "POINT (4.5 0)"
        observed_wkt = self.lattice_network.node2geom[0][1].wkt
        self.assertEqual(observed_wkt, known_wkt)

    def test_lattice_network_node2coords(self):
        known_lookup = [
            [0, [(4.5, 0.0)]],
            [1, [(4.5, 4.5)]],
            [2, [(4.5, 9.0)]],
            [3, [(0.0, 4.5)]],
            [4, [(9.0, 4.5)]],
        ]
        observed_lookup = self.lattice_network.node2coords
        self.assertEqual(observed_lookup, known_lookup)

    def test_lattice_network_s_ids(self):
        known_ids = [0, 1, 2, 3]
        observed_ids = self.lattice_network.s_ids
        self.assertEqual(observed_ids, known_ids)

    def test_lattice_network_n_ids(self):
        known_ids = [0, 1, 2, 3, 4]
        observed_ids = self.lattice_network.n_ids
        self.assertEqual(observed_ids, known_ids)

    def test_lattice_network_n_segm(self):
        known_segm_count, observed_segm_count = 4, self.lattice_network.n_segm
        self.assertEqual(observed_segm_count, known_segm_count)

    def test_lattice_network_n_node(self):
        known_node_count, observed_node_count = 5, self.lattice_network.n_node
        self.assertEqual(observed_node_count, known_node_count)

    def test_lattice_network_length(self):
        known_length, observed_length = 18.0, self.lattice_network.network_length
        self.assertEqual(observed_length, known_length)

    def test_lattice_node2degree(self):
        known_node2degree = [[0, [1]], [1, [4]], [2, [1]], [3, [1]], [4, [1]]]
        observed_node2degree = self.lattice_network.node2degree
        self.assertEqual(observed_node2degree, known_node2degree)

    def test_lattice_ndata_degree(self):
        known_degree = [1, 4, 1, 1, 1]
        observed_degree = list(self.lattice_network.n_data["degree"])
        self.assertEqual(observed_degree, known_degree)


class TestNetworkDefineGraphElementsLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.Network(
            s_data=self.lattice, record_geom=True, def_graph_elems=True
        )

    def test_lattice_network_segm2elem(self):
        known_elements = [[0, "leaf"], [1, "leaf"], [2, "leaf"], [3, "leaf"]]
        observed_elements = self.lattice_network.segm2elem
        self.assertEqual(observed_elements, known_elements)

    def test_lattice_network_sdata_segm2elem(self):
        known_elements = ["leaf", "leaf", "leaf", "leaf"]
        observed_elements = list(self.lattice_network.s_data["graph_elem"])
        self.assertEqual(observed_elements, known_elements)

    def test_lattice_network_node2elem(self):
        known_elements = [
            [0, "leaf"],
            [1, "branch"],
            [2, "leaf"],
            [3, "leaf"],
            [4, "leaf"],
        ]
        observed_elements = self.lattice_network.node2elem
        self.assertEqual(observed_elements, known_elements)

    def test_lattice_network_ndata_segm2elem(self):
        known_elements = ["leaf", "branch", "leaf", "leaf", "leaf"]
        observed_elements = list(self.lattice_network.n_data["graph_elem"])
        self.assertEqual(observed_elements, known_elements)


class TestNetworkSimplifyBarb(unittest.TestCase):
    def setUp(self):
        lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1, wbox=True)
        self.barb = lattice[~lattice["SegID"].isin([1, 2, 5, 7, 9, 10])]
        kws = {"record_components": True, "record_geom": True, "def_graph_elems": True}
        self.network = tigernet.Network(s_data=self.barb, **kws)
        # copy testing
        self.graph = self.network.simplify_network(**kws)
        # inplace
        self.network.simplify_network(inplace=True, **kws)

    # def test_..._copy(self):

    # def test_..._inplace(self):



"""

if __name__ == "__main__":
    unittest.main()

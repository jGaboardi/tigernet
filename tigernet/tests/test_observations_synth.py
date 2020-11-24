"""Synthetic observation data testing.
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


class TestSyntheticObservationsSegmentRandomLattice1x1(unittest.TestCase):
    def setUp(self):
        # instantiate network
        lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        network = tigernet.Network(s_data=lattice, record_geom=True)

        # generate synthetic observations
        obs = tigernet.generate_obs(5, network.s_data)
        obs["obs_id"] = ["a", "b", "c", "d", "e"]

        # build kd tree
        net_nodes_kdtree = network.nodes_kdtree()

        # associate observations with the network
        args = network, obs.copy(), net_nodes_kdtree
        kwargs = {"df_name": "obs1", "df_key": "obs_id"}
        net_obs = tigernet.Observations(*args, **kwargs)

    def test_obs2coords(self):
        known_obs2coords = {
            (0, "a"): (4.939321535345923, 6.436704297351775),
            (1, "b"): (5.4248703846447945, 4.903948646972072),
            (2, "c"): (3.8128931940501425, 5.813047017599905),
            (3, "d"): (3.9382849013642325, 8.025957007038718),
            (4, "e"): (8.672964844509263, 3.4509736694319995),
        }
        observed_obs2coords = self.net_obs.obs2coords
        for k, v in known_obs2coords.items():
            self.assertAlmostEqual(observed_obs2coords[k], v)

    def test_obs2segm(self):
        known_obs2segm = {"a": 1, "b": 3, "c": 1, "d": 1, "e": 3}
        observed_obs2segm = self.net_obs.obs2segm
        self.assertEqual(observed_obs2segm, known_obs2segm)

    def test_snapped_points_df_dist_a(self):
        known_dist_a = [
            1.9367042973517747,
            0.9248703846447945,
            1.3130470175999047,
            3.5259570070387185,
            4.172964844509263,
        ]
        observed_dist_a = list(self.net_obs.snapped_points["dist_a"])
        self.assertAlmostEqual(observed_dist_a, known_dist_a)

    def test_snapped_points_df_dist_b(self):
        known_dist_b = [
            2.563295702648225,
            3.5751296153552055,
            3.1869529824000953,
            0.9740429929612815,
            0.32703515549073714,
        ]
        observed_dist_b = list(self.net_obs.snapped_points["dist_b"])
        self.assertAlmostEqual(observed_dist_b, known_dist_b)

    def test_snapped_points_df_node_a(self):
        known_node_a = [1, 1, 1, 1, 1]
        observed_node_a = list(self.net_obs.snapped_points["node_a"])
        self.assertEqual(observed_node_a, known_node_a)

    def test_snapped_points_df_node_b(self):
        known_node_b = [2, 4, 2, 2, 4]
        observed_node_b = list(self.net_obs.snapped_points["node_b"])
        self.assertEqual(observed_node_b, known_node_b)

    def test_snapped_points_df_dist2line(self):
        known_dist2line = [
            0.4393215353459228,
            0.4039486469720721,
            0.6871068059498575,
            0.5617150986357675,
            1.0490263305680005,
        ]
        observed_dist2line = list(self.net_obs.snapped_points["dist2line"])
        self.assertAlmostEqual(observed_dist2line, known_dist2line)


class TestSyntheticObservationsNodeRandomLattice1x1(unittest.TestCase):
    def setUp(self):
        # instantiate network
        lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        network = tigernet.Network(s_data=lattice, record_geom=True)

        # generate synthetic observations
        obs = tigernet.generate_obs(5, network.s_data)
        obs["obs_id"] = ["a", "b", "c", "d", "e"]

        # build kd tree
        net_nodes_kdtree = network.nodes_kdtree()

        # associate observations with the network
        args = network, obs.copy(), net_nodes_kdtree
        kwargs = {"df_name": "obs1", "df_key": "obs_id", "snap_to": "nodes"}
        net_obs = tigernet.Observations(*args, **kwargs)

    def test_obs2coords(self):
        known_obs2coords = {
            (0, "a"): (4.939321535345923, 6.436704297351775),
            (1, "b"): (5.4248703846447945, 4.903948646972072),
            (2, "c"): (3.8128931940501425, 5.813047017599905),
            (3, "d"): (3.9382849013642325, 8.025957007038718),
            (4, "e"): (8.672964844509263, 3.4509736694319995),
        }
        observed_obs2coords = self.net_obs.obs2coords
        for k, v in known_obs2coords.items():
            self.assertAlmostEqual(observed_obs2coords[k], v)

    def test_obs2node(self):
        known_obs2node = {"a": 1, "b": 1, "c": 1, "d": 2, "e": 4}
        observed_obs2node = self.net_obs.obs2node
        self.assertEqual(observed_obs2node, known_obs2node)

    def test_snapped_points_df_dist2node(self):
        known_dist2node = [
            1.9859070841304562,
            1.0092372059053203,
            1.4819609418640627,
            1.1244036660258458,
            1.098821293546778,
        ]
        observed_dist2node = list(self.net_obs.snapped_points["dist2node"])
        self.assertAlmostEqual(observed_dist2node, known_dist2node)


class TestSyntheticObservationsSegmentRandomEmpirical(unittest.TestCase):
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
        network = tigernet.Network(**kwargs)

        # simplify network
        kws = {"record_components": True, "record_geom": True, "def_graph_elems": True}
        network.simplify_network(inplace=True, **kws)

        # generate synthetic observations
        obs = tigernet.generate_obs(500, network.s_data)
        obs["obs_id"] = obs.index

        # build kd tree
        net_nodes_kdtree = network.nodes_kdtree()

        # associate observations with the network
        args = network, obs.copy(), net_nodes_kdtree
        kwargs = {"df_name": "obs1", "df_key": "obs_id"}
        net_obs = tigernet.Observations(*args, **kwargs)

    def test_obs2coords(self):
        known_obs2coords = [
            ((495, 495), (621033.3213594754, 164941.80269090834)),
            ((496, 496), (621819.5720103906, 165514.3885859197)),
            ((497, 497), (623654.2570885622, 164241.2803142736)),
            ((498, 498), (622851.6060250874, 166857.07354681785)),
            ((499, 499), (621816.24144166, 166044.17761455863)),
        ]
        observed_obs2coords = self.net_obs.obs2coords
        for k, v in known_obs2coords:
            self.assertAlmostEqual(observed_obs2coords[k], v)

    def test_obs2segm(self):
        known_obs2segm = [(495, 150), (496, 230), (497, 84), (498, 91), (499, 105)]
        observed_obs2segm = list(self.net_obs.obs2segm.items())[-5:]
        self.assertEqual(observed_obs2segm, known_obs2segm)

    def test_snapped_points_df_dist_a(self):
        known_dist_a = [
            210.40526565933823,
            118.30357725098324,
            34.12778222322711,
            120.39577375386378,
            0.0,
        ]
        observed_dist_a = list(self.net_obs.snapped_points["dist_a"])[-5:]
        self.assertAlmostEqual(observed_dist_a, known_dist_a)

    def test_snapped_points_df_dist_b(self):
        known_dist_b = [
            342.6965551431302,
            0.0,
            86.50490751040633,
            58.25005873237134,
            152.0185068774602,
        ]
        observed_dist_b = list(self.net_obs.snapped_points["dist_b"])[-5:]
        self.assertAlmostEqual(observed_dist_b, known_dist_b)

    def test_snapped_points_df_node_a(self):
        known_node_a = [186, 86, 122, 132, 151]
        observed_node_a = list(self.net_obs.snapped_points["node_a"])[-5:]
        self.assertEqual(observed_node_a, known_node_a)

    def test_snapped_points_df_node_b(self):
        known_node_b = [193, 245, 48, 133, 22]
        observed_node_b = list(self.net_obs.snapped_points["node_b"])[-5:]
        self.assertEqual(observed_node_b, known_node_b)

    def test_snapped_points_df_dist2line(self):
        known_dist2line = [
            147.05576410321171,
            298.0459114928476,
            2.914177304108527,
            160.72592517096817,
            300.2025615374258,
        ]
        observed_dist2line = list(self.net_obs.snapped_points["dist2line"])[-5:]
        self.assertAlmostEqual(observed_dist2line, known_dist2line)


class TestSyntheticObservationsSegmentRandomEmpirical(unittest.TestCase):
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
        network = tigernet.Network(**kwargs)

        # simplify network
        kws = {"record_components": True, "record_geom": True, "def_graph_elems": True}
        network.simplify_network(inplace=True, **kws)

        # generate synthetic observations
        obs = tigernet.generate_obs(500, network.s_data)
        obs["obs_id"] = obs.index

        # build kd tree
        net_nodes_kdtree = network.nodes_kdtree()

        # associate observations with the network
        args = network, obs.copy(), net_nodes_kdtree
        kwargs = {"df_name": "obs1", "df_key": "obs_id", "snap_to": "nodes"}
        net_obs = tigernet.Observations(*args, **kwargs)

    def test_obs2coords(self):
        known_obs2coords = [
            ((495, 495), (621033.3213594754, 164941.80269090834)),
            ((496, 496), (621819.5720103906, 165514.3885859197)),
            ((497, 497), (623654.2570885622, 164241.2803142736)),
            ((498, 498), (622851.6060250874, 166857.07354681785)),
            ((499, 499), (621816.24144166, 166044.17761455863)),
        ]
        observed_obs2coords = self.net_obs.obs2coords
        for k, v in known_obs2coords:
            self.assertAlmostEqual(observed_obs2coords[k], v)

    def test_obs2node(self):
        known_obs2node = [(495, 192), (496, 245), (497, 122), (498, 133), (499, 151)]
        observed_obs2node = self.net_obs.obs2node
        for k, v in known_obs2node:
            self.assertAlmostEqual(observed_obs2node[k], v)

    def test_snapped_points_df_dist2node(self):
        known_dist2node = [
            233.41263770566138,
            298.0459114928476,
            34.25197729818704,
            170.95581991959833,
            300.2025615374258,
        ]
        observed_dist2node = list(self.net_obs.snapped_points["dist2node"])[-5:]
        self.assertAlmostEqual(observed_dist2node, known_dist2node)


if __name__ == "__main__":
    unittest.main()

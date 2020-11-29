"""Synthetic observation data testing.
"""

import copy
import unittest
import numpy

import tigernet
from .network_objects import network_lattice_1x1_geomelem
from .network_objects import network_empirical_simplified


class TestSyntheticObservationsSegmentRandomLattice1x1(unittest.TestCase):
    def setUp(self):
        network = network_lattice_1x1_geomelem

        # generate synthetic observations
        obs = tigernet.generate_obs(5, network.s_data)
        obs["obs_id"] = ["a", "b", "c", "d", "e"]

        # build kd tree
        net_nodes_kdtree = network.nodes_kdtree()

        # associate observations with the network
        args = network, obs.copy(), net_nodes_kdtree
        kwargs = {"df_name": "obs1", "df_key": "obs_id"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

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
        network = network_lattice_1x1_geomelem

        # generate synthetic observations
        obs = tigernet.generate_obs(5, network.s_data)
        obs["obs_id"] = ["a", "b", "c", "d", "e"]

        # build kd tree
        net_nodes_kdtree = network.nodes_kdtree()

        # associate observations with the network
        args = network, obs.copy(), net_nodes_kdtree
        kwargs = {"df_name": "obs1", "df_key": "obs_id", "snap_to": "nodes"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

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
        network = copy.deepcopy(network_empirical_simplified)

        # generate synthetic observations
        obs = tigernet.generate_obs(500, network.s_data)
        obs["obs_id"] = obs.index

        # build kd tree
        net_nodes_kdtree = network.nodes_kdtree()

        # associate observations with the network
        args = network, obs.copy(), net_nodes_kdtree
        kwargs = {"df_name": "obs1", "df_key": "obs_id"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

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
            obs = numpy.array(observed_obs2coords[k])
            numpy.testing.assert_array_almost_equal(obs, numpy.array(v))

    def test_obs2segm(self):
        known_obs2segm = [(495, 150), (496, 230), (497, 84), (498, 91), (499, 105)]
        observed_obs2segm = list(self.net_obs.obs2segm.items())[-5:]
        self.assertEqual(observed_obs2segm, known_obs2segm)

    def test_snapped_points_df_dist_a(self):
        known_dist_a = numpy.array(
            [
                210.40526565933823,
                118.30357725098324,
                34.12778222322711,
                120.39577375386378,
                0.0,
            ]
        )
        observed_dist_a = list(self.net_obs.snapped_points["dist_a"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_a), known_dist_a
        )

    def test_snapped_points_df_dist_b(self):
        known_dist_b = numpy.array(
            [
                342.6965551431302,
                0.0,
                86.50490751040633,
                58.25005873237134,
                152.0185068774602,
            ]
        )
        observed_dist_b = list(self.net_obs.snapped_points["dist_b"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_b), known_dist_b
        )

    def test_snapped_points_df_node_a(self):
        known_node_a = [186, 86, 122, 132, 151]
        observed_node_a = list(self.net_obs.snapped_points["node_a"])[-5:]
        self.assertEqual(observed_node_a, known_node_a)

    def test_snapped_points_df_node_b(self):
        known_node_b = [193, 245, 48, 133, 22]
        observed_node_b = list(self.net_obs.snapped_points["node_b"])[-5:]
        self.assertEqual(observed_node_b, known_node_b)

    def test_snapped_points_df_dist2line(self):
        known_dist2line = numpy.array(
            [
                147.05576410321171,
                298.0459114928476,
                2.914177304108527,
                160.72592517096817,
                300.2025615374258,
            ]
        )
        observed_dist2line = list(self.net_obs.snapped_points["dist2line"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist2line), known_dist2line
        )


class TestSyntheticObservationsNodeRandomEmpirical(unittest.TestCase):
    def setUp(self):
        network = copy.deepcopy(network_empirical_simplified)

        # generate synthetic observations
        obs = tigernet.generate_obs(500, network.s_data)
        obs["obs_id"] = obs.index

        # build kd tree
        net_nodes_kdtree = network.nodes_kdtree()

        # associate observations with the network
        args = network, obs.copy(), net_nodes_kdtree
        kwargs = {"df_name": "obs1", "df_key": "obs_id", "snap_to": "nodes"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

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
            numpy.testing.assert_array_almost_equal(
                numpy.array(observed_obs2coords[k]), numpy.array(v)
            )

    def test_obs2node(self):
        known_obs2node = [(495, 192), (496, 245), (497, 122), (498, 133), (499, 151)]
        observed_obs2node = self.net_obs.obs2node
        for k, v in known_obs2node:
            self.assertAlmostEqual(observed_obs2node[k], v)

    def test_snapped_points_df_dist2node(self):
        known_dist2node = numpy.array(
            [
                233.41263770566138,
                298.0459114928476,
                34.25197729818704,
                170.95581991959833,
                300.2025615374258,
            ]
        )
        observed_dist2node = list(self.net_obs.snapped_points["dist2node"])[-5:]

        numpy.testing.assert_array_almost_equal(
            numpy.array(numpy.array(observed_dist2node)), known_dist2node
        )


if __name__ == "__main__":
    unittest.main()
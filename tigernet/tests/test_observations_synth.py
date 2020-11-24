"""Synthetic observation data testing.
"""

import tigernet
import unittest
import numpy


class TestObservationsSegmentRandomLattice1x1(unittest.TestCase):
    def setUp(self):
        # instantiate network
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.Network(s_data=self.lattice, record_geom=True)

        # generate synthetic observations
        self.obs = tigernet.generate_obs(5, self.lattice_network.s_data)
        self.obs["obs_id"] = ["a", "b", "c", "d", "e"]

        # build kd tree
        net_nodes_kdtree = self.lattice_network.nodes_kdtree()

        # associate observations with the network
        self.net_obs = tigernet.Observations(
            self.lattice_network,
            self.obs.copy(),
            net_nodes_kdtree,
            df_name="obs1",
            df_key="obs_id",
        )

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


class TestObservationsNodeRandomLattice1x1(unittest.TestCase):
    def setUp(self):
        # instantiate network
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.Network(s_data=self.lattice, record_geom=True)

        # generate synthetic observations
        self.obs = tigernet.generate_obs(5, self.lattice_network.s_data)
        self.obs["obs_id"] = ["a", "b", "c", "d", "e"]

        # build kd tree
        self.net_nodes_kdtree = self.lattice_network.nodes_kdtree()

        # associate observations with the network
        self.net_obs = tigernet.Observations(
            self.lattice_network,
            self.obs.copy(),
            self.net_nodes_kdtree,
            df_name="obs1",
            df_key="obs_id",
            snap_to="nodes",
        )

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


if __name__ == "__main__":
    unittest.main()

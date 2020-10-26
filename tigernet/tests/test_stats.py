"""Testing for tigernet.py
"""

import tigernet

import unittest

import numpy
from shapely.geometry import LineString

##########################################################################################
# Synthetic testing
##########################################################################################


class TestTigerNetStatsBarb(unittest.TestCase):
    def setUp(self):
        lat = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1, wbox=True)
        lat = lat[~lat["SegID"].isin([3, 5, 8])]
        rec = {
            "geometry": LineString(((4.5, 9), (4.5, 13.5))),
            "SegID": 13,
            "MTFCC": "S1400",
        }
        barb = lat.append(rec, ignore_index=True)

        # full network
        self.network = tigernet.TigerNet(s_data=barb.copy())
        self.network.calc_net_stats()

        # simplified network
        self.graph = self.network.simplify_network()
        self.graph.calc_net_stats()

    def test_barb_network_sinuosity(self):
        known_sinuosity = [1.0] * 10
        observed_sinuosity = list(self.network.s_data["sinuosity"])
        self.assertEqual(observed_sinuosity, known_sinuosity)

    def test_barb_network_sinuosity_stats(self):
        known_max = 1.0
        observed_max = self.network.max_sinuosity
        self.assertEqual(observed_max, known_max)

        known_min = 1.0
        observed_min = self.network.min_sinuosity
        self.assertEqual(observed_min, known_min)

        known_mean = 1.0
        observed_mean = self.network.mean_sinuosity
        self.assertEqual(observed_mean, known_mean)

        known_std = 0.0
        observed_std = self.network.std_sinuosity
        self.assertEqual(observed_std, known_std)

    def test_barb_graph_sinuosity(self):
        known_sinuosity = [2.23606797749979, numpy.inf, 1.0]
        observed_sinuosity = list(self.graph.s_data["sinuosity"])
        self.assertEqual(observed_sinuosity, known_sinuosity)

    def test_barb_graph_sinuosity_stats(self):
        known_max = 2.23606797749979
        observed_max = self.graph.max_sinuosity
        self.assertEqual(observed_max, known_max)

        known_min = 1.0
        observed_min = self.graph.min_sinuosity
        self.assertEqual(observed_min, known_min)

        known_mean = 1.8240453183331933
        observed_mean = self.graph.mean_sinuosity
        self.assertEqual(observed_mean, known_mean)

        known_std = 0.71364417954618
        observed_std = self.graph.std_sinuosity
        self.assertEqual(observed_std, known_std)


##########################################################################################
# Empirical testing
##########################################################################################


class TestTigerNetStatsEmpirical(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == "__main__":
    unittest.main()

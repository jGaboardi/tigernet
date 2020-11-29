"""Testing for tigernet.py
"""

import copy
import unittest
import numpy
import geopandas
from shapely.geometry import LineString


import tigernet
from .network_objects import network_empirical_lcc
from .network_objects import network_empirical_simplified
from .network_objects import network_empirical_simplified_wcm


##########################################################################################
# Synthetic testing
##########################################################################################


class TestNetworkStatsBarb(unittest.TestCase):
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
        self.network = tigernet.Network(s_data=barb.copy())
        with self.assertWarns(UserWarning):
            self.network.calc_net_stats()

        # simplified network
        self.graph = self.network.simplify_network()
        with self.assertWarns(UserWarning):
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

    def test_barb_network_node_degree_stats(self):
        known_max = 4
        observed_max = self.network.max_node_degree
        self.assertEqual(observed_max, known_max)

        known_min = 1
        observed_min = self.network.min_node_degree
        self.assertEqual(observed_min, known_min)

        known_mean = 2.0
        observed_mean = self.network.mean_node_degree
        self.assertEqual(observed_mean, known_mean)

        known_std = 0.816496580927726
        observed_std = self.network.std_node_degree
        self.assertEqual(observed_std, known_std)

    def test_barb_graph_node_degree_stats(self):
        known_max = 4
        observed_max = self.graph.max_node_degree
        self.assertEqual(observed_max, known_max)

        known_min = 1
        observed_min = self.graph.min_node_degree
        self.assertEqual(observed_min, known_min)

        known_mean = 2.0
        observed_mean = self.graph.mean_node_degree
        self.assertEqual(observed_mean, known_mean)

        known_std = 1.7320508075688772
        observed_std = self.graph.std_node_degree
        self.assertEqual(observed_std, known_std)


class TestNetworkStatsSineLine(unittest.TestCase):
    def setUp(self):
        sine = tigernet.generate_sine_lines()
        sine = tigernet.generate_sine_lines()
        sine = sine[sine["SegID"].isin([0, 1, 2, 3])]

        # full network
        self.network = tigernet.Network(s_data=sine.copy())
        with self.assertWarns(UserWarning):
            self.network.calc_net_stats()

        # simplified network
        self.graph = self.network.simplify_network()
        with self.assertWarns(UserWarning):
            self.graph.calc_net_stats()

    def test_sine_network_sinuosity(self):
        known_sinuosity = [
            1.1913994275103448,
            1.0377484765201541,
            1.0714252226602858,
            1.1885699897294775,
        ]
        observed_sinuosity = list(self.network.s_data["sinuosity"])
        self.assertEqual(observed_sinuosity, known_sinuosity)

    def test_sine_network_sinuosity_stats(self):
        known_max = 1.1913994275103448
        observed_max = self.network.max_sinuosity
        self.assertEqual(observed_max, known_max)

        known_min = 1.0377484765201541
        observed_min = self.network.min_sinuosity
        self.assertEqual(observed_min, known_min)

        known_mean = 1.1222857791050656
        observed_mean = self.network.mean_sinuosity
        self.assertEqual(observed_mean, known_mean)

        known_std = 0.07938019212245889
        observed_std = self.network.std_sinuosity
        self.assertEqual(observed_std, known_std)

    def test_sine_graph_sinuosity(self):
        known_sinuosity = [1.2105497715794307, 1.2105497715794304]
        observed_sinuosity = list(self.graph.s_data["sinuosity"])
        self.assertEqual(observed_sinuosity, known_sinuosity)

    def test_sine_graph_sinuosity_stats(self):
        known_max = 1.2105497715794307
        observed_max = self.graph.max_sinuosity
        self.assertEqual(observed_max, known_max)

        known_min = 1.2105497715794304
        observed_min = self.graph.min_sinuosity
        self.assertEqual(observed_min, known_min)

        known_mean = 1.2105497715794304
        observed_mean = self.graph.mean_sinuosity
        self.assertEqual(observed_mean, known_mean)

        known_std = 2.220446049250313e-16
        observed_std = self.graph.std_sinuosity
        self.assertEqual(observed_std, known_std)

    def test_sine_network_node_degree_stats(self):
        known_max = 2
        observed_max = self.network.max_node_degree
        self.assertEqual(observed_max, known_max)

        known_min = 1
        observed_min = self.network.min_node_degree
        self.assertEqual(observed_min, known_min)

        known_mean = 1.3333333333333333
        observed_mean = self.network.mean_node_degree
        self.assertEqual(observed_mean, known_mean)

        known_std = 0.5163977794943223
        observed_std = self.network.std_node_degree
        self.assertEqual(observed_std, known_std)

    def test_sine_graph_node_degree_stats(self):
        known_max = 1
        observed_max = self.graph.max_node_degree
        self.assertEqual(observed_max, known_max)

        known_min = 1
        observed_min = self.graph.min_node_degree
        self.assertEqual(observed_min, known_min)

        known_mean = 1.0
        observed_mean = self.graph.mean_node_degree
        self.assertEqual(observed_mean, known_mean)

        known_std = 0.0
        observed_std = self.graph.std_node_degree
        self.assertEqual(observed_std, known_std)


class TestNetworkConnectivityLattice1x1(unittest.TestCase):
    def setUp(self):
        lat1 = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        kws = {"n_hori_lines": 1, "n_vert_lines": 1, "bounds": [6, 6, 8, 8]}
        lat2 = tigernet.generate_lattice(**kws)
        self.lats = lat1.append(lat2)
        self.lats.reset_index(drop=True, inplace=True)

    def test_lattice_network_wcomps_connectivity(self):
        # with recorded components
        net_wcomps = tigernet.Network(s_data=self.lats.copy(), record_components=True)
        with self.assertWarns(UserWarning):
            net_wcomps.calc_net_stats(conn_stat="all")

        known_alpha = 0.0
        observed_alpha = net_wcomps.alpha
        self.assertEqual(observed_alpha, known_alpha)

        known_beta = 0.8
        observed_beta = net_wcomps.beta
        self.assertEqual(observed_beta, known_beta)

        known_gamma = 0.3333333333333333
        observed_gamma = net_wcomps.gamma
        self.assertEqual(observed_gamma, known_gamma)

        known_eta = 2.75
        observed_eta = net_wcomps.eta
        self.assertEqual(observed_eta, known_eta)

    def test_lattice_network_wcomps_alpha(self):
        # with recorded components
        net_wcomps = tigernet.Network(s_data=self.lats.copy(), record_components=True)
        with self.assertWarns(UserWarning):
            net_wcomps.calc_net_stats(conn_stat="alpha")

        known_alpha = 0.0
        observed_alpha = net_wcomps.alpha
        self.assertEqual(observed_alpha, known_alpha)

    def test_lattice_network_wcomps_beta(self):
        # with recorded components
        net_wcomps = tigernet.Network(s_data=self.lats.copy(), record_components=True)
        with self.assertWarns(UserWarning):
            net_wcomps.calc_net_stats(conn_stat="beta")

        known_beta = 0.8
        observed_beta = net_wcomps.beta
        self.assertEqual(observed_beta, known_beta)

    def test_lattice_network_wcomps_gamma(self):
        # with recorded components
        net_wcomps = tigernet.Network(s_data=self.lats.copy(), record_components=True)
        with self.assertWarns(UserWarning):
            net_wcomps.calc_net_stats(conn_stat="gamma")

        known_gamma = 0.3333333333333333
        observed_gamma = net_wcomps.gamma
        self.assertEqual(observed_gamma, known_gamma)

    def test_lattice_network_wcomps_eta(self):
        # with recorded components
        net_wcomps = tigernet.Network(s_data=self.lats.copy(), record_components=True)
        with self.assertWarns(UserWarning):
            net_wcomps.calc_net_stats(conn_stat="eta")

        known_eta = 2.75
        observed_eta = net_wcomps.eta
        self.assertEqual(observed_eta, known_eta)


class TestNetworkEntropyLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lat = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)

    def test_lattice_network_segment_entropy_xvariation(self):
        net = tigernet.Network(s_data=self.lat.copy())
        net.calc_entropy("MTFCC", "s_data")

        known_entropies = {"S1400": 0.0}
        observed_entropies = net.mtfcc_entropies
        self.assertEqual(observed_entropies, known_entropies)

        known_entropy = -0.0
        observed_entropy = net.network_mtfcc_entropy
        self.assertEqual(observed_entropy, known_entropy)

    def test_lattice_network_segment_entropy_wvariation(self):
        _lat = self.lat.copy()
        _lat["MTFCC"] = ["S1100", "S1200", "S1400", "S1700"]
        net = tigernet.Network(s_data=_lat)
        net.calc_entropy("MTFCC", "s_data")

        known_entropies = {"S1100": -0.5, "S1200": -0.5, "S1400": -0.5, "S1700": -0.5}
        observed_entropies = net.mtfcc_entropies
        self.assertEqual(observed_entropies, known_entropies)

        known_entropy = 2.0
        observed_entropy = net.network_mtfcc_entropy
        self.assertEqual(observed_entropy, known_entropy)

    def test_lattice_network_node_entropy(self):
        _lat = self.lat.copy()
        net = tigernet.Network(s_data=_lat)
        net.calc_entropy("degree", "n_data")

        known_entropies = {1: -0.2575424759098898, 4: -0.46438561897747244}
        observed_entropies = net.degree_entropies
        for k, v in known_entropies.items():
            self.assertAlmostEqual(observed_entropies[k], v)

        known_entropy = 0.7219280948873623
        observed_entropy = net.network_degree_entropy
        self.assertAlmostEqual(observed_entropy, known_entropy)


class TestNetworkDistanceMetricsLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lat = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.net = tigernet.Network(s_data=self.lat)
        self.net.cost_matrix()
        self.net.calc_net_stats()

    def test_network_radius(self):
        known_pair, known_radius = [(0, 1, 1, 1, 1, 2, 3, 4), 4.5]
        observed_pair, observed_radius = self.net.radius
        self.assertEqual(observed_pair, known_pair)
        self.assertAlmostEqual(observed_radius, known_radius)

    def test_network_diameter(self):
        known_pair, known_diameter = [(0, 0, 0, 2, 2, 2, 3, 3, 3, 4, 4, 4), 9.0]
        observed_pair, observed_diameter = self.net.diameter
        self.assertEqual(observed_pair, known_pair)
        self.assertAlmostEqual(observed_diameter, known_diameter)

    def test_network_total_network_distance(self):
        known_distance = 144.0
        observed_distance = self.net.d_net
        self.assertAlmostEqual(observed_distance, known_distance)

    def test_network_total_euclidean_distance(self):
        known_distance = 122.91168824543142
        observed_distance = self.net.d_euc
        self.assertAlmostEqual(observed_distance, known_distance, 3)

    def test_network_circuity(self):
        known_circuity = 1.17157287525381
        observed_circuity = self.net.circuity
        self.assertAlmostEqual(observed_circuity, known_circuity, 3)


##########################################################################################
# Empirical testing
##########################################################################################


class TestNetworkStatsEmpirical(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_simplified_wcm)
        self.network.calc_net_stats()

    def test_network_sinuosity(self):
        known_sinuosity = [
            1.1814786403587332,
            1.0,
            1.0002181434117707,
            1.000964242197008,
            1.1293480862834322,
        ]
        observed_sinuosity = list(self.network.s_data["sinuosity"][:5])
        for k, o in zip(known_sinuosity, observed_sinuosity):
            self.assertAlmostEqual(o, k)

    def test_network_sinuosity_stats(self):
        known_max = 4.479497558172366
        observed_max = self.network.max_sinuosity
        self.assertAlmostEqual(observed_max, known_max)

        known_min = 1.0
        observed_min = self.network.min_sinuosity
        self.assertEqual(observed_min, known_min)

        known_mean = 1.1300036198574728
        observed_mean = self.network.mean_sinuosity
        self.assertAlmostEqual(observed_mean, known_mean)

        known_std = 0.48197915355062204
        observed_std = self.network.std_sinuosity
        self.assertAlmostEqual(observed_std, known_std)

    def test_network_node_degree_stats(self):
        known_max = 5
        observed_max = self.network.max_node_degree
        self.assertEqual(observed_max, known_max)

        known_min = 1
        observed_min = self.network.min_node_degree
        self.assertEqual(observed_min, known_min)

        known_mean = 2.4125874125874125
        observed_mean = self.network.mean_node_degree
        self.assertAlmostEqual(observed_mean, known_mean)

        known_std = 1.0650997569862783
        observed_std = self.network.std_node_degree
        self.assertAlmostEqual(observed_std, known_std)


class TestNetworkConnectivityEmpirical(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_simplified)
        with self.assertWarns(UserWarning):
            self.network.calc_net_stats(conn_stat="all")

    def test_lattice_network_wcomps_connectivity(self):
        known_alpha = 0.10582010582010581
        observed_alpha = self.network.alpha
        self.assertAlmostEqual(observed_alpha, known_alpha)

        known_beta = 1.2062937062937062
        observed_beta = self.network.beta
        self.assertAlmostEqual(observed_beta, known_beta)

        known_gamma = 0.40492957746478875
        observed_gamma = self.network.gamma
        self.assertAlmostEqual(observed_gamma, known_gamma)

        known_eta = 217.00458598445144
        observed_eta = self.network.eta
        self.assertAlmostEqual(observed_eta, known_eta)


class TestNetworkEntropyEmpirical(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_lcc)
        self.network.calc_entropy("MTFCC", "s_data")
        self.network.calc_entropy("degree", "n_data")

    def test_network_entropy_variation(self):
        known_entropies = {
            "S1630": -0.15869726894257252,
            "S1100": -0.08968927494169666,
            "S1400": -0.2700938933248304,
            "S1200": -0.42847006925102654,
        }
        observed_entropies = self.network.mtfcc_entropies
        for mtfcc_type, known_entropy_val in known_entropies.items():
            self.assertAlmostEqual(observed_entropies[mtfcc_type], known_entropy_val)

        known_entropy = 0.9469505064601262
        observed_entropy = self.network.network_mtfcc_entropy
        self.assertAlmostEqual(observed_entropy, known_entropy)

    def test_network_node_entropy(self):
        known_entropies = {
            3: -0.5144720846413561,
            5: -0.024261331884622785,
            4: -0.2796123512080418,
            1: -0.5148474076720833,
            2: -0.4433974870650428,
        }
        observed_entropies = self.network.degree_entropies
        for k, v in known_entropies.items():
            self.assertAlmostEqual(observed_entropies[k], v)

        known_entropy = 1.7765906624711467
        observed_entropy = self.network.network_degree_entropy
        self.assertAlmostEqual(observed_entropy, known_entropy)


class TestNetworkDistanceMetricsEmpiricalGDF(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_simplified_wcm)
        self.network.calc_net_stats()

    def test_network_radius(self):
        known_pair, known_radius = [(130, 147), 9.943247035238207]
        observed_pair, observed_radius = self.network.radius
        self.assertEqual(observed_pair, known_pair)
        self.assertAlmostEqual(observed_radius, known_radius, 3)

    def test_network_diameter(self):
        known_pair, known_diameter = [(120, 11), 7519.207911226202]
        observed_pair, observed_diameter = self.network.diameter
        self.assertEqual(observed_pair, known_pair)
        self.assertAlmostEqual(observed_diameter, known_diameter, 3)

    def test_network_total_network_distance(self):
        known_distance = 223504355.29578546
        observed_distance = self.network.d_net
        self.assertAlmostEqual(observed_distance, known_distance, 3)

    def test_network_total_euclidean_distance(self):
        known_distance = 143326889.90408167
        observed_distance = self.network.d_euc
        self.assertAlmostEqual(observed_distance, known_distance, 3)

    def test_network_circuity(self):
        known_circuity = 1.5594028130057156
        observed_circuity = self.network.circuity
        self.assertAlmostEqual(observed_circuity, known_circuity, 3)


if __name__ == "__main__":
    unittest.main()

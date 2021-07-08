"""Synthetic observation data testing.
"""

import copy
import unittest
import numpy

import tigernet
from .network_objects import network_empirical_simplified_wcm

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


####################################################################################
############################# ORIG-XXXX--Segments ##################################
####################################################################################


class TestEmpiricalObservationsOrigToXXXXSegments(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_simplified_wcm)

        # empirical observations
        obs = tigernet.testing_data("CensusBlocks_Leon_FL_2010")

        # associate observations with the network
        args = self.network, obs.copy()
        kwargs = {"df_name": "obs1", "df_key": "GEOID"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_net_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 3285.69020577, 3883.24256199, 3581.0882225],
                [3285.69020577, 0.0, 645.01865974, 3264.87455121],
                [3883.24256199, 645.01865974, 0.0, 3909.89321095],
                [3581.0882225, 3264.87455121, 3909.89321095, 0.0],
            ]
        )
        args = copy.deepcopy(self.net_obs), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": None,
            "snap_dist": False,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(
            observed_mtx[:4, :4], known_mtx, decimal=DECIMAL
        )

        known_mtx_sum = 22099816.17479256
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 3389.37545944, 3975.65875829, 3687.64355986],
                [3389.37545944, 0.0, 727.84658824, 3361.84162077],
                [3975.65875829, 727.84658824, 0.0, 3995.59122314],
                [3687.64355986, 3361.84162077, 3995.59122314, 0.0],
            ]
        )
        args = copy.deepcopy(self.net_obs), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": None,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(
            observed_mtx[:4, :4], known_mtx, decimal=DECIMAL
        )

        known_mtx_sum = 23230504.28050229
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)


####################################################################################
############################# ORIG-XXXX--Nodes #####################################
####################################################################################


class TestEmpiricalObservationsOrigToXXXXNodes(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_simplified_wcm)

        # empirical observations
        obs = tigernet.testing_data("CensusBlocks_Leon_FL_2010")

        # associate observations with the network
        args = self.network, obs.copy()
        kwargs = {"df_name": "obs1", "df_key": "GEOID", "snap_to": "nodes"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_net_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 3317.88793311, 4003.17771763, 3387.79485659],
                [3317.88793311, 0.0, 940.71263515, 3273.42506072],
                [4003.17771763, 940.71263515, 0.0, 3836.53500068],
                [3387.79485659, 3273.42506072, 3836.53500068, 0.0],
            ]
        )
        args = self.net_obs, self.network
        kwargs = {
            "destination_observations": None,
            "snap_dist": False,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(
            observed_mtx[:4, :4], known_mtx, decimal=DECIMAL
        )

        known_mtx_sum = 22077455.792563077
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 3501.0453002, 4171.59260934, 3611.91094868],
                [3501.0453002, 0.0, 1027.01398167, 3415.42760763],
                [4171.59260934, 1027.01398167, 0.0, 3963.79507221],
                [3611.91094868, 3415.42760763, 3963.79507221, 0.0],
            ]
        )
        args = self.net_obs, self.network
        kwargs = {
            "destination_observations": None,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(
            observed_mtx[:4, :4], known_mtx, decimal=DECIMAL
        )

        known_mtx_sum = 23904433.77183481
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)


####################################################################################
############################# ORIG-DEST--Segments ##################################
####################################################################################


class TestEmpiricalObservationsOrigToDestSegments(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_simplified_wcm)

        # empirical origins
        obs1 = tigernet.testing_data("CensusBlocks_Leon_FL_2010")
        # empirical destinations
        obs2 = tigernet.testing_data("WeightedParcels_Leon_FL_2010")

        # associate origins with the network
        args = self.network, obs1.copy()
        kwargs = {"df_name": "obs1", "df_key": "GEOID"}
        self.net_obs1 = tigernet.Observations(*args, **kwargs)

        # associate destinations with the network
        args = self.network, obs2.copy()
        kwargs = {"df_name": "obs2", "df_key": "PARCEL_ID"}
        self.net_obs2 = tigernet.Observations(*args, **kwargs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_net_no_snap(self):
        known_mtx = numpy.array(
            [
                [2308.71804518, 1823.15385777, 1629.12550059, 1597.86997059],
                [2412.05369383, 2167.34025136, 2869.67781642, 3774.02731159],
                [3057.07235357, 2604.44689325, 3306.78445831, 4371.5796678],
                [2295.60401708, 5094.9154798, 4900.88712262, 3106.05917224],
            ]
        )
        args = copy.deepcopy(self.net_obs1), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": False,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(
            observed_mtx[:4, :4], known_mtx, decimal=DECIMAL
        )

        known_mtx_sum = 469638820.3745194
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [2396.63085253, 1918.5571568, 1727.08007354, 1702.91268741],
                [2490.37823338, 2253.15528259, 2958.04412157, 3869.4817606],
                [3124.12783575, 2678.99286711, 3383.88170609, 4455.76505945],
                [2376.79864032, 5183.60059472, 4992.12351146, 3204.38370495],
            ]
        )
        args = copy.deepcopy(self.net_obs1), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(
            observed_mtx[:4, :4], known_mtx, decimal=DECIMAL
        )

        known_mtx_sum = 489306860.5320058
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)


####################################################################################
############################# ORIG-DEST--Nodes #####################################
####################################################################################


class TestEmpiricalObservationsOrigToDestNodes(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_simplified_wcm)

        # empirical origins
        obs1 = tigernet.testing_data("CensusBlocks_Leon_FL_2010")
        # empirical destinations
        obs2 = tigernet.testing_data("WeightedParcels_Leon_FL_2010")

        # associate origins with the network
        args = self.network, obs1.copy()
        kwargs = {"df_name": "obs1", "df_key": "GEOID", "snap_to": "nodes"}
        self.net_obs1 = tigernet.Observations(*args, **kwargs)

        # associate destinations with the network
        args = self.network, obs2.copy()
        kwargs = {"df_name": "obs2", "df_key": "PARCEL_ID", "snap_to": "nodes"}
        self.net_obs2 = tigernet.Observations(*args, **kwargs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_net_no_snap(self):
        known_mtx = numpy.array(
            [
                [2064.59726946, 1899.48030691, 1749.06065623, 1567.86895727],
                [2562.49978182, 2285.24409928, 2943.97295784, 3776.2240256],
                [3125.60972178, 2648.05559975, 3306.78445831, 4461.51381012],
                [2062.69976958, 4977.94856303, 4827.52891235, 3122.6351043],
            ]
        )
        args = self.net_obs1, self.network
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": False,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(
            observed_mtx[:4, :4], known_mtx, decimal=DECIMAL
        )

        known_shape = (92, 1969)
        observed_shape = observed_mtx.shape
        self.assertEqual(observed_shape, observed_shape)

        known_mtx_sum = 467794809.4908291
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [2336.13968088, 2090.68486173, 1923.01392459, 1802.63808338],
                [2751.92864805, 2394.33510891, 3035.812681, 3928.87960652],
                [3300.29611263, 2742.40413399, 3383.88170609, 4599.42691566],
                [2293.08736082, 5127.99829766, 4960.32736052, 3316.24941022],
            ]
        )
        args = self.net_obs1, self.network
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(
            observed_mtx[:4, :4], known_mtx, decimal=DECIMAL
        )

        known_shape = (92, 1969)
        observed_shape = observed_mtx.shape
        self.assertEqual(observed_shape, observed_shape)

        known_mtx_sum = 503083944.4664013
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)


if __name__ == "__main__":
    unittest.main()

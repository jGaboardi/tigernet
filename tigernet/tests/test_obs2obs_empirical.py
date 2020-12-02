"""Synthetic observation data testing.
"""

import copy
import unittest
import numpy

import tigernet
from .network_objects import network_empirical_simplified_wcm


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
        numpy.testing.assert_array_almost_equal(observed_mtx[:4, :4], known_mtx)

        known_mtx_sum = 22144649.49077968
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)

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
        numpy.testing.assert_array_almost_equal(observed_mtx[:4, :4], known_mtx)

        known_mtx_sum = 23283690.15943207
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
        numpy.testing.assert_array_almost_equal(observed_mtx[:4, :4], known_mtx)

        known_mtx_sum = 22077455.792563077
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)

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
        numpy.testing.assert_array_almost_equal(observed_mtx[:4, :4], known_mtx)

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

    def test_net_no_snap(self):
        known_mtx = numpy.array(
            [
                [2308.55782266, 1823.52901858, 1629.12550059, 1597.1534076],
                [2412.21391635, 2166.96509055, 2869.67781642, 3773.3107486],
                [3057.23257609, 2604.07173243, 3306.78445831, 4370.86310481],
                [2295.44379456, 5095.29064061, 4900.88712262, 3105.34260925],
            ]
        )
        args = copy.deepcopy(self.net_obs1), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": False,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx[:4, :4], known_mtx)

        known_mtx_sum = 470126303.68085647
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)

    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [2395.75317806, 1918.29966974, 1726.83999895, 1702.36034626],
                [2489.82100395, 2252.14747391, 2957.80404698, 3868.92941945],
                [3123.57060632, 2677.98505843, 3383.6416315, 4455.2127183],
                [2375.92096584, 5183.34310766, 4991.88343687, 3203.8313638],
            ]
        )
        args = copy.deepcopy(self.net_obs1), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx[:4, :4], known_mtx)

        known_mtx_sum = 489927946.32060623
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
        numpy.testing.assert_array_almost_equal(observed_mtx[:4, :4], known_mtx)

        known_mtx_sum = 467758744.4634375
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)

    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [2335.4438419, 2090.54629746, 1922.77385, 1803.34707265],
                [2751.23280906, 2394.19654464, 3035.57260641, 3929.58859578],
                [3299.60027364, 2742.26556972, 3383.6416315, 4600.13590492],
                [2292.39152184, 5127.85973339, 4960.08728593, 3316.95839949],
            ]
        )
        args = self.net_obs1, self.network
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx[:4, :4], known_mtx)

        known_mtx_sum = 503078065.92697436
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum, delta=1)


if __name__ == "__main__":
    unittest.main()

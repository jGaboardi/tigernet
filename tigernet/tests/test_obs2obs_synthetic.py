"""Synthetic observation data testing.
"""

import copy
import unittest
import geopandas
import numpy
from shapely.geometry import Point

import tigernet
from .network_objects import network_lattice_1x1_small


####################################################################################
############################# ORIG-XXXX--Segments ##################################
####################################################################################


class TestSyntheticObservationsOrigToXXXXSegments(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_lattice_1x1_small)

        # generate synthetic observations
        pts = [Point(1, 1), Point(3, 1), Point(1, 3), Point(3, 3)]
        obs = geopandas.GeoDataFrame({"obs_id": ["a", "b", "c", "d"]}, geometry=pts)

        # associate observations with the network
        args = self.network, obs.copy()
        kwargs = {"df_name": "obs1", "df_key": "obs_id"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

    def test_net_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0],
            ]
        )
        args = copy.deepcopy(self.net_obs), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": None,
            "snap_dist": False,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 16.0
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 2.0, 4.0, 4.0],
                [2.0, 0.0, 4.0, 4.0],
                [4.0, 4.0, 0.0, 2.0],
                [4.0, 4.0, 2.0, 0.0],
            ]
        )
        args = copy.deepcopy(self.net_obs), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": None,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 40.0
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_euc_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0],
            ]
        )
        args = copy.deepcopy(self.net_obs), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": None,
            "snap_dist": False,
            "distance_type": "euclidean",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 16.0
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_euc_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0],
            ]
        )
        args = copy.deepcopy(self.net_obs), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": None,
            "snap_dist": True,
            "distance_type": "euclidean",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 16.0
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)


####################################################################################
############################# ORIG-XXXX--Nodes #####################################
####################################################################################


class TestSyntheticObservationsOrigToXXXXNodes(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_lattice_1x1_small)

        # generate synthetic observations
        pts = [Point(1, 1), Point(3, 1), Point(1, 3), Point(3, 3)]
        obs = geopandas.GeoDataFrame({"obs_id": ["a", "b", "c", "d"]}, geometry=pts)

        # associate observations with the network
        args = self.network, obs.copy()
        kwargs = {"df_name": "obs1", "df_key": "obs_id", "snap_to": "nodes"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

    def test_net_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        args = self.net_obs, self.network
        kwargs = {
            "destination_observations": None,
            "snap_dist": False,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 0.0
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 2.82842712, 2.82842712, 2.82842712],
                [2.82842712, 0.0, 2.82842712, 2.82842712],
                [2.82842712, 2.82842712, 0.0, 2.82842712],
                [2.82842712, 2.82842712, 2.82842712, 0.0],
            ]
        )
        args = self.net_obs, self.network
        kwargs = {
            "destination_observations": None,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 33.941125496954285
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_euc_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        args = self.net_obs, self.network
        kwargs = {
            "destination_observations": None,
            "snap_dist": False,
            "distance_type": "euclidean",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 0.0
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_euc_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        args = self.net_obs, self.network
        kwargs = {
            "destination_observations": None,
            "snap_dist": True,
            "distance_type": "euclidean",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 0.0
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)


####################################################################################
############################# ORIG-DEST--Segments ##################################
####################################################################################


class TestSyntheticObservationsOrigToDestSegments(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_lattice_1x1_small)

        # generate synthetic origins
        obs1 = tigernet.generate_obs(5, self.network.s_data)
        obs1["obs_id"] = ["a", "b", "c", "d", "e"]
        # generate synthetic destinations
        obs2 = tigernet.generate_obs(3, self.network.s_data, seed=1)
        obs2["obs_id"] = ["z", "y", "x"]

        # associate origins with the network
        args = self.network, obs1.copy()
        kwargs = {"df_name": "obs1", "df_key": "obs_id"}
        self.net_obs1 = tigernet.Observations(*args, **kwargs)

        # associate destinations with the network
        args = self.network, obs2.copy()
        kwargs = {"df_name": "obs2", "df_key": "obs_id"}
        self.net_obs2 = tigernet.Observations(*args, **kwargs)

    def test_net_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.02054051, 2.86029997, 2.49140309],
                [1.29235148, 2.41059601, 2.04169913],
                [0.29772152, 2.58311895, 2.21422207],
                [0.68579403, 3.5666345, 3.19773762],
                [2.73594902, 3.85419354, 3.48529666],
            ]
        )
        args = copy.deepcopy(self.net_obs1), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": False,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 33.737558095596384
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [0.54770651, 3.84622369, 4.09963354],
                [1.80379619, 3.38079845, 3.63420829],
                [0.93501431, 3.67916947, 3.93257931],
                [1.26735717, 4.60695537, 4.86036522],
                [3.53409492, 5.11109718, 5.36450702],
            ]
        )
        args = copy.deepcopy(self.net_obs1), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 50.60350662252801
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_euc_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.02054051, 2.17694135, 2.49140309],
                [0.97244594, 2.41059601, 1.68165696],
                [0.29772152, 2.08296224, 2.21422207],
                [0.68579403, 2.54046208, 3.19773762],
                [2.05339149, 3.85419354, 2.46956183],
            ]
        )
        args = copy.deepcopy(self.net_obs1), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": False,
            "distance_type": "euclidean",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 29.14963026553491
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_euc_snap(self):
        known_mtx = numpy.array(
            [
                [0.02054051, 2.17694135, 2.49140309],
                [0.97244594, 2.41059601, 1.68165696],
                [0.29772152, 2.08296224, 2.21422207],
                [0.68579403, 2.54046208, 3.19773762],
                [2.05339149, 3.85419354, 2.46956183],
            ]
        )
        args = copy.deepcopy(self.net_obs1), copy.deepcopy(self.network)
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": True,
            "distance_type": "euclidean",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 29.14963026553491
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)


####################################################################################
############################# ORIG-DEST--Nodes #####################################
####################################################################################


class TestSyntheticObservationsOrigToDestNodes(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_lattice_1x1_small)

        # generate synthetic origins
        obs1 = tigernet.generate_obs(5, self.network.s_data)
        obs1["obs_id"] = ["a", "b", "c", "d", "e"]
        # generate synthetic destinations
        obs2 = tigernet.generate_obs(3, self.network.s_data, seed=1)
        obs2["obs_id"] = ["z", "y", "x"]

        # associate origins with the network
        args = self.network, obs1.copy()
        kwargs = {"df_name": "obs1", "df_key": "obs_id", "snap_to": "nodes"}
        self.net_obs1 = tigernet.Observations(*args, **kwargs)

        # associate destinations with the network
        args = self.network, obs2.copy()
        kwargs = {"df_name": "obs2", "df_key": "obs_id", "snap_to": "nodes"}
        self.net_obs2 = tigernet.Observations(*args, **kwargs)

    def test_net_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 2.0, 2.0],
                [0.0, 2.0, 2.0],
                [0.0, 2.0, 2.0],
                [2.0, 4.0, 4.0],
                [2.0, 4.0, 4.0],
            ]
        )
        args = self.net_obs1, self.network
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": False,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 32.0
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_net_snap(self):
        known_mtx = numpy.array(
            [
                [1.8243534, 3.67329521, 4.34307909],
                [1.3902779, 3.23921971, 3.90900359],
                [1.60037734, 3.44931915, 4.11910303],
                [3.44146299, 5.2904048, 5.96018868],
                [3.43009305, 5.27903486, 5.94881874],
            ]
        )
        args = self.net_obs1, self.network
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": True,
            "distance_type": "network",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 56.89803154575887
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_euc_no_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 2.0, 2.0],
                [0.0, 2.0, 2.0],
                [0.0, 2.0, 2.0],
                [2.0, 2.82842712, 4.0],
                [2.0, 4.0, 2.82842712],
            ]
        )
        args = self.net_obs1, self.network
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": False,
            "distance_type": "euclidean",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 29.656854249492383
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)

    def test_euc_snap(self):
        known_mtx = numpy.array(
            [
                [0.0, 2.0, 2.0],
                [0.0, 2.0, 2.0],
                [0.0, 2.0, 2.0],
                [2.0, 2.82842712, 4.0],
                [2.0, 4.0, 2.82842712],
            ]
        )
        args = self.net_obs1, self.network
        kwargs = {
            "destination_observations": self.net_obs2,
            "snap_dist": True,
            "distance_type": "euclidean",
        }
        observed_mtx = tigernet.obs2obs_cost_matrix(*args, **kwargs)
        numpy.testing.assert_array_almost_equal(observed_mtx, known_mtx)

        known_mtx_sum = 29.656854249492383
        observed_mtx_sum = observed_mtx.sum()
        self.assertAlmostEqual(observed_mtx_sum, known_mtx_sum)


if __name__ == "__main__":
    unittest.main()

"""emprical observation data testing.
"""

import copy
import unittest
import numpy

import tigernet
from .network_objects import network_empirical_simplified

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
################################## EMPIR-EMPIR #####################################
####################################################################################


class TestEmpiricalObservationsSegment(unittest.TestCase):
    def setUp(self):
        network = copy.deepcopy(network_empirical_simplified)

        # generate synthetic observations
        obs = tigernet.testing_data("WeightedParcels_Leon_FL_2010")

        # associate observations with the network
        args = network, obs.copy()
        kwargs = {"df_name": "obs1", "df_key": "PARCEL_ID"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

    def test_obs2coords(self):
        known_obs2coords = [
            ((1964, "1117160000020"), (623164.6468275142, 164563.56861681942)),
            ((1965, "1117160000010"), (623217.6552926502, 164543.54240330344)),
            ((1966, "2113200600000"), (621263.5870570062, 165290.3686240851)),
            ((1967, "2113370000020"), (621277.5546571331, 165245.949495386)),
            ((1968, "2113200920000"), (621301.4738957918, 165288.5749146659)),
        ]
        observed_obs2coords = self.net_obs.obs2coords
        for k, v in known_obs2coords:
            obs = numpy.array(observed_obs2coords[k])
            numpy.testing.assert_array_almost_equal(
                obs, numpy.array(v), decimal=DECIMAL
            )

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_obs2segm(self):
        known_obs2segm = [
            ("1117160000020", 326),
            ("1117160000010", 326),
            ("2113200600000", 94),
            ("2113370000020", 297),
            ("2113200920000", 94),
        ]
        observed_obs2segm = list(self.net_obs.obs2segm.items())[-5:]
        self.assertEqual(observed_obs2segm, known_obs2segm)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_dist_a(self):
        known_dist_a = numpy.array(
            [
                131.88044673243306,
                32.97512515277735,
                98.65586950150907,
                55.098820557205926,
                185.54560775073523,
            ]
        )
        observed_dist_a = list(self.net_obs.snapped_points["dist_a"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_a), known_dist_a, decimal=DECIMAL
        )

        known_dist_a_mean = 150.35622473717447
        observed_dist_a_mean = self.net_obs.snapped_points["dist_a"].mean()
        self.assertAlmostEqual(observed_dist_a_mean, known_dist_a_mean)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_dist_b(self):
        known_dist_b = numpy.array(
            [
                157.15709861024837,
                256.06242018990406,
                152.9136827435795,
                32.797412043076804,
                66.02394449435334,
            ]
        )
        observed_dist_b = list(self.net_obs.snapped_points["dist_b"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_b), known_dist_b, decimal=DECIMAL
        )

        known_dist_b_mean = 134.64877859913972
        observed_dist_b_mean = self.net_obs.snapped_points["dist_b"].mean()
        self.assertAlmostEqual(observed_dist_b_mean, known_dist_b_mean)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_node_a(self):
        known_node_a = [109, 109, 137, 64, 137]
        observed_node_a = list(self.net_obs.snapped_points["node_a"])[-5:]
        self.assertEqual(observed_node_a, known_node_a)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_node_b(self):
        known_node_b = [279, 279, 64, 271, 64]
        observed_node_b = list(self.net_obs.snapped_points["node_b"])[-5:]
        self.assertEqual(observed_node_b, known_node_b)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_dist2line(self):
        known_dist2line = numpy.array(
            [
                36.671640315034786,
                17.984434828655004,
                29.53995199409558,
                29.974400866302904,
                25.331194851980122,
            ]
        )
        observed_dist2line = list(self.net_obs.snapped_points["dist2line"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist2line), known_dist2line, decimal=DECIMAL
        )

        known_dist2line_mean = 41.04646957730281
        observed_dist2ine_mean = self.net_obs.snapped_points["dist2line"].mean()
        self.assertAlmostEqual(observed_dist2ine_mean, known_dist2line_mean)


class TestEmpiricalObservationsNodeEmpirical(unittest.TestCase):
    def setUp(self):
        network = copy.deepcopy(network_empirical_simplified)

        # generate synthetic observations
        obs = tigernet.testing_data("WeightedParcels_Leon_FL_2010")

        # associate observations with the network
        args = network, obs.copy()
        kwargs = {"df_name": "obs1", "df_key": "PARCEL_ID", "snap_to": "nodes"}
        self.net_obs = tigernet.Observations(*args, **kwargs)

    def test_obs2coords(self):
        known_obs2coords = [
            ((1964, "1117160000020"), (623164.6468275142, 164563.56861681942)),
            ((1965, "1117160000010"), (623217.6552926502, 164543.54240330344)),
            ((1966, "2113200600000"), (621263.5870570062, 165290.3686240851)),
            ((1967, "2113370000020"), (621277.5546571331, 165245.949495386)),
            ((1968, "2113200920000"), (621301.4738957918, 165288.5749146659)),
        ]
        observed_obs2coords = self.net_obs.obs2coords
        for k, v in known_obs2coords:
            numpy.testing.assert_array_almost_equal(
                numpy.array(observed_obs2coords[k]), numpy.array(v)
            )

    def test_obs2node(self):
        known_obs2node = [
            ("1117160000020", 278),
            ("1117160000010", 109),
            ("2113200600000", 271),
            ("2113370000020", 271),
            ("2113200920000", 64),
        ]
        observed_obs2node = self.net_obs.obs2node
        for k, v in known_obs2node:
            self.assertAlmostEqual(observed_obs2node[k], v)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_dist2node(self):
        known_dist2node = numpy.array(
            [
                75.64960013443921,
                46.835114475515894,
                77.50991311066547,
                44.43123837837936,
                71.89657171345416,
            ]
        )
        observed_dist2node = list(self.net_obs.snapped_points["dist2node"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(numpy.array(observed_dist2node)),
            known_dist2node,
            decimal=DECIMAL,
        )

        known_dist2node_mean = 85.6959085474719
        observed_dist2node_mean = self.net_obs.snapped_points["dist2node"].mean()
        self.assertAlmostEqual(observed_dist2node_mean, known_dist2node_mean)


####################################################################################
######################## EMPIR-EMPIR RESTRICTED ####################################
####################################################################################


class TestEmpiricalObservationsSegmentEmpiricalRestricted(unittest.TestCase):
    def setUp(self):
        network = copy.deepcopy(network_empirical_simplified)

        # generate synthetic observations
        obs = tigernet.testing_data("WeightedParcels_Leon_FL_2010")

        # associate observations with the network
        args = network, obs.copy()
        kwargs = {"df_name": "obs1", "df_key": "PARCEL_ID"}
        kwargs.update({"restrict_col": "MTFCC"})
        kwargs.update({"remove_restricted": ["S1100", "S1630", "S1640"]})
        self.net_obs = tigernet.Observations(*args, **kwargs)

    def test_obs2coords(self):
        known_obs2coords = [
            ((1964, "1117160000020"), (623164.6468275142, 164563.56861681942)),
            ((1965, "1117160000010"), (623217.6552926502, 164543.54240330344)),
            ((1966, "2113200600000"), (621263.5870570062, 165290.3686240851)),
            ((1967, "2113370000020"), (621277.5546571331, 165245.949495386)),
            ((1968, "2113200920000"), (621301.4738957918, 165288.5749146659)),
        ]
        observed_obs2coords = self.net_obs.obs2coords
        for k, v in known_obs2coords:
            obs = numpy.array(observed_obs2coords[k])
            numpy.testing.assert_array_almost_equal(obs, numpy.array(v))

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_obs2segm(self):
        known_obs2segm = [
            ("1117160000020", 326),
            ("1117160000010", 326),
            ("2113200600000", 94),
            ("2113370000020", 297),
            ("2113200920000", 94),
        ]
        observed_obs2segm = list(self.net_obs.obs2segm.items())[-5:]
        self.assertEqual(observed_obs2segm, known_obs2segm)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_dist_a(self):
        known_dist_a = numpy.array(
            [
                131.88044673243306,
                32.97512515277735,
                98.65586950150907,
                55.098820557205926,
                185.54560775073523,
            ]
        )
        observed_dist_a = list(self.net_obs.snapped_points["dist_a"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_a), known_dist_a, decimal=DECIMAL
        )

        known_dist_a_mean = 146.85511322662327
        observed_dist_a_mean = self.net_obs.snapped_points["dist_a"].mean()
        self.assertAlmostEqual(observed_dist_a_mean, known_dist_a_mean)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_dist_b(self):
        known_dist_b = numpy.array(
            [
                157.15709861024837,
                256.06242018990406,
                152.9136827435795,
                32.797412043076804,
                66.02394449435334,
            ]
        )
        observed_dist_b = list(self.net_obs.snapped_points["dist_b"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_b), known_dist_b, decimal=DECIMAL
        )

        known_dist_b_mean = 133.53613454081363
        observed_dist_b_mean = self.net_obs.snapped_points["dist_b"].mean()
        self.assertAlmostEqual(observed_dist_b_mean, known_dist_b_mean)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_node_a(self):
        known_node_a = [109, 109, 137, 64, 137]
        observed_node_a = list(self.net_obs.snapped_points["node_a"])[-5:]
        self.assertEqual(observed_node_a, known_node_a)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_node_b(self):
        known_node_b = [279, 279, 64, 271, 64]
        observed_node_b = list(self.net_obs.snapped_points["node_b"])[-5:]
        self.assertEqual(observed_node_b, known_node_b)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_dist2line(self):
        known_dist2line = numpy.array(
            [
                36.671640315034786,
                17.984434828655004,
                29.53995199409558,
                29.974400866302904,
                25.331194851980122,
            ]
        )
        observed_dist2line = list(self.net_obs.snapped_points["dist2line"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist2line), known_dist2line, decimal=DECIMAL
        )

        known_dist2line_mean = 41.33531671731001
        observed_dist2ine_mean = self.net_obs.snapped_points["dist2line"].mean()
        self.assertAlmostEqual(observed_dist2ine_mean, known_dist2line_mean)


class TestEmpiricalObservationsNodeEmpiricalRestricted(unittest.TestCase):
    def setUp(self):
        network = copy.deepcopy(network_empirical_simplified)

        # generate synthetic observations
        obs = tigernet.testing_data("WeightedParcels_Leon_FL_2010")

        # associate observations with the network
        args = network, obs.copy()
        kwargs = {"df_name": "obs1", "df_key": "PARCEL_ID", "snap_to": "nodes"}
        kwargs.update({"restrict_col": "MTFCC"})
        kwargs.update({"remove_restricted": ["S1100", "S1630", "S1640"]})
        self.net_obs = tigernet.Observations(*args, **kwargs)

    def test_obs2coords(self):
        known_obs2coords = [
            ((1964, "1117160000020"), (623164.6468275142, 164563.56861681942)),
            ((1965, "1117160000010"), (623217.6552926502, 164543.54240330344)),
            ((1966, "2113200600000"), (621263.5870570062, 165290.3686240851)),
            ((1967, "2113370000020"), (621277.5546571331, 165245.949495386)),
            ((1968, "2113200920000"), (621301.4738957918, 165288.5749146659)),
        ]
        observed_obs2coords = self.net_obs.obs2coords
        for k, v in known_obs2coords:
            numpy.testing.assert_array_almost_equal(
                numpy.array(observed_obs2coords[k]), numpy.array(v), decimal=DECIMAL
            )

    def test_obs2node(self):
        known_obs2node = [
            ("1117160000020", 278),
            ("1117160000010", 109),
            ("2113200600000", 271),
            ("2113370000020", 271),
            ("2113200920000", 64),
        ]
        observed_obs2node = self.net_obs.obs2node
        for k, v in known_obs2node:
            self.assertAlmostEqual(observed_obs2node[k], v)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_snapped_points_df_dist2node(self):
        known_dist2node = numpy.array(
            [
                75.64960013443921,
                46.835114475515894,
                77.50991311066547,
                44.43123837837936,
                71.89657171345416,
            ]
        )
        observed_dist2node = list(self.net_obs.snapped_points["dist2node"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(numpy.array(observed_dist2node)),
            known_dist2node,
            decimal=DECIMAL,
        )

        known_dist2node_mean = 85.79855239395815
        observed_dist2node_mean = self.net_obs.snapped_points["dist2node"].mean()
        self.assertAlmostEqual(observed_dist2node_mean, known_dist2node_mean)


####################################################################################
######################### Segment-to-Population ####################################
####################################################################################


class TestEmpiricalObservationsSegm2Pop(unittest.TestCase):
    def setUp(self):
        network = copy.deepcopy(network_empirical_simplified)

        # generate synthetic observations ---------------------------------------- 1
        self.obs1 = tigernet.testing_data("WeightedParcels_Leon_FL_2010")

        # associate observations with the network
        args = network, self.obs1.copy()
        kwargs = {"df_name": "obs1", "df_key": "PARCEL_ID", "snap_to": "segments"}
        kwargs.update({"obs_pop": "SUM_EST_PO", "restrict_col": "MTFCC"})
        kwargs.update({"remove_restricted": ["S1100", "S1630", "S1640"]})
        self.net_obs1 = tigernet.Observations(*args, **kwargs)

        # generate synthetic observations ---------------------------------------- 2
        self.obs2 = tigernet.testing_data("CensusBlocks_Leon_FL_2010")

        # associate observations with the network
        args = network, self.obs2.copy()
        kwargs = {"df_name": "obs2", "df_key": "GEOID", "snap_to": "segments"}
        kwargs.update({"obs_pop": "POP100", "restrict_col": "MTFCC"})
        kwargs.update({"remove_restricted": ["S1100", "S1630", "S1640"]})
        self.net_obs2 = tigernet.Observations(*args, **kwargs)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_segm2pop_1(self):
        known_segm2pop = [
            (336, 4.85357142857),
            (337, 27.62499999998),
            (338, 0.0),
            (339, 1.8),
            (340, 20.85714285714),
            (341, 19.710526315760003),
            (342, 9.904761904760003),
            (343, 46.666666666600015),
            (344, 11.666666666649999),
            (345, 12.125),
        ]
        observed_segm2pop = self.net_obs1.segm2pop
        for k, v in known_segm2pop:
            self.assertAlmostEqual(observed_segm2pop[k], v)

        known_pop_sum = self.obs1["SUM_EST_PO"].sum()
        known_segm2pop_sum = 4874.156151922997
        observed_segm2pop_sum = sum(self.net_obs1.segm2pop.values())
        self.assertAlmostEqual(known_pop_sum, known_segm2pop_sum)
        self.assertAlmostEqual(observed_segm2pop_sum, known_segm2pop_sum)

    @unittest.skipIf(WINDOWS, "Skipping Windows due to precision issues.")
    def test_segm2pop_2(self):
        known_segm2pop = [
            (336, 0),
            (337, 31),
            (338, 0),
            (339, 0),
            (340, 0),
            (341, 0),
            (342, 52),
            (343, 0),
            (344, 189),
            (345, 0),
        ]
        observed_segm2pop = self.net_obs2.segm2pop
        for k, v in known_segm2pop:
            self.assertAlmostEqual(observed_segm2pop[k], v)

        known_pop_sum = self.obs2["POP100"].sum()
        known_segm2pop_sum = 3791
        observed_segm2pop_sum = sum(self.net_obs2.segm2pop.values())
        self.assertAlmostEqual(known_pop_sum, known_segm2pop_sum)
        self.assertAlmostEqual(observed_segm2pop_sum, known_segm2pop_sum)


if __name__ == "__main__":
    unittest.main()

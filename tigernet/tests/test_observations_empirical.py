"""emprical observation data testing.
"""

import copy
import unittest
import numpy

import tigernet
from .network_objects import network_empirical_simplified


####################################################################################
################################## EMPIR-EMPIR #####################################
####################################################################################


class TestSEmpiricalObservationsSegment(unittest.TestCase):
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
            numpy.testing.assert_array_almost_equal(obs, numpy.array(v))

    def test_obs2segm(self):
        known_obs2segm = [
            ("1117160000020", 324),
            ("1117160000010", 116),
            ("2113200600000", 94),
            ("2113370000020", 297),
            ("2113200920000", 94),
        ]
        observed_obs2segm = list(self.net_obs.obs2segm.items())[-5:]
        self.assertEqual(observed_obs2segm, known_obs2segm)

    def test_snapped_points_df_dist_a(self):
        known_dist_a = numpy.array(
            [
                100.6590409536673,
                1.9196222374448804,
                98.65586950148744,
                55.098820557786,
                185.54560775124625,
            ]
        )
        observed_dist_a = list(self.net_obs.snapped_points["dist_a"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_a), known_dist_a
        )

        known_dist_a_mean = 150.44896275323015
        observed_dist_a_mean = self.net_obs.snapped_points["dist_a"].mean()
        self.assertAlmostEqual(observed_dist_a_mean, known_dist_a_mean)

    def test_snapped_points_df_dist_b(self):
        known_dist_b = numpy.array(
            [
                7.402633071489845,
                126.88698767323638,
                152.91368274543564,
                32.79741204175603,
                66.02394449567683,
            ]
        )
        observed_dist_b = list(self.net_obs.snapped_points["dist_b"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_b), known_dist_b
        )

        known_dist_b_mean = 134.26040850106116
        observed_dist_b_mean = self.net_obs.snapped_points["dist_b"].mean()
        self.assertAlmostEqual(observed_dist_b_mean, known_dist_b_mean)

    def test_snapped_points_df_node_a(self):
        known_node_a = [108, 109, 137, 64, 137]
        observed_node_a = list(self.net_obs.snapped_points["node_a"])[-5:]
        self.assertEqual(observed_node_a, known_node_a)

    def test_snapped_points_df_node_b(self):
        known_node_b = [278, 161, 64, 271, 64]
        observed_node_b = list(self.net_obs.snapped_points["node_b"])[-5:]
        self.assertEqual(observed_node_b, known_node_b)

    def test_snapped_points_df_dist2line(self):
        known_dist2line = numpy.array(
            [
                75.28653946161977,
                46.795758337733,
                29.539951996916958,
                29.974400864691226,
                25.33119485173932,
            ]
        )
        observed_dist2line = list(self.net_obs.snapped_points["dist2line"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist2line), known_dist2line
        )

        known_dist2line_mean = 41.294713493852406
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
            numpy.array(numpy.array(observed_dist2node)), known_dist2node
        )

        known_dist2node_mean = 85.82639474374425
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

    def test_obs2segm(self):
        known_obs2segm = [
            ("1117160000020", 324),
            ("1117160000010", 116),
            ("2113200600000", 94),
            ("2113370000020", 297),
            ("2113200920000", 94),
        ]
        observed_obs2segm = list(self.net_obs.obs2segm.items())[-5:]
        self.assertEqual(observed_obs2segm, known_obs2segm)

    def test_snapped_points_df_dist_a(self):
        known_dist_a = numpy.array(
            [
                100.6590409536673,
                1.9196222374448804,
                98.65586950148744,
                55.098820557786,
                185.54560775124625,
            ]
        )
        observed_dist_a = list(self.net_obs.snapped_points["dist_a"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_a), known_dist_a
        )

        known_dist_a_mean = 146.94785124268932
        observed_dist_a_mean = self.net_obs.snapped_points["dist_a"].mean()
        self.assertAlmostEqual(observed_dist_a_mean, known_dist_a_mean)

    def test_snapped_points_df_dist_b(self):
        known_dist_b = numpy.array(
            [
                7.402633071489845,
                126.88698767323638,
                152.91368274543564,
                32.79741204175603,
                66.02394449567683,
            ]
        )
        observed_dist_b = list(self.net_obs.snapped_points["dist_b"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_b), known_dist_b
        )

        known_dist_b_mean = 133.1477644427244
        observed_dist_b_mean = self.net_obs.snapped_points["dist_b"].mean()
        self.assertAlmostEqual(observed_dist_b_mean, known_dist_b_mean)

    def test_snapped_points_df_node_a(self):
        known_node_a = [108, 109, 137, 64, 137]
        observed_node_a = list(self.net_obs.snapped_points["node_a"])[-5:]
        self.assertEqual(observed_node_a, known_node_a)

    def test_snapped_points_df_node_b(self):
        known_node_b = [278, 161, 64, 271, 64]
        observed_node_b = list(self.net_obs.snapped_points["node_b"])[-5:]
        self.assertEqual(observed_node_b, known_node_b)

    def test_snapped_points_df_dist2line(self):
        known_dist2line = numpy.array(
            [
                75.28653946161977,
                46.795758337733,
                29.539951996916958,
                29.974400864691226,
                25.33119485173932,
            ]
        )
        observed_dist2line = list(self.net_obs.snapped_points["dist2line"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist2line), known_dist2line
        )

        known_dist2line_mean = 41.58356063385915
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
            numpy.array(numpy.array(observed_dist2node)), known_dist2node
        )

        known_dist2node_mean = 85.92903859023075
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

    def test_segm2pop_1(self):
        known_segm2pop = [
            (335, 4.85357142857),
            (336, 27.62499999998),
            (337, 0.0),
            (338, 1.8),
            (339, 20.85714285714),
            (340, 19.710526315760003),
            (341, 9.904761904760003),
            (342, 46.666666666600015),
            (343, 11.666666666649999),
            (344, 12.125),
        ]
        observed_segm2pop = self.net_obs1.segm2pop
        for k, v in known_segm2pop:
            self.assertAlmostEqual(observed_segm2pop[k], v)

        known_pop_sum = self.obs1["SUM_EST_PO"].sum()
        known_segm2pop_sum = 4874.156151922997
        observed_segm2pop_sum = sum(self.net_obs1.segm2pop.values())
        self.assertAlmostEqual(known_pop_sum, known_segm2pop_sum)
        self.assertAlmostEqual(observed_segm2pop_sum, known_segm2pop_sum)

    def test_segm2pop_2(self):
        known_segm2pop = [
            (335, 0),
            (336, 31),
            (337, 0),
            (338, 0),
            (339, 0),
            (340, 0),
            (341, 52),
            (342, 0),
            (343, 189),
            (344, 0),
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

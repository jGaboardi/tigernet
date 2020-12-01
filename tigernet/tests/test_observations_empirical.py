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
            ((1964, "1117160000020"), (623164.270749338, 164564.20005694253)),
            ((1965, "1117160000010"), (623217.2792266451, 164544.17383878413)),
            ((1966, "2113200600000"), (621263.2105423947, 165291.0002326349)),
            ((1967, "2113370000020"), (621277.1781457132, 165246.58109367805)),
            ((1968, "2113200920000"), (621301.0973898807, 165289.20652278655)),
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
                101.22125208369172,
                2.355757530573965,
                98.25770372915204,
                55.45498202154211,
                184.93666598339948,
            ]
        )
        observed_dist_a = list(self.net_obs.snapped_points["dist_a"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_a), known_dist_a
        )

        known_dist_a_mean = 151.2405677949956
        observed_dist_a_mean = self.net_obs.snapped_points["dist_a"].mean()
        self.assertAlmostEqual(observed_dist_a_mean, known_dist_a_mean)

    def test_snapped_points_df_dist_b(self):
        known_dist_b = numpy.array(
            [
                6.840421941465422,
                126.45085238010729,
                153.31184851777104,
                32.441250577999924,
                66.6328862635236,
            ]
        )
        observed_dist_b = list(self.net_obs.snapped_points["dist_b"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_b), known_dist_b
        )

        known_dist_b_mean = 134.3391469790589
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
                75.75989986241024,
                47.38730073895094,
                28.92176381317788,
                30.617694159772974,
                25.74335433857507,
            ]
        )
        observed_dist2line = list(self.net_obs.snapped_points["dist2line"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist2line), known_dist2line
        )

        known_dist2line_mean = 41.28516261058276
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
            ((1964, "1117160000020"), (623164.270749338, 164564.20005694253)),
            ((1965, "1117160000010"), (623217.2792266451, 164544.17383878413)),
            ((1966, "2113200600000"), (621263.2105423947, 165291.0002326349)),
            ((1967, "2113370000020"), (621277.1781457132, 165246.58109367805)),
            ((1968, "2113200920000"), (621301.0973898807, 165289.20652278655)),
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
                76.06808660339182,
                47.44582030980183,
                78.03947131820428,
                44.60804786945554,
                72.62844699189847,
            ]
        )
        observed_dist2node = list(self.net_obs.snapped_points["dist2node"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(numpy.array(observed_dist2node)), known_dist2node
        )

        known_dist2node_mean = 85.86254846615837
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
            ((1964, "1117160000020"), (623164.270749338, 164564.20005694253)),
            ((1965, "1117160000010"), (623217.2792266451, 164544.17383878413)),
            ((1966, "2113200600000"), (621263.2105423947, 165291.0002326349)),
            ((1967, "2113370000020"), (621277.1781457132, 165246.58109367805)),
            ((1968, "2113200920000"), (621301.0973898807, 165289.20652278655)),
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
                101.22125208369172,
                2.355757530573965,
                98.25770372915204,
                55.45498202154211,
                184.93666598339948,
            ]
        )
        observed_dist_a = list(self.net_obs.snapped_points["dist_a"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_a), known_dist_a
        )

        known_dist_a_mean = 146.55829989861797
        observed_dist_a_mean = self.net_obs.snapped_points["dist_a"].mean()
        self.assertAlmostEqual(observed_dist_a_mean, known_dist_a_mean)

    def test_snapped_points_df_dist_b(self):
        known_dist_b = numpy.array(
            [
                6.840421941465422,
                126.45085238010729,
                153.31184851777104,
                32.441250577999924,
                66.6328862635236,
            ]
        )
        observed_dist_b = list(self.net_obs.snapped_points["dist_b"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist_b), known_dist_b
        )

        known_dist_b_mean = 133.1811533652186
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
                75.75989986241024,
                47.38730073895094,
                28.92176381317788,
                30.617694159772974,
                25.74335433857507,
            ]
        )
        observed_dist2line = list(self.net_obs.snapped_points["dist2line"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(observed_dist2line), known_dist2line
        )

        known_dist2line_mean = 41.57135787648275
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
            ((1964, "1117160000020"), (623164.270749338, 164564.20005694253)),
            ((1965, "1117160000010"), (623217.2792266451, 164544.17383878413)),
            ((1966, "2113200600000"), (621263.2105423947, 165291.0002326349)),
            ((1967, "2113370000020"), (621277.1781457132, 165246.58109367805)),
            ((1968, "2113200920000"), (621301.0973898807, 165289.20652278655)),
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
                76.06808660339182,
                47.44582030980183,
                78.03947131820428,
                44.60804786945554,
                72.62844699189847,
            ]
        )
        observed_dist2node = list(self.net_obs.snapped_points["dist2node"])[-5:]
        numpy.testing.assert_array_almost_equal(
            numpy.array(numpy.array(observed_dist2node)), known_dist2node
        )

        known_dist2node_mean = 85.96394724233582
        observed_dist2node_mean = self.net_obs.snapped_points["dist2node"].mean()
        self.assertAlmostEqual(observed_dist2node_mean, known_dist2node_mean)


if __name__ == "__main__":
    unittest.main()

"""Data generation for testing for tigernet.py
"""

import unittest
import numpy

import tigernet
from .network_objects import network_lattice_1x1_no_args
from .network_objects import network_empirical_simplified


class TestNetworkDataGenerationCreation(unittest.TestCase):
    def setUp(self):
        b1133, nh, nv = [-1, -1, 3, 3], 3, 4
        self._34_kws = {"bounds": b1133, "n_hori_lines": nh, "n_vert_lines": nv}

    def test_generate_sine_lines(self):
        sine = tigernet.generate_sine_lines()
        observed_length = sine.loc[(sine["SegID"] == 0), "geometry"].squeeze().length
        known_length = 11.774626215766602
        self.assertAlmostEqual(observed_length, known_length, 10)

    def test_generate_lattice_2x2_xbox_0099(self):
        lat = tigernet.generate_lattice()
        observed_length = lat.length.sum()
        known_length = 36.0
        self.assertEqual(observed_length, known_length)

    def test_generate_lattice_2x2_wbox_0099(self):
        lat = tigernet.generate_lattice(wbox=True)
        observed_length = lat.length.sum()
        known_length = 72.0
        self.assertEqual(observed_length, known_length)

    def test_generate_lattice_3x4_xbox_neg1neg133(self):
        lat = tigernet.generate_lattice(**self._34_kws)
        observed_length = lat.length.sum()
        known_length = 27.8
        self.assertEqual(observed_length, known_length)

    def test_generate_lattice_2x2_wbox_neg1neg133(self):
        lat = tigernet.generate_lattice(wbox=True, **self._34_kws)
        observed_length = lat.length.sum()
        known_length = 44.2
        self.assertEqual(observed_length, known_length)


class TestNetworkDataGenerationRead(unittest.TestCase):
    def setUp(self):
        self.dset = "Edges_Leon_FL_2010"

    def test_good_read_no_bbox(self):
        known_recs = 37798
        bbox = None
        observed_recs = tigernet.testing_data(self.dset, bbox=bbox).shape[0]
        self.assertEqual(observed_recs, known_recs)

    def test_good_read_from_coords(self):
        known_recs = 4
        bbox = (-84.34, 30.4935, -84.3378, 30.494)
        observed_recs = tigernet.testing_data(self.dset, bbox=bbox).shape[0]
        self.assertEqual(observed_recs, known_recs)


class TestObservationDataGenerationSynthetic(unittest.TestCase):
    def setUp(self):
        self.network = network_lattice_1x1_no_args

    def test_generate_observations_totalbounds(self):
        n_obs = 5
        obs = tigernet.generate_obs(n_obs, self.network.s_data)
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (4.939321535345923, 6.436704297351775),
                (5.4248703846447945, 4.903948646972072),
                (3.8128931940501425, 5.813047017599905),
                (3.9382849013642325, 8.025957007038718),
                (8.672964844509263, 3.4509736694319995),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_generate_observations_inbuffer(self):
        n_obs = 5
        obs = tigernet.generate_obs(n_obs, self.network.s_data, near_net=0.5)
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (4.939321535345923, 6.436704297351775),
                (5.4248703846447945, 4.903948646972072),
                (7.125525342743981, 4.76005427777614),
                (4.153314260276387, 7.024762586578099),
                (4.696634895750645, 3.731957459914712),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)


class TestObservationDataGenerationEmpirical(unittest.TestCase):
    def setUp(self):
        self.network = network_empirical_simplified

    def test_generate_observations_totalbounds(self):

        n_obs = 500
        obs = tigernet.generate_obs(n_obs, self.network.s_data)
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (622974.1796832045, 166162.55760926675),
                (623169.2985719838, 165632.6814708112),
                (622521.5219401979, 165946.95826632337),
                (622571.9108776418, 166711.9648736473),
                (624474.552580049, 165130.38564160923),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])[:5]
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_generate_observations_inbuffer(self):
        n_obs = 500
        obs = tigernet.generate_obs(n_obs, self.network.s_data, near_net=30)
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (622571.9108776418, 166711.9648736473),
                (624474.552580049, 165130.38564160923),
                (623803.6385443554, 166644.26001242414),
                (622876.6555154349, 165227.52219813256),
                (623400.7775349629, 166023.94388077687),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])[:5]
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_generate_observations_inbuffer_restrict(self):
        n_obs = 500
        obs = tigernet.generate_obs(
            n_obs, self.network.s_data, restrict=("MTFCC", ["S1100"]), near_net=30
        )
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (622571.9108776418, 166711.9648736473),
                (624474.552580049, 165130.38564160923),
                (623803.6385443554, 166644.26001242414),
                (622876.6555154349, 165227.52219813256),
                (623400.7775349629, 166023.94388077687),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])[:5]
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)


if __name__ == "__main__":
    unittest.main()

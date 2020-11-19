"""Testing for tigernet.py
"""

import tigernet
import unittest
import numpy


class TestNetworkDataGeneration(unittest.TestCase):
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


class TestObservationDataGenerationSynth(unittest.TestCase):
    def setUp(self):
        lat = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.net = tigernet.Network(s_data=lat)

    def test_generate_observations_totalbounds(self):
        n_obs = 5
        obs = tigernet.generate_obs(n_obs, self.net.s_data)
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
        obs = tigernet.generate_obs(n_obs, self.net.s_data, near_net=0.5)
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


if __name__ == "__main__":
    unittest.main()

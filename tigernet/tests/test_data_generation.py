"""Testing for tigernet.py
"""

import tigernet
import unittest


class TestDataGeneration(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

"""Testing for tigernet.py
"""

import tigernet
import unittest


class TestDataGeneration(unittest.TestCase):
    def setUp(self):
        pass

    def test_generate_sine_lines(self):
        sine = tigernet.generate_sine_lines()
        observed_length = sine.loc[(sine["SegID"] == 0), "geometry"].squeeze().length
        known_length = 11.774626215766602
        self.assertAlmostEqual(observed_length, known_length, 10)

    def test_generate_lattice(self):
        lat = tigernet.generate_lattice()
        observed_length = lat.length.sum()
        known_length = 36.0
        self.assertEqual(observed_length, known_length)

    def test_generate_lattice_wbox(self):
        lat = tigernet.generate_lattice(wbox=True)
        observed_length = lat.length.sum()
        known_length = 72.0
        self.assertEqual(observed_length, known_length)


if __name__ == "__main__":
    unittest.main()

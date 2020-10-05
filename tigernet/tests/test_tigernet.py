"""Testing for tigernet.py
"""

import tigernet
import geopandas
import numpy
from libpysal import cg

import unittest


class TestTigerNet(unittest.TestCase):
    def setUp(self):
        pass

    def test_generate_sine_lines(self):
        sine = tigernet.generate_sine_lines()
        observed_length = sine.loc[(sine["SegID"] == 0), "geometry"].squeeze().length
        known_length = 11.774626215766602
        self.assertAlmostEqual(observed_length, known_length, 10)


if __name__ == "__main__":
    unittest.main()

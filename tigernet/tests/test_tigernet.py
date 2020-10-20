"""Testing for tigernet.py
"""

import tigernet
import geopandas
import numpy
from libpysal import cg

import unittest


class TestTigerNetLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)

    def test_1x1_lattice_network(self):
        lattice_network = tigernet.TigerNet(segmdata=self.lattice)

        known_segments = 4
        observed_segments = lattice_network.segmdata.shape[0]
        self.assertEqual(observed_segments, known_segments)

        known_length = 18.0
        observed_length = lattice_network.segmdata.length.sum()
        self.assertEqual(observed_length, known_length)


class TestTigerNetEmpirical(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == "__main__":
    unittest.main()

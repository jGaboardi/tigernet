"""Testing for tigernet.py
"""

import tigernet
import geopandas
import numpy
from libpysal import cg

import unittest


class TestTigerNetNoData(unittest.TestCase):
    def setUp(self):
        pass

    def test_no_segmdata(self):
        with self.assertRaises(ValueError):
            tigernet.TigerNet(segmdata=None)


class TestTigerNetLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.TigerNet(segmdata=self.lattice)

    def test_lattice_network_segmdata(self):
        segmdata = self.lattice_network.segmdata

        known_segments = 4
        observed_segments = segmdata.shape[0]
        self.assertEqual(observed_segments, known_segments)

        known_length = 18.0
        observed_length = segmdata.length.sum()
        self.assertEqual(observed_length, known_length)

    def test_lattice_network_segmdata_ids(self):
        segmdata = self.lattice_network.segmdata

        known_ids = [0, 1, 2, 3]
        observed_ids = list(segmdata["SegID"])
        self.assertEqual(observed_ids, known_ids)

    def test_lattice_network_segm2xyid(self):
        known_xyids = [3, ["x4.5y4.5", "x9.0y4.5"]]
        observed_xyids = self.lattice_network.segm2xyid[-1]
        self.assertEqual(observed_xyids, known_xyids)


class TestTigerNetEmpirical(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == "__main__":
    unittest.main()

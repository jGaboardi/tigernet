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
            tigernet.TigerNet(s_data=None)


class TestTigerNetLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.TigerNet(s_data=self.lattice)

    def test_lattice_network_sdata(self):
        sdata = self.lattice_network.s_data

        known_segments = 4
        observed_segments = sdata.shape[0]
        self.assertEqual(observed_segments, known_segments)

        known_length = 18.0
        observed_length = sdata.length.sum()
        self.assertEqual(observed_length, known_length)

    def test_lattice_network_ndata(self):
        ndata = self.lattice_network.n_data

        known_nodes = 5
        observed_nodes = ndata.shape[0]
        self.assertEqual(observed_nodes, known_nodes)

        known_bounds = [0.0, 0.0, 9.0, 9.0]
        observed_bounds = list(ndata.total_bounds)
        self.assertEqual(observed_bounds, known_bounds)

    def test_lattice_network_sdata_ids(self):
        sdata = self.lattice_network.s_data

        known_ids = [0, 1, 2, 3]
        observed_ids = list(sdata["SegID"])
        self.assertEqual(observed_ids, known_ids)

    def test_lattice_network_ndata_ids(self):
        ndata = self.lattice_network.n_data

        known_ids = [0, 1, 2, 3, 4]
        observed_ids = list(ndata["NodeID"])
        self.assertEqual(observed_ids, known_ids)

    def test_lattice_network_segm2xyid(self):
        known_xyid = [3, ["x4.5y4.5", "x9.0y4.5"]]
        observed_xyid = self.lattice_network.segm2xyid[-1]
        self.assertEqual(observed_xyid, known_xyid)

    def test_lattice_network_node2xyid(self):
        known_xyid = [4, ["x9.0y4.5"]]
        observed_xyid = self.lattice_network.node2xyid[-1]
        self.assertEqual(observed_xyid, known_xyid)


class TestTigerNetEmpirical(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == "__main__":
    unittest.main()

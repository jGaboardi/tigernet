"""Testing for tigernet.py
"""

import tigernet

import unittest


class TestTigerNetBuildLattice1x1(unittest.TestCase):
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


class TestTigerNetTopologyLattice1x1(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.TigerNet(s_data=self.lattice)

    def test_lattice_network_segm2node(self):
        known_segm2node = [[0, [0, 1]], [1, [1, 2]], [2, [1, 3]], [3, [1, 4]]]
        observed_segm2node = self.lattice_network.segm2node
        self.assertEqual(observed_segm2node, known_segm2node)

    def test_lattice_network_node2segm(self):
        known_node2segm = [[0, [0]], [1, [0, 1, 2, 3]], [2, [1]], [3, [2]], [4, [3]]]
        observed_node2segm = self.lattice_network.node2segm
        self.assertEqual(observed_node2segm, known_node2segm)

    def test_lattice_network_segm2segm(self):
        known_segm2segm = [
            [0, [1, 2, 3]],
            [1, [0, 2, 3]],
            [2, [0, 1, 3]],
            [3, [0, 1, 2]],
        ]
        observed_segm2segm = self.lattice_network.segm2segm
        self.assertEqual(observed_segm2segm, known_segm2segm)

    def test_lattice_network_node2node(self):
        known_node2node = [[0, [1]], [1, [0, 2, 3, 4]], [2, [1]], [3, [1]], [4, [1]]]
        observed_node2node = self.lattice_network.node2node
        self.assertEqual(observed_node2node, known_node2node)


class TestTigerNetBuildEmpirical(unittest.TestCase):
    def setUp(self):
        pass


class TestTigerNetTopologyEmpirical(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == "__main__":
    unittest.main()

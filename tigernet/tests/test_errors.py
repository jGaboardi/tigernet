"""Testing for tigernet.py
"""

import tigernet
from .. import utils

import copy
import unittest


class TestTigerNetErrors(unittest.TestCase):
    def setUp(self):
        pass

    def test_no_segmdata(self):
        with self.assertRaises(ValueError):
            tigernet.TigerNet(s_data=None)


class TestUtilsErrors(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.TigerNet(s_data=self.lattice)

    def test_get_neighbors(self):
        with self.assertRaises(ValueError):
            utils.get_neighbors(None, None, astype=None)

        with self.assertRaises(TypeError):
            utils.get_neighbors(None, None, astype=str)

    def test_assert_2_neighs(self):
        _net = copy.deepcopy(self.lattice_network)
        _net.segm2node.append([999, [888, 777, 666]])
        with self.assertRaises(AssertionError):
            tigernet.utils.assert_2_neighs(_net)


if __name__ == "__main__":
    unittest.main()

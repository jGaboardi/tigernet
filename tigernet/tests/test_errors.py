"""Testing for tigernet.py
"""

import tigernet
from .. import utils

import copy
import unittest


class TestNeworkErrors(unittest.TestCase):
    def setUp(self):
        pass

    def test_no_segmdata(self):
        with self.assertRaises(ValueError):
            tigernet.Network(s_data=None)


class TestStatsErrors(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)

    def test_bad_alpha(self):
        with self.assertRaises(AttributeError):
            lattice_network = tigernet.Network(s_data=self.lattice.copy())
            lattice_network.calc_net_stats(conn_stat="alpha")

    def test_bad_alpha(self):
        with self.assertRaises(ValueError):
            lattice_network = tigernet.Network(s_data=self.lattice.copy())
            lattice_network.calc_net_stats(conn_stat="omega")


class TestUtilsErrors(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.Network(s_data=self.lattice)

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

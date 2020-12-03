"""Testing errors in tigernet.py
"""


import copy
import geopandas
import unittest

import tigernet
from .. import utils
from .network_objects import network_lattice_1x1_no_args
from .network_objects import network_lattice_1x1_geomelem
from .network_objects import network_empirical_lcc


class TestObservationsErrors(unittest.TestCase):
    def test_no_segm2geom(self):
        network = copy.deepcopy(network_lattice_1x1_no_args)
        with self.assertRaises(AttributeError):
            tigernet.Observations(network, None, None)

    def test_bad_snap_to(self):
        network = copy.deepcopy(network_lattice_1x1_geomelem)
        with self.assertRaises(ValueError):
            tigernet.Observations(network, None, None, snap_to="network")


class TestCostMatrixErrors(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_lcc)

    def test_non_sequential_ids(self):
        with self.assertRaises(IndexError):
            self.network.cost_matrix()


class TestStatsErrors(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_lattice_1x1_no_args)

    def test_bad_alpha(self):
        with self.assertRaises(AttributeError):
            self.network.calc_net_stats(conn_stat="alpha")

    def test_bad_stat(self):
        with self.assertRaises(ValueError):
            self.network.calc_net_stats(conn_stat="omega")

    def test_no_ccs_for_alpa(self):
        with self.assertRaises(AttributeError):
            self.network.calc_net_stats(conn_stat="all")

    def test_no_circuity(self):
        with self.assertWarns(UserWarning):
            self.network.calc_net_stats()


class TestUtilsErrors(unittest.TestCase):
    def test_bad_xwalk_column(self):
        with self.assertRaises(ValueError):
            tigernet.utils.xwalk(geopandas.GeoDataFrame(), c1="bad1", c2="bad2")

    def test_assert_2_neighs(self):
        _net = copy.deepcopy(network_lattice_1x1_no_args)
        _net.segm2node[999] = [888, 777, 666]
        with self.assertRaises(AssertionError):
            tigernet.utils.assert_2_neighs(_net)

    def test_bad_branch_or_leaf(self):
        _net = copy.deepcopy(network_lattice_1x1_no_args)
        with self.assertRaises(ValueError):
            tigernet.utils.branch_or_leaf(_net, geom_type="rhombus")


if __name__ == "__main__":
    unittest.main()

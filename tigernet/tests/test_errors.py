"""Testing errors in tigernet.py
"""


import copy
import geopandas
from shapely.geometry import Point
import unittest

import tigernet
from .. import utils
from .network_objects import network_lattice_1x1_no_args
from .network_objects import network_lattice_1x1_geomelem
from .network_objects import network_empirical_lcc


class TestCostMatrixErrors(unittest.TestCase):
    def setUp(self):
        self.network = copy.deepcopy(network_empirical_lcc)

    def test_non_sequential_ids(self):
        with self.assertRaises(IndexError):
            self.network.cost_matrix()


class TestDataGenerationErrors(unittest.TestCase):
    def setUp(self):
        self.dset = "Edges_Leon_FL_2010"

    def test_read_data_bbox_bad_str(self):
        with self.assertRaises(ValueError):
            tigernet.testing_data(self.dset, bbox="bad_description")

    def test_read_data_bbox_not_iterable(self):
        with self.assertRaises(ValueError):
            tigernet.testing_data(self.dset, bbox=2020)

    def test_read_data_bbox_5_coords(self):
        with self.assertRaises(ValueError):
            tigernet.testing_data(self.dset, bbox=[1.0, 1.0, 2.0, 2.0, 2.0])

    def test_read_data_bbox_4_int_coords(self):
        with self.assertRaises(ValueError):
            tigernet.testing_data(self.dset, bbox=[1, 1, 2, 2])

    def test_read_data_bbox_5_int_coords(self):
        with self.assertRaises(ValueError):
            tigernet.testing_data(self.dset, bbox=[1, 1, 2, 2, 2])


class TestInfoErrors(unittest.TestCase):
    def test_get_discard_segms_bad_year(self):
        with self.assertRaises(KeyError):
            tigernet.get_discard_segms("1500", "12", "073")

    def test_get_discard_segms_no_info(self):
        with self.assertRaises(KeyError):
            tigernet.get_discard_segms("2000", "12", "073")


class TestObservationsErrors(unittest.TestCase):
    def test_no_segm2geom(self):
        network = copy.deepcopy(network_lattice_1x1_no_args)
        with self.assertRaises(AttributeError):
            tigernet.Observations(network, None, None)

    def test_bad_snap_to(self):
        network = copy.deepcopy(network_lattice_1x1_geomelem)
        with self.assertRaises(ValueError):
            tigernet.Observations(network, None, None, snap_to="network")


class TestObs2ObsErrors(unittest.TestCase):
    def test_no_cost_matrix(self):
        network = copy.deepcopy(network_lattice_1x1_geomelem)
        pts = [Point(1, 1), Point(3, 1), Point(1, 3), Point(3, 3)]
        od_data = {"obs_id": ["a", "b", "c", "d"]}
        obs = geopandas.GeoDataFrame(od_data, geometry=pts)
        args = network, obs.copy()
        kwargs = {"df_name": "obs1", "df_key": "obs_id"}
        observations = tigernet.Observations(*args, **kwargs)
        with self.assertRaises(AttributeError):
            tigernet.obs2obs_cost_matrix(observations, network)


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

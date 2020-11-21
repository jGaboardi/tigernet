"""Testing for tigernet.py
"""

import tigernet
from .. import utils

import copy
import unittest
import geopandas


# get the roads shapefile as a GeoDataFrame
bbox = (-84.279, 30.480, -84.245, 30.505)
f = "zip://test_data/Edges_Leon_FL_2010.zip!Edges_Leon_FL_2010.shp"
gdf = geopandas.read_file(f, bbox=bbox)
gdf = gdf.to_crs("epsg:2779")

# filter out only roads
yes_roads = gdf["ROADFLG"] == "Y"
roads = gdf[yes_roads].copy()

# Tiger attributes primary and secondary
ATTR1, ATTR2 = "MTFCC", "TLID"

# segment welding and splitting stipulations --------------------------------------------
INTRST = "S1100"  # interstates mtfcc code
RAMP = "S1630"  # ramp mtfcc code
SERV_DR = "S1640"  # service drive mtfcc code
SPLIT_GRP = "FULLNAME"  # grouped by this variable
SPLIT_BY = [RAMP, SERV_DR]  # split interstates by ramps & service
SKIP_RESTR = True  # no weld retry if still MLS


class TestNeworkErrors(unittest.TestCase):
    def setUp(self):
        pass

    def test_no_segmdata(self):
        with self.assertRaises(ValueError):
            tigernet.Network(s_data=None)


class TestCostMatrixErrors(unittest.TestCase):
    def setUp(self):
        # set up the network instantiation parameters
        discard_segs = None
        kwargs = {"s_data": roads.copy(), "from_raw": True}
        attr_kws = {"attr1": ATTR1, "attr2": ATTR2}
        kwargs.update(attr_kws)
        comp_kws = {"record_components": True, "largest_component": True}
        kwargs.update(comp_kws)
        geom_kws = {"record_geom": True, "calc_len": True}
        kwargs.update(geom_kws)
        mtfcc_kws = {"discard_segs": discard_segs, "skip_restr": SKIP_RESTR}
        mtfcc_kws.update({"mtfcc_split": INTRST, "mtfcc_intrst": INTRST})
        mtfcc_kws.update({"mtfcc_split_grp": SPLIT_GRP, "mtfcc_ramp": RAMP})
        mtfcc_kws.update({"mtfcc_split_by": SPLIT_BY, "mtfcc_serv": SERV_DR})
        kwargs.update(mtfcc_kws)

        # create a network isntance
        self.network = tigernet.Network(**kwargs)

    def test_non_sequential_ids(self):
        with self.assertRaises(IndexError):
            self.network.cost_matrix()


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

    def test_no_circuity(self):
        with self.assertWarns(UserWarning):
            lattice_network = tigernet.Network(s_data=self.lattice.copy())
            lattice_network.calc_net_stats()


class TestUtilsErrors(unittest.TestCase):
    def setUp(self):
        self.lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.lattice_network = tigernet.Network(s_data=self.lattice)

    def test_bad_xwalk_column(self):
        with self.assertRaises(ValueError):
            tigernet.utils.xwalk(geopandas.GeoDataFrame(), c1="bad1", c2="bad2")

    def test_assert_2_neighs(self):
        _net = copy.deepcopy(self.lattice_network)
        _net.segm2node[999] = [888, 777, 666]
        with self.assertRaises(AssertionError):
            tigernet.utils.assert_2_neighs(_net)

    def test_bad_branch_or_leaf(self):
        _net = copy.deepcopy(self.lattice_network)
        with self.assertRaises(ValueError):
            tigernet.utils.branch_or_leaf(_net, geom_type="rhombus")


if __name__ == "__main__":
    unittest.main()

"""Testing for tigernet/utils.py
"""

import tigernet
from .. import utils

import copy
import numpy
import operator
import pandas
import unittest
from shapely.geometry import MultiLineString

from .network_objects import network_lattice_1x1_no_args


class TestUtilsAddLength(unittest.TestCase):
    def test_add_length(self):
        known_length = 18.0
        h1v1 = {"n_hori_lines": 1, "n_vert_lines": 1}
        gdf = tigernet.generate_lattice(**h1v1)
        gdf["length"] = gdf.geometry.length
        gdf = utils.add_length(gdf, len_col="length", geo_col="geometry")
        observed_length = gdf["length"].sum()
        self.assertEqual(observed_length, known_length)


class TestUtilsDropGeoms(unittest.TestCase):
    def test__drop_geoms(self):
        known_values = numpy.array([["c"], ["e"]])
        geom_df = pandas.DataFrame({"points": ["a", "b", "c", "d", "e"]})
        observed_values = utils._drop_geoms(geom_df, [[0, 1], [None, 3]]).values
        numpy.testing.assert_array_equal(observed_values, known_values)


class TestUtilsWeldingFuncs(unittest.TestCase):
    def test__weld_MultiLineString_1(self):
        known_weld_wkt = "LINESTRING (0 0, 0 1, 1 1)"
        mls = MultiLineString((((0, 0), (0, 1)), ((0, 1), (1, 1))))
        observed_weld_wkt = utils._weld_MultiLineString(mls).wkt
        self.assertEqual(observed_weld_wkt, known_weld_wkt)

    def test__weld_MultiLineString_2(self):
        known_weld_wkt = "LINESTRING (0 0, 0 1, 1 1)"
        mls = MultiLineString((((0, 0), (0, 0.99999999)), ((0, 1), (1, 1))))
        observed_weld_wkt = utils._weld_MultiLineString(mls).wkt
        self.assertEqual(observed_weld_wkt, known_weld_wkt)

    def test__weld_MultiLineString_3(self):
        known_weld_wkt = "LINESTRING (1 0, 1 1, 0 1, 0 0)"
        mls = MultiLineString(
            (((0, 1.00000001), (0, 0)), ((1, 1), (0, 1)), ((1, 0), (1, 1)))
        )
        observed_weld_wkt = utils._weld_MultiLineString(mls).wkt
        self.assertEqual(observed_weld_wkt, known_weld_wkt)


class TestUtilsFilterFuncs(unittest.TestCase):
    def setUp(self):
        self.df = pandas.DataFrame().from_dict(
            {"v1": ["x"] * 2 + ["y"] * 2 + ["z"] * 2, "v2": ["a"] * 3 + ["b"] * 3}
        )

    def test_record_filter_mval_in(self):
        known_values = numpy.array([["x", "a"], ["x", "a"], ["y", "a"], ["y", "b"]])
        kws = {"column": "v1", "mval": ["x", "y"], "oper": "in"}
        observed_values = utils.record_filter(self.df.copy(), **kws).values
        numpy.testing.assert_array_equal(observed_values, known_values)

    def test_record_filter_mval_out(self):
        known_values = numpy.array([["z", "b"], ["z", "b"]])
        kws = {"column": "v1", "mval": ["x", "y"], "oper": "out"}
        observed_values = utils.record_filter(self.df.copy(), **kws).values
        numpy.testing.assert_array_equal(observed_values, known_values)

    def test_record_filter_mval_index(self):
        known_values = numpy.array([["y", "b"], ["z", "b"], ["z", "b"]])
        kws = {"column": "index", "mval": [0, 1, 2], "oper": "out"}
        observed_values = utils.record_filter(self.df.copy(), **kws).values
        numpy.testing.assert_array_equal(observed_values, known_values)

    def test_record_filter_sval(self):
        known_values = numpy.array([["x", "a"], ["x", "a"], ["y", "a"]])
        kws = {"column": "v2", "sval": "a", "oper": operator.eq}
        observed_values = utils.record_filter(self.df.copy(), **kws).values
        numpy.testing.assert_array_equal(observed_values, known_values)


class TestUtilRsestrictionWelder(unittest.TestCase):
    def setUp(self):
        pass

    def test_restriction_welder_key_error_synth(self):
        known_return = None
        _net = copy.deepcopy(network_lattice_1x1_no_args)
        ATTR1, ATTR2, SPLIT_GRP, SKIP_RESTR = "MTFCC", "TLID", "FULLNAME", True
        INTRST, RAMP, SERV_DR = "S1100", "S1630", "S1640"
        SPLIT_BY = [RAMP, SERV_DR]
        _net.attr1, _net.attr2, _net.skip_restr = ATTR1, ATTR2, SKIP_RESTR
        _net.mtfcc_split, _net.mtfcc_intrst = INTRST, INTRST
        _net.mtfcc_split_grp, _net.mtfcc_ramp = SPLIT_GRP, RAMP
        _net.mtfcc_split_by, _net.mtfcc_serv = SPLIT_BY, SERV_DR
        observed_return = utils.restriction_welder(_net)
        self.assertEqual(observed_return, known_return)


if __name__ == "__main__":
    unittest.main()

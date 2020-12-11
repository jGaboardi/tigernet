"""Testing for tigernet/utils.py
"""

import tigernet
from .. import utils

import copy
import geopandas
import numpy
import operator
import pandas
import unittest
from shapely.geometry import LineString, MultiLineString

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


class TestUtilGetLargestCCSNoSmallKeys(unittest.TestCase):
    def test_get_largest_cc_no_small_keys(self):
        known_largest = {1: [0, 1, 2, 3, 4]}
        ccs = {0: [0, 1, 2, 3]}
        ccs.update(known_largest)
        observed_largest = utils.get_largest_cc(ccs, smallkeys=False)
        self.assertEqual(observed_largest, known_largest)


class TestUtilRingCorrection(unittest.TestCase):
    def test_ring_correction_no_correction(self):
        class SynthNetwork:
            def __init__(self):
                self.geo_col = "geometry"
                self.sid_name = "SegID"
                self.xyid = "xyID"

        known_ring_corrections = 0
        net = SynthNetwork()
        line = LineString(((0, 0), (1, 1)))
        ring = LineString(((2, 2), (3, 2), (2.5, 3), (2, 2)))
        gdf = geopandas.GeoDataFrame(geometry=[line, ring])
        gdf["ring"] = ["False", "True"]
        utils.ring_correction(net, gdf.copy())
        observed_ring_corrections = net.corrected_rings
        self.assertEqual(observed_ring_corrections, known_ring_corrections)


class TestUtilGetIntersectingGeoms(unittest.TestCase):
    def test_get_intersecting_geoms_2_dfs_wbool(self):
        class SynthNetwork:
            def __init__(self):
                self.geo_col = "geometry"

        known_len, known_type = 2, tuple
        net = SynthNetwork()
        line = LineString(((0, 0), (1, 1)))
        ring = LineString(((2, 2), (3, 2), (2.5, 3), (2, 2)))
        gdf1 = geopandas.GeoDataFrame(geometry=[line])
        gdf2 = geopandas.GeoDataFrame(geometry=[ring])
        kws = {"df1": gdf1, "geom1": 0, "df2": gdf2, "geom2": 0, "wbool": True}
        observed = utils.get_intersecting_geoms(net, **kws)
        observed_len, observed_type = len(observed), type(observed)
        self.assertEqual(observed_len, known_len)
        self.assertEqual(observed_type, known_type)

    def test_get_intersecting_geoms_2_dfs_xbool(self):
        class SynthNetwork:
            def __init__(self):
                self.geo_col = "geometry"

        known_len, known_type = 0, geopandas.geodataframe.GeoDataFrame
        net = SynthNetwork()
        line = LineString(((0, 0), (1, 1)))
        ring = LineString(((2, 2), (3, 2), (2.5, 3), (2, 2)))
        gdf1 = geopandas.GeoDataFrame(geometry=[line])
        gdf2 = geopandas.GeoDataFrame(geometry=[ring])
        kws = {"df1": gdf1, "geom1": 0, "df2": gdf2, "geom2": 0, "wbool": False}
        observed = utils.get_intersecting_geoms(net, **kws)
        observed_len, observed_type = len(observed), type(observed)
        self.assertEqual(observed_len, known_len)
        self.assertEqual(observed_type, known_type)


class TestUtilSplitLine(unittest.TestCase):
    def test_split_line_unaltered(self):
        known_unaltered = LineString(((0, 1), (1, 1)))
        tgeoms = [LineString(((0, 0), (0, 2)))] + [known_unaltered]
        tcase = geopandas.GeoDataFrame(geometry=tgeoms)
        idx, geom = 1, "geometry"
        loi = tcase.loc[idx, geom]
        observed_unaltered = utils._split_line(loi, idx, df=tcase, geo_col=geom)
        for eidx, xy in enumerate(known_unaltered.xy):
            for cidx, coord in enumerate(xy):
                self.assertEqual(observed_unaltered[0].xy[eidx][cidx], coord)


if __name__ == "__main__":
    unittest.main()

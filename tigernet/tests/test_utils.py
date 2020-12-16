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
    def setUp(self):
        self.g = "geometry"
        # Case 1: unaltered / Case 1: split
        self.line1, self.line1_idx = LineString(((10, 11), (11, 11))), 0
        self.line2, self.line2_idx = LineString(((10, 10), (10, 12))), 1
        geoms = [self.line1, self.line2]
        rings = ["False", "False"]
        # Case 1: basic ring
        self.line3, self.line3_idx = LineString(((0, 1), (1, 1))), 2
        self.line4 = LineString(((1, 0.5), (2, 0.5), (2, 2), (1, 2), (1, 0.5)))
        self.line4_idx = 3
        geoms += [self.line3, self.line4]
        rings += ["False", "True"]
        # Case 2: horseshoe
        self.line5 = LineString(((0.25, 1), (0.25, 1.5), (0.75, 1.5), (0.75, 1)))
        self.line5_idx = 4
        geoms += [self.line5]
        rings += ["False"]
        # Case 2: s-crossover
        # ---> # line3, line5
        # Case 3
        self.line6, self.line6_idx = LineString(((2.5, 0.5), (2.5, 1.5))), 5
        self.line7, self.line7_idx = LineString(((2.5, 1), (2.5, 2))), 6
        geoms += [self.line6, self.line7]
        rings += ["False", "False"]
        # Case 4
        self.line8, self.line8_idx = LineString(((0, 3), (3, 3))), 7
        self.line9 = LineString(
            ((0.5, 3), (1, 3), (1, 3.5), (2, 3.5), (2, 3), (2.5, 3))
        )
        self.line9_idx = 8
        geoms += [self.line8, self.line9]
        rings += ["False", "False"]
        # Case 5: complex intersection - line & point
        self.line10, self.line10_idx = LineString(((5, 3), (8, 3))), 9
        self.line11 = LineString(((5.5, 3), (6, 3), (6, 3.5), (7, 3.5), (7, 3)))
        self.line11_idx = 10
        geoms += [self.line10, self.line11]
        rings += ["False", "False"]
        # Case 5: complex intersection - line & line
        self.line12, self.line12_idx = LineString(((0, 6), (3, 6))), 11
        self.line13, self.line13_idx = LineString(((0, 7), (3, 7))), 12
        self.line14 = LineString(
            ((0.5, 6), (1, 6), (1, 6.5), (2, 6.5), (2, 7), (2.5, 7))
        )
        self.line14_idx = 13
        geoms += [self.line12, self.line13, self.line14]
        rings += ["False", "False", "False"]
        # Case 5: complex intersection - line & line & point
        self.line15, self.line15_idx = LineString(((5, 6), (8, 6))), 14
        self.line16, self.line16_idx = LineString(((5, 7), (8, 7))), 15
        coords17 = [(5.5, 6), (6, 6), (6, 6.5), (7, 6.5)]
        coords17 += [(7, 7), (7.5, 7), (7.5, 6), (7, 6)]
        self.line17 = LineString(coords17)
        self.line17_idx = 16
        geoms += [self.line15, self.line16, self.line17]
        rings += ["False", "False", "False"]
        # Case 5: complex intersection - line & line & line (camel humps)
        self.line18, self.line18_idx = LineString(((0, 9), (3, 9))), 17
        self.line19 = LineString(
            (
                (0.25, 9),
                (0.5, 9),
                (0.5, 9.5),
                (1, 9.5),
                (1, 9),
                (1.5, 9),
                (1.5, 9.5),
                (2, 9.5),
                (2, 9),
                (2.25, 9),
                (2.25, 8.5),
                (2.75, 8.5),
                (2.75, 9.25),
            )
        )
        self.line19_idx = 17
        geoms += [self.line18, self.line19]
        rings += ["False", "False"]

        self.gdf = geopandas.GeoDataFrame(geometry=geoms)
        self.gdf["ring"] = rings

    def test_split_line_case1_unaltered_line(self):
        known_unaltered_line = self.line1
        idx = self.line1_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_unaltered_line = utils._split_line(*args, **kwargs)
        for eidx, xy in enumerate(known_unaltered_line.xy):
            for cidx, coord in enumerate(xy):
                self.assertEqual(observed_unaltered_line[0].xy[eidx][cidx], coord)

    def test_split_line_case1_basic_ring(self):
        known_basic_ring_split = self.line4
        idx = self.line4_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_basic_ring_split = utils._split_line(*args, ring_road=True, **kwargs)
        for eidx, xy in enumerate(known_basic_ring_split.xy):
            for cidx, coord in enumerate(xy):
                self.assertEqual(observed_basic_ring_split[0].xy[eidx][cidx], coord)

    def test_split_line_case1_basic_split(self):
        known_basic_split = [
            [[10.0, 10.0], [10.0, 11.0]],
            [[10.0, 10.0], [11.0, 12.0]],
        ]
        idx = self.line2_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_basic_split = utils._split_line(*args, **kwargs)
        for lidx, xy in enumerate(known_basic_split):
            for cidx, coord in enumerate(xy):
                self.assertEqual(list(observed_basic_split[lidx].xy[cidx]), coord)

    def test_split_line_case2_horseshoe(self):
        known_horseshoe_split = [[[0.25, 0.25, 0.75, 0.75], [1.0, 1.5, 1.5, 1.0]]]
        idx = self.line5_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_horseshoe_split = utils._split_line(*args, **kwargs)
        for lidx, xy in enumerate(known_horseshoe_split):
            for cidx, coord in enumerate(xy):
                self.assertEqual(list(observed_horseshoe_split[lidx].xy[cidx]), coord)

    def test_split_line_case2_standard_multipoint(self):
        known_smp_split = [
            [[0.0, 0.25], [1.0, 1.0]],
            [[0.25, 0.75], [1.0, 1.0]],
            [[0.75, 1.0], [1.0, 1.0]],
        ]
        idx = self.line3_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_smp_split = utils._split_line(*args, **kwargs)
        for lidx, xy in enumerate(known_smp_split):
            for cidx, coord in enumerate(xy):
                self.assertEqual(list(observed_smp_split[lidx].xy[cidx]), coord)

    def test_split_line_case3(self):
        known_case3 = [[[2.5, 2.5], [0.5, 1.0]], [[2.5, 2.5], [1.0, 1.5]]]
        idx = self.line6_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_case3 = tigernet.utils._split_line(*args, **kwargs)
        for lidx, xy in enumerate(known_case3):
            for cidx, coord in enumerate(xy):
                self.assertEqual(list(observed_case3[lidx].xy[cidx]), coord)

    def test_split_line_case4(self):
        known_case4 = [
            [[0.0, 1.0], [3.0, 3.0]],
            [[1.0, 2.0], [3.0, 3.0]],
            [[2.0, 3.0], [3.0, 3.0]],
        ]
        idx = self.line8_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_case4 = tigernet.utils._split_line(*args, **kwargs)
        for lidx, xy in enumerate(known_case4):
            for cidx, coord in enumerate(xy):
                self.assertEqual(list(observed_case4[lidx].xy[cidx]), coord)

    def test_split_line_case51(self):
        known_case51 = [
            [[5.0, 5.5], [3.0, 3.0]],
            [[5.5, 6.0], [3.0, 3.0]],
            [[6.0, 7.0], [3.0, 3.0]],
            [[7.0, 8.0], [3.0, 3.0]],
        ]
        idx = self.line10_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_case51 = tigernet.utils._split_line(*args, **kwargs)
        for lidx, xy in enumerate(known_case51):
            for cidx, coord in enumerate(xy):
                self.assertEqual(list(observed_case51[lidx].xy[cidx]), coord)

    def test_split_line_case52(self):
        known_case52 = [
            [[0.5, 1.0, 1.0], [6.0, 6.0, 6.0]],
            [[1.0, 1.0, 2.0, 2.0, 2.0], [6.0, 6.5, 6.5, 7.0, 7.0]],
            [[2.0, 2.5], [7.0, 7.0]],
        ]
        idx = self.line14_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_case52 = tigernet.utils._split_line(*args, **kwargs)
        for lidx, xy in enumerate(known_case52):
            for cidx, coord in enumerate(xy):
                self.assertEqual(list(observed_case52[lidx].xy[cidx]), coord)

    def test_split_line_case53(self):
        known_case53 = [
            [[5.5, 6.0, 6.0], [6.0, 6.0, 6.0]],
            [[6.0, 6.0, 7.0, 7.0, 7.5, 7.5, 7.0], [6.0, 6.5, 6.5, 7.0, 7.0, 6.0, 6.0]],
        ]
        idx = self.line17_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_case53 = tigernet.utils._split_line(*args, **kwargs)
        for lidx, xy in enumerate(known_case53):
            for cidx, coord in enumerate(xy):
                self.assertEqual(list(observed_case53[lidx].xy[cidx]), coord)

    def test_split_line_case54(self):
        known_case54 = [
            [[0.0, 0.25], [9.0, 9.0]],
            [[0.25, 0.5], [9.0, 9.0]],
            [[0.5, 1.0], [9.0, 9.0]],
            [[1.0, 1.5], [9.0, 9.0]],
            [[1.5, 2.0], [9.0, 9.0]],
            [[2.0, 2.25], [9.0, 9.0]],
            [[2.25, 2.75], [9.0, 9.0]],
            [[2.75, 3.0], [9.0, 9.0]],
        ]
        idx = self.line19_idx
        loi = self.gdf.loc[idx, self.g]
        args, kwargs = (loi, idx), {"df": self.gdf, "geo_col": self.g}
        observed_case54 = tigernet.utils._split_line(*args, **kwargs)
        for lidx, xy in enumerate(known_case54):
            for cidx, coord in enumerate(xy):
                self.assertEqual(list(observed_case54[lidx].xy[cidx]), coord)


if __name__ == "__main__":
    unittest.main()

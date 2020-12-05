"""Testing for tigernet/utils.py
"""

from .. import utils

import numpy
import operator
import pandas
import unittest
from shapely.geometry import MultiLineString


class TestUtilsWeldingFuncs(unittest.TestCase):
    def setUp(self):
        pass

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


if __name__ == "__main__":
    unittest.main()

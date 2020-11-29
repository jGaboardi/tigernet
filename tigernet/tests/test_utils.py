"""Testing for tigernet/utils.py
"""

from .. import utils

import unittest
from shapely.geometry import MultiLineString


class TestUtils(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

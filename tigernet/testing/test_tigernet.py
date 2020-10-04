"""Testing for tigernet.py
"""

import tigernet
import geopandas
import numpy
from libpysal import cg

import unittest


class TestTigerNet(unittest.TestCase):
    """
    """

    def setUp(self):

        self.filler = "1"

    def test_filler(self):

        observed = self.filler
        known = "1"

        self.assertEqual(observed, known)


if __name__ == "__main__":
    unittest.main()

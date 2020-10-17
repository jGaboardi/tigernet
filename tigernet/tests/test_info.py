"""Testing for tigernet.py
"""

import tigernet
import unittest


class TestInformationRetrieval(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_discard_mtfcc_by_desc(self):
        observed_discard_mtfcc_types = tigernet.get_discard_mtfcc_by_desc()
        walkways = "Walkway/Pedestrian Trail"
        self.assertEqual(observed_discard_mtfcc_types[-2], walkways)

    def test_get_mtfcc_types(self):
        mtfcc_types = tigernet.get_mtfcc_types()
        observed_mtfcc_keys = list(mtfcc_types.keys())
        known_mtfcc_keys = [
            "S1100",
            "S1200",
            "S1400",
            "S1500",
            "S1630",
            "S1640",
            "S1710",
            "S1720",
            "S1730",
            "S1740",
            "S1750",
            "S1780",
            "S1820",
            "S1830",
            "S2000",
        ]
        self.assertEqual(observed_mtfcc_keys, known_mtfcc_keys)


if __name__ == "__main__":
    unittest.main()

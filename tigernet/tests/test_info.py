"""Testing information retrieval in tigernet.py
"""

import tigernet
import unittest


class TestInformationRetrieval(unittest.TestCase):
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

    def test_discard_segments(self):
        # 1-------------------------------------------------------------------------
        known_shape_after_discard = 47
        kws = {"bbox": "discard", "direc": "test_data"}
        gdf = tigernet.testing_data("Edges_Leon_FL_2010", **kws)
        discard = tigernet.get_discard_segms("2010", "12", "073")
        observed_shape_after_discard = gdf[~gdf["TLID"].isin(discard)].shape[0]
        self.assertEqual(observed_shape_after_discard, known_shape_after_discard)

        # 2-------------------------------------------------------------------------
        known_shape_after_discard = 23
        # filter out only roads
        gdf = tigernet.testing_data("Edges_Leon_FL_2010", **kws)
        yes_roads = gdf["MTFCC"].str.startswith("S")
        roads = gdf[yes_roads].copy()
        # Tiger attributes primary and secondary
        ATTR1, ATTR2 = "MTFCC", "TLID"
        # segment welding and splitting stipulations
        INTRST = "S1100"  # interstates mtfcc code
        RAMP = "S1630"  # ramp mtfcc code
        SERV_DR = "S1640"  # service drive mtfcc code
        SPLIT_GRP = "FULLNAME"  # grouped by this variable
        SPLIT_BY = [RAMP, SERV_DR]  # split interstates by ramps & service
        SKIP_RESTR = True  # no weld retry if still MLS
        kwargs = {"from_raw": True, "attr1": ATTR1, "attr2": ATTR2}
        comp_kws = {"record_components": True, "largest_component": True}
        kwargs.update(comp_kws)
        geom_kws = {"record_geom": True, "calc_len": True}
        kwargs.update(geom_kws)
        mtfcc_kws = {"skip_restr": SKIP_RESTR}
        mtfcc_kws.update({"mtfcc_split": INTRST, "mtfcc_intrst": INTRST})
        mtfcc_kws.update({"mtfcc_split_grp": SPLIT_GRP, "mtfcc_ramp": RAMP})
        mtfcc_kws.update({"mtfcc_split_by": SPLIT_BY, "mtfcc_serv": SERV_DR})
        kwargs.update(mtfcc_kws)
        network = tigernet.Network(roads, discard_segs=("2010", "12", "073"), **kwargs)
        network.simplify_network(
            record_components=True,
            record_geom=True,
            largest_component=False,
            def_graph_elems=True,
            inplace=True,
        )
        observed_shape_after_discard = network.s_data.shape[0]
        self.assertEqual(observed_shape_after_discard, known_shape_after_discard)


if __name__ == "__main__":
    unittest.main()

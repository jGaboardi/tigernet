"""Testing for tigernet.py
"""

import tigernet
import unittest
import numpy
import geopandas


# get the roads shapefile as a GeoDataFrame
bbox = (-84.279, 30.480, -84.245, 30.505)
f = "zip://test_data/Edges_Leon_FL_2010.zip!Edges_Leon_FL_2010.shp"
gdf = geopandas.read_file(f, bbox=bbox)
gdf = gdf.to_crs("epsg:2779")

# filter out only roads
# yes_roads = gdf["ROADFLG"] == "Y"
# roads = gdf[yes_roads].copy()
yes_roads = gdf["MTFCC"].str.startswith("S")
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


class TestNetworkDataGeneration(unittest.TestCase):
    def setUp(self):
        b1133, nh, nv = [-1, -1, 3, 3], 3, 4
        self._34_kws = {"bounds": b1133, "n_hori_lines": nh, "n_vert_lines": nv}

    def test_generate_sine_lines(self):
        sine = tigernet.generate_sine_lines()
        observed_length = sine.loc[(sine["SegID"] == 0), "geometry"].squeeze().length
        known_length = 11.774626215766602
        self.assertAlmostEqual(observed_length, known_length, 10)

    def test_generate_lattice_2x2_xbox_0099(self):
        lat = tigernet.generate_lattice()
        observed_length = lat.length.sum()
        known_length = 36.0
        self.assertEqual(observed_length, known_length)

    def test_generate_lattice_2x2_wbox_0099(self):
        lat = tigernet.generate_lattice(wbox=True)
        observed_length = lat.length.sum()
        known_length = 72.0
        self.assertEqual(observed_length, known_length)

    def test_generate_lattice_3x4_xbox_neg1neg133(self):
        lat = tigernet.generate_lattice(**self._34_kws)
        observed_length = lat.length.sum()
        known_length = 27.8
        self.assertEqual(observed_length, known_length)

    def test_generate_lattice_2x2_wbox_neg1neg133(self):
        lat = tigernet.generate_lattice(wbox=True, **self._34_kws)
        observed_length = lat.length.sum()
        known_length = 44.2
        self.assertEqual(observed_length, known_length)


class TestObservationDataGenerationSynthetic(unittest.TestCase):
    def setUp(self):
        lat = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        self.network = tigernet.Network(s_data=lat)

    def test_generate_observations_totalbounds(self):
        n_obs = 5
        obs = tigernet.generate_obs(n_obs, self.network.s_data)
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (4.939321535345923, 6.436704297351775),
                (5.4248703846447945, 4.903948646972072),
                (3.8128931940501425, 5.813047017599905),
                (3.9382849013642325, 8.025957007038718),
                (8.672964844509263, 3.4509736694319995),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_generate_observations_inbuffer(self):
        n_obs = 5
        obs = tigernet.generate_obs(n_obs, self.network.s_data, near_net=0.5)
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (4.939321535345923, 6.436704297351775),
                (5.4248703846447945, 4.903948646972072),
                (7.125525342743981, 4.76005427777614),
                (4.153314260276387, 7.024762586578099),
                (4.696634895750645, 3.731957459914712),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)


class TestObservationDataGenerationEmpirical(unittest.TestCase):
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

        # create a network instance
        self.network = tigernet.Network(**kwargs)

        # simplify network
        kws = {"record_components": True, "record_geom": True, "def_graph_elems": True}
        self.network.simplify_network(inplace=True, **kws)

    def test_generate_observations_totalbounds(self):

        n_obs = 500
        obs = tigernet.generate_obs(n_obs, self.network.s_data)
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (622974.1796832045, 166162.55760926675),
                (623169.2985719838, 165632.6814708112),
                (622521.5219401979, 165946.95826632337),
                (622571.9108776418, 166711.9648736473),
                (624474.552580049, 165130.38564160923),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])[:5]
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_generate_observations_inbuffer(self):
        n_obs = 500
        obs = tigernet.generate_obs(n_obs, self.network.s_data, near_net=30)
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (622571.9108776418, 166711.9648736473),
                (624474.552580049, 165130.38564160923),
                (623803.6385443554, 166644.26001242414),
                (622876.6555154349, 165227.52219813256),
                (623400.7775349629, 166023.94388077687),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])[:5]
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)

    def test_generate_observations_inbuffer_restrict(self):
        n_obs = 500
        obs = tigernet.generate_obs(
            n_obs, self.network.s_data, restrict=("MTFCC", ["S1100"]), near_net=30
        )
        known_n_obs = n_obs
        observed_n_obs = obs.shape[0]
        self.assertEqual(observed_n_obs, known_n_obs)

        known_coords = numpy.array(
            [
                (622571.9108776418, 166711.9648736473),
                (624474.552580049, 165130.38564160923),
                (623803.6385443554, 166644.26001242414),
                (622876.6555154349, 165227.52219813256),
                (623400.7775349629, 166023.94388077687),
            ]
        )
        observed_coords = numpy.array([(p.x, p.y) for p in obs.geometry])[:5]
        numpy.testing.assert_array_almost_equal(observed_coords, known_coords)


if __name__ == "__main__":
    unittest.main()

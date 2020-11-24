"""KDTree data testing.
"""

import tigernet
import unittest
import geopandas
import numpy


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


class TestKDTreeLattice1x1(unittest.TestCase):
    def setUp(self):
        # instantiate network
        lattice = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        network = tigernet.Network(s_data=lattice, record_geom=True)

        # build kd tree
        self.net_nodes_kdtree = network.nodes_kdtree()

        # build kd tree -- from coordinates only
        self.net_nodes_kdtree_coords_only = network.nodes_kdtree(only_coords=True)

    def test_net_nodes_kdtree_data(self):
        known_kdtree_data = numpy.array(
            [[4.5, 0.0], [4.5, 4.5], [4.5, 9.0], [0.0, 4.5], [9.0, 4.5]]
        )
        observed_kdtree_data = self.net_nodes_kdtree.data
        numpy.testing.assert_array_equal(observed_kdtree_data, known_kdtree_data)

    def test_net_nodes_kdtree_query_dists(self):
        known_kdtree_query_dists = numpy.array([0.0, 4.5, 6.36396103])
        observed_kdtree_query_dists = numpy.array(
            self.net_nodes_kdtree.query([4.5, 0.0], k=3)
        )[0, :]
        numpy.testing.assert_array_almost_equal(
            observed_kdtree_query_dists, known_kdtree_query_dists
        )

    def test_net_nodes_kdtree_query_indices(self):
        known_kdtree_query_indices = numpy.array([0.0, 1.0, 3.0])
        observed_kdtree_query_indices = numpy.array(
            self.net_nodes_kdtree.query([4.5, 0.0], k=3)
        )[1, :]
        numpy.testing.assert_array_equal(
            observed_kdtree_query_indices, known_kdtree_query_indices
        )

    def test_net_nodes_kdtree_data_only_coords(self):
        known_kdtree_data = numpy.array(
            [[4.5, 0.0], [4.5, 4.5], [4.5, 9.0], [0.0, 4.5], [9.0, 4.5]]
        )
        observed_kdtree_data = self.net_nodes_kdtree.data
        numpy.testing.assert_array_equal(observed_kdtree_data, known_kdtree_data)

    def test_net_nodes_kdtree_query_dists_only_coords(self):
        known_kdtree_query_dists = numpy.array([0.0, 4.5, 6.36396103])
        observed_kdtree_query_dists = numpy.array(
            self.net_nodes_kdtree.query([4.5, 0.0], k=3)
        )[0, :]
        numpy.testing.assert_array_almost_equal(
            observed_kdtree_query_dists, known_kdtree_query_dists
        )

    def test_net_nodes_kdtree_query_indices_only_coords(self):
        known_kdtree_query_indices = numpy.array([0.0, 1.0, 3.0])
        observed_kdtree_query_indices = numpy.array(
            self.net_nodes_kdtree.query([4.5, 0.0], k=3)
        )[1, :]
        numpy.testing.assert_array_equal(
            observed_kdtree_query_indices, known_kdtree_query_indices
        )


class TestKDTreeEmpirical(unittest.TestCase):
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
        network = tigernet.Network(**kwargs)

        # simplify network
        kws = {"record_components": True, "record_geom": True, "def_graph_elems": True}
        network.simplify_network(inplace=True, **kws)

        # build kd tree
        self.net_nodes_kdtree = network.nodes_kdtree()

    def test_net_nodes_kdtree_data(self):
        known_kdtree_data = numpy.array(
            [
                [624484.3321678, 165959.33602943],
                [621472.15969302, 164456.34801777],
                [623580.79853327, 164531.84479675],
                [623531.61385791, 164455.13730139],
                [622213.77398256, 166384.29556895],
            ]
        )
        observed_kdtree_data = self.net_nodes_kdtree.data[-5:, :]
        numpy.testing.assert_array_almost_equal(observed_kdtree_data, known_kdtree_data)

    def test_net_nodes_kdtree_query_dists(self):
        known_kdtree_query_dists = numpy.array([0.0, 138.50270336, 369.19792667])
        observed_kdtree_query_dists = numpy.array(
            self.net_nodes_kdtree.query(self.net_nodes_kdtree.data[-1, :], k=3)
        )[0, :]
        numpy.testing.assert_array_almost_equal(
            observed_kdtree_query_dists, known_kdtree_query_dists
        )

    def test_net_nodes_kdtree_query_indices(self):
        known_kdtree_query_indices = numpy.array([285.0, 195.0, 196.0])
        observed_kdtree_query_indices = numpy.array(
            self.net_nodes_kdtree.query(self.net_nodes_kdtree.data[-1, :], k=3)
        )[1, :]
        numpy.testing.assert_array_equal(
            observed_kdtree_query_indices, known_kdtree_query_indices
        )


if __name__ == "__main__":
    unittest.main()

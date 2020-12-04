"""Network objects for testing.
"""

import copy
import tigernet


###############################################################################
############################### Synthetic networks ############################
###############################################################################
# --------------------------------------------------------------------------
# One 1x1 lattice network
#   used in:
#       - test_tigernet_synthetic.TestNetworkBuildLattice1x1
#       - test_tigernet_synthetic.TestNetworkTopologyLattice1x1
#       - test_data_generation.TestObservationDataGenerationSynthetic
#       - test_errors.TestObservationsErrors
#       - test_errors.TestUtilsErrors
#       - test_errors.TestStatsErrors
h1v1 = {"n_hori_lines": 1, "n_vert_lines": 1}
lattice = tigernet.generate_lattice(**h1v1)
network_lattice_1x1_no_args = tigernet.Network(lattice)

# --------------------------------------------------------------------------
# Two 1x1 lattices network (both components)
#   used in:
#       - test_tigernet_synthetic.TestNetworkComponentsLattice1x1
lattice1 = tigernet.generate_lattice(**h1v1)
lattice2 = tigernet.generate_lattice(bounds=[6, 6, 8, 8], **h1v1)
lattice = lattice1.append(lattice2)
lattice.reset_index(drop=True, inplace=True)
kws = {"record_components": True}
network_lattice_2x1x1_all = tigernet.Network(lattice, **kws)

# --------------------------------------------------------------------------
# Two 1x1 lattices network (both components) with a calculated cost matrix (and paths)
#   used in:
#       - test_cost_matrix.TestNetworkCostMatrixLattice1x1_2
network_lattice_2x1x1_wcm_attr = copy.deepcopy(network_lattice_2x1x1_all)
network_lattice_2x1x1_wcm_attr.cost_matrix()
network_lattice_2x1x1_wpaths_attr = copy.deepcopy(network_lattice_2x1x1_all)
network_lattice_2x1x1_wpaths_attr.cost_matrix(wpaths=True)
network_lattice_2x1x1_wcm_var = copy.deepcopy(network_lattice_2x1x1_all).cost_matrix(
    asattr=False
)
_, network_lattice_2x1x1_wpaths_var = copy.deepcopy(
    network_lattice_2x1x1_all
).cost_matrix(wpaths=True, asattr=False)

# --------------------------------------------------------------------------
# Two 1x1 lattices network (largest component)
#   used in:
#       - test_tigernet_synthetic.TestNetworkComponentsLattice1x1
kws.update({"largest_component": True})
network_lattice_2x1x1_largest = tigernet.Network(lattice, **kws)

# --------------------------------------------------------------------------
# One 1x1 lattice network (with recorded geometry and defined graph elements)
#   used in:
#       - test_tigernet_synthetic.TestNetworkAssociationsLattice1x1
#       - test_tigernet_synthetic.TestNetworkDefineGraphElementsLattice1x1
#       - test_kdtree.TestKDTreeLattice1x1
#       - test_errors.TestObservationsErrors
#       - test_observations_synthetic.TestSyntheticObservationsSegmentRandomLattice1x1
#       - test_observations_synthetic.TestSyntheticObservationsNodeRandomLattice1x1
#       - test_observations_synthetic.TestSyntheticObservationsSegmentRandomLattice1x1Restricted
#       - test_observations_synthetic.TestSyntheticObservationsNodeRandomLattice1x1Restricted
lattice = tigernet.generate_lattice(**h1v1)
kws = {"record_geom": True, "def_graph_elems": True}
network_lattice_1x1_geomelem = tigernet.Network(lattice, **kws)

# --------------------------------------------------------------------------
# # One 1x1 lattice network with a calculated cost matrix (and paths)
#   used in:
#       - test_cost_matrix.TestNetworkCostMatrixLattice1x1_1
network_lattice_1x1_wcm_attr = copy.deepcopy(network_lattice_1x1_no_args)
network_lattice_1x1_wcm_attr.cost_matrix()
network_lattice_1x1_wpaths_attr = copy.deepcopy(network_lattice_1x1_no_args)
network_lattice_1x1_wpaths_attr.cost_matrix(wpaths=True)
network_lattice_1x1_wcm_var = copy.deepcopy(network_lattice_1x1_no_args).cost_matrix(
    asattr=False
)
_, network_lattice_1x1_wpaths_var = copy.deepcopy(
    network_lattice_1x1_no_args
).cost_matrix(wpaths=True, asattr=False)

# --------------------------------------------------------------------------
# Copied and simplified inplace barb network
# (with recorded components, recorded geometry and defined graph elements)
#   used in:
#       - test_tigernet_synthetic.TestNetworkSimplifyBarb
barb = tigernet.generate_lattice(wbox=True, **h1v1)
barb = barb[~barb["SegID"].isin([1, 2, 5, 7, 9, 10])]
kws.update({"record_components": True})
network_barb = tigernet.Network(barb, **kws)
# copy
graph_barb = network_barb.simplify_network(**kws)
# inplace
network_barb.simplify_network(inplace=True, **kws)

# --------------------------------------------------------------------------
# Copied and simplified inplace barb network with a calculated cost matrix (and paths)
#   used in:
#       - test_cost_matrix.TestNetworkCostMatrixSimplifyBarb
graph_barb_wcm_copy_attr = copy.deepcopy(graph_barb)
graph_barb_wcm_copy_attr.cost_matrix()
graph_barb_wpaths_copy_attr = copy.deepcopy(graph_barb)
graph_barb_wpaths_copy_attr.cost_matrix(wpaths=True)
graph_barb_wcm_copy_var = copy.deepcopy(graph_barb).cost_matrix(asattr=False)
_, graph_barb_wpaths_copy_var = copy.deepcopy(graph_barb).cost_matrix(
    wpaths=True, asattr=False
)

network_barb_wcm_inplace_attr = copy.deepcopy(network_barb)
network_barb_wcm_inplace_attr.cost_matrix()
network_barb_wpaths_inplace_attr = copy.deepcopy(network_barb)
network_barb_wpaths_inplace_attr.cost_matrix(wpaths=True)
network_barb_wcm_inplace_var = copy.deepcopy(network_barb).cost_matrix(asattr=False)
_, network_barb_wpaths_inplace_var = copy.deepcopy(network_barb).cost_matrix(
    wpaths=True, asattr=False
)

# --------------------------------------------------------------------------
# One 1x1 lattice network with smaller bounds
#   used in:
#       - test_obs2obs_synthetic.TestSyntheticObservationsOrigToXXXXSegments
#       - test_obs2obs_synthetic.TestSyntheticObservationsOrigToXXXXNodes
#       - test_obs2obs_synthetic.TestSyntheticObservationsOrigToDestSegments
#       - test_obs2obs_synthetic.TestSyntheticObservationsOrigToDestNodes
h1v1 = {"n_hori_lines": 1, "n_vert_lines": 1}
lattice = tigernet.generate_lattice(bounds=[0, 0, 4, 4], **h1v1)
network_lattice_1x1_small = tigernet.Network(lattice, **kws)
network_lattice_1x1_small.cost_matrix()


###############################################################################
############################### Empirical networks ############################
###############################################################################


# get the roads shapefile as a GeoDataFrame
gdf = tigernet.testing_data("Edges_Leon_FL_2010")

# filter out only roads
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

# --------------------------------------------------------------------------
# Full empirical network (all args, all components)
#   used in:
#       - test_tigernet_empirical_gdf.TestNetworkComponentsEmpiricalGDF
discard_segs = None
kwargs = {"from_raw": True, "attr1": ATTR1, "attr2": ATTR2}
comp_kws = {"record_components": True}
kwargs.update(comp_kws)
geom_kws = {"record_geom": True, "calc_len": True}
kwargs.update(geom_kws)
mtfcc_kws = {"discard_segs": discard_segs, "skip_restr": SKIP_RESTR}
mtfcc_kws.update({"mtfcc_split": INTRST, "mtfcc_intrst": INTRST})
mtfcc_kws.update({"mtfcc_split_grp": SPLIT_GRP, "mtfcc_ramp": RAMP})
mtfcc_kws.update({"mtfcc_split_by": SPLIT_BY, "mtfcc_serv": SERV_DR})
kwargs.update(mtfcc_kws)
network_empirical_full = tigernet.Network(roads.copy(), **kwargs)

# --------------------------------------------------------------------------
# Full empirical network (all args, largest component)
#   used in:
#       - test_stats.TestNetworkEntropyEmpirical
#       - test_tigernet_empirical_gdf.TestNetworkBuildEmpiricalGDF
#       - test_tigernet_empirical_gdf.TestNetworkTopologyEmpiricalGDF
#       - test_tigernet_empirical_gdf.TestNetworkComponentsEmpiricalGDF
#       - test_erros.TestCostMatrixErrors
kwargs.update({"largest_component": True, "def_graph_elems": True})
network_empirical_lcc = tigernet.Network(roads.copy(), **kwargs)

# --------------------------------------------------------------------------
# Simplified empirical network (all args, largest component)
#   used in:
#       - test_data_generation.TestObservationDataGenerationSynthetic
#       - test_stats.TestNetworkConnectivityEmpirical
#       - test_tigernet_empirical_gdf.TestNetworkDistanceMetricsEmpiricalGDF
#       - test_tigernet_empirical_gdf.TestNetworkAssociationsEmpiricalGDF
#       - test_tigernet_empirical_gdf.TestNetworkDefineGraphElementsEmpiricalGDF
#       - test_tigernet_empirical_gdf.TestNetworkSimplifyEmpiricalGDF
#       - test_kdtree.TestKDTreeEmpirical
#       - test_observations_synthetic.TestSyntheticObservationsSegmentRandomEmpirical
#       - test_observations_synthetic.TestSyntheticObservationsNodeRandomEmpirical
network_empirical_simplified = copy.deepcopy(network_empirical_lcc)
kws = {"record_components": True, "record_geom": True, "def_graph_elems": True}
# copy
graph_empirical_simplified = network_empirical_simplified.simplify_network(**kws)
# inplace
network_empirical_simplified.simplify_network(inplace=True, **kws)

# --------------------------------------------------------------------------
# Simplified empirical network (all args, largest component) **WITH COST MATRIX**
#   used in:
#       - test_stats.TestNetworkStatsEmpirical
#       - test_stats.TestNetworkDistanceMetricsEmpiricalGDF
#       - test_cost_matrix.TestNetworkCostMatrixEmpircalGDF
#       - test_observations_empirical.TestEmpiricalObservationsOrigToXXXXSegments
#       - test_observations_empirical.TestEmpiricalObservationsOrigToXXXXNodes
network_empirical_simplified_wcm = copy.deepcopy(network_empirical_simplified)
network_empirical_simplified_wcm.cost_matrix(wpaths=True)

# --------------------------------------------------------------------------

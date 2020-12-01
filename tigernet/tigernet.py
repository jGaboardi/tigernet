"""
"""

from . import utils
from . import stats
from . import info

import copy
import warnings

from libpysal import cg


__author__ = "James D. Gaboardi <jgaboardi@gmail.com>"


class Network:
    def __init__(
        self,
        network_instance=None,
        s_data=None,
        n_data=None,
        from_raw=False,
        tiger_edges=True,
        sid_name="SegID",
        nid_name="NodeID",
        geo_col="geometry",
        xyid="xyid",
        len_col="length",
        tnid="TNID",
        tnidf="TNIDF",
        tnidt="TNIDT",
        attr1=None,
        attr2=None,
        mtfcc_types=None,
        mtfcc_discard=None,
        discard_segs=None,
        edge_subsets=None,
        mtfcc_split=None,
        mtfcc_intrst=None,
        mtfcc_ramp=None,
        mtfcc_serv=None,
        mtfcc_split_by=None,
        mtfcc_split_grp=None,
        skip_restr=False,
        calc_len=False,
        record_components=False,
        record_geom=False,
        largest_component=False,
        calc_stats=False,
        def_graph_elems=False,
    ):
        """
        Parameters
        ----------
        tnid : str
            TIGER/Line node ID variable used for working with
            TIGER/Line edges. Default is ``'TNID'``.
        tnidf : str
            TIGER/Line 'From Node' variable used for building topology
            in TIGER/Line edges. Default is ``'TNIDF'``.
        tnidt : str
             TIGER/Line 'To Node' variable used for building topology in
             TIGER/Line edges. Default is ``'TNIDT'``.
        s_data : {str, geopandas.GeoDataFrame}
            Path to segments data or a dataframe itself.
        n_data : {str, geopandas.GeoDataFrame, None}
            Nodes data. Default is ``None``.
        sid_name : str
            Segment column name. Default is ``'SegID'``.
        nid_name : str
            Node column name. Default is ``'NodeID'``.





        geo_col

        from_raw



        attr1 : str
            Auxillary variable being used. Default is ``None``.
        attr2 : str
            Auxillary variable being used. Either ``'TLID'`` for tiger edges
            or ``'LINEARID'`` for tiger roads. Default is ``None``.
        mtfcc_types : dict
            MTFCC road type descriptions. Default is ``None``.
            from [utils.get_mtfcc_types()]
        mtfcc_discard : list
            MTFCC types (by code) to discard. Default is ``None``.
            from [utils.get_discard_mtfcc_by_desc()]
        discard_segs : list
            specifc segment ids to discard. Default is ``None``.
            from [utils.discard_troublemakers()]
        xyid : str
            Combined x-coord + y-coords string ID. Default is ``'xyid'``.
        len_col : str
            Length column name. Default is ``'length'``.
        tiger_edges : bool
            Using TIGER/Line edges file. Default is ``True``.
        edge_subsets : list
            {type:{'col':column, 'oper':operator, 'val':value}}
            i.e. -- {'edge': {'col':'ROADFLG', 'val':'Y'}}
        mtfcc_split : str
            MTFCC codes for segments to weld and then split during the
            line splitting process. Default is None.
        mtfcc_intrst : str
            MTFCC codes for interstates. Default is None.
        mtfcc_ramp : str
            MTFCC codes for on ramps. Default is None.
        mtfcc_serv : str
            MTFCC codes for service drives. Default is None.
        mtfcc_split_grp : str
            after subseting this road type, group by this attribute
            before welding. Default is None.
        mtfcc_split_by : list
            MTFCC codes to eventually split the segments of
            `mtfcc_no_split` with. Default is None.
        skip_restr : bool
            skip re-welding restricted segments. Used when woring with
            TIGER/Lines. Default is False.
        calc_len : bool
            calculated length and add column. Default is False.
        record_components : bool
            Record connected components in graph. This is used for teasing out the
            largest connected component. Default is ``False``.
        largest_component : bool
            Keep only the largest connected component in the graph. Default is ``False``.
        record_geom : bool
            create associated between IDs and shapely geometries.
            Default is False.
        calc_stats : bool
            calculate network stats. Default is False.
        def_graph_elems : bool
            define graph elements. Default is False.

        Methods : Attributes
        --------------------
        __init__ : s_data
        build_network : --
        build_base : n_data, segm2xyid, node2xyid
        build_topology : segm2node, node2segment, segm2segm, node2node
        build_components : segm_cc, cc_lens, node_cc, longest_segm_cc,
            largest_segm_cc, largest_node_cc, n_ccs
        build_associations : s_ids, n_ids, n_segm, n_node, segm2len,
            network_length, node2degree, segm2tlid
        define_graph_elements : segm2elem, node2elem
        simplify_network : --
        add_node : --
        add_edge : --

        network_cost_matrix : diameter, radius, d_net, d_euc, circuity,
            n2n_euclidean, n2n_algo, n2n_matrix, n2n_paths
        calc_net_stats : max_sinuosity, min_sinuosity,
            net_mean_sinuosity, net_std_sinuosity, max_node_degree,
            min_node_degree, mean_node_degree, std_node_degree, alpha,
            beta, gamma, eta, entropies_mtfcc, entropy_mtfcc

        ############### sauce.setup_raw : raw_data_info
        ############### sauce.ring_correction : corrected_rings
        ############### sauce.line_splitter : lines_split
        ############### sauce.seg_welder : welded_mls
        ############### sauce.cleanse_supercycle : cleanse_cycles, scrubbed
        ############### sauce.geom_assoc : segm2geom, node2geom
        ############### sauce.coords_assoc : segm2coords, node2coords
        ############### sauce.get_stats_frame : network_stats

        Examples
        --------

        >>> import tigernet
        >>> lat = tigernet.generate_lattice(n_hori_lines=1, n_vert_lines=1)
        >>> net = tigernet.Network(s_data=lat)
        >>> net.s_data[["SegID", "MTFCC", "length", "xyid", "s_neigh", "n_neigh"]]
           SegID  MTFCC  length                      xyid    s_neigh n_neigh
        0      0  S1400     4.5  ['x4.5y0.0', 'x4.5y4.5']  [1, 2, 3]  [0, 1]
        1      1  S1400     4.5  ['x4.5y4.5', 'x4.5y9.0']  [0, 2, 3]  [1, 2]
        2      2  S1400     4.5  ['x0.0y4.5', 'x4.5y4.5']  [0, 1, 3]  [1, 3]
        3      3  S1400     4.5  ['x4.5y4.5', 'x9.0y4.5']  [0, 1, 2]  [1, 4]

        >>> net.n_data[["NodeID", "xyid", "s_neigh", "n_neigh", "degree"]]
           NodeID          xyid       s_neigh       n_neigh  degree
        0       0  ['x4.5y0.0']           [0]           [1]       1
        1       1  ['x4.5y4.5']  [0, 1, 2, 3]  [0, 2, 3, 4]       4
        2       2  ['x4.5y9.0']           [1]           [1]       1
        3       3  ['x0.0y4.5']           [2]           [1]       1
        4       4  ['x9.0y4.5']           [3]           [1]       1

        >>> net.segm2xyid[0]
        ['x4.5y0.0', 'x4.5y4.5']

        >>> net.node2xyid[0]
        ['x4.5y0.0']

        >>> net.segm2node[3]
        [1, 4]

        >>> net.node2segm[4]
        [3]

        >>> net.segm2segm[3]
        [0, 1, 2]

        >>> net.node2node[4]
        [1]

        """

        self.s_data = s_data
        self.is_gdf = hasattr(self.s_data, "geometry")
        IS_TIGEREDGES = hasattr(self.s_data, "MTFCC") or tiger_edges == True
        if IS_TIGEREDGES:
            self.tiger_edges = True
        else:
            self.tiger_edges = False

        if not self.is_gdf and not self.s_data and not network_instance:
            msg = "The 'segmdata' parameters must be set, "
            msg += "either as a 'str' or 'geopandas.GeoDataFrame', *OR* "
            msg += "the 'network_instance' parameter must be set."
            raise ValueError(msg)

        if network_instance:
            self = network_instance
        else:
            self.xyid, self.from_raw = xyid, from_raw
            self.sid_name, self.nid_name = sid_name, nid_name
            self.geo_col, self.len_col = geo_col, len_col

            #########################################################################
            # self.tiger_edges = tiger_edges
            # if self.tiger_edges:
            #    self.census_data = True
            # else:
            #    self.census_data = False
            #########################################################################

            if IS_TIGEREDGES:

                # TIGER variable attributes
                self.tnid, self.tnidf, self.tnidt = tnid, tnidf, tnidt
                self.attr1, self.attr2, self.tlid = attr1, attr2, attr2

                self.mtfcc_types = info.get_mtfcc_types()
                self.mtfcc_discard = info.get_discard_mtfcc_by_desc()
                self.discard_segs = discard_segs

                # This reads in and prepares/cleans a segments geodataframe
                if self.from_raw:
                    if self.is_gdf or self.s_data.endswith(".shp"):
                        # self.edge_subsets = edge_subsets

                        # ---------- make these something like ``mtfcc_kwargs``
                        self.mtfcc_split = mtfcc_split
                        self.mtfcc_intrst = mtfcc_intrst
                        self.mtfcc_ramp = mtfcc_ramp
                        self.mtfcc_serv = mtfcc_serv
                        self.mtfcc_split_grp = mtfcc_split_grp
                        self.mtfcc_split_by = mtfcc_split_by
                        self.skip_restr = skip_restr

                    else:
                        raise RuntimeError("Unknown line data.")

                    # freshly cleaned segments
                    utils.tiger_netprep(self, calc_len=calc_len)

            # build a network object from segments
            self.build_network(
                self.s_data,
                record_components=record_components,
                largest_component=largest_component,
                record_geom=record_geom,
                def_graph_elems=def_graph_elems,
            )

    ###########################################################################
    ########################    end __init__    ###############################
    ###########################################################################

    def build_network(
        self,
        s_data,
        record_components=False,
        record_geom=False,
        largest_component=False,
        def_graph_elems=False,
    ):
        """Top-level method for full network object creation from a
        geopandas.GeoDataFrame of lines.

        Parameters
        ----------
        s_data : geopandas.GeoDataFrame
            Segments data.
        record_components : bool
            Find rooted connected components in the network (``True``),
            or ignore (``False``). Default is ``False``.
        largest_component : bool
            Keep only the largest connected compnent of the network
            (``True``), or keep all components (``False``). Default is ``False``.
        record_geom : bool
            Create an id to geometry lookup (``True``), or ignore (``False``).
            Default is ``False``.
        def_graph_elems : bool
            Define each element of the graph as either a branch
            [connected to two or more other elements], or a leaf
            [connected to only one other element] (``True``), or ignore
            (``False``). Default is ``False``.

        """

        self.build_base(s_data)
        self.build_topology()
        if record_components:
            self.build_components(largest_cc=largest_component)
        self.build_associations(record_geom=record_geom)
        if def_graph_elems:
            self.define_graph_elements()

    def build_base(self, s_data):
        """Extract nodes from segment endpoints and relate
        segments and nodes to a location ID (``xyid``)

        Parameters
        ----------
        s_data : geopandas.GeoDataFrame
            Segments data.

        """

        # Instantiate segments dataframe as part of TigerNetwork class
        self.s_data = s_data
        del s_data
        self.s_data.reset_index(drop=True, inplace=True)
        self.s_data = utils.add_ids(self.s_data, id_name=self.sid_name)
        if not self.len_col in self.s_data.columns:
            self.s_data[self.len_col] = getattr(self.s_data, self.len_col)

        # create segment xyid
        self.segm2xyid = utils.generate_xyid(
            df=self.s_data, geom_type="segm", geo_col=self.geo_col
        )
        _skws = {"idx": self.sid_name, "col": self.xyid, "data": self.segm2xyid}
        self.s_data = utils.fill_frame(self.s_data, **_skws)

        # Instantiate nodes dataframe as part of NetworkClass
        self.n_data = utils.extract_nodes(self)
        self.n_data.reset_index(drop=True, inplace=True)

        # create permanent node xyid
        self.node2xyid = utils.generate_xyid(
            df=self.n_data, geom_type="node", geo_col=self.geo_col
        )
        _nkws = {"idx": self.nid_name, "col": self.xyid, "data": self.node2xyid}
        self.n_data = utils.fill_frame(self.n_data, **_nkws)

        # set segment & node ID lists and counts elements
        utils.set_ids(self)

    def build_topology(self):
        """Relate all graph elements."""

        # Associate segments with neighboring nodes
        _pri_sec = {"primary": self.segm2xyid, "secondary": self.node2xyid}
        self.segm2node = utils.associate(assoc="segm2node", **_pri_sec)

        # Associate nodes with neighboring segments
        _pri_sec = {"primary": self.node2xyid, "secondary": self.segm2xyid}
        self.node2segm = utils.associate(assoc="node2segm", **_pri_sec)

        # Associate segments with neighboring segments
        _args = self.segm2node, self.node2segm
        self.segm2segm = utils.get_neighbors(*_args, valtype=list)

        # Associate nodes with neighboring nodes
        _args = self.node2segm, self.segm2node
        self.node2node = utils.get_neighbors(*_args, valtype=list)

        # 1. Catch cases w/ >= 3 neighboring nodes for a segment and throw an error.
        # 2. Catch rings and add start & end node.
        self = utils.assert_2_neighs(self)

        # fill dataframe with seg2seg
        _skws = {"idx": self.sid_name, "col": "s_neigh", "data": self.segm2segm}
        self.s_data = utils.fill_frame(self.s_data, **_skws)

        # fill dataframe with seg2node
        _skws = {"idx": self.sid_name, "col": "n_neigh", "data": self.segm2node}
        self.s_data = utils.fill_frame(self.s_data, **_skws)

        # fill dataframe with node2seg
        _nkws = {"idx": self.nid_name, "col": "s_neigh", "data": self.node2segm}
        self.n_data = utils.fill_frame(self.n_data, **_nkws)

        # fill dataframe with node2node
        _nkws = {"idx": self.nid_name, "col": "n_neigh", "data": self.node2node}
        self.n_data = utils.fill_frame(self.n_data, **_nkws)

    def build_components(self, largest_cc=False):
        """Find the rooted connected components of the graph (either largest or longest).
        *** Must choose either largest or longest. If both ``largest_cc`` and
        ``longest_cc`` are ``True``, ``largest_cc`` will be selected by default. ***

        Parameters
        ----------
        largest_cc : bool
            Keep only the largest connected component (the most
            edges/nodes) in the graph. Default is ``False``.

        """

        ### Segms -- Connected Components
        # -- Count
        self.segm_cc = utils.get_roots(self.segm2segm)
        _skws = {"idx": self.sid_name, "col": "CC", "data": self.segm_cc}
        self.s_data = utils.fill_frame(self.s_data, **_skws)

        # -- Length
        # fill connected component len column in dataframe and return dict
        self.cc_lens = utils.get_cc_len(self, len_col=self.len_col)

        ### Node -- Connected Components
        self.node_cc = utils.get_roots(self.node2node)
        _nkws = {"idx": self.nid_name, "col": "CC", "data": self.node_cc}
        self.n_data = utils.fill_frame(self.n_data, **_nkws)

        # Extract largest CCs
        if largest_cc:
            # largest CC by count (nodes & segments) -- and return small keys
            self.largest_segm_cc, segm_smallkeys = utils.get_largest_cc(
                self.segm_cc, smallkeys=True
            )
            self.largest_node_cc, node_smallkeys = utils.get_largest_cc(
                self.node_cc, smallkeys=True
            )

            # Keep only the largest connected component
            utils.update_adj(self, segm_smallkeys, node_smallkeys)

            lcck = list(self.largest_segm_cc.keys())[0]
            self.cc_lens = {k: v for k, v in self.cc_lens.items() if k == lcck}

            # set segment & node ID lists and counts elements
            utils.set_ids(self)

        # Count connected components in network
        self.n_ccs = len(self.cc_lens.keys())

    def build_associations(self, record_geom=False):
        """Associate graph elements with geometries, coordinates,
        segment lengths, node degrees, and other information.

        Parameters
        ----------
        record_geom : bool
            Create an ID-to-geometry lookup (``True``). Default is ``False``.

        """

        # associate segments & nodes with geometries and coordinates
        if record_geom:
            utils.geom_assoc(self)
        utils.geom_assoc(self, coords=True)

        # associate segments with length
        self.segm2len = utils.xwalk(self.s_data, c1=self.sid_name, c2=self.len_col)

        # total length
        self.network_length = sum([v for k, v in self.segm2len.items()])

        # Calculate degree for n_ids -- incident segs +1; incident loops +2
        self.node2degree = utils.calc_valency(self, col="n_neigh")
        self.n_data["degree"] = self.n_data[self.nid_name].map(self.node2degree)

        # Create segment to TIGER/Line ID lookup
        try:
            if self.tiger_edges:
                self.segm2tlid = utils.xwalk(
                    self.s_data, c1=self.sid_name, c2=self.tlid
                )
        except KeyError:
            pass

    def define_graph_elements(self):
        """Define all segments and nodes as either a leaf (incident with
        one other element) or a branch (incident with more than one
        other grpah elemet).
        """

        self.segm2elem = utils.branch_or_leaf(self, geom_type="segm")
        _kws = {"idx": self.sid_name, "col": "graph_elem", "data": self.segm2elem}
        self.s_data = utils.fill_frame(self.s_data, **_kws)

        self.node2elem = utils.branch_or_leaf(self, geom_type="node")
        _kws = {"idx": self.nid_name, "col": "graph_elem", "data": self.node2elem}
        self.n_data = utils.fill_frame(self.n_data, **_kws)

    def simplify_network(
        self,
        record_components=False,
        largest_component=False,
        record_geom=False,
        def_graph_elems=False,
        inplace=False,
    ):
        """Remove all non-articulation points in the network.

        Parameters
        ----------
        record_components : bool
            Record connected components in graph. This is used for teasing out the
            largest connected component. Default is ``False``.
        largest_component : bool
            Keep only the largest connected component in the graph. Default is ``False``.
        record_geom : bool
            Create associated between IDs and shapely geometries. Default is ``False``.
        def_graph_elems : bool
            Define graph elements. Default is ``False``.
        inplace : bool
            Overwrite the original network with the simplified. Default is ``False``.

        Returns
        -------
        simp_net : geopandas.GeoDataFrame
            The simplified network (if ``inplace`` is set to ``False``).

        """

        if not inplace:
            simp_net = copy.deepcopy(self)
        else:
            simp_net = self

        # Create simplified road segments (remove non-articulation points)
        simp_segms = utils.simplify(simp_net)

        # Reset index and SegIDX to match
        simp_segms.reset_index(drop=True, inplace=True)
        simp_segms = utils.add_ids(simp_segms, id_name=simp_net.sid_name)
        simp_segms = utils.label_rings(simp_segms, geo_col=simp_net.geo_col)
        simp_segms = utils.ring_correction(simp_net, simp_segms)

        # add xyid
        segm2xyid = utils.generate_xyid(
            df=simp_segms, geom_type="segm", geo_col=simp_net.geo_col
        )
        simp_segms = utils.fill_frame(simp_segms, col=simp_net.xyid, data=segm2xyid)

        # build a network object from simplified segments
        simp_net.build_network(
            simp_segms,
            record_geom=record_geom,
            record_components=record_components,
            largest_component=largest_component,
            def_graph_elems=def_graph_elems,
        )

        if not inplace:
            return simp_net

    def calc_net_stats(self, conn_stat=None):
        """Calculate network analyis descriptive statistics.

        Parameters
        ----------
        conn_stat : {None, str}
            Either ``'alpha'``, ``'beta'``, ``'gamma'``, ``'eta'``.
            Set to ``'all'`` toc calculate all available statistics.
            For descriptions see ``stats.connectivity()``.

        """

        def _set_stat(s: str, xnccs: bool):
            """Calculate a connectivity stat and set as attribute."""

            # anonymous function for calculating and setting stats
            setter = lambda _s: setattr(self, _s, stats.connectivity(self, measure=_s))

            if (s == "alpha" and not xnccs) or s != "alpha":
                setter(s)
            elif s == "alpha" and xnccs:
                msg = "\nConnected components must be calculated"
                msg += " for alpha connectivity.\nCall the"
                msg += " 'build_components' method and run again."
                raise AttributeError(msg)

        # Calculate the sinuosity of network segments and provide descriptive stats
        stats.calc_sinuosity(self)

        # Set node degree attributes
        stats.set_node_degree(self)

        # network connectivity stats
        if conn_stat:
            # check if connected components are present
            if not hasattr(self, "n_ccs"):
                x_n_ccs = True
            else:
                x_n_ccs = False
            _available_stats = ["alpha", "beta", "gamma", "eta"]
            _cs = conn_stat.lower()
            if _cs == "all":
                for _as in _available_stats:
                    _set_stat(_as, x_n_ccs)
            elif _cs in _available_stats:
                _set_stat(_cs, x_n_ccs)
            else:
                raise ValueError("Connectivity measure '%s' not supported." % _cs)

        # network diameter, radius, circuity
        mtx_str = "n2n_matrix"
        if not hasattr(self, mtx_str):
            msg = "The 'Network' has no '%s' attribute. " % mtx_str
            msg += "Run 'cost_matrix()' and try again."
            warnings.warn(msg)
        else:
            mtx = getattr(self, mtx_str)

            # network diameter -- longest shortest path
            self.diameter = stats.dist_metric(mtx, "max")

            # network radius -- shortest shortest path
            self.radius = stats.dist_metric(mtx, "min")

            # circuity
            stats.circuity(self)

    def calc_entropy(self, ent_col, frame_name):
        """Network entropy statistics. For descriptions see ``stats.entropies()``.

        Parameters
        ----------
        ent_col : str
            The column name in ``frame_name`` to calculate entropy on.
        frame_name : str
            The name of the network element dataframe.

        """

        frame = getattr(self, frame_name)
        if frame_name == "s_data":
            n_elems = self.n_segm
        else:
            n_elems = self.n_node

        # calculate local network element entropies
        indiv_entropies = stats.entropies(ent_col, frame, n_elems)
        attr_name = "%s_entropies" % ent_col.lower()
        setattr(self, attr_name, indiv_entropies)

        # calculate global network entropy
        _entropy = [v for k, v in list(getattr(self, attr_name).items())]
        network_entropy = sum(_entropy) * -1.0
        attr_name = "network_%s_entropy" % ent_col.lower()
        setattr(self, attr_name, network_entropy)

    def cost_matrix(self, wpaths=False, asattr=True, validate_symmetry=True):
        """Network node-to-node cost matrix calculation with options for generating
        shortest paths along tree. For best results the network should be simplified
        prior to running this method.

        Parameters
        ----------
        wpaths : bool
            Generate shortest paths tree. Default is ``False``.
        asattr : bool
            Set ``n2n_matrix`` and ``paths`` as attributes of ``Network`` if ``True``,
            otherwise return them. Default is ``True``.
        validate_symmetry : bool
            Validate matrix symmetry. Default is ``False``.

        Returns
        -------
        n2n_matrix : numpy.ndarray
            Shortest path costs between all nodes.
        paths : dict
            Graph traveral paths.

        """

        # check whether IDs are consecutive and sequential
        i1 = self.s_ids[-1]
        i2 = self.n_segm - 1
        i3 = list(self.s_data[self.sid_name])[-1]
        simplified = i1 == i2 == i3
        if not simplified:
            msg = "Network element IDs are not consecutive/sequential. "
            msg += "Simplify the network and try again."
            raise IndexError(msg)

        n2n_matrix, paths = utils.shortest_path(self, gp=wpaths)

        if validate_symmetry:
            # validate symmetry
            if n2n_matrix[0][0] == 0.0:
                if not utils._check_symmetric(n2n_matrix, tol=1e-8):
                    msg = "The all-to-all cost matrix is not symmetric."
                    raise ValueError(msg)

        if asattr:
            self.n2n_matrix = n2n_matrix
            if wpaths:
                self.n2n_paths = paths
        else:
            if wpaths:
                return n2n_matrix, paths
            else:
                return n2n_matrix

    def nodes_kdtree(self, only_coords=False):
        """Build a kdtree from the network node coords for observations lookup.

        Parameters
        ----------
        only_coords : bool
            Flag for only coordinated being passed in.

        Returns
        -------
        kdtree : scipy.spatial.kdtree.KDTree
            All network nodes lookup.

        """

        if only_coords:
            geoms = self.n_data[self.geo_col]
            coords = list(zip(geoms.x, geoms.y))
            kdtree = cg.KDTree(coords)
        else:
            coords = [coords[0] for (node, coords) in list(self.node2coords.items())]
            kdtree = cg.KDTree(coords)

        return kdtree


class Observations:
    """Near-network observations.

    Parameters
    ----------
    net : tigernet.Network
        Network object.
    df : geopandas.GeoDataFrame
        Observation points dataframe.
    df_name : str
        Dataframe name. Default is ``None``.
    df_key : {str, int}
        Dataframe key column name. Default is ``'index'``.
    df_pop : ...........
        .......................... find where this is used..........#########################
    simulated : bool
        Empir. or sim. points along network segments. Default is ``False``.
    restrict_col : str
        Column name for segment restriction stipulation. Default is ``None``.
    remove_restricted : list
        Restricted segment types. Default is ``None``.
    k : int
        Number of nearest neighbors to query. Default is ``5``.
    tol : float
        Snapping to line tolerance. Default is ``.01``.
    snap_to : str
        Snap points to either segments of nodes. Default is ``'segments'``.
    no_pop : list ###########################################################################
        Observations that do not include a population measure.
        Default is ``['FireStations', 'FireStationsSynthetic']``.

    Methods : Attributes
    --------------------
    study_area : str
        Study area within county.
    sid_name : str
        Segment id column name.
    xyid : str
        Combined x-coord + y-coords string ID.
    obs2coords : list
        Observation index and attribute id lookup of coordinates.
    snapped_points : geopandas.GeoDataFrame
        Snapped point representation.
    obs2segm : dict
        Observation id (key) to segment id.

    """

    def __init__(
        self,
        net,
        df,
        df_name=None,
        df_key=None,
        df_pop=None,
        simulated=False,
        restrict_col=None,
        remove_restricted=None,
        k=5,
        tol=0.01,
        snap_to="segments",
        geo_col="geometry",
        # no_pop=["FireStations", "FireStationsSynthetic", "SegmMidpoints"],
    ):

        if not hasattr(net, "segm2geom"):
            msg = "The 'segm2geom' attribute is not present for the network. "
            msg += "Instantiate the network again with 'record_geom=True' "
            msg += "before re-running."
            raise AttributeError(msg)

        snap_to = snap_to.lower()
        valid_snap_values = ["segments", "nodes"]
        if not snap_to in valid_snap_values:
            msg = "The 'snap_to' parameter is set to '%s'. " % snap_to
            msg += "Valid values are: %s." % valid_snap_values
            raise ValueError(msg)

        # remove restricted network segments
        if remove_restricted:
            kws = {"restr": remove_restricted, "col": restrict_col}
            net = utils.remove_restricted(net, **kws)

        # build kdtree
        kd_tree = net.nodes_kdtree()

        self.sid_name = net.sid_name
        self.df = df
        self.geo_col = geo_col
        self.df_name = df_name
        self.df_key = df_key
        self.xyid = net.xyid
        self.k = k
        self.kd_tree = kd_tree
        self.tol = tol
        self.snap_to = snap_to

        # create observation to coordinate xwalk
        self.obs2coords = utils.get_obs2coords(self)

        # snap points and return dataframe
        self.snapped_points = utils.snap_to_nearest(self, net=net)

        # create observation-to-segment lookup
        if self.snap_to == "segments":
            k, s = self.snapped_points[self.df_key], self.snapped_points["assoc_segm"]
            self.obs2segm = dict(zip(k, s))
        else:
            k, n = self.snapped_points[self.df_key], self.snapped_points["assoc_node"]
            self.obs2node = dict(zip(k, n))

        # if not self.df_name in no_pop:
        #
        #    try:
        #        self.snapped_points[df_pop] = self.df[df_pop]
        #    except KeyError:
        #        try:
        #            df_pop = "POP100_syn"
        #            self.snapped_points[df_pop] = self.df[df_pop]
        #        except KeyError:
        #            df_pop = "synth_pop"
        #            self.snapped_points[df_pop] = self.df[df_pop]
        #
        #    # create a segment-to-population tracker
        #    # this will vary with each different method employed
        #    self.segm2pop = {
        #        seg: self.snapped_points.loc[
        #            (self.snapped_points["assoc_segm"] == seg), df_pop
        #        ].sum()
        #        for seg in net.s_ids
        #    }

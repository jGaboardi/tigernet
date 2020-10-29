"""
"""

from . import utils
from . import stats

import copy


__author__ = "James D. Gaboardi <jgaboardi@gmail.com>"


class Network:
    def __init__(
        self,
        tnid="TNID",
        tnidf="TNIDF",
        tnidt="TNIDT",
        network_instance=None,
        s_data=None,
        n_data=None,
        sid_name="SegID",
        nid_name="NodeID",
        geo_col="geometry",
        # proj_init=None,
        # proj_trans=None,
        # proj_units=None,
        # inter=None,
        attr1=None,
        attr2=None,
        # study_area=None,
        # county=None,
        # state=None,
        # year=None,
        # place_time=None,
        mtfcc_types=None,
        mtfcc_discard=None,
        discard_segs=None,
        xyid="xyid",
        len_col="length",
        tiger_edges=True,  ######################################################
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
        gen_matrix=False,
        mtx_to_csv=None,
        gen_paths=False,
        paths_to_csv=None,
        gen_adjmtx=False,
        adjmtx_to_csv=None,
        algo=None,
        def_graph_elems=False,
        simplify=False,
        save_full=False,
        full_net_segms=None,
        full_net_nodes=None,
        save_simplified=False,
        simp_net_segms=None,
        simp_net_nodes=None,
        remove_gdfs=False,
        file_type=".shp",
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


        #proj_init : int
        #    initial projection. Default is None.
        #proj_trans : int
        #    transformed projection. Default is None.
        #proj_units : str
        #    unit of transformed projection. Default is None.

        attr1 : str
            Auxillary variable being used. Default is ``None``.
        attr2 : str
            Auxillary variable being used. Either ``'TLID'`` for tiger edges
            or ``'LINEARID'`` for tiger roads. Default is ``None``.

        #inter : str
        #    file path to intermediary data. Default is None.
        #study_area : str
        #    study area within county. Default is None.
        #county : str
        #    county of interest. Default is None.
        #state : str
        #    state of interest. Default is None.
        #year : str
        #    data collection year. Default is None.
        #place_time : str
        #    place and time descriptor. Default is None. e.g.
        #    '_Leon_FL_2010'

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
        gen_matrix : bool
            calculate a cost matrix. Default is False.
        mtx_to_csv : str
            file path to save the cost matrix. Default is None.
        gen_paths : bool
            calculate shortest path trees. Default is False.
        paths_to_csv : str
            file path to save the shortest path trees. Default is None.
        gen_adjmtx : bool
            calculate node adjacency matrix. Default is False.
        adjmtx_to_csv : str
            file path to save the adjacency matrix. Default is None.
        algo : str
            shortest path algorithm. Default is None.
        def_graph_elems : bool
            define graph elements. Default is False.
        simplify : bool
            remove all non-articulation points from the network object.
            Default is False.
        save_full : bool
            save out the full network objects. Default is False.
        full_net_segms : str
            path and file name to save out the full network segments.
            Default is None.
        full_net_nodes : str
            path and file name to save out the full network nodes.
            Default is None.
        save_simplified : bool
            save out the simplified network objects. Default is False.
        simp_net_segms : str
            path and file name to save out the simplified network
            segments. Default is None.
        simp_net_nodes : str
            path and file name to save out the simplified network nodes.
            Default is None.
        remove_gdfs : bool
            remove dataframes from network object following  network
            simplification. Default is False.

        file_type : str
            file extension. Default is ``'.shp'``.

        Methods : Attributes
        --------------------
        __init__ : segmdata, census_data
        build_network : --
        build_base : s_data, n_data, segm2xyid, node2xyid
        build_topology : segm2node, node2segment, segm2segm, node2node
        build_components : segm_cc, cc_lens, node_cc, longest_segm_cc,
            largest_segm_cc, largest_node_cc, n_ccs
        build_associations : s_ids, n_ids, n_segm, n_node, segm2len,
            network_length, node2degree, segm2tlid
        define_graph_elements : segm2elem, node2elem
        simplify_network : --
        add_node : --
        add_edge : --
        adjacency_matrix : n2n_adjmtx
        network_cost_matrix : diameter, radius, d_net, d_euc, circuity,
            n2n_euclidean, n2n_algo, n2n_matrix, n2n_paths
        calc_net_stats : max_sinuosity, min_sinuosity,
            net_mean_sinuosity, net_std_sinuosity, max_node_degree,
            min_node_degree, mean_node_degree, std_node_degree, alpha,
            beta, gamma, eta, entropies_mtfcc, entropy_mtfcc,
            actual_object_sizes, actual_total_size

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
        >>> net.s_data
                                                geometry  SegID  MTFCC  length                      xyid    s_neigh n_neigh
        0  LINESTRING (4.50000 0.00000, 4.50000 4.50000)      0  S1400     4.5  ['x4.5y0.0', 'x4.5y4.5']  [1, 2, 3]  [0, 1]
        1  LINESTRING (4.50000 4.50000, 4.50000 9.00000)      1  S1400     4.5  ['x4.5y4.5', 'x4.5y9.0']  [0, 2, 3]  [1, 2]
        2  LINESTRING (0.00000 4.50000, 4.50000 4.50000)      2  S1400     4.5  ['x0.0y4.5', 'x4.5y4.5']  [0, 1, 3]  [1, 3]
        3  LINESTRING (4.50000 4.50000, 9.00000 4.50000)      3  S1400     4.5  ['x4.5y4.5', 'x9.0y4.5']  [0, 1, 2]  [1, 4]

        >>> net.n_data
                          geometry  NodeID          xyid       s_neigh       n_neigh  degree
        0  POINT (4.50000 0.00000)       0  ['x4.5y0.0']           [0]           [1]       1
        1  POINT (4.50000 4.50000)       1  ['x4.5y4.5']  [0, 1, 2, 3]  [0, 2, 3, 4]       4
        2  POINT (4.50000 9.00000)       2  ['x4.5y9.0']           [1]           [1]       1
        3  POINT (0.00000 4.50000)       3  ['x0.0y4.5']           [2]           [1]       1
        4  POINT (9.00000 4.50000)       4  ['x9.0y4.5']           [3]           [1]       1

        >>> net.segm2xyid[0]
        [0, ['x4.5y0.0', 'x4.5y4.5']]

        >>> net.node2xyid[0]
        [0, ['x4.5y0.0']]

        >>> net.segm2node[-1]
        [3, [1, 4]]

        >>> net.node2segm[-1]
        [4, [3]]

        >>> net.segm2segm[-1]
        [3, [0, 1, 2]]

        >>> net.node2node[-1]
        [4, [1]]

        """

        IS_GDF = hasattr(s_data, "geometry")

        if not IS_GDF and not s_data:
            msg = "The 'segmdata' parameters must be set, "
            msg += "either as a 'str' or 'geopandas.GeoDataFrame'."
            raise ValueError(msg)

        if network_instance:
            self = network_instance
        else:
            self.tnid, self.tnidf, self.tnidt = tnid, tnidf, tnidt
            self.sid_name, self.nid_name = sid_name, nid_name
            self.geo_col, self.len_col = geo_col, len_col
            # self.proj_init, self.proj_trans = proj_init, proj_trans
            # self.proj_units = proj_units
            self.xyid = xyid
            # self.inter = inter
            # self.file_type = file_type
            self.mtfcc_types = mtfcc_types
            self.mtfcc_discard = mtfcc_discard
            self.tiger_edges = tiger_edges
            # self.tiger_roads = tiger_roads
            self.discard_segs = discard_segs
            if self.tiger_edges:  #############################################
                self.census_data = True
            else:
                self.census_data = False

            if self.census_data:
                # TIGER variable attributes
                self.attr1 = attr1
                self.attr2 = attr2
                # self.study_area = study_area
                # self.county = county
                # self.state = state
                # self.year = year
                # self.place_time = place_time
                self.s_data = s_data
                if self.tiger_edges:
                    self.tlid = self.attr2

            # This reads in and prepares/cleans a segments geodataframe
            if not hasattr(s_data, self.geo_col) and self.census_data:
                if self.tiger_edges:
                    self.edge_subsets = edge_subsets
                    self.mtfcc_split = mtfcc_split
                    self.mtfcc_intrst = mtfcc_intrst
                    self.mtfcc_ramp = mtfcc_ramp
                    self.mtfcc_serv = mtfcc_serv
                    self.mtfcc_split_grp = mtfcc_split_grp
                    self.mtfcc_split_by = mtfcc_split_by
                    self.skip_restr = skip_restr
                    # fetch path to raw tiger edge data is available
                    # of local machine, otherwise download from
                    # https://www2.census.gov/
                    # raw_file = sauce.get_raw_tiger_edges(self)
                else:
                    raise RuntimeError("Unknown line data.")

                ######################################################################### do after work out `build_network`
                # freshly cleaned segments geodataframe
                # segmdata = sauce.tiger_netprep(
                #    self, in_file=raw_file, calc_len=calc_len
                # )
                #########################################################################

            # build a network object from segments
            self.build_network(
                s_data,
                record_components=record_components,
                largest_component=largest_component,
                record_geom=record_geom,
                def_graph_elems=def_graph_elems,
            )

        """
        ################# simplify the network
        ################if simplify:
        ################    # create simplified segments geodataframe
        ################    simplified_segms = self.simplify_network(self)
        ################    # build a network object from simplified segments
        ################    self.build_network(simplified_segms, record_geom=record_geom,
        ################                       record_components=record_components,
        ################                       largest_component=largest_component,
        ################                       def_graph_elems=def_graph_elems)
        ################    
        ################    if save_simplified:
        ################        self.s_data.to_file(simp_net_segms+self.file_type)
        ################        self.n_data.to_file(simp_net_nodes+self.file_type)
        
        # create node to node adjacency matrix
        if gen_adjmtx:
            self.adjacency_matrix(adjmtx_to_csv)
        
        # Create and save out an all to all network node cost matrix
        if gen_matrix:
            
            if not algo:
                raise Exception('Set algorithm for cost matrix calculation.')
            
            self.n2n_algo = algo
            self.network_cost_matrix(mtx_to_csv=mtx_to_csv, gpths=gen_paths,
                                     paths_to_csv=paths_to_csv,
                                     calc_stats=calc_stats)
        # calculate descriptive network stats
        if calc_stats:
            self.calc_net_stats()
        
        if remove_gdfs:
            self.remove_frames()
        """

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
        self.segm2segm = utils.get_neighbors(*_args, astype=list)

        # Associate nodes with neighboring nodes
        _args = self.node2segm, self.segm2node
        self.node2node = utils.get_neighbors(*_args, astype=list)

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

            lcck = self.largest_segm_cc[0]
            self.cc_lens = {k: vs for k, vs in self.cc_lens.items() if k == lcck}

        # Count connected components in network
        self.n_ccs = len(self.segm_cc)

    def build_associations(self, record_geom=False):
        """Associate graph elements with geometries, coordinates,
        segment lengths, node degrees, and other information.

        Parameters
        ----------
        record_geom : bool
            Create an ID-to-geometry lookup (``True``). Default is ``False``.

        """

        if record_geom:
            utils.geom_assoc(self)
        utils.geom_assoc(self, coords=True)

        # id lists
        self.s_ids = list(self.s_data[self.sid_name])
        self.n_ids = list(self.n_data[self.nid_name])

        # permanent segment count & node count
        self.n_segm, self.n_node = len(self.s_ids), len(self.n_ids)

        # Associate segments with length
        self.segm2len = utils.xwalk(self.s_data, c1=self.sid_name, c2=self.len_col)

        # total length
        self.network_length = sum([v for (k, v) in self.segm2len])

        # Calculate degree for n_ids -- incident segs +1; incident loops +2
        self.node2degree = utils.calc_valency(self, col="n_neigh")
        self.n_data["degree"] = [n2d[1][0] for n2d in self.node2degree]

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
                    if _as == "alpha" and not x_n_ccs or _as != "alpha":
                        setattr(self, _as, stats.connectivity(self, measure=_as))
                    elif _as == "alpha" and x_n_ccs:
                        msg = "\nConnected components must be calculated"
                        msg += " for alpha connectivity.\nCall the"
                        msg += " 'build_components' method and run again."
                        raise AttributeError(msg)
                    else:
                        msg = "Connectivity measure '%s' not supported." % _cs
                        raise ValueError(msg)
            elif _cs in _available_stats:
                setattr(self, _cs, stats.connectivity(self, measure=_cs))
            else:
                raise ValueError("Connectivity measure '%s' not supported." % _cs)

        """
        # network connectivity stats
        
        self.entropies_mtfcc = sauce.entropy(self) #return dict
        entropy = [v for k,v in list(self.entropies_mtfcc.items())]
        self.entropy_mtfcc = sum(entropy)*-1.
        
        # create dataframe of descriptive network stats
        if hasattr(self, 'n2n_matrix'):
            sauce.get_stats_frame(self)
        
        """

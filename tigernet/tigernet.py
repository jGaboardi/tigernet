"""
"""

from . import utils

__author__ = "James D. Gaboardi <jgaboardi@gmail.com>"


class TigerNet:
    def __init__(
        self,
        file_type=".shp",
        tnid="TNID",
        tnidf="TNIDF",
        tnidt="TNIDT",
        network_instance=None,
        s_data=None,
        n_data=None,
        sid_name="SegID",
        nid_name="NodeID",
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
        tiger_edges=True,
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
    ):
        """
        Parameters
        ----------
        file_type : str
            file extension. Default is ``'.shp'``.
        tnid : str
            TIGER/Line node ID variable used for working with
            TIGER/Line edges. Default is ``'TNID'``.
        tnidf : str
            TIGER/Line 'From Node' variable used for building topology
            in TIGER/Line edges. Default is ``'TNIDF'``.
        tnidt : str
             TIGER/Line 'To Node' variable used for building topology in
             TIGER/Line edges. Default is ``'TNIDT'``.
        s_data : str **OR** geopandas.GeoDataFrame
            Path to segments data or a dataframe itself.
        n_data : str **OR** geopandas.GeoDataFrame
            Nodes data. Default is ``None``.
        sid_name : str
            Segment column name. Default is ``'SegID'``.
        nid_name : str
            Node column name. Default is ``'NodeID'``.
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
            record connected components in graph. This is used for
            teasing out the largests connected component.
            Default is False.
        largest_component : bool
            keep only the largest connected component in the graph.
            Default is False.
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

        Methods : Attributes
        --------------------
        __init__ : segmdata, census_data
        build_network : --
        build_base : s_data, n_data, segm2xyid, node2xyid
        build_topology : segm2node, node2segment, segm2segm, node2node
        build_components : segm_cc, cc_lens, node_cc, longest_segm_cc,
            largest_segm_cc, largest_node_cc, n_edge_cc
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
        >>> net = tigernet.TigerNet(s_data=lat)
        >>> net.s_data
                                                geometry  SegID  MTFCC                      xyid
        0  LINESTRING (4.50000 0.00000, 4.50000 4.50000)      0  S1400  ['x4.5y0.0', 'x4.5y4.5']
        1  LINESTRING (4.50000 4.50000, 4.50000 9.00000)      1  S1400  ['x4.5y4.5', 'x4.5y9.0']
        2  LINESTRING (0.00000 4.50000, 4.50000 4.50000)      2  S1400  ['x0.0y4.5', 'x4.5y4.5']
        3  LINESTRING (4.50000 4.50000, 9.00000 4.50000)      3  S1400  ['x4.5y4.5', 'x9.0y4.5']

        >>> net.n_data
                          geometry  NodeID          xyid
        0  POINT (4.50000 0.00000)       0  ['x4.5y0.0']
        1  POINT (4.50000 4.50000)       1  ['x4.5y4.5']
        2  POINT (4.50000 9.00000)       2  ['x4.5y9.0']
        3  POINT (0.00000 4.50000)       3  ['x0.0y4.5']
        4  POINT (9.00000 4.50000)       4  ['x9.0y4.5']

        >>> net.segm2xyid[0]
        [0, ['x4.5y0.0', 'x4.5y4.5']]

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
            self.len_col = len_col
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
            if self.tiger_edges:
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
            if not hasattr(s_data, "geometry") and self.census_data:
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
        # self.build_topology()
        # if record_components:
        #    self.build_components(largest_cc=largest_component)
        # self.build_associations(record_geom=record_geom)
        # if def_graph_elems:
        #    self.define_graph_elements()

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

        # create segment xyid
        self.segm2xyid = utils.generate_xyid(df=self.s_data, geom_type="segm")
        self.s_data = utils.fill_frame(
            self.s_data, idx=self.sid_name, col=self.xyid, data=self.segm2xyid
        )

        # Instantiate nodes dataframe as part of NetworkClass
        self.n_data = utils.extract_nodes(self)
        self.n_data.reset_index(drop=True, inplace=True)

        # create permanent node xyid
        self.node2xyid = utils.generate_xyid(df=self.n_data, geom_type="node")
        self.n_data = utils.fill_frame(
            self.n_data, idx=self.nid_name, col=self.xyid, data=self.node2xyid
        )

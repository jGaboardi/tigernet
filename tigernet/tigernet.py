"""Network topology via TIGER/Line Edges.
"""

from . import utils
from . import stats
from . import info
from .generate_data import generate_xyid

import copy
import warnings

from libpysal import cg


__author__ = "James D. Gaboardi <jgaboardi@gmail.com>"


class Network:
    def __init__(
        self,
        s_data,
        from_raw=False,
        sid_name="SegID",
        nid_name="NodeID",
        geo_col="geometry",
        len_col="length",
        xyid="xyid",
        tnid="TNID",
        tnidf="TNIDF",
        tnidt="TNIDT",
        attr1=None,
        attr2=None,
        mtfcc_types=None,
        mtfcc_discard=None,
        discard_segs=None,
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
        s_data : geopandas.GeoDataFrame
            Segments dataframe.
        from_raw : bool
            Input ``s_data`` is raw TIGER/Line Edge data. Default is ``False``.
        sid_name : str
            Segment column name. Default is ``'SegID'``.
        nid_name : str
            Node column name. Default is ``'NodeID'``.
        geo_col : str
            Geometry column name. Default is ``'geometry'``.
        len_col : str
            Length column name. Default is ``'length'``.
        xyid : str
            Combined x-coord + y-coords string ID. Default is ``'xyid'``.
        tnid : str
            TIGER/Line node ID variable used for working with
            TIGER/Line edges. Default is ``'TNID'``.
        tnidf : str
            TIGER/Line 'From Node' variable used for building topology
            in TIGER/Line edges. Default is ``'TNIDF'``.
        tnidt : str
             TIGER/Line 'To Node' variable used for building topology in
             TIGER/Line edges. Default is ``'TNIDT'``.
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
        mtfcc_split : str
            MTFCC codes for segments to weld and then split during the
            line splitting process. Default is ``None``.
        mtfcc_intrst : str
            MTFCC codes for interstates. Default is ``None``.
        mtfcc_ramp : str
            MTFCC codes for on ramps. Default is ``None``.
        mtfcc_serv : str
            MTFCC codes for service drives. Default is ``None``.
        mtfcc_split_by : list
            MTFCC codes to eventually split the segments of
            `mtfcc_no_split` with. Default is ``None``.
        mtfcc_split_grp : str
            After subseting this road type, group by this attribute
            before welding. Default is ``None``.
        skip_restr : bool
            Skip re-welding restricted segments. Default is ``False``.
        calc_len : bool
            Calculate length and add column. Default is ``False``.
        record_components : bool
            Record connected components in graph. This is used for teasing out the
            largest connected component. Default is ``False``.
        record_geom : bool
            Create associated between IDs and shapely geometries.
            Default is ``False``.
        largest_component : bool
            Keep only the largest connected component in the graph. Default is ``False``.
        calc_stats : bool
            Calculate network stats. Default is ``False``.
        def_graph_elems : bool
            Define graph elements. Default is ``False``.

        Attributes
        ----------
        segm2xyid : dict
            Segment to xyID lookup.
        node2xyid : dict
            Node to xyID lookup.
        segm2node : dict
            Segment to node lookup.
        node2segm : dict
            Node to segment lookup.
        segm2segm : dict
            Segment to segment lookup.
        node2node : dict
            Node to node lookup.
        segm_cc : dict
            Root segment ID to connected component segment IDs lookup.
        cc_lens : dict
            Root segment ID to connected component length lookup.
        node_cc : dict
            Root node ID to connected component node IDs lookup.
        largest_segm_cc : dict
            Root segment ID to largest connected component segment IDs lookup.
        largest_node_cc : dict
            Root node ID to largest connected component node IDs lookup.
        n_ccs : int
            The number of connected components.
        s_ids : list
            Segment IDs.
        n_ids : list
            Node IDs.
        n_segm : int
            Network segment count.
        n_node : int
            Network node count.
        segm2len : dict
            Segment to segment length lookup.
        network_length : float
            Full network length.
        node2degree : dict
            Node to node degree lookup.
        segm2tlid : dict
            Segment to TIGER/Line ID lookup.
        segm2elem : dict
            Segment to network element lookup.
        node2elem : dict
            Node to network element lookup.
        diameter : float
            The longest shortest path between two nodes in the network.
        radius : float
            The shortest path between two nodes in the network.
        d_net : float
            Cumulative network diameter.
        d_euc : float
            Cumulative euclidean diameter.
        circuity : float
            Network circuity. See ``stats.circuity()``.
        n2n_matrix : numpy.array
            All node-to-node shortest path lengths in the network.
        n2n_paths : dict
            All node-to-node shortest paths in the network.
        max_sinuosity : float
            Maximum segment sinuosity.
        min_sinuosity : float
            Minimum segment sinuosity.
        net_mean_sinuosity : float
            Network segment sinuosity mean.
        net_std_sinuosity : float
            Network segment sinuosity standard deviation.
        max_node_degree : int
            Maximum node degree.
        min_node_degree : int
            Minimum node degree.
        mean_node_degree : float
            Network node degree mean.
        std_node_degree : float
            Network node degree standard deviation.
        alpha : float
            Network alpha measure. See ``stats.connectivity()``.
        beta : float
            Network beta measure. See ``stats.connectivity()``.
        gamma : float
            Network gamma measure. See ``stats.connectivity()``.
        eta : float
            Network eta measure. See ``stats.connectivity()``.
        entropies_{} : dict
            Segment/Node ID to {variable/attribute} entropies.
        network_entropy_{} : float
            Network {variable/attribute} entropy.
        corrected_rings : int
            Number of corrected rings in the network.
        lines_split : int
            Number of split lines in the network.
        welded_mls : int
            Number of welded multilinestrings in the network.
        segm2geom : dict
            Segment to geometry lookup.
        node2geom : dict
            Node to geometry lookup.
        segm2coords : dict
            Segment to endpoint coordinates lookup.
        node2coords : dict
            Node to coordinates lookup.

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
        self.xyid, self.from_raw = xyid, from_raw
        self.sid_name, self.nid_name = sid_name, nid_name
        self.geo_col, self.len_col = geo_col, len_col

        # TIGER variable attributes
        self.tnid, self.tnidf, self.tnidt = tnid, tnidf, tnidt
        self.attr1, self.attr2, self.tlid = attr1, attr2, attr2

        # discard segments if desired
        if discard_segs:
            ds = info.get_discard_segms(*discard_segs)
            self.s_data = self.s_data[~self.s_data[self.tlid].isin(ds)]

        self.mtfcc_types = info.get_mtfcc_types()
        self.mtfcc_discard = info.get_discard_mtfcc_by_desc()
        self.discard_segs = discard_segs

        # This reads in and prepares/cleans a segments geodataframe
        if self.from_raw:
            self.mtfcc_split = mtfcc_split
            self.mtfcc_intrst = mtfcc_intrst
            self.mtfcc_ramp = mtfcc_ramp
            self.mtfcc_serv = mtfcc_serv
            self.mtfcc_split_grp = mtfcc_split_grp
            self.mtfcc_split_by = mtfcc_split_by
            self.skip_restr = skip_restr

            # freshly cleaned segments
            utils.tiger_netprep(self, calc_len)

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
        segments and nodes to a location ID (``xyid``).

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
        self.segm2xyid = generate_xyid(
            df=self.s_data, geom_type="segm", geo_col=self.geo_col
        )
        _skws = {"idx": self.sid_name, "col": self.xyid}
        self.s_data = utils.fill_frame(self.s_data, self.segm2xyid, **_skws)

        # Instantiate nodes dataframe as part of NetworkClass
        self.n_data = utils.extract_nodes(self)
        self.n_data.reset_index(drop=True, inplace=True)

        # create permanent node xyid
        self.node2xyid = generate_xyid(
            df=self.n_data, geom_type="node", geo_col=self.geo_col
        )
        _nkws = {"idx": self.nid_name, "col": self.xyid}
        self.n_data = utils.fill_frame(self.n_data, self.node2xyid, **_nkws)

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
        self.segm2segm = utils.get_neighbors(self.segm2node, self.node2segm)

        # Associate nodes with neighboring nodes
        self.node2node = utils.get_neighbors(self.node2segm, self.segm2node)

        # 1. Catch cases w/ >= 3 neighboring nodes for a segment and throw an error.
        # 2. Catch rings and add start & end node.
        self = utils.assert_2_neighs(self)

        # fill dataframe with seg2seg
        _skws = {"idx": self.sid_name, "col": "s_neigh"}
        self.s_data = utils.fill_frame(self.s_data, self.segm2segm, **_skws)

        # fill dataframe with seg2node
        _skws = {"idx": self.sid_name, "col": "n_neigh"}
        self.s_data = utils.fill_frame(self.s_data, self.segm2node, **_skws)

        # fill dataframe with node2seg
        _nkws = {"idx": self.nid_name, "col": "s_neigh"}
        self.n_data = utils.fill_frame(self.n_data, self.node2segm, **_nkws)

        # fill dataframe with node2node
        _nkws = {"idx": self.nid_name, "col": "n_neigh"}
        self.n_data = utils.fill_frame(self.n_data, self.node2node, **_nkws)

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
        _skws = {"idx": self.sid_name, "col": "CC"}
        self.s_data = utils.fill_frame(self.s_data, self.segm_cc, **_skws)

        # -- Length
        # fill connected component len column in dataframe and return dict
        self.cc_lens = utils.get_cc_len(self, len_col=self.len_col)

        ### Node -- Connected Components
        self.node_cc = utils.get_roots(self.node2node)
        _nkws = {"idx": self.nid_name, "col": "CC"}
        self.n_data = utils.fill_frame(self.n_data, self.node_cc, **_nkws)

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
        if self.tlid:
            self.segm2tlid = utils.xwalk(self.s_data, c1=self.sid_name, c2=self.tlid)

    def define_graph_elements(self):
        """Define all segments and nodes as either a leaf (incident with one other
        element) or a branch (incident with more than one other graph element).
        """

        self.segm2elem = utils.branch_or_leaf(self, geom_type="segm")
        _kws = {"idx": self.sid_name, "col": "graph_elem"}
        self.s_data = utils.fill_frame(self.s_data, self.segm2elem, **_kws)

        self.node2elem = utils.branch_or_leaf(self, geom_type="node")
        _kws = {"idx": self.nid_name, "col": "graph_elem"}
        self.n_data = utils.fill_frame(self.n_data, self.node2elem, **_kws)

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
        segm2xyid = generate_xyid(
            df=simp_segms, geom_type="segm", geo_col=simp_net.geo_col
        )
        simp_segms = utils.fill_frame(simp_segms, segm2xyid, col=simp_net.xyid)

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
            else:
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

    def cost_matrix(self, wpaths=False, asattr=True):
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

        # calculate shortest path length and records paths if desired
        n2n_matrix, paths = utils.shortest_path(self, gp=wpaths)

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
    simulated : bool
        Empir. or sim. points along network segments. Default is ``False``.
    restrict_col : str
        Column name for segment restriction stipulation. Default is ``None``.
    remove_restricted : list
        Restricted segment types. Default is ``None``.
    obs_pop : str
        Population column of the observations dataframe. Default is ``None``.
    k : int
        Number of nearest neighbors to query. Default is ``5``.
    tol : float
        Snapping to line tolerance. Default is ``.01``.
    snap_to : str
        Snap points to either segments of nodes. Default is ``'segments'``.

    Attributes
    ----------
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
        obs_pop=None,
        k=5,
        tol=0.01,
        snap_to="segments",
        geo_col="geometry",
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

        ########################################################################
        ############### ------------- this would be better as 'label_restricted'
        ############### --- actually removing the network segments is bad design
        ############### --- ... the segments ARE still part of the network......
        ########################################################################

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

        # create a segment-to-population tracker
        if self.snap_to == "segments" and obs_pop:
            self.snapped_points[obs_pop] = self.df[obs_pop]
            self.segm2pop = {
                seg: self.snapped_points.loc[
                    (self.snapped_points["assoc_segm"] == seg), obs_pop
                ].sum()
                for seg in net.s_ids
            }


def obs2obs_cost_matrix(
    origin_observations,
    network,
    destination_observations=None,
    snap_dist=True,
    distance_type="network",
):
    """Calculate a cost matrix from (n) observations to (m) observations.

    Parameters
    ----------
    origin_observations : tigernet.Observations
    network : tigernet.Network
    destination_observations : tigernet.Observations
        Destination observations. Default is ``None``.
    snap_dist : str
        Include the distance to observations from the network. Default is ``True``.
    distance_type : str
        Type of distance cost matrix. Default is ``'network'``.
        Option is ``'euclidean'``.

    Returns
    -------
    n2m_matrix : numpy.ndarray
        'nXm' cost matrix.

    """

    # ensure the network object has an associated cost matrix
    mtx_str = "n2n_matrix"
    if not hasattr(network, mtx_str):
        msg = "The 'Network' has no '%s' attribute. " % mtx_str
        msg += "Run 'cost_matrix()' and try again."
        raise AttributeError(msg)
    else:
        network_matrix = getattr(network, mtx_str)

    # set the origin point dataframe
    orig_obs = origin_observations.snapped_points

    # set cost matrix as symmetric if no destination pattern is specified
    if not destination_observations:
        dest_obs = orig_obs
        symmetric = True
    else:
        dest_obs = destination_observations.snapped_points
        symmetric = False

    # determine whether calculating distance from
    # segments locations or from the nearest node
    assoc = origin_observations.snap_to
    if assoc == "nodes":
        assoc_col = "assoc_node"
        from_nodes = True
    else:
        assoc_col = "assoc_segm"
        from_nodes = False

    # declare distance to the network (if desired)
    if snap_dist:
        if from_nodes:
            snap_dist = "dist2node"
        else:
            snap_dist = "dist2line"

    # declare columns that should be checked for numerical values
    numeric_cols = [assoc_col]
    if not from_nodes:
        numeric_cols += ["dist_a", "dist_b"]
    if snap_dist:
        numeric_cols += [snap_dist]

    # set the xy-id name from the ``Network``
    xyid = network.xyid

    # generate the cost matrix
    n2m_matrix = utils.obs2obs_costs(
        orig_obs,
        dest=dest_obs,
        symmetric=symmetric,
        network_matrix=network_matrix,
        xyid=xyid,
        from_nodes=from_nodes,
        snap_dist=snap_dist,
        dist_type=distance_type,
        assoc_col=assoc_col,
        numeric_cols=numeric_cols,
    )

    return n2m_matrix

"""
"""

from ast import literal_eval
import copy

import geopandas
import numpy
import pandas
from shapely.geometry import Point, MultiPoint
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import GeometryCollection
from shapely.ops import linemerge


from shapely.geometry import Point, MultiPoint
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import GeometryCollection
from shapely.ops import linemerge, polygonize

# used to supress warning in addIDX()
geopandas.pd.set_option("mode.chained_assignment", None)


def _get_lat_lines(hspace, vspace, withbox, bounds, hori=True):
    """Generate line segments for a lattice.

    Parameters
    ----------
    hspace : list
        Horizontal spacing.
    vspace : list
        Vertical spacing.
    withbox : bool
        Include outer rim.
    bounds : list
        area bounds in the form of ``[x1,y1,x2,y2]``.
    hori : bool
        Generate horizontal line segments.
        Default is ``True``. ``False`` generates vertical segments.

    Returns
    -------
    lines : list
        All vertical or horizontal line segments in the grid.
    """

    # Initialize starting and ending horizontal indices
    h_start_at, h_end_at = 0, len(hspace)

    # Initialize starting and ending vertical indices
    v_start_at, v_end_at = 0, len(vspace)

    # set inital index track back to 0
    y_minus = 0
    x_minus = 0

    if hori:  # start track back at 1 for horizontal lines
        x_minus = 1
        if not withbox:  # do not include borders
            v_start_at += 1
            v_end_at -= 1

    else:  # start track back at 1 for vertical lines
        y_minus = 1
        if not withbox:  # do not include borders
            h_start_at += 1
            h_end_at -= 1

    # Create empty line list and fill
    lines = []

    # for element in the horizontal index
    for hplus in range(h_start_at, h_end_at):

        # for element in the vertical index
        for vplus in range(v_start_at, v_end_at):

            # ignore if a -1 index
            if hplus - x_minus == -1 or vplus - y_minus == -1:
                continue
            else:
                # Point 1 (start point + previous slot in
                #          horizontal or vertical space index)
                p1x = bounds[0] + hspace[hplus - x_minus]
                p1y = bounds[1] + vspace[vplus - y_minus]
                p1 = Point(p1x, p1y)

                # Point 2 (start point + current slot in
                #          horizontal or vertical space index)
                p2x = bounds[0] + hspace[hplus]
                p2y = bounds[1] + vspace[vplus]
                p2 = Point(p2x, p2y)

                # LineString
                lines.append(LineString((p1, p2)))
    return lines


def add_ids(frame, id_name=None):
    """add an idx column to a dataframe

    Parameters
    ----------
    frame : geopandas.GeoDataFrame
        dataframe of geometries
    id_name : str
        name of id column. Default is None.

    Returns
    -------
    frame : geopandas.GeoDataFrame
        updated dataframe of geometries
    """

    frame[id_name] = [idx for idx in range(frame.shape[0])]
    frame[id_name] = frame[id_name].astype(int)

    return frame


def generate_xyid(df=None, geom_type="node", geo_col=None):
    """Create a string xy id.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Geometry dataframe. Default is ``None``.
    geom_type : str
        Either ``'node'`` of ``'segm'``. Default is ``'node'``.
    geo_col : str
        Geometry column name. Default is ``None``.

    Returns
    -------
    xyid : list
        List of combined x-coord + y-coords strings.

    """

    xyid = []

    for idx, geom in enumerate(df[geo_col]):

        if geom_type == "segm":
            xys = ["x" + str(x) + "y" + str(y) for (x, y) in geom.coords[:]]
            xyid.append([idx, xys])

        # try to make the xyid from a polygon
        if geom_type == "node":
            try:
                xy = "x" + str(geom.centroid.x) + "y" + str(geom.centroid.y)

            # if the geometry is not polygon, but already point
            except AttributeError:
                try:
                    xy = "x" + str(geom.x) + "y" + str(geom.y)
                except:
                    print("geom:", type(geom))
                    print(dir(geom))
                    raise AttributeError(
                        "geom has neither attribute:\n"
                        + "\t\t- `.centroid.[coord]`\n"
                        + "\t\t- `.[coord]`"
                    )

            xyid.append([idx, [xy]])

    return xyid


def fill_frame(frame, full=False, idx="index", col=None, data=None, add_factor=0):
    """Fill a dataframe with a column of data.

    Parameters
    ----------
    frame : geopandas.GeoDataFrame
        Geometry dataframe.
    full : bool
        Create a new column (``False``) or a new frame (``True``).
        Default is ``False``.
    idx : str
        Index column name. Default is ``'index'``.
    col : str or list
         New column name(s). Default is ``None``.
    data : list *OR* dict
        List of data to fill the column. Default is ``None``.
        *OR*
        dict of data to fill the records. Default is ``None``.
    add_factor : int
        Used when dataframe index does not start at ``0``.
        Default is ``0``.

    Returns
    -------
    frame : geopandas.GeoDataFrame
        The updated geometry dataframe.

    """

    # create a full geopandas.GeoDataFrame
    if full:
        frame = geopandas.GeoDataFrame.from_dict(data, orient="index")

    # write a single column in a geopandas.GeoDataFrame
    else:
        frame[col] = numpy.nan
        for (k, v) in data:
            k += add_factor

            if col == "CC":
                frame.loc[frame[idx].isin(v), col] = k

            elif idx == "index":
                frame.loc[k, col] = str(v)

            else:
                frame.loc[(frame[idx] == k), col] = str(v)

        if col == "CC":
            frame[col] = frame[col].astype("category").astype(int)

    return frame


def _drop_geoms(net, gdf, geoms, series=False):
    """Drop a subset of geometries from a geopandas dataframe.

    Parameters
    ----------
    net : tigernet.Network
    gdf : geopandas.GeoDataFrame
        Dataframe of geometries to search.
    geoms : list
        Either a list or a list of lists.
    series : bool
        Search a geoseries. Default is ``False``.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Retained geometries in a dataframe.

    """

    if series:
        drop_geoms = geoms
    else:
        drop_geoms = set([item for sublist in geoms for item in sublist])

    if None in drop_geoms:
        drop_geoms.remove(None)

    gdf = gdf[~gdf.index.isin(drop_geoms)]

    return gdf


def extract_nodes(net):
    """Extract ``n_ids`` from line segments and return them in a geodataframe.

    Parameters
    ----------
    net : tigernet.Network

    Returns
    -------
    nodedf : geopandas.GeoDataFrame
        Node dataframe.

    """

    def _drop_covered_nodes(net, ndf):
        """Keep only the top node in stack of overlapping nodes.

        Parameters
        ----------
        net : tigernet.Network
        ndf : geopandas.GeoDataFrame
            Node dataframe.

        Returns
        -------
        nodedf : geopandas.GeoDataFrame
            Updated node dataframe.

        """

        scanned, drop = set(), set()

        # keep only the top node in a node stack.
        for n in ndf.index:
            if n not in scanned:
                unscanned = ndf[~ndf.index.isin(scanned)]
                xyid = unscanned[net.xyid][n]

                # intersecting node idxs
                iidxs = list(unscanned[unscanned[net.xyid] == xyid].index)
                iidxs.remove(n)
                if iidxs:
                    drop.update(iidxs)
                    scanned.update(drop)
                scanned.add(n)

        ndf = _drop_geoms(net, ndf, drop, series=True)
        ndf = add_ids(ndf, id_name=net.nid_name)

        return ndf

    sdf, nodes = net.s_data, []
    sdf_ring_flag = hasattr(sdf, "ring")

    # create n_ids and give the segment attribute data
    for seg in sdf.index:
        seggeom = sdf.loc[seg, net.geo_col]
        if sdf_ring_flag and sdf["ring"][seg] == "True":
            x, y = seggeom.coords.xy
            nodes.append(create_node(x, y))
        else:
            xy1, xy2 = seggeom.boundary[0], seggeom.boundary[1]
            nodes.extend([xy1, xy2])
    nodedf = geopandas.GeoDataFrame(geometry=nodes)
    nodedf = add_ids(nodedf, id_name=net.nid_name)

    if sdf.crs:
        nodedf.crs = sdf.crs

    # Give an initial string 'xy' ID
    prelim_xy_id = generate_xyid(df=nodedf, geom_type="node", geo_col=net.geo_col)
    nodedf = fill_frame(nodedf, idx=net.nid_name, col=net.xyid, data=prelim_xy_id)

    # drop all node but the top in the stack
    nodedf = _drop_covered_nodes(net, nodedf)
    nodedf.reset_index(drop=True, inplace=True)

    return nodedf


def associate(
    primary=None,
    secondary=None,
    assoc=None,
    initial_weld=False,
    net=None,
    df=None,
    ss=None,
):
    """Create 2 dictioanries of neighor relationships (``x2y`` and ``y2x``).
    *OR* create one list of ``x2y`` neighor relationships.

    Parameters
    ----------
    primary : list
        Primary data in the form: ``[idx, [xyID1, xyID2,...]]``.
        Default is ``None``.
    secondary : list
        Secondary data in the form: ``[idx, [xyID1, xyID2,...]]``.
        Default is ``None``.
    assoc : str
        Either ``'node2seg'`` or ``'seg2node'``. Default is ``None``.
    initial_weld : bool
        Welding subset of restricted access road segments. Used in
        ``cleanse_supercycle()``. Default is ``False``.
    net : tigernet.Network
    df : geopandas.GeoDataFrame
        restricted streets susbet dataframe. Default is ``None``.
    ss : geopandas.GeoDataFrame
        Subset of restricted streets susbet. Default is ``None``.

    Returns
    -------
    segm_dict : dict
        Neighoring elements in the form: ``{seg1, [node1, node2]}``.
    node_dict : dict
        Neighoring elements in the form: ``{node1, [seg1, seg2]}``.
    topos_list : list
        Neighoring elements in the form: ``[x1, [y1, y2]]``.

    """

    if initial_weld:
        segm_dict = {}

        for idx in df.index:
            neigh = [ss[net.tnidf][idx], ss[net.tnidt][idx]]
            segm_dict[ss[net.attr2][idx]] = neigh

        # get nodes
        ss_nodes = set()

        for sidx, nidx in list(segm_dict.items()):
            for nx in nidx:
                ss_nodes.add(nx)
        node_dict = {}

        for node_idx in ss_nodes:
            node_dict[node_idx] = set()
            for seg_idx, nodes_idx in list(segm_dict.items()):
                if node_idx in nodes_idx:
                    node_dict[node_idx].add(seg_idx)

        return segm_dict, node_dict

    topos_list = []

    for primary_idx, primary_info in enumerate(primary):
        topos = [primary_idx, []]

        for secondary_idx, secondary_info in enumerate(secondary):
            secondary_idxs = []

            # first and last point of the segment in string format for primary_info
            # in 'segm2node' and secondary_info in 'node2segm'
            if assoc == "segm2node":
                s10p10 = secondary_info[1][0] == primary_info[1][0]
                s10p11 = secondary_info[1][0] == primary_info[1][-1]
                if s10p10 or s10p11:
                    secondary_idxs.append(secondary_idx)

            if assoc == "node2segm":
                p10s10 = primary_info[1][0] == secondary_info[1][0]
                p10s11 = primary_info[1][0] == secondary_info[1][-1]
                if p10s10 or p10s11:
                    secondary_idxs.append(secondary_idx)

            topos[1].extend(secondary_idxs)

        topos_list.extend([topos])

    return topos_list


def get_neighbors(x2y, y2x, astype=None):
    """Get all neighboring graph elements of the same type.

    Parameters
    ----------
    x2y : list or dict
        Element type1 to element type2 crosswalk.
    y2x : list or dict
        Element type2 to element type1 crosswalk.
    astype : list or dict
        Return the lookup as either type. Default is ``None``.

    Returns
    -------
    x2x : list or dict
        Element type1 to element type1 crosswalk *OR* element type2 to element
        type2 crosswalk in the form: ``{x1, [x2,x3]}`` *OR* ``[x1, [x2,x3]]``.
    """

    if not astype:
        raise ValueError("The `astype` parameter must be set.")

    elif astype == dict:
        x2x = {}
        for k, vn in list(x2y.items()):
            x2x[k] = set()
            for v in vn:
                x2x[k].update(y2x[v])
                x2x[k].discard(k)

    elif astype == list:
        x2x = []
        for (k, vn) in x2y:
            x2x.append([k, set()])
            for v in vn:
                x2x[k][1].update(y2x[v][1])
                x2x[k][1].discard(k)
        x2x = [[k, list(v)] for (k, v) in x2x]

    else:
        raise TypeError(str(type), "not a valid type for `astype` parameter.")

    return x2x


def xwalk(df, c1=None, c2=None, stipulation=None, geo_col=None):
    """Create adjacency crosswalks as lists.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        The geometry dataframe.
    c1 : str
        The column name. Default is ``None``.
    c2 : str
        The column name. Default is ``None``.
    stipulation : str
        Default is ``None``.
    geo_col : str
        The name of the geometry column. Default is ``None``.

    Returns
    -------
    xw : list
        The adjacency crosswalk.

    """

    if c2 in ["nodeNeighs", "segmNeighs"]:
        xw = [[df[c1][ix], literal_eval(df[c2][ix])] for ix in df.index]

    if c2 in ["degree", "length", "TLID"]:
        xw = [[df[c1][ix], df[c2][ix]] for ix in df.index]

    if c2 == geo_col and not stipulation:
        xw = [[df[c1][ix], df[c2][ix]] for ix in df.index]

    if c2 == geo_col and stipulation == "coords":
        xw = [[df[c1][ix], df[c2][ix].coords[:]] for ix in df.index]

    return xw


def assert_2_neighs(net):
    """
    1. Raise an error if a segment has more that 2 neighbor nodes.
    2. If the road has one neighbor node then it is a ring road.
        In this case give the ring road a copy of the one nighbor.

    Parameters
    ----------
    net : tigernet.Network

    """

    more_2_neighs = [k for (k, v) in net.segm2node if len(v) > 2]

    if more_2_neighs:
        msg = "Adjacency value corruption. The segments listed below are incident with "
        msg += " more than two nodes.\n\nProblem segment IDs: %s" % str(more_2_neighs)
        raise AssertionError(msg)

    rings = [k for (k, v) in net.segm2node if len(v) < 2]

    if rings:
        for ring in rings:
            n1 = net.segm2node[ring][1][0]
            net.segm2node[ring][1] = [n1, n1]

    return net


def get_roots(adj):
    """Create a rooted object that stores connected components.

    Parameters
    ----------
    adj : list
        Record of adjacency.

    Returns
    -------
    ccs : list
        Rooted connected components

    """

    def _find_root_depth(obj: int, root: list) -> tuple:
        """Find the root and it's depth."""
        while obj != root[obj][0]:
            obj = root[obj][0]
        # (int, int)
        obj_rootdepth = obj, root[obj][1]
        return obj_rootdepth

    if type(adj) == dict:
        adj = [[idx, list(cc)] for idx, cc in list(adj.items())]

    # 1. set all objects within the root lookup to zero
    root = {i: (i, 0) for (i, neighs) in adj}

    # 2. iterate through each combination of neighbors
    for (i, neighs) in adj:
        for j in neighs:

            # 2-A. find the root of i and its depth
            root_of_i, depth_of_i = _find_root_depth(i, root)

            # 2-B. find the root of j and its depth
            root_of_j, depth_of_j = _find_root_depth(j, root)

            # 2-C. set each object as either root or non root
            if root_of_i != root_of_j:
                _min, _max = root_of_i, root_of_j
                if depth_of_i > depth_of_j:
                    _min, _max = root_of_j, root_of_i
                root[_max] = _max, max(root[_min][1] + 1, root[_max][1])
                root[_min] = (root[_max][0], -1)

    # 3. create empty list entry for each rooted connected component
    ccs = {i: [] for (i, neighs) in adj if root[i][0] == i}

    # 4. fill each list with the components
    [ccs[_find_root_depth(i, root)[0]].append(i) for (i, neighs) in adj]
    ccs = [list(cc) for cc in list(ccs.items())]

    return ccs


def get_cc_len(net, len_col=None):
    """return the geodataframe with the length of each associated
    connected component in a new column.

    Parameters
    ----------
    net : tigernet.Network
    len_col : str
        The name of the length column. Default is ``None``.

    Returns
    -------
    cc_lens : dict
        ``{ID:length}`` for each connected component in graph.

    """

    net.s_data["ccLength"] = numpy.nan
    cc_lens = {}

    for (k, v) in net.segm_cc:
        new_v, segment_ids = v, v
        new_v = net.s_data[net.s_data[net.sid_name].isin(new_v)]
        new_v = new_v[len_col].sum()
        net.s_data.loc[net.s_data[net.sid_name].isin(v), "ccLength"] = new_v
        cc_lens[k] = [new_v, segment_ids]

    return cc_lens


def get_largest_cc(ccs, smallkeys=True):
    """Return the largest component object.

    Parameters
    ----------
    ccs : list
        A list of connected components.
    smallkeys : bool
        Return the keys of the smaller components. Default is ``True``.

    Returns
    -------
    results : list, tuple
        Either ``[largestKey, largestValues]`` if ``smallkeys=False`` or
        ``[largestKey, largestValues], non_largest`` if ``smallkeys=True``.

    """

    largest = max(ccs, key=lambda k: len(k[1]))
    largest_key = largest[0]
    largest_values = largest[1]

    results = [largest_key, largest_values]

    if smallkeys:
        non_largest = []
        for (cck, ccvs) in ccs:
            if cck is not largest_key:
                non_largest.extend(ccvs)
        results = [largest_key, largest_values], non_largest

    return results


def update_adj(net, seg_keys, node_keys):
    """Update adjacency relationships between segments and nodes.

    Parameters
    ----------
    net : tigernet.Network
    seg_keys : list
        Segment keys to remove from adjacency.
    node_keys : list
        Node keys to remove from adjacency.

    """

    # update all crosswalk dictionaries
    net.segm2segm = remove_adj(net.segm2segm, seg_keys)
    net.segm2node = remove_adj(net.segm2node, seg_keys)
    net.node2node = remove_adj(net.node2node, node_keys)
    net.node2segm = remove_adj(net.node2segm, node_keys)

    # Keep only the largest connected component
    net.segm_cc = net.largest_segm_cc
    net.node_cc = net.largest_node_cc

    # Set component ID to dataframe
    net.s_data = net.s_data[net.s_data[net.sid_name].isin(net.segm_cc[1])]
    net.s_data.reset_index(drop=True, inplace=True)
    net.n_data = net.n_data[net.n_data[net.nid_name].isin(net.node_cc[1])]
    net.n_data.reset_index(drop=True, inplace=True)


def remove_adj(e2e, remove):
    """Remove adjacent elements from list of IDs.

    Parameters
    ----------
    e2e : list
        The element-to-element relationship list. This is either an
        'x2x' relationship or 'x2y' relationship.
    remove : list
        The keys to remove from list.

    Returns
    -------
    e2e : list
        The updated e2e relationship list.

    """
    e2e = [[k, vs] for (k, vs) in e2e if k not in set(remove)]
    return e2e


def geom_assoc(net, coords=False):
    """Associate nodes and segments with geometry or coordinates.

    Parameters
    ----------
    net : tigernet.Network
    coords : bool
        Associate with coordinates (``True``). Default is ``False``.

    """

    _kws = {"c2": net.geo_col, "geo_col": net.geo_col}
    if not coords:
        net.segm2geom = xwalk(net.s_data, c1=net.sid_name, **_kws)
        net.node2geom = xwalk(net.n_data, c1=net.nid_name, **_kws)
    else:
        _kws.update({"stipulation": "coords"})
        net.segm2coords = xwalk(net.s_data, c1=net.sid_name, **_kws)
        net.node2coords = xwalk(net.n_data, c1=net.nid_name, **_kws)


def calc_valency(net, col=None):
    """Calculate the valency of each node and return a lookup.

    Parameters
    ----------
    net : tigernet.Network
    col : str
        The node neighbors column. Default is ``None``.

    Returns
    -------
    n2d : list
        The node-to-degree lookup.

    """

    n2d = []
    for (node, segs) in net.node2segm:
        loops = 0
        for s in segs:
            segv = net.s_data[net.sid_name] == s
            neighs = literal_eval(net.s_data.loc[segv, col].values[0])
            if neighs[0] != neighs[1]:
                continue
            if neighs[0] == neighs[1]:
                loops += 1
        degree = len(segs) + loops
        n2d.append([node, [degree]])

    return n2d


def branch_or_leaf(net, geom_type=None):
    """Define each graph element (either segment or node) as either
    branch or leaf. Branches are nodes with degree 2 or higher, or
    segments with both incident nodes of degree 2 or higher
    (a.k.a. internal elements). Leaves are nodes with degree 1 or
    less, or segments with one incident node of degree 1 (a.k.a
    external elements). Branches are 'core' elements, while leaves
    can be thought of as 'dead-ends'.

    Parameters
    ----------
    net : tigernet.Network
    geom_type : str
        ``'segm'`` or ``'node'``.

    Returns
    -------
    geom2ge : list
        Geometry ID-to-graph element type crosswalk.

    """

    if geom_type == "segm":
        id_list = net.s_ids
    elif geom_type == "node":
        id_list = net.n_ids
    else:
        msg = "'geom_type' of %s not valid." % geom_type
        raise ValueError(msg)

    geom2ge = []
    for idx in id_list:
        if geom_type == "segm":
            n1, n2 = net.segm2node[idx][1][0], net.segm2node[idx][1][1]
            n1d, n2d = net.node2degree[n1][1][0], net.node2degree[n2][1][0]

            if n1d == 1 or n2d == 1:
                graph_element = "leaf"
            else:
                graph_element = "branch"
        if geom_type == "node":
            nd = net.node2degree[idx][1][0]
            if nd == 1:
                graph_element = "leaf"
            else:
                graph_element = "branch"
        geom2ge.append([idx, graph_element])

    return geom2ge


def simplify(net):
    """Remove all non-articulation objects.

    Parameters
    ----------
    net : tigernet.Network

    Returns
    -------
    segs : geopandas.GeoDataFrame
        Simplified segments dataframe.

    """

    # locate all non-articulation points
    na_objs = _locate_naps(net)

    # remove all non-articulation points
    segs = _simplifysegs(net, na_objs)

    return segs


def _locate_naps(net):
    """Locate all non-articulation points in order to simplfy graph.

    Parameters
    ----------
    net : tigernet.Network

    Returns
    -------
    napts : dict
        Dictionary of non-articulation points and segments.

    """

    # subset only degree-2 nodes
    degree_two_nodes = set([n for (n, d) in net.node2degree if 2 in d])

    # recreate n2n xwalk
    new_n2n = {k: v for (k, v) in net.node2node}
    two2two = {k: new_n2n[k] for k in degree_two_nodes}

    # get set intersection of degree-2 node neighbors
    for k, vs in list(two2two.items()):
        two2two[k] = list(degree_two_nodes.intersection(set(vs)))

    # convert back to list
    two2two = [[k, vs] for k, vs in list(two2two.items())]

    # created rooted non-articulation nodes object
    rooted_napts, napts, napts_count = get_roots(two2two), {}, 0
    for (k, v) in rooted_napts:
        napts_count += 1
        napts[napts_count] = {net.nid_name: v}

    # add segment info to rooted non-articulation point object
    for napt_count, napt_info in list(napts.items()):
        napt = []
        for napt_node in napt_info[net.nid_name]:
            napt.extend([i[1] for i in net.node2segm if i[0] == napt_node])

        # if more than one pair of segments in napt
        napt = set([seg for segs in napt for seg in segs])
        napts[napt_count].update({net.sid_name: napt})

    return napts


def _simplifysegs(net, na_objs):
    """Drop nodes and weld bridge segments.

    Parameters
    ----------
    net : tigernet.Network
    na_objs : dict
        Non-articulation point information.

    Returns
    -------
    net.s_data : geopandas.GeoDataFrame
        Simplified segments dataframe.
    """

    nsn = net.sid_name

    # for each bridge
    for na_objs_sidx, na_objs_info in list(na_objs.items()):

        # get the dominant SegIDX and dataframe index
        inherit_attrs_from, idx = _get_hacky_index(net, na_objs_info)

        # set total length to 0 and instantiate an empty segments list
        total_length, geoms = 0.0, []

        # add the length of each segment to total_length and add
        # the segment to geoms
        for segm in na_objs_info[nsn]:
            seg_loc = net.s_data[nsn] == segm
            total_length += net.s_data.loc[seg_loc, net.len_col].squeeze()
            geom = net.s_data.loc[seg_loc, net.geo_col].squeeze()
            geoms.append(geom)

        # take the dominant line segment id out of the `remove` list
        na_objs_info[nsn].remove(inherit_attrs_from)

        # add new total length cell value
        net.s_data.loc[idx, net.len_col] = total_length

        # add new welded line segment of dominant and non-dominant lines
        welded_line = _weld_MultiLineString(geoms)
        net.s_data.loc[idx, net.geo_col] = welded_line

        # remove all non-dominant line segments from the dataframe
        net.s_data = net.s_data[~net.s_data[nsn].isin(na_objs_info[nsn])]

    return net.s_data


def _get_hacky_index(net, ni):
    """VERY hacky function to get back dataframe index due to 
    trouble with using -->
    df.loc[(df[sidx] == segidx), 'geometry'\
                                  = _weldMultiLineString(geoms)
    *** See issue at
    https://github.com/pandas-dev/pandas/issues/28924
    
    *** see also:
        _weld_MultiLineString()
        tigernet.TigerNetPoints.snap_to_nearest._record_snapped_points._casc2point
    
    Parameters
    ----------
    net : tigernet.Network
    ni : dict
        Non-articulation point information.
    
    Returns
    -------
    inherit_attrs_from : int
        Segment ID.
    idx : int
        Dataframe index value.

    """

    df, lc, sid = net.s_data, net.len_col, net.sid_name

    # get maximum length
    _df = df[df[sid].isin(ni[sid])]
    max_len = max([_df.loc[(_df[sid] == segm), lc].squeeze() for segm in ni[sid]])

    # inherit attributes from the longest segment (SegIDX)
    inherit_attrs_from = _df.loc[(_df[lc] == max_len), sid].squeeze()
    try:
        try:
            if inherit_attrs_from.shape[0] > 1:
                # inherit_attrs_from is: geopandas.GeoDataFrame
                inherit_attrs_from = inherit_attrs_from[:1].index[0]
        except IndexError:
            # inherit_attrs_from is: numpy.int64
            pass
    except AttributeError:
        # inherit_attrs_from is: int
        pass

    # get the df index of 'SegIDX == inherit_attrs_from' and maxLen
    dominant, longest = (_df[sid] == inherit_attrs_from), (_df[lc] == max_len)
    idx = _df.loc[dominant & longest].index[0]

    return inherit_attrs_from, idx


def _weld_MultiLineString(multilinestring, weld_multi=True, skip_restr=True):
    """weld a shapely.MultiLineString into a shapely.LineString

    Parameters
    ----------
    multilinestring : shapely.geometry.MultiLineString
        Segment (collection) to weld.
    weld_multi :bool
        If welded line is still a multiline segment, then determine if
        the segments of the multiline are almost equal. Default is ``True``.
    skip_restr : bool
        Skip re-welding restricted segments. Default is ``True``.

    Returns
    -------
    welded : shapely.geometry.LineString
        freshly welded segment (collection).
    """

    welded = linemerge(multilinestring)

    # Due to minute rounding (.00000001 meters) some line vertices can
    # be off, thus creating a MultiLineString where shapely thinks two
    # LineString objects don't actually touch where, in fact, they do.
    # The following loop iterates through each pair of LineString
    # objects sequentially to  determine if their endpoints are
    # 'almost equal' instead of exactly equal. When the endpoints are
    # 'almost equal' the starting point of the second line is duplicated
    # as the ending point of the first line before the lines are
    # welded together.
    if type(welded) == MultiLineString and weld_multi and skip_restr:
        line_count, new_lines = len(welded), {}
        for line1 in range(line_count):
            for line2 in range(line1 + 1, line_count):

                # geometries
                L1, L2 = welded[line1], welded[line2]
                # starting and endpoints
                sp1, ep1 = L1.boundary[0], L1.boundary[1]
                sp2, ep2 = L2.boundary[0], L2.boundary[1]

                # if equal move along
                if ep1.equals(sp2) or sp1.equals(ep2):
                    continue

                # if either sets are almost equal pass along the
                # altered first line and the original second line
                if ep1.almost_equals(sp2) or sp1.almost_equals(ep2):
                    if ep1.almost_equals(sp2):
                        new_line = LineString(L1.coords[:-1] + L2.coords[:1])
                    if sp1.almost_equals(ep2):
                        new_line = LineString(L2.coords[-1:] + L1.coords[1:])
                    new_lines[line1] = new_line

        # convert welded multiline to list
        welded = list(welded)
        for idx, line in list(new_lines.items()):
            welded[idx] = line

        # re-weld
        welded = linemerge(welded)

    return welded


def label_rings(df, geo_col=None):
    """Label each line segment as ring (``'True'``) or not (``'False'``).

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Dataframe of geometries.
    geo_col : str
        Geometry column name. Default is ``None``.

    Returns
    -------
    df : geopandas.GeoDataFrame
        Updated dataframe of geometries.

    """

    df["ring"] = ["False"] * df.shape[0]
    for idx in df.index:
        if df[geo_col][idx].is_ring:
            df["ring"][idx] = "True"

    return df


def ring_correction(net, df):
    """Ring roads should start and end with the point at which it intersects with
    another road segment. This algorithm find instances where rings roads are digitized
    incorrectly, which results in ring roads having their endpoints somewhere in the
    middle of the line, then corrects the loop by updating the geometry. Length and
    attributes of the original line segment are not changed.

    Parameters
    ----------
    net : tigernet.Network
    df : geopandas.GeoDataFrame
        Dataframe of road segments.

    Returns
    -------
    df : geopandas.GeoDataFrame
        Updated dataframe of road segments.

    """

    # subset only ring roads
    rings_df = df[df["ring"] == "True"]
    ringsidx, corrected_rings = rings_df.index, 0

    for idx in ringsidx:
        LOI = rings_df[net.geo_col][idx]

        # get idividual ring road - normal road pairs intersection
        i_geoms = get_intersecting_geoms(net, df1=df, geom1=idx, wbool=False)
        i_geoms = i_geoms[i_geoms.index != idx]

        # rings that are not connected to the network will be removed
        if i_geoms.shape[0] < 1:
            continue
        node = i_geoms[net.geo_col][:1].intersection(LOI).values[0]

        # if pre cleaned and segments still overlap
        if type(node) != Point:
            continue

        node_coords = list(zip(node.xy[0], node.xy[1]))
        line_coords = list(zip(LOI.coords.xy[0], LOI.coords.xy[1]))

        # if problem ring road
        # (e.g. the endpoint is not the intersection)
        if node_coords[0] != line_coords[0]:
            updated_line = _correct_ring(node_coords, line_coords)

            # update dataframe record
            df[net.geo_col][idx] = updated_line
            corrected_rings += 1

    df.reset_index(drop=True, inplace=True)
    df = add_ids(df, id_name=net.sid_name)

    # add updated xyid
    segm2xyid = generate_xyid(df=df, geom_type="segm", geo_col=net.geo_col)
    df = fill_frame(df, col=net.xyid, data=segm2xyid)

    # corrected ring road count
    net.corrected_rings = corrected_rings

    return df


def _correct_ring(node_coords, line_coords):
    """Helper function for ``ring_correction()``.

    Parameters
    ----------
    node_coords : list
        xy tuple for a node.
    line_coords  : list
        All xy tuples for a line.

    Returns
    -------
    updated_line : shapely.LineString
        Ring road updated so it begins and ends at the intersecting node.

    """

    for itemidx, coord in enumerate(line_coords):
        # find the index of the intersecting coord in the line
        if coord == node_coords[0]:
            break

    # adjust the line coordinates for the true ring start/end
    updated_line = LineString(line_coords[itemidx:] + line_coords[1 : itemidx + 1])

    return updated_line


def get_intersecting_geoms(net, df1=None, geom1=None, df2=None, geom2=None, wbool=True):
    """Return the subset of intersecting geometries from within a geodataframe.

    Parameters
    ----------
    net : tigernet.Network
    df1 : geopandas.GeoDataFrame
        Primary dataframe. Default is ``None``.
    geom1 : int
        Geometry index. Default is ``None``.
    df2 : geopandas.GeoDataFrame
        Secondary dataframe . Default is ``None``.
    geom2 : int
        Geometry index. Default is ``None``.
    wbool : bool
        Return a boolean object for intersections. Default is ``True``.
    geo_col : str
        Geometry column name. Default is ``None``.

    Returns
    -------
    i_geom : geopandas.GeoDataFrame
        Intersecting geometry subset.
    i_bool : numpy.array
        Optional return of booleans for intersecting geoms.

    """

    # if there *IS NO* dataframe 2 in play
    if not hasattr(df2, net.geo_col):
        i_bool = df1.intersects(df1[net.geo_col][geom1])

    # if there *IS* dataframe 2 in play
    else:
        i_bool = df1.intersects(df2[net.geo_col][geom2])
    i_geom = df1[i_bool]

    if wbool:
        return i_bool, i_geom

    else:
        return i_geom


def create_node(x, y):
    """Create a node along the network.

    Parameters
    ----------
    x : {float, int}
        The x coordinate of a point.
    y : {float, int}
        The y coordinate of a point.

    Returns
    -------
    _node : shapely.geoemtry.Point
        Instantiated node.

    """

    _node = Point(list(zip(x, y))[0])

    return _node


def euc_calc(net, col=None):
    """Calculate the euclidean distance between two line endpoints
    for each line in a set of line segments.

    Parameters
    ----------
    net : tigernet.Network
    col : str
        new column name. Default is None.

    Returns
    -------
    df : geopandas.GeoDataFrame
        updated segments dataframe

    """

    net.s_data[col] = numpy.nan
    for (seg_k, (n1, n2)) in net.segm2node:
        p1, p2 = net.node2coords[n1][1][0], net.node2coords[n2][1][0]
        ed = _euc_dist(p1, p2)
        net.s_data.loc[(net.s_data[net.sid_name] == seg_k), col] = ed

    return net.s_data


def _euc_dist(p1, p2):
    """Calculate the euclidean distance between two line endpoints.

    Parameters
    ----------
    p1 : {float, int}
        The start point of a line.
    p2 : {float, int}
        The end point of a line.

    Returns
    -------
    euc : float
        The euclidean distance between two line endpoints.

    """

    euc = numpy.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    return euc


###############################################################################
################ TIGER/Line clean up functionality ############################
###############################################################################


def tiger_netprep(net, in_file=None, calc_len=False):
    """Scrub a raw TIGER/Line EDGES file at the county level to prep for network.

    Parameters
    ----------
    net : tigernet.Network
    calc_len : bool
        Calculate length and add column. Default is ``False``.

    """

    # Reproject roads and subset by road type
    initial_subset(net, net.s_data, calc_len=calc_len)

    # Correcting ring roads
    net.s_data = ring_correction(net, net.s_data)

    # Cleanse SuperCycle -- Before splitting weld interstate segment pieces
    cleanse_kws = {"calc_len": calc_len, "inherit_attrs": True}
    net.s_data = cleanse_supercycle(net, net.s_data, **cleanse_kws)


def initial_subset(net, raw_file, calc_len=False):
    """Initalize a network data cleanse from raw tiger line files

    Parameters
    ----------
    net : tigernet.Network
    raw_data : str
        Directory and file name for to find raw tiger data
    calc_len : bool

    """

    # Read in raw TIGER street data
    if not net.is_gdf:
        net.s_data = geopandas.read_file(net.s_data)

    if calc_len:
        net.s_data = add_length(net.s_data, len_col=net.len_col, geo_col=net.geo_col)

    # remove trouble maker segments
    if net.discard_segs:
        net.s_data = net.s_data[~net.s_data[net.attr2].isin(net.discard_segs)]

    # Add three new MTFCC columns for feature class, description, and rank
    if net.mtfcc_types:
        mtfcc_cols = ["FClass", "Desc"]
        for c in mtfcc_cols:
            net.s_data[c] = [
                net.mtfcc_types[mtfcc][c] for mtfcc in net.s_data[net.attr1]
            ]

    # Subset roads
    if "FClass" in net.s_data.columns and net.mtfcc_discard:
        _kws = {"column": "FClass", "mval": net.mtfcc_discard, "oper": "out"}
        net.s_data = record_filter(net.s_data, **_kws)
    net.s_data.reset_index(drop=True, inplace=True)
    net.s_data = label_rings(net.s_data, geo_col=net.geo_col)

    # create segment xyID
    segm2xyid = generate_xyid(df=net.s_data, geom_type="segm", geo_col=net.geo_col)
    net.s_data = fill_frame(net.s_data, col=net.xyid, data=segm2xyid)


def add_length(frame, len_col=None, geo_col=None):
    """Add length column to a dataframe.

    Parameters
    ----------
    frame : geopandas.GeoDataFrame
        Dataframe of geometries.
    len_col : str
        Length column name in dataframe. Default is ``None``.
    geo_col : str
        Geometry column name. Default is ``None``.

    Returns
    -------
    frame : geopandas.GeoDataFrame
        Updated dataframe of geometries.

    """

    if list(frame.columns).__contains__(len_col):
        frame = frame.drop(len_col, axis=1)
    frame[len_col] = [frame[geo_col][idx].length for idx in frame.index]

    return frame


def record_filter(df, column=None, sval=None, mval=None, oper=None):
    """Used in phase 2 with incidents

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Dataframe of incident records.
    oper : operator object, str
        ``{(operator.eq, operator.ne)``, ``('in', 'out')}``.
    sval : str, int, float, bool, etc.
        Single value to filter.
    mval : list
        Multiple values to filter.

    Returns
    -------
    df : geopandas.GeoDataFrame
        dataframe of incident records

    """

    # use index or specific column
    if column == "index":
        frame_col = df.index
    else:
        frame_col = df[column]

    # single value in column
    if not sval == None:
        return df[oper(frame_col, sval)].copy()

    # multiple values in column
    if not mval == None:
        if oper == "in":
            return df[frame_col.isin(mval)].copy()
        if oper == "out":
            return df[~frame_col.isin(mval)].copy()


def label_rings(df, geo_col=None):
    """Label each line segment as ring (``True``) or not (``False``).

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Dataframe of geometries.
    geo_col : str
        Geometry column name. Default is ``None``.

    Returns
    -------
    df : geopandas.GeoDataFrame
        Dataframe of geometries.

    """

    df["ring"] = ["False"] * df.shape[0]
    for idx in df.index:
        if df[geo_col][idx].is_ring:
            df["ring"][idx] = "True"

    return df


def cleanse_supercycle(net, gdf, inherit_attrs=False, calc_len=True):
    """One iteration of a cleanse supercycle; then repeat as necessary.
    1. Drop equal geoms; 2. Drop contained geoms; 3. Split line segments

    Parameters
    ----------
    net : tigernet.Network
    gdf : geopandas.GeoDataFrame
        Streets dataframe.
    inherit_attrs : bool
        Inherit attributes from the dominant line segment. Default is ``False``.
    calc_len : bool
        Calculate length and add column. Default is ``True``.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Updated streets dataframe.

    """

    # weld together restricted segments (e.g. Interstates)
    net.s_data = restriction_welder(net)
    # split segments at known intersections
    net.s_data = line_splitter(net, calc_len=calc_len, inherit_attrs=inherit_attrs)

    # Re-lablel Rings
    net.s_data = label_rings(net.s_data, geo_col=net.geo_col)
    net.s_data.reset_index(inplace=True, drop=True)

    return net.s_data


def restriction_welder(net):
    """Weld each set of restricted segments (e.g. interstates).

    Parameters
    ----------
    net : tigernet.Network

    Returns
    -------
    net.s_data : geopandas.GeoDataFrame
        Updated streets dataframe.

    """

    # make restricted subset
    restr_ss = net.s_data[net.s_data[net.attr1] == net.mtfcc_split]

    try:
        restr_names = [str(grp) for grp in restr_ss[net.mtfcc_split_grp].unique()]
    except KeyError:
        return

    # create a sub-subset for each group (e.g. interstate)
    for grp in restr_names:
        ss = restr_ss[restr_ss[net.mtfcc_split_grp] == grp]

        # get restriction segments to restriction nodes lookup dict
        # and restriction nodes to restriction segments lookup dict
        s2n, n2s = associate(initial_weld=True, net=net, df=restr_ss, ss=ss)

        # x2x topologies
        s2s = get_neighbors(s2n, n2s, astype=dict)
        n2n = get_neighbors(n2s, s2n, astype=dict)

        # get rooted connected components
        s2s_cc = get_roots(s2s)

        # weld together segments from each component of the group
        for cc in s2s_cc:
            keep_id, all_ids = cc[0], cc[1]
            drop_ids = copy.deepcopy(all_ids)
            drop_ids.remove(keep_id)

            # subset of specific segment to weld
            weld_ss = ss[ss[net.attr2].isin(all_ids)]
            weld = list(weld_ss.geometry)
            weld = _weld_MultiLineString(weld, skip_restr=net.skip_restr)

            # if the new segment if a LineString set the new, welded
            # geometry to the `keep_id` index of the dataframe
            if type(weld) == LineString:
                index = weld_ss.loc[(weld_ss[net.attr2] == keep_id)].index[0]
                net.s_data.loc[index, net.geo_col] = weld

            # if the weld resulted in a MultiLineString remove ids from
            # from `drop_ids` and set to new for each n+1 new segment.
            if type(weld) == MultiLineString:
                unique_segs = len(weld)
                keeps_ids = [keep_id] + drop_ids[: unique_segs - 1]
                index = list(weld_ss[weld_ss[net.attr2].isin(keeps_ids)].index)
                for idx, seg in enumerate(weld):
                    net.s_data.loc[index[idx], net.geo_col] = seg
                for idx in keeps_ids:
                    if idx in drop_ids:
                        drop_ids.remove(idx)

            # remove original segments used to create the new, welded
            # segment(s) from the full segments dataframe
            net.s_data = net.s_data[~net.s_data[net.attr2].isin(drop_ids)]

    net.s_data.reset_index(inplace=True, drop=True)

    return net.s_data


def line_splitter(net, inherit_attrs=False, calc_len=False, road_type="MTFCC"):
    """Top-level function for spliting line segments.

    Parameters
    ----------
    net : tigernet.Network
    inherit_attrs : bool
        Inherit attributes from the dominant line segment. Default is ``False``.
    calc_len : bool
        Calculate length and add column. Default is ``False``.
    road_type : str
        Column to use for grouping road types. Default is ``'MTFCC'``.

    Returns
    -------
    split_lines : geopandas.GeoDataFrame
        All line segments including unsplit lines.

    """

    # it `net.mtfcc_split` is string put it into a list
    if not hasattr(net.mtfcc_split, "__iter__"):
        net.mtfcc_split = [net.mtfcc_split]

    # create subset of segments to split and not split
    if net.mtfcc_split_by and net.mtfcc_split:
        if type(net.mtfcc_split) == list:
            subset_codes = net.mtfcc_split_by + net.mtfcc_split
        elif type(net.mtfcc_split) == str:
            subset_codes = net.mtfcc_split_by + [net.mtfcc_split]
        non_subset = net.s_data[~net.s_data[road_type].isin(subset_codes)]
        net.s_data = net.s_data[net.s_data[road_type].isin(subset_codes)]

    if inherit_attrs:
        drop_cols = [net.len_col, net.geo_col, net.xyid, "ring"]
        attrs = [col for col in net.s_data.columns if not drop_cols.__contains__(col)]
        attr_vals = {attr: [] for attr in attrs}

    # Iterate over dataframe to find intersecting and split
    split_lines = []
    count_lines_split = 0
    for loi_idx in net.s_data.index:

        # Working with TIGER/Line *EDGES*
        if net.mtfcc_split_by and net.mtfcc_split:

            # if a line segment used for splitting
            # but not to be split itself
            if net.s_data[road_type][loi_idx] not in net.mtfcc_split:
                split_lines.extend([net.s_data[net.geo_col][loi_idx]])

                # fill dictionary with attribute values
                if inherit_attrs:
                    for attr in attrs:
                        fill_val = [net.s_data[attr][loi_idx]]
                        attr_vals[attr].extend(fill_val)
                continue

        # get segs from the dataset that intersect
        # with the Line Of Interest
        intersecting = get_intersecting_geoms(
            net, df1=net.s_data, geom1=loi_idx, wbool=False
        )
        intersecting = intersecting[intersecting.index != loi_idx]

        # Working with TIGER/Line *ROADS*
        if not net.mtfcc_split_by and not net.mtfcc_split:

            # if the LOI is an interstate only pass in
            # the ramps for splitting
            if net.s_data[road_type][loi_idx] == net.mtfcc_intrst:
                intersecting = intersecting[
                    (intersecting[road_type] == net.mtfcc_ramp)
                    | (intersecting[road_type] == net.mtfcc_serv)
                    | (intersecting[road_type] == net.mtfcc_intrst)
                ]

            # if LOI not ramp and interstates in dataframe
            # filter them out
            elif (
                list(intersecting[road_type]).__contains__(net.mtfcc_intrst)
                and net.s_data[road_type][loi_idx] != net.mtfcc_ramp
            ):
                intersecting = intersecting[intersecting[road_type] != net.mtfcc_intrst]

        # if There are no intersecting segments
        if intersecting.shape[0] == 0:
            continue

        # ring road bool
        ring_road = literal_eval(net.s_data["ring"][loi_idx])

        # actual line split call happens here
        new_lines = _split_line(
            net.s_data[net.geo_col][loi_idx],
            loi_idx,
            df=intersecting,
            ring_road=ring_road,
            geo_col=net.geo_col,
        )
        n_lines = len(new_lines)
        if n_lines > 1:
            count_lines_split += 1
        split_lines.extend(new_lines)

        # fill dictionary with attribute values
        if inherit_attrs:
            for attr in attrs:
                fill_val = [net.s_data[attr][loi_idx]]
                attr_vals[attr].extend(fill_val * n_lines)

    # create dataframe
    split_lines = geopandas.GeoDataFrame(split_lines, columns=[net.geo_col])

    # fill dataframe with attribute values
    if inherit_attrs:
        for attr in attrs:
            split_lines[attr] = attr_vals[attr]

    if calc_len:
        split_lines = add_length(split_lines, len_col=net.len_col, geo_col=net.geo_col)

    # recombine EDGES subset and non subset segment lists
    if net.mtfcc_split_by and net.mtfcc_split:
        # combine newly split suset segments with all segments
        split_lines = split_lines.append(non_subset, sort=False)
        split_lines.reset_index(inplace=True, drop=True)
    split_lines = add_ids(split_lines, id_name=net.sid_name)

    # add updated xyid
    segm2xyid = generate_xyid(df=split_lines, geom_type="segm", geo_col=net.geo_col)
    split_lines = fill_frame(split_lines, col=net.xyid, data=segm2xyid)
    split_lines = label_rings(split_lines, geo_col=net.geo_col)

    # number of lines split
    net.lines_split = count_lines_split

    return split_lines


def _split_line(loi, idx, df=None, geo_col=None, ring_road=False):
    """middle level function for spliting line segements

    Parameters
    ----------
    loi : shapely.LineString
        The line segment in question.
    idx : int
        The index number of the LOI.
    df : geopandas.GeoDataFrame
        The dataframe of line segments.
    ring_road : bool
        (``True``) if ring road. (``False``) if not. Default is ``False``.
    geo_col : str
        The geometry column name. Default is ``None``.

    Returns
    -------
    new_lines : list
        A list of new lines generated from splitting.

    """

    intersectinglines = df[df.index != idx]  # all lines not LOI

    # Unary Union for intersection determination
    intersectinglines = intersectinglines[geo_col].unary_union

    # Intersections of LOI and the Unary Union
    breaks = loi.intersection(intersectinglines)

    # find and return points on the line to split if any exist
    unaltered, breaks, ring_endpoint, basic_ring, complex_ring = _find_break_locs(
        loi=loi, breaks=breaks, ring_road=ring_road
    )

    if unaltered:
        return unaltered

    # Line breaking
    if not type(breaks) == list:
        breaks = [breaks]

    new_lines = _create_split_lines(
        breaks=breaks,
        loi=loi,
        ring_road=ring_road,
        basic_ring=basic_ring,
        complex_ring=complex_ring,
        ring_endpoint=ring_endpoint,
    )

    return new_lines


def _create_split_lines(
    breaks=None,
    ring_endpoint=None,
    loi=None,
    ring_road=False,
    basic_ring=False,
    complex_ring=False,
):
    """Deep function from splitting a single line segment along break points.

    Parameters
    ----------
    breaks : list
        The point to break a line. Default is ``None``.
    ring_endpoint : shapely.Point
        The endpoint of a ring road. Default is ``None``.
    loi : shapely.geometry.LineString
        The line of interest.
    ring_road : bool
        Boolean for 'is' or 'is not' a ring road. Default is ``False``.
    basic_ring : bool
        is or is not a basic ring road. This indicates a 'normal' ring
        road in which there is one endpoint. Default is False.
    complex_ring : bool
        is or is not a complex ring road. This indicates a any situation
        not deemed a 'basic' ring. Default is False.

    Returns
    -------
    new_lines : list
        A list of new lines generated from splitting.

    """

    points = [Point(xy) for xy in breaks]

    # First coords of line
    coords = list(loi.coords)

    # Keep list coords where to cut (cuts = 1)
    cuts = [0] * len(coords)
    cuts[0] = 1
    cuts[-1] = 1

    # Add the coords from the points
    coords += [list(p.coords)[0] for p in points]
    cuts += [1] * len(points)

    # Calculate the distance along the line for each point
    dists = [loi.project(Point(p)) for p in coords]

    # sort the coords/cuts based on the distances
    # see http://stackoverflow.com/questions/6618515/
    #     sorting-list-based-on-values-from-another-list
    coords = [p for (d, p) in sorted(zip(dists, coords))]
    cuts = [p for (d, p) in sorted(zip(dists, cuts))]
    if ring_road:  # ensure there is an endpoint for rings
        if basic_ring:
            coords = ring_endpoint + coords + ring_endpoint
        if complex_ring:
            archetelos = [loi.coords[0]]  # beginning and ending of ring
            coords = archetelos + coords + archetelos
        cuts = [1] + cuts + [1]

    # generate the lines
    if cuts[-1] != 1:  # ensure there is an endpoint for rings
        cuts += [1]
    new_lines = []

    for i in range(len(coords) - 1):
        if cuts[i] == 1:
            # find next element in cuts == 1 starting from index i + 1
            j = cuts.index(1, i + 1)
            new_line = LineString(coords[i : j + 1])
            if new_line.is_valid:
                new_lines.append(new_line)
    return new_lines


def _find_break_locs(loi=None, breaks=None, ring_road=False):
    """Locate points along a line segment where breaks need to be made.

    Parameters
    ----------
    loi : shapely.geometry.LineString
        The line of interest.
    breaks : list
        The point to break a line. Default is ``None``.
    ring_road : bool
        Boolean for 'is' or 'is not' a ring road. Default is ``False``.

    Returns
    -------
    unaltered : None or list
        List of one unaltered LineString.
    breaks : None of list
        The point to break a line. Default is ``None``.
    ring_endpoint : shapely.Point
        The endpoint of a ring road. Default is ``None``.
    basic_ring : bool
        is or is not a basic ring road. This indicates a 'normal' ring
        road in which there is one endpoint.
    complex_ring : bool
        is or is not a complex ring road. This indicates a any
        situation not deemed a 'basic' ring.

    """

    intersection_type = type(breaks)
    unaltered = None
    ring_endpoint = None
    basic_ring = False
    complex_ring = False

    # Case 1
    # - Single point from a line intersects the LOI
    # loop roads & 'typical' intersections
    if intersection_type == Point:
        ring_endpoint = [breaks]
        if ring_road == False:
            if breaks == loi.boundary[0] or breaks == loi.boundary[1]:
                return [loi], None, None, None, None
        if ring_road:
            basic_ring = True
            # Do nothing, return the exact ring geometry
            return [loi], None, None, None, None
        else:
            breaks = _make_break_locs(breaks=breaks, standard=True)

    # Case 2
    # - Multiple points from one line intersect the LOI
    # horseshoe roads, multiple intersections of one line and LOI
    elif intersection_type == MultiPoint:
        if ring_road:
            complex_ring = True
            breaks = _make_break_locs(breaks=breaks)
        # horseshoe
        elif breaks == loi.boundary:
            return [loi], None, None, None, None
        else:
            breaks = _make_break_locs(loi=loi, breaks=breaks)

    # Case 3
    # - Overlapping line segments along one stretch of road
    # multiple names, etc. for a section of roadway which was then
    # digitized as separate, stacked entities.
    elif intersection_type == LineString:
        breaks = _make_break_locs(loi=loi, breaks=breaks, line=True)

    # Case 4
    # - Overlapping line segments along multiple stretches of
    # road multiple names, etc. for multiple sections of roadway which
    # were then digitized as separate, stacked entities.
    elif intersection_type == MultiLineString:
        breaks = _make_break_locs(loi=loi, breaks=breaks, mline=True)

    # Case 5
    # - Complex intersection of points and Lines
    # anomaly in digitization / unclear
    elif intersection_type == GeometryCollection:
        # points and line in the geometry collection
        pts_in_gc = []
        lns_in_gc = []

        # if only one line intersection with a point intersection
        multiple_line_intersections = False
        for idx, geom in enumerate(breaks):
            # collect points
            if type(geom) == Point or type(geom) == MultiPoint:
                pts_in_gc.append(geom)
            # collect line(s)
            if type(geom) == LineString or type(geom) == MultiLineString:
                lns_in_gc.append(geom)

        # get split indices in line based on touching geometry
        split_index = []
        iter_limit = len(lns_in_gc) - 1
        for i in range(iter_limit):
            j = i + 1
            current_geom = lns_in_gc[i]
            next_geom = lns_in_gc[j]
            # comparing incremental geometry pairs
            # if touching: do nothing
            if current_geom.touches(next_geom):
                continue
            else:  # if don't touch add a split index
                split_index.append(j)
                multiple_line_intersections = True

        # if there are multiple line intersections between two
        # lines split the segments at the line intersections
        if multiple_line_intersections:
            split_list = []
            for split in split_index:
                if split == split_index[0]:
                    prev_split = split
                    # first split
                    section = lns_in_gc[:split]
                else:
                    # 2nd to n-1 split
                    section = lns_in_gc[prev_split:split]
                # add split line segment
                split_list.append(section)
                # only one split
                if split_index[0] == split_index[-1]:
                    split_list.append(lns_in_gc[split:])
                # last split
                elif split == split_index[-1]:
                    split_list.append(lns_in_gc[split:])
            lns_in_gc = split_list

        # otherwise if there are not multiple line intersections...
        if not multiple_line_intersections:
            welded_line = _weld_MultiLineString(lns_in_gc)
            pts_in_gc.extend([welded_line.boundary[0], welded_line.boundary[1]])
        elif multiple_line_intersections:
            for geoms in lns_in_gc:
                if len(geoms) == 1:
                    pts_in_gc.extend([geoms[0].boundary[0], geoms[0].boundary[1]])
                else:
                    welded_line = _weld_MultiLineString(geoms)
                    pts_in_gc.extend([welded_line.boundary[0], welded_line.boundary[1]])

        breaks = _make_break_locs(loi=loi, breaks=pts_in_gc)

    return unaltered, breaks, ring_endpoint, basic_ring, complex_ring


def _make_break_locs(breaks=None, standard=False, loi=None, line=False, mline=False):
    """Record the points along a line where breaks needs to be made.

    Parameters
    ----------
    breaks : {shapely.Point, shapely.LineString}
        The object to use for breaking the segment. Default is ``None``.
    standard : bool
        This indicates a single point break. Default is ``False``.
    loi : shapely.LineString coordinates
        The coordinates along a line. Default is ``None``.
    line : bool
        Boolean for 'is a LineString'. Default is ``False``.
    mline : bool
        Boolean for 'is a MultiLineString'. Default is ``False``.

    Returns
    -------
    break_points : list
        The geometries of points to break a line.

    """

    if breaks and standard:
        break_points = [Point(breaks.coords[:][0])]

    elif breaks and not standard:
        if line:
            breaks = [breaks.boundary[0], breaks.boundary[1]]
        if mline:
            lns_in_mls = [l for l in breaks]

            # created welded line, but only referencing
            # for break location
            welded_line = _weld_MultiLineString(lns_in_mls)
            breaks = [welded_line.boundary[0], welded_line.boundary[1]]
        break_points = [Point(point.coords[:][0]) for point in breaks]

    return break_points

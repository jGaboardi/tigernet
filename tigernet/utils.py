"""
"""

import geopandas
import numpy
import pandas
from shapely.geometry import Point, LineString

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


def generate_xyid(df=None, geom_type="node"):
    """Create a string xy id.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Geometry dataframe. Default is ``None``.
    geom_type : str
        Either ``'node'`` of ``'segm'``. Default is ``'node'``.

    Returns
    -------
    xyid : list
        List of combined x-coord + y-coords strings.

    """

    xyid = []

    for idx, geom in enumerate(df.geometry):

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
    net : tigernet.TigerNet
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
    net : tigernet.TigerNet

    Returns
    -------
    nodedf : geopandas.GeoDataFrame
        Node dataframe.

    """

    def _drop_covered_nodes(net, ndf):
        """Keep only the top node in stack of overlapping nodes.

        Parameters
        ----------
        net : tigernet.TigerNet
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
        seggeom = sdf.loc[seg, "geometry"]
        if sdf_ring_flag and sdf["ring"][seg]:
            xs, ys = seggeom.coords.xy
            nodes.append(create_node(xs, ys))
        else:
            b1, b2 = seggeom.boundary[0], seggeom.boundary[1]
            nodes.extend([b1, b2])
    nodedf = geopandas.GeoDataFrame(geometry=nodes)
    nodedf = add_ids(nodedf, id_name=net.nid_name)

    if sdf.crs:
        nodedf.crs = sdf.crs

    # Give an initial string 'xy' ID
    prelim_xy_id = generate_xyid(df=nodedf, geom_type="node")
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
    net : tigernet.TigerNet
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


def assert_2_neighs(net):
    """
    1. Raise an error if a segment has more that 2 neighbor nodes.
    2. If the road has one neighbor node then it is a ring road.
        In this case give the ring road a copy of the one nighbor.

    Parameters
    ----------
    net : tigernet.TigerNet

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
    net : tigernet.TigerNet
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
        new_v = getattr(
            new_v, len_col
        ).sum()  ##################################### issue
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
    net : tigernet.TigerNet
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


'''
def record_filter(df, column=None, sval=None, mval=None, oper=None):
    """used in phase 2 with incidents
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        dataframe of incident records
    oper : operator object *OR* str
        {(operator.eq, operator.ne), ('in', 'out')}
    sval : str, int, float, bool, etc.
        single value to filter
    mval : list
        multiple values to filter
    
    Returns
    -------
    df : geopandas.GeoDataFrame
        dataframe of incident records
    """
    
    # use index or specific column
    if column == 'index':
        frame_col = df.index
    else:
        frame_col = df[column]
    
    # single value in column
    if not sval == None:
        return df[oper(frame_col, sval)].copy()
    
    # multiple values in column
    if not mval == None:
        if oper == 'in':
            return df[frame_col.isin(mval)].copy()
        if oper == 'out':
            return df[~frame_col.isin(mval)].copy()



def geom_to_float(df, xval=None, yval=None, geom_type=None):
    """convert a geometric point object to single floats
    for inclusion in a dataframe column.
    
    Parameters
    ----------
    df : geopandas.GeoDataframe
        initial dataframe
    xval : str
        x coordinate column name. Default is None.
    yval : str
        y coordinate column name. Default is None.
    geom_type : str
        geometry type to transform into. Currently either cent
        (centroid) or repp (representative point).
    
    Returns
    -------
    df : geopandas.GeoDataframe
        updated dataframe
    """
    
    geoms = {'cent':'centroid', 'repp':'representative_point'} 
    
    try:
        # for centroids
        df[xval] = [getattr(p, geoms[geom_type]).x for p in df.geometry]
        df[yval] = [getattr(p, geoms[geom_type]).y for p in df.geometry]
    
    except AttributeError:
        try:
            # for representative points
            df[xval] = [getattr(p, geoms[geom_type])().x for p in df.geometry]
            df[yval] = [getattr(p, geoms[geom_type])().y for p in df.geometry]
        except:
            raise AttributeError(geoms[geom_type]+' attribute not present.')
    
    df.drop([xval, yval], axis=1, inplace=True)
    
    return df


def get_fips(st, ct):
    """return cenus FIPS codes for states and counties
    
    Parameters
    ----------
    st : str
        state name
    ct : str
        county name
    
    Returns
    -------
    sf, cf  : tuple
        state and county fips
    """
    
    if st.lower() == 'fl' and ct.lower() == 'leon':
        sf, cf = '12', '073'
    
    else:
        fips_csv_path = '../us_county_fips_2010.csv'
        us_df = pd.read_csv(fips_csv_path, dtype={'ST_FIPS': str,
                                                  'CT_FIPS': str})
        us_df['CT'] = us_df['CT_Name'].apply(lambda x:\
                                            ''.join(x.split(' ')[:-1]).lower())
        st_df = us_df[us_df.ST_Post == st.upper()]
        record = st_df[st_df.CT == ct.replace(' ', '').lower()]
        sf, cf = record.ST_FIPS.values[0], record.CT_FIPS.values[0]
    
    return sf, cf



'''

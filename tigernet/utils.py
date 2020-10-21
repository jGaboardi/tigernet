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


def set_crs(df, proj_init=None, proj_trans=None, crs=None):
    """Set and transform the coordinate
    reference system of a geodataframe.
    
    Parameters
    ----------
    df : geopandas.GeoDataframe
        geodataframe being transformed
    proj_init : int
        intial coordinate reference system. default is None.
    proj_trans : int
        transformed coordinate reference system. default is None.
    crs : dict
        crs from another geodataframe
    
    Returns
    -------
    df : geopandas.GeoDataframe
        transformed geodataframe
    """
    
    if proj_init:
        df.crs = {'init': 'epsg:'+str(proj_init)}
    
    if proj_trans:
        df = df.to_crs(epsg=int(proj_trans))
    
    if crs:
        df = df.to_crs(crs)
    
    return df


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

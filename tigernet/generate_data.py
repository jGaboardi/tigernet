"""
"""

import geopandas
import numpy
from shapely.geometry import Point, LineString

from .utils import _get_lat_lines


__author__ = "James D. Gaboardi <jgaboardi@gmail.com>"


def testing_data(f, to_crs="epsg:2779", bbox="general", direc="test_data"):
    """Read in a prepared dataset for testing/tutorial.

    Parameters
    ----------
    f : str
        The name of the shapefile/dataset.
    to_crs : {str, None}
        Transform to this coordinate reference system. Default is ``'epsg:2779'``.
    bbox : {str, list, tuple, None}
        Filter records by this bounding box. If ``str``, this should be one
        of the following: ``'general'``,``'discard'``, ***********************************
        If ``list`` or ``tuple`` this must be four coordinates in the form
        (minx, miny, maxx, maxy).
        If ``None``, the complete data file will be read in.
        Default is ``'general'``, which corresponds
        to the Waverly Hills area of Tallahassee.
    direc : str
        File directory. Default is ``'test_data'``.

    Returns
    -------
    _gdf : geopandas.GeoDataFrame
        The dataset.

    """

    if type(bbox) == str:
        if bbox == "general":
            bbox = (-84.279, 30.480, -84.245, 30.505)
        elif bbox == "discard":
            bbox = (-84.2525, 30.4412, -84.2472, 30.4528)
        elif bbox == "bad_ring":
            pass
        else:
            msg = "'bbox' value of '%s' not supported." % bbox
            raise ValueError(msg)
    else:
        if bbox:
            good_coords = False
            is_iterable = True if type(bbox) in [tuple, list] else False
            try:
                has_4_elements = True if len(bbox) == 4 else False
            except TypeError:
                has_4_elements = False
            if is_iterable and has_4_elements:
                float_coords = [type(c) == float for c in bbox]
                coords_are_floats = True if all(float_coords) else False
                if coords_are_floats:
                    bbox = bbox
                    good_coords = True
            if not good_coords:
                msg = "There is a problem with the 'bbox' values: %s" % bbox
                raise ValueError(msg)

    base = "zip://%s/%s.zip!%s.shp"
    infile = base % (direc, f, f)
    _gdf = geopandas.read_file(infile, bbox=bbox)
    _gdf = _gdf.to_crs(to_crs)
    return _gdf


def generate_sine_lines(sid_name="SegID", mtfcc="MTFCC", mtfcc_label="S1400"):
    """Generate and write out connected, parallel sine functions.

    Parameters
    ----------
    sid_name : str
        Segment column name. Default is ``'SegID'``.
    mtfcc : str
        MTFCC dataframe column name. Default is ``'MTFCC'``.
    mtfcc_label : str
        Feature class code. Default is ``'S1400'``.

    Returns
    -------
    sine_arcs : geopandas.GeoDataFrame
        The segments comprising the example sine data.

    """

    # create sin arcs
    xs1 = numpy.linspace(1, 14, 100)

    # sine line 1(a)
    l1_xys = xs1[:75]
    xys = zip(l1_xys, numpy.sin(l1_xys))
    pts = [Point(x, y) for (x, y) in xys]
    l1 = LineString(pts)

    # sine line 1(b)
    l2_xys = xs1[74:]
    xys = zip(l2_xys, numpy.sin(l2_xys))
    pts = [Point(x, y) for (x, y) in xys]
    l2 = LineString(pts)

    # sine line 2(a)
    l3_xys = xs1[:25]
    xys = zip(l3_xys, numpy.sin(l3_xys) + 5)
    pts = [Point(x, y) for (x, y) in xys]
    l3 = LineString(pts)

    # sine line 2(b)
    l4_xys = xs1[24:]
    xys = zip(l4_xys, numpy.sin(l4_xys) + 5)
    pts = [Point(x, y) for (x, y) in xys]
    l4 = LineString(pts)

    # inner bounding lines
    l5 = LineString((Point(l2.coords[0]), Point(l2.coords[0][0], 2.5)))
    l6 = LineString((Point(l4.coords[0]), Point(l4.coords[0][0], 2.5)))
    l7 = LineString((Point(l5.coords[-1]), Point(l6.coords[-1])))
    l1xy, l2xy = l1.coords[0], l2.coords[-1]
    l8 = LineString([Point(p) for p in [l1xy, (1, -2), (14, -2), l2xy]])
    l3xy, l4xy = l3.coords[0], l4.coords[-1]
    l9 = LineString([Point(p) for p in [l3xy, (1, 7), (14, 7), l4xy]])
    l10 = LineString((Point(l8.coords[0]), Point(l9.coords[0])))
    l11 = LineString((Point(l8.coords[-1]), Point(l9.coords[-1])))

    # all LineString objects in the example
    lines = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11]

    # outer bounding lines
    lines += [LineString(((0, -3), (0, 8))), LineString(((0, 8), (15, 8)))]
    lines += [LineString(((15, 8), (15, -3))), LineString(((15, -3), (0, -3)))]

    # create GeoDataFrame and set column names
    sine_arcs = geopandas.GeoDataFrame(geometry=lines)
    sine_arcs[sid_name] = sine_arcs.index
    sine_arcs[mtfcc] = [mtfcc_label] * sine_arcs.shape[0]

    return sine_arcs


def generate_lattice(
    sid_name="SegID",
    mtfcc="MTFCC",
    mtfcc_label="S1400",
    bounds=None,
    n_hori_lines=None,
    n_vert_lines=None,
    wbox=False,
):
    """Generate a graph theoretic lattice.

    Parameters
    ----------
    sid_name : str
        Segment column name. Default is ``'SegID'``.
    mtfcc : str
        MTFCC dataframe column name. Default is ``'MTFCC'``.
    mtfcc_label : str
        Feature class code. Default is ``'S1400'``.
    bounds : list
        Area bounds in the form of ``[x1,y1,x2,y2]``.
    n_hori_lines : int
        Count of horizontal lines. Default is ``None``.
    n_vert_lines : int
        Count of vertical lines. Default is ``None``.
    wbox : bool
        Include outer bounding box segments. Default is ``False``.

    Returns
    -------
    lat_arcs : geopandas.GeoDataFrame
        The segments comprising the example lattice data.
    """

    # set grid parameters if not declared
    if not bounds:
        bounds = [0, 0, 9, 9]
    if not n_hori_lines:
        n_hori_lines = 2
    if not n_vert_lines:
        n_vert_lines = 2

    # bounding box line lengths
    h_length, v_length = bounds[2] - bounds[0], bounds[3] - bounds[1]

    # horizontal and vertical increments
    h_incr, v_incr = (
        h_length / float(n_hori_lines + 1),
        v_length / float(n_vert_lines + 1),
    )

    # define the horizontal and vertical space
    hspace = [h_incr * slot for slot in range(n_vert_lines + 2)]
    vspace = [v_incr * slot for slot in range(n_hori_lines + 2)]

    # get vertical and horizontal lines
    horis = _get_lat_lines(hspace, vspace, wbox, bounds)
    verts = _get_lat_lines(hspace, vspace, wbox, bounds, hori=False)

    # combine into one list
    geoms = verts + horis

    # instantiate dataframe
    lat_arcs = geopandas.GeoDataFrame(geometry=geoms)

    lat_arcs[sid_name] = lat_arcs.index + 1
    lat_arcs[mtfcc] = [mtfcc_label] * lat_arcs.shape[0]

    return lat_arcs


def generate_obs(npts, s_df, near_net=None, restrict=None, seed=0):
    """Generate random point observations.

    Parameters
    ----------
    npts : int
        The number of random points desired.
    s_df : geopandas.GeoDataFrame
        Network line segments.
    near_net : {int, float}
        The distance from segments to generate point. If set to  ``None``
        the full area in ``s_df.total_bounds`` is used. Default is ``None``.
    restrict : list
        Do not generate points along these MTFCC types. Default is ``None``.
    seed : int
        The random seed for point generation. Default is ``0``

    Returns
    -------
    rand_obs : geopandas.GeoDataFrame
        Randomly generated point observations.

    """

    # extract bounds
    minx, miny, maxx, maxy = s_df.total_bounds

    # set random seed and declare random alias
    numpy.random.seed(seed)
    nru = numpy.random.uniform

    # declare anonymous function for point generation
    rand_pt = lambda xy: Point(nru(minx, maxx), nru(miny, maxy))

    # generate point within a specified proximity to the network
    if near_net:
        if restrict:
            # remove restricted segments from consideration
            segms = s_df[~s_df[restrict[0]].isin(restrict[1])].copy()
        else:
            segms = s_df.copy()
        # union of network line segments buffer
        in_space = segms.buffer(near_net).unary_union
        # intersection checker
        intersector = lambda pt: pt.intersects(in_space)
        rand_pts = []
        while len(rand_pts) != npts:
            point = rand_pt(0)
            if intersector(point):
                rand_pts.append(point)

    # generate points with the total segment bounds
    else:
        rand_pts = [rand_pt(pt) for pt in range(npts)]

    # instantiate as a GeoDataFrame
    rand_obs = geopandas.GeoDataFrame(geometry=rand_pts)

    return rand_obs

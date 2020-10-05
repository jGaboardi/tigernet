"""
"""

import geopandas
import numpy
import pandas
from shapely.geometry import Point, LineString


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


def generate_xyid(df=None, geom_type='node'):
    """create a string xy id
    
    Parameters
    ----------
    df : geopandas.GeoDataFrame
        geometry dataframe. Default is None.
    geom_type : str
        either node of segm. Default is 'node'.
    
    Returns
    -------
    xyid : list
        list of combined x-coord + y-coords strings
    """
    
    xyid = []
    
    for idx, geom in enumerate(df.geometry):
        
        if geom_type == 'segm':
            xys = ['x'+str(x)+'y'+str(y) for (x,y) in geom.coords[:]]
            xyid.append([idx, xys])
        
        # try to make the xyid from a polygon
        if geom_type == 'node':
            try:
                xy = 'x'+str(geom.centroid.x)+'y'+str(geom.centroid.y)
            
            # if the geometry is not polygon, but already point
            except AttributeError:
                try:
                    xy = 'x'+str(geom.x)+'y'+str(geom.y)
                except:
                    print('geom:', type(geom))
                    print(dir(geom))
                    raise AttributeError('geom has neither attribute:\n'\
                                         +'\t\t- `.centroid.[coord]`\n'\
                                         +'\t\t- `.[coord]`')
            
            xyid.append([idx,[xy]])
    
    return xyid


def fill_frame(frame, full=False, idx='index',
               col=None, data=None, add_factor=0):
    """fill a dataframe with a column of data
    
    Parameters
    ----------
    frame : geopandas.GeoDataFrame
        geometry dataframe
    full : bool
        create a new column (False) or a new frame (True).
        Default is False.
    idx : str
        index column name. Default is 'index'.
    col : str or list
         New column name(s). Default is None.
    data : list *OR* dict
        list of data to fill the column. Default is None. OR
        dict of data to fill the records. Default is None.
    add_factor : int
        used when dataframe index does not start at zero.
        Default is zero.
    
    Returns
    -------
    frame : geopandas.GeoDataFrame
        updated geometry dataframe
    """
    
    # create a full geopandas.GeoDataFrame
    if full:
        out_frame = gpd.GeoDataFrame.from_dict(data, orient='index')
        
        return out_frame
    
    # write a single column in a geopandas.GeoDataFrame
    else:
        frame[col] = np.nan
        for (k,v) in data:
            k += add_factor
            
            if col == 'CC':
                frame.loc[frame[idx].isin(v), col] = k
            
            elif idx == 'index':
                frame.loc[k, col] = str(v)
            
            else:
                frame.loc[(frame[idx] == k), col] = str(v)
        
        if col == 'CC':
            frame[col] = frame[col].astype('category').astype(int)
        
        return frame


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


def get_discard_mtfcc_by_desc():
    """discard these road types from the mtfcc categories
    """
    
    return ['Bike Path or Trail', 'Parking Lot Road',  'Alley',\
            'Vehicular Trail (4WD)', 'Walkway/Pedestrian Trail',\
            'Private Road for service vehicles (logging, oil fields, '\
            +'ranches, etc.)']


def get_mtfcc_types():
    """read in dictionary of MTFCC road type descriptions
    https://www.census.gov/geo/reference/mtfcc.html
    
    ******* Ranks are subjective *******
    
    """
   
    mtfcc_types = {'S1100':{'FClass':'Primary Road',
                             'Desc': 'Primary roads are generally divided, '\
                                     +'limited-access highways within the '\
                                     +'interstate highway system or under '\
                                     +'state management, and are '\
                                     +'distinguished by the presence of '\
                                     +'interchanges. These highways are '\
                                     +'accessible by ramps and may include '\
                                     +'some toll highways.'},
                   'S1200':{'FClass':'Secondary Road',
                             'Desc': 'Secondary roads are main arteries, '\
                                     +'usually in the U.S. Highway, State '\
                                     +'Highway or County Highway system. '\
                                     +'These roads have one or more lanes of '\
                                     +'traffic in each direction, may or may '\
                                     +'not be divided, and usually have '\
                                     +'at-grade intersections with many '\
                                     +'other roads and driveways. They often '\
                                     +'have both a local name and a route '\
                                     +'number.'},
                   'S1400':{'FClass':'Local Neighborhood Road, '\
                                      +'Rural Road, City Street',
                             'Desc': 'Generally a paved non-arterial street, '\
                                     +'road, or byway that usually has a '\
                                     +'single lane of traffic in each '\
                                     +'direction. Roads in this feature '\
                                     +'class may be privately or publicly '\
                                     +'maintained. Scenic park roads would '\
                                     +'be included in this feature class, '\
                                     +'as would (depending on the region of '\
                                     +'the country) some unpaved roads.'},
                   'S1500':{'FClass':'Vehicular Trail (4WD)',
                             'Desc': 'An unpaved dirt trail where a '\
                                     +'four-wheel drive vehicle is required. '\
                                     +'These vehicular trails are found '\
                                     +'almost exclusively in very rural '\
                                     +'areas. Minor, unpaved roads usable by '\
                                     +'ordinary cars and trucks belong in '\
                                     +'the S1400 category.'},
                   'S1630':{'FClass':'Ramp',
                             'Desc': 'A road that allows controlled access '\
                                     +'from adjacent roads onto a limited '\
                                     +'access highway, often in the form of '\
                                     +'a cloverleaf interchange. These roads '\
                                     +'are unaddressable.'},
                   'S1640':{'FClass':'Service Drive usually along a limited '\
                                      +'access highway',
                             'Desc': 'A road, usually paralleling a limited '\
                                     +'access highway, that provides access '\
                                     +'to structures along the highway. '\
                                     +'These roads can be named and may '\
                                     +'intersect with other roads.'},
                   'S1710':{'FClass':'Walkway/Pedestrian Trail',
                             'Desc': 'A path that is used for walking, being '\
                                     +'either too narrow for or legally '\
                                     +'restricted from vehicular traffic.'},
                   'S1720':{'FClass':'Stairway',
                             'Desc': 'A pedestrian passageway from one level '\
                                     +'to another by a series of steps.'},
                   'S1730':{'FClass':'Alley',
                             'Desc': 'A service road that does not generally '\
                                     +'have associated addressed structures '\
                                     +'and is usually unnamed. It is located '\
                                     +'at the rear of buildings and '\
                                     +'properties and is used for '\
                                     +'deliveries.'},
                   'S1740':{'FClass':'Private Road for service vehicles '\
                                      + '(logging, oil fields, ranches, etc.)',
                             'Desc': 'A road within private property that is '\
                                     +'privately maintained for service, '\
                                     +'extractive, or other purposes. These '\
                                     +'roads are often unnamed.'},
                   'S1750':{'FClass':'Internal U.S. Census Bureau use',
                             'Desc': 'Internal U.S. Census Bureau use'},
                   'S1780':{'FClass':'Parking Lot Road',
                             'Desc': 'The main travel route for vehicles '\
                                     +'through a paved parking area.'},
                   'S1820':{'FClass':'Bike Path or Trail',
                             'Desc': 'A path that is used for manual or '\
                                     +'small, motorized bicycles, being '\
                                     +'either too narrow for or legally '\
                                     +'restricted from vehicular traffic.'},
                   'S1830':{'FClass':'Bridle Path',
                             'Desc': 'A path that is used for horses, being '\
                                     +'either too narrow for or legally '\
                                     +'restricted from vehicular traffic.'},
                   'S2000':{'FClass':'Road Median',
                             'Desc': 'The unpaved area or barrier between '\
                                     +'the carriageways of a divided road.'}
                   }
    
    return mtfcc_types
'''

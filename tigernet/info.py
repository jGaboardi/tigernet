def get_discard_mtfcc_by_desc() -> list:
    """Discard these road types from the mtfcc categories."""

    return [
        "Bike Path or Trail",
        "Parking Lot Road",
        "Alley",
        "Vehicular Trail (4WD)",
        "Walkway/Pedestrian Trail",
        "Private Road for service vehicles (logging, oil fields, ranches, etc.)",
    ]


def get_mtfcc_types() -> dict:
    """Return a dictionary of MTFCC road type descriptions.
    See [https://www.census.gov/geo/reference/mtfcc.html] for reference.
    """

    mtfcc_types = {
        "S1100": {
            "FClass": "Primary Road",
            "Desc": "Primary roads are generally divided, "
            + "limited-access highways within the "
            + "interstate highway system or under "
            + "state management, and are "
            + "distinguished by the presence of "
            + "interchanges. These highways are "
            + "accessible by ramps and may include "
            + "some toll highways.",
        },
        "S1200": {
            "FClass": "Secondary Road",
            "Desc": "Secondary roads are main arteries, "
            + "usually in the U.S. Highway, State "
            + "Highway or County Highway system. "
            + "These roads have one or more lanes of "
            + "traffic in each direction, may or may "
            + "not be divided, and usually have "
            + "at-grade intersections with many "
            + "other roads and driveways. They often "
            + "have both a local name and a route "
            + "number.",
        },
        "S1400": {
            "FClass": "Local Neighborhood Road, " + "Rural Road, City Street",
            "Desc": "Generally a paved non-arterial street, "
            + "road, or byway that usually has a "
            + "single lane of traffic in each "
            + "direction. Roads in this feature "
            + "class may be privately or publicly "
            + "maintained. Scenic park roads would "
            + "be included in this feature class, "
            + "as would (depending on the region of "
            + "the country) some unpaved roads.",
        },
        "S1500": {
            "FClass": "Vehicular Trail (4WD)",
            "Desc": "An unpaved dirt trail where a "
            + "four-wheel drive vehicle is required. "
            + "These vehicular trails are found "
            + "almost exclusively in very rural "
            + "areas. Minor, unpaved roads usable by "
            + "ordinary cars and trucks belong in "
            + "the S1400 category.",
        },
        "S1630": {
            "FClass": "Ramp",
            "Desc": "A road that allows controlled access "
            + "from adjacent roads onto a limited "
            + "access highway, often in the form of "
            + "a cloverleaf interchange. These roads "
            + "are unaddressable.",
        },
        "S1640": {
            "FClass": "Service Drive usually along a limited " + "access highway",
            "Desc": "A road, usually paralleling a limited "
            + "access highway, that provides access "
            + "to structures along the highway. "
            + "These roads can be named and may "
            + "intersect with other roads.",
        },
        "S1710": {
            "FClass": "Walkway/Pedestrian Trail",
            "Desc": "A path that is used for walking, being "
            + "either too narrow for or legally "
            + "restricted from vehicular traffic.",
        },
        "S1720": {
            "FClass": "Stairway",
            "Desc": "A pedestrian passageway from one level "
            + "to another by a series of steps.",
        },
        "S1730": {
            "FClass": "Alley",
            "Desc": "A service road that does not generally "
            + "have associated addressed structures "
            + "and is usually unnamed. It is located "
            + "at the rear of buildings and "
            + "properties and is used for "
            + "deliveries.",
        },
        "S1740": {
            "FClass": "Private Road for service vehicles "
            + "(logging, oil fields, ranches, etc.)",
            "Desc": "A road within private property that is "
            + "privately maintained for service, "
            + "extractive, or other purposes. These "
            + "roads are often unnamed.",
        },
        "S1750": {
            "FClass": "Internal U.S. Census Bureau use",
            "Desc": "Internal U.S. Census Bureau use",
        },
        "S1780": {
            "FClass": "Parking Lot Road",
            "Desc": "The main travel route for vehicles "
            + "through a paved parking area.",
        },
        "S1820": {
            "FClass": "Bike Path or Trail",
            "Desc": "A path that is used for manual or "
            + "small, motorized bicycles, being "
            + "either too narrow for or legally "
            + "restricted from vehicular traffic.",
        },
        "S1830": {
            "FClass": "Bridle Path",
            "Desc": "A path that is used for horses, being "
            + "either too narrow for or legally "
            + "restricted from vehicular traffic.",
        },
        "S2000": {
            "FClass": "Road Median",
            "Desc": "The unpaved area or barrier between "
            + "the carriageways of a divided road.",
        },
    }

    return mtfcc_types


def get_discard_segms(year, state_fips, county_fips):
    """This function provides a catalogue of known troublemaker TIGER/Line
    Edges stored at year --> state --> county level. Currently, only
    2010 TIGER/Line Edges for Leon County, Florida are supported.

    Parameters
    ----------
    year : str
        TIGER/Line Edges year.
    state_fips : str
        State FIPS code.
    county_fips : str
        County FIPS code.

    Returns
    -------
    discard : list
        Network segments that are known to be troublemakers.

    """

    catalogue = {
        "2010": {
            "12": {
                "073": [
                    618799725,
                    618799786,
                    618799785,
                    634069404,
                    618799771,
                    618799763,
                    610197711,
                    82844664,
                    82844666,
                    82844213,
                    82844669,
                    82844657,
                    82844670,
                    82844652,
                    82844673,
                ]
            }
        },
        "2000": {"12": {"073": []}},
    }

    try:
        discard = catalogue[year][state_fips][county_fips]
        if not discard:
            raise KeyError
    except KeyError:
        params = year, state_fips, county_fips
        msg = "There was a problem with the query. Check parameters: %s, %s, %s"
        msg += msg % params
        raise KeyError(msg)

    return discard

import geopandas as gpd
from pyproj import Proj, transform
from shapely.geometry import Polygon

PROJ_LAT_LON = Proj(4326, preserve_units=False)


def bounds_to_polygons(bounds_list, max_bounds=None):
    """
    Creating a geodataframe with polygons based on a list of bounds.

    Parameters
    ----------
    bounds_list : list
        List of rasterio bounds objects (or a list with the same order).
    max_bounds : list, optional
        Rasterio bounds object containing the maximum bounds that are enforced. By default it is None, which means no
        type checking is done. They keyword global gives: [-180,-90,180,90].

    Returns
    -------
    gdf : geopandas.geodataframe
        Geodataframe containing the polygons corresponding to the bounds that were given
    """
    # todo; this function can be optimised super much by vectorizing the rational
    # rename bounds_list to bounds

    if max_bounds == "global":
        max_bounds = [-180, -90, 180, 90]

    # todo; rewrite
    assert type(bounds_list) in (list, tuple), "List of bounds not iterable"

    # create geodataframe with columns to store the bounds and the polygons
    gdf = gpd.GeoDataFrame(columns=("bounds", "geometry"))

    for i, bounds in enumerate(bounds_list):
        if not isinstance(max_bounds, type(None)):
            # make sure the bounds don't fall of the earth
            left = max(bounds[0], max_bounds[0])
            bottom = max(bounds[1], max_bounds[1])
            right = min(bounds[2], max_bounds[2])
            top = min(bounds[3], max_bounds[3])
        else:
            left, bottom, right, top = bounds

        # create a list with the x coordinates of the corners
        box_x = [left, right, right, left]
        # create a list with the y coordinates of the corners
        box_y = [top, top, bottom, bottom]

        # create a shapely Polygon from those lists
        polygon = Polygon(zip(box_x, box_y))

        # store bounds and polygon in the dataframe
        gdf.loc[i] = [bounds, polygon]

    return gdf


def bounds_to_platecarree(proj, bounds):
    """This functionality is deprecated: likely switch to
    >>> transformer = Transform.from_proj(proj, PROJ_LAT_LON, always_xy=True)
    >>> return transformer(bounds[0], bounds[1]), transformer(bounds[2], bounds[3])
    """
    return (*transform(proj, PROJ_LAT_LON, bounds[0], bounds[1], always_xy=True),
            *transform(proj, PROJ_LAT_LON, bounds[2], bounds[3], always_xy=True))


def bounds_to_data_projection(proj, bounds):
    return (*transform(PROJ_LAT_LON, proj, bounds[0], bounds[1], always_xy=True),
            *transform(PROJ_LAT_LON, proj, bounds[2], bounds[3], always_xy=True))

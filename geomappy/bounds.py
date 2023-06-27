from typing import Tuple

import geopandas as gpd  # type: ignore
import numpy as np
from pyproj import Proj
from rasterio.coords import BoundingBox  # type: ignore
from shapely.geometry import Polygon  # type: ignore

PROJ_LAT_LON = Proj(4326, preserve_units=False)


def bounds_to_polygons(bounds_list: Tuple[BoundingBox]):
    """
    Creating a geodataframe with polygons based on a list of bounds.

    Parameters
    ----------
    bounds_list : list
        List of rasterio BoundingBox objects (or a list with the same order).

    Returns
    -------
    gdf : geopandas.GeoDataframe
        GeoDataframe containing the polygons corresponding to the bounds that were given
    """
    if not isinstance(bounds_list, (list, tuple, np.ndarray)):
        raise TypeError("List of bounds not iterable")

    # create geodataframe with columns to store the bounds and the polygons
    gdf = gpd.GeoDataFrame(columns=("bounds", "geometry"))

    for i, bounds in enumerate(bounds_list):
        left, bottom, right, top = bounds
        box_x = [left, right, right, left]
        box_y = [top, top, bottom, bottom]
        polygon = Polygon(zip(box_x, box_y))
        gdf.loc[i] = [bounds, polygon]

    return gdf.set_geometry('geometry')

import geopandas as gpd
from rasterio.coords import BoundingBox
from shapely.geometry import Polygon


def bounds_to_polygons(bounds: tuple[BoundingBox]) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(index=range(len(bounds)), columns=('bounds', 'geometry'))

    for i, bounds in enumerate(bounds):
        left, bottom, right, top = bounds
        box_x = [left, right, right, left]
        box_y = [top, top, bottom, bottom]
        polygon = Polygon(zip(box_x, box_y))
        gdf.loc[i, 'bounds'] = bounds
        gdf.loc[i, 'geometry'] = polygon

    return gdf


def extent_from_bounds(
    bounds: BoundingBox | tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return bounds[0], bounds[2], bounds[1], bounds[3]

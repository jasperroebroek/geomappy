from typing import Tuple, Optional, Union

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import pyplot as plt
from rasterio.coords import BoundingBox

from geomappy.basemap import basemap, add_gridlines, add_ticks
from geomappy.bounds import bounds_to_polygons
from geomappy.types import Number


def plot_world(bounds: Union[Tuple[Number, Number, Number, Number]], ax: Optional[plt.Axes] = None,
               extent: Union[Tuple[Number, Number, Number, Number], str] = 'global',
               projection: ccrs.Projection = ccrs.PlateCarree(),
               extent_projection: ccrs.Projection = ccrs.PlateCarree(),
               bounds_projection: ccrs.Projection = ccrs.PlateCarree(),
               **kwargs) -> GeoAxes:
    """
    Plots the outer bounds on a world map

    Parameters
    ----------
    bounds: list of int
        Bounds to be plotted on the map
    ax : :obj:`~matplotlib.axes.Axes`, optional
        Axes on which to plot the figure
    extent : list, optional
        Takes a four number list, or rasterio bounds object. It constrains the world view
        to a specific view. If not lat-lon, the extent_projection needs to be specified
    extent_projection: cartopy.CRS
        Projection of the extent and bounds. The default ccrs.PlateCarree().
    projection : cartopy.CRS
        Projection of the plot. The default ccrs.PlateCarree().
    kwargs
        Arguments for the Basemap function

    Returns
    -------
    GeoAxis
    """
    ticks = kwargs.pop('ticks', 30)
    lines = kwargs.pop('lines', 30)

    ax = basemap(ax=ax, projection=projection, **kwargs)

    if extent == 'global':
        ax.set_global()
    else:
        ax.set_extent(extent, crs=extent_projection)
    ax.coastlines()

    add_gridlines(ax, lines)
    add_ticks(ax, ticks)

    gdf = bounds_to_polygons((BoundingBox(*bounds),))
    gdf.plot(ax=ax, edgecolor="red", facecolor="none", zorder=2, transform=bounds_projection)

    return ax

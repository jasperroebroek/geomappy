from typing import Optional, Union, Tuple

import cartopy.crs as ccrs
import geopandas as gpd
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import pyplot as plt

from geomappy.basemap import basemap as basemap_function, add_gridlines, add_ticks
from geomappy.shapes import plot_shapes, plot_classified_shapes
from geomappy.types import Number, OptionalLegend
from geomappy.utils import add_method, change_between_bounds_and_extent
from geomappy.world import plot_world


def _plot_combined_shapes(classified, df, *,
                          figsize: Tuple[int, int] = (8, 8), ax: Optional[plt.Axes] = None, basemap: bool = True,
                          lines: Union[Number, Tuple[Number, Number], Tuple[Tuple[Number], Tuple[Number]]] = 30,
                          ticks: Union[Number, Tuple[Number, Number], Tuple[Tuple[Number], Tuple[Number]]] = 30,
                          fontsize: int = 10, **kwargs) -> Tuple[plt.Axes, OptionalLegend]:
    if isinstance(ax, GeoAxes):
        basemap = True

    if basemap:
        projection = df.get_cartopy_projection()
        ax = basemap_function(ax=ax, projection=projection, figsize=figsize)
        ax.set_extent(df.get_extent(), crs=projection)
        ax.coastlines()
        add_gridlines(ax, lines)
        add_ticks(ax, ticks, fontsize=fontsize)
        kwargs['transform'] = projection

    if classified:
        return plot_classified_shapes(df=df, ax=ax, figsize=figsize, **kwargs)
    else:
        return plot_shapes(df=df, ax=ax, figsize=figsize, **kwargs)


@add_method("get_extent", gpd.GeoDataFrame, gpd.GeoSeries)
def gpd_get_extent(self):
    return change_between_bounds_and_extent(self.total_bounds)


@add_method("get_cartopy_projection", gpd.GeoDataFrame, gpd.GeoSeries)
def gpd_get_cartopy_projection(self):
    if self.crs is None:
        raise ValueError("No projection is set on the DataArray")

    crs = self.crs

    if crs.to_epsg() == 4326:
        return ccrs.PlateCarree()
    elif crs.to_epsg() == 3857:
        return ccrs.Mercator()
    elif crs.to_epsg() is not None:
        return ccrs.epsg(crs.to_epsg())
    else:
        return ccrs.Projection(crs)


@add_method("plot_shapes", gpd.GeoDataFrame, gpd.GeoSeries)
def gpd_plot_shapes(self, *, ax: Optional[plt.Axes] = None, **kwargs):
    """
    Plot data from a GeoDataFrame. This function is exposed as ``plot_shapes`` in a ``GeoDataFrame``.

    Parameters
    ----------
    basemap : bool, optional
        Plot a basemap behind the data.
    figsize : tuple, optional
        Matplotlib figsize parameter
    ax : Axes, optional
        Matplotlib axes/ Cartopy GeoAxes where plot is drawn on. If not provided, it will be created on the fly,
        based on the 'basemap' and 'projection' parameters.
    lines : int, tuple of ints, or tuple of tuples of ints, optional
        parameter that describes the distance between two gridlines in lat-lon terms. The default 30
        means that every 30 degrees a gridline gets drawn. See add_gridlines
    ticks : int, tuple of ints, or tuple of tuples of ints, optional
        parameter that describes the distance between two tick labels in lat-lon terms. The default 30
        means that every 30 degrees a tick gets placed. see add_ticks
    fontsize : float/tuple, optional
        fontsize for both the lon/lat ticks and the ticks on the colorbar if one number, if a list of is passed it
        represents the basemap fontsize and the colorbar fontsize.
    kwargs
        kwargs going to the plot_classified_shapes() command

    Returns
    -------
    (:obj:`~matplotlib.axes.Axes`, legend)
        Axes and legend. The legend depends on the `legend` parameter and can be None.
    """
    if isinstance(self, type(None)):
        return plot_shapes(ax=ax, **kwargs)
    elif isinstance(self, gpd.GeoDataFrame):
        return _plot_combined_shapes(classified=False, ax=ax, df=self, **kwargs)
    elif isinstance(self, gpd.GeoSeries):
        return _plot_combined_shapes(classified=False, ax=ax, df=gpd.GeoDataFrame(self, crs=self.crs), **kwargs)
    else:
        raise TypeError("This method does not support positional arguments")


@add_method("plot_classified_shapes", gpd.GeoDataFrame, gpd.GeoSeries)
def gpd_plot_classified_shapes(self, *, ax: Optional[plt.Axes] = None, **kwargs):
    """
    Plot data from a GeoDataFrame. This function is exposed as ``plot_classified_shapes`` in a ``GeoDataFrame``.

    Parameters
    ----------
    basemap : bool, optional
        Plot a basemap behind the data.
    figsize : tuple, optional
        Matplotlib figsize parameter
    ax : Axes, optional
        Matplotlib axes/ Cartopy GeoAxes where plot is drawn on. If not provided, it will be created on the fly,
        based on the 'basemap' and 'projection' parameters.
    lines : int, tuple of ints, or tuple of tuples of ints, optional
        parameter that describes the distance between two gridlines in lat-lon terms. The default 30
        means that every 30 degrees a gridline gets drawn. See add_gridlines
    ticks : int, tuple of ints, or tuple of tuples of ints, optional
        parameter that describes the distance between two tick labels in lat-lon terms. The default 30
        means that every 30 degrees a tick gets placed. see add_ticks
    fontsize : float/tuple, optional
        fontsize for both the lon/lat ticks and the ticks on the colorbar if one number, if a list of is passed it
        represents the basemap fontsize and the colorbar fontsize.
    kwargs
        kwargs going to the plot_classified_shapes() command

    Returns
    -------
    (:obj:`~matplotlib.axes.Axes`, legend)
        Axes and legend. The legend depends on the `legend` parameter and can be None.
    """
    if isinstance(self, type(None)):
        return plot_classified_shapes(ax=ax, **kwargs)
    elif isinstance(self, gpd.GeoDataFrame):
        return _plot_combined_shapes(classified=True, ax=ax, df=self, **kwargs)
    elif isinstance(self, gpd.GeoSeries):
        return _plot_combined_shapes(classified=True, ax=ax, df=gpd.GeoDataFrame(self, crs=self.crs), **kwargs)
    else:
        raise TypeError("This method does not support positional arguments")


@add_method("plot_world", gpd.GeoDataFrame, gpd.GeoSeries)
def gpd_plot_world(self, ax: Optional[plt.Axes] = None, extent: Union[Tuple[int, int, int, int], str] = 'global',
                   projection: ccrs.Projection = ccrs.PlateCarree(),
                   extent_projection: ccrs.Projection = ccrs.PlateCarree(), **kwargs) -> GeoAxes:
    """
    Plots the outer bounds of a GeoDataFrame on a world map.

    Parameters
    ----------
    ax : :obj:`~matplotlib.axes.Axes`, optional
        Axes on which to plot the figure
    extent : list, optional
        Takes a four number list, or rasterio bounds object. It constrains the world view
        to a specific view. If not lat-lon, the extent_projection needs to be specified
    extent_projection: cartopy.CRS
        Projection of the extent. The default ccrs.PlateCarree().
    projection : cartopy.CRS
        Projection of the plot. The default ccrs.PlateCarree().
    kwargs
        arguments for the Basemap function

    Returns
    -------
    GeoAxes
    """
    bounds = self.total_bounds
    plot_world(bounds, ax=ax, extent=extent, projection=projection, extent_projection=extent_projection,
               bounds_projection=self.get_cartopy_projection(), **kwargs)


@add_method("plot_file", gpd.GeoDataFrame, gpd.GeoSeries)
def gpd_plot_file(self, **kwargs) -> GeoAxes:
    """
    Plots a map of the outer bounds of GeoDataFrame.

    Parameters
    ----------
    kwargs
        arguments for the :func:`geomappy.gpd_plot_world` function

    Returns
    -------
    GeoAxes
    """
    return self.plot_world(extent=self.get_extent(), projection=self.get_cartopy_projection(),
                           extent_projection=self.get_cartopy_projection(), **kwargs)

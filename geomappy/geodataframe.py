import geopandas as gpd
import cartopy.crs as ccrs
import numpy as np
from shapely.geometry import Polygon

from .bounds import bounds_to_polygons
from .plotting import plot_shapes as plot_shapes, plot_classified_shapes
from .basemap import basemap as basemap_function
from .utils import add_method

# TODO; geo_clip


def _plot_combined_shapes(classified, *, df=None, basemap=False, figsize=(10, 10), ax=None, projection=None, xticks=30,
                          yticks=30, fontsize=10, extent=None, basemap_kwargs=None, **kwargs):
    if df is None:
        raise TypeError("Internal error: df not received")

    if not isinstance(fontsize, (tuple, list)):
        fontsize = (fontsize, fontsize)
    kwargs.update({'fontsize': fontsize[1]})

    if basemap:
        if isinstance(basemap_kwargs, type(None)):
            basemap_kwargs = {}

        if 'xticks' not in basemap_kwargs:
            basemap_kwargs.update({'xticks': xticks})
        if 'yticks' not in basemap_kwargs:
            basemap_kwargs.update({'yticks': yticks})
        if 'fontsize' not in basemap_kwargs:
            basemap_kwargs.update({'fontsize': fontsize[0]})

        if hasattr(ax, 'projection'):
            projection = ax.projection
        elif projection is not None:
            pass
        elif df.crs is not None:
            projection = df.get_cartopy_projection()
        else:
            raise ValueError("No projection is provided, which is necessary when plotting a basemap. Search order is "
                             "a projection set on a provided Axes, the projection parameter, the CRS of the dataframe.")

        if extent is None:
            extent = df.total_bounds.tolist()
            basemap_kwargs.update({"extent_projection": df.get_cartopy_projection()})

        ax = basemap_function(extent, ax=ax, projection=projection, figsize=figsize, **basemap_kwargs)
        kwargs.update({'transform': projection})

        crs_proj4 = projection.proj4_init
        extent = ax.get_extent()
        df = df.to_crs(crs_proj4).clip(Polygon(
            ((extent[0], extent[2]), (extent[0], extent[3]), (extent[1], extent[3]), (extent[1], extent[2]))
        ))

    if classified:
        return plot_classified_shapes(df=df, ax=ax, figsize=figsize, **kwargs)
    else:
        return plot_shapes(df=df, ax=ax, figsize=figsize, **kwargs)


@add_method("get_cartopy_projection", gpd.GeoDataFrame, gpd.GeoSeries)
def gpd_get_cartopy_projection(self=None):
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
def gpd_plot_shapes(self=None, *, ax=None, **kwargs):
    """
    Plot data from a GeoDataFrame. This function is exposed as ``plot_shapes`` in a ``GeoDataFrame``.

    Parameters
    ----------
    self : GeoDataFrame
        This is not supposed to be used directly, but as the self parameter when called from a GeoDataFrame directly
    basemap : bool, optional
        Plot a basemap behind the data.
    figsize : tuple, optional
        Matplotlib figsize parameter
    ax : Axes, optional
        Matplotlib axes/ Cartopy GeoAxes where plot is drawn on. If not provided, it will be created on the fly,
        based on the 'basemap' and 'projection' parameters.
    projection : cartopy CRS, optional
        A basemap will be drawn to this projection. If ax already contains a Cartopy GeoAxes, this parameter will be
        ignored.
    xticks : float or list, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used.
    yticks : float or list, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used.
    fontsize : float/tuple, optional
        fontsize for both the lon/lat ticks and the ticks on the colorbar if one number, if a list of is passed it
        represents the basemap fontsize and the colorbar fontsize.
    extent : list or string, optional
        Extent of the plot, if not provided it will plot the extent belonging to dataframe. If the string "global" is
        used, the maximum extent of the projection is set. Is only active with basemap switched on. Extent is defined
        in lotlan by default, but can be switched to something different by passing extent_projection to basemap_kwargs.
    basemap_kwargs : dict, optional
        kwargs going to the basemap function
    kwargs
        kwargs going to the plot_shapes() command

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
def gpd_plot_classified_shapes(self=None, *, ax=None, **kwargs):
    """
    Plot data from a GeoDataFrame. This function is exposed as ``plot_classified_shapes`` in a ``GeoDataFrame``.

    Parameters
    ----------
    self : GeoDataFrame
        This is not supposed to be used directly, but as the self parameter when called from a GeoDataFrame directly
    basemap : bool, optional
        Plot a basemap behind the data.
    figsize : tuple, optional
        Matplotlib figsize parameter
    ax : Axes, optional
        Matplotlib axes/ Cartopy GeoAxes where plot is drawn on. If not provided, it will be created on the fly,
        based on the 'basemap' and 'projection' parameters.
    projection : cartopy CRS, optional
        A basemap will be drawn to this projection. If ax already contains a Cartopy GeoAxes, this parameter will be
        ignored.
    xticks : float or list, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used.
    yticks : float or list, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used.
    fontsize : float/tuple, optional
        fontsize for both the lon/lat ticks and the ticks on the colorbar if one number, if a list of is passed it
        represents the basemap fontsize and the colorbar fontsize.
    extent : list or string, optional
        Extent of the plot, if not provided it will plot the extent belonging to dataframe. If the string "global" is
        used, the maximum extent of the projection is set. Is only active with basemap switched on. Extent is defined
        in lotlan by default, but can be switched to something different by passing extent_projection to basemap_kwargs.
    basemap_kwargs : dict, optional
        kwargs going to the basemap function
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
def gpd_plot_world(self=None, projection=None, ax=None, extent="global", **kwargs):
    """
    Plots the outer bounds of a GeoDataFrame on a world map.

    Parameters
    ----------
    self : GeoDataFrame
        This is not supposed to be used directly, but as the self parameter when called from a GeoDataFrame directly
    ax : :obj:`~matplotlib.axes.Axes`, optional
        Axes on which to plot the figure
    extent : list, optional
        Takes a four number list, or rasterio bounds object. It constrains the world view
        to a specific view. If not lat-lon, the extent_projection needs to be specified
    projection : cartopy.CRS
        Projection of the plot. The default is None, which yields ccrs.PlateCarree(). It accepts
        'native', which plots the map on the projection of the data.
    kwargs
        arguments for the Basemap function

    Returns
    -------
    GeoAxis
    """
    if projection == "native":
        projection = self.get_cartopy_projection()

    ax = basemap_function(extent, ax=ax, projection=projection, **kwargs)

    bounds = self.total_bounds
    gdf = bounds_to_polygons([bounds])
    gdf.crs = self.crs
    gdf.plot(ax=ax, edgecolor="green", facecolor="none", transform=self.get_cartopy_projection(), zorder=2)

    return ax


@add_method("plot_file", gpd.GeoDataFrame, gpd.GeoSeries)
def gpd_plot_file(self=None, **kwargs):
    """
    Plots a map of the outer bounds of GeoDataFrame.

    Parameters
    ----------
    self : GeoDataFrame
        This is not supposed to be used directly, but as the self parameter when called from a GeoDataFrame directly
    kwargs
        arguments for the :func:`geomappy.gpd_plot_world` function

    Returns
    -------
    GeoAxis
    """
    ax = self.plot_world(extent=self.total_bounds, projection=self.get_cartopy_projection(),
                         extent_projection=self.get_cartopy_projection(), **kwargs)

    return ax

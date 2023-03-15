import pyproj
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
from rasterio.enums import Resampling
from geomappy.bounds import bounds_to_polygons
from geomappy.plotting import plot_raster
from geomappy.plotting import plot_classified_raster
from geomappy.basemap import basemap as basemap_function
from geomappy.utils import add_method

# TODO; make a geo_clip function supporting lon_lat
# TODO; clip -> reproject -> clip (Transformer().transform_bounds())
# TODO; automatically switch basemap on when a geoaxes is provided

def _plot_combined_raster(classified, *, da=None, basemap=False, figsize=(10, 10), ax=None, projection=None, xticks=30,
                          yticks=30, fontsize=10, extent=None, basemap_kwargs=None, resampling=Resampling.nearest,
                          **kwargs):
    if da.ndim == 2:
        pass
    elif da.ndim == 3:
        if da.shape[0] == 1:
            da = da[0]
        elif da.shape[0] == 3:
            # todo; this does not work
            da = da.transpose(da.rio.y_dim, da.rio.x_dim, '...')
        else:
            raise IndexError("Bands can only be 1 or 3, corresponding to values or RGB")
    else:
        raise IndexError("Only 2 or 3 dimensional DataArrays are accepted")

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
        elif da.rio.crs is not None:
            projection = da.get_cartopy_projection()
        else:
            raise ValueError("No projection is provided, which is necessary when plotting a basemap. Search order is "
                             "a projection set on a provided Axes, the projection parameter, the CRS of the dataframe.")

        if extent is None:
            extent = da.rio._internal_bounds()
            basemap_kwargs.update({"extent_projection": da.get_cartopy_projection()})

        ax = basemap_function(extent, ax=ax, projection=projection, figsize=figsize, **basemap_kwargs)

        minx, maxx, miny, maxy = ax.get_extent()

        da = (
            da.rio.reproject(projection, resampling=resampling)
              .rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        )

    a = da.to_masked_array()

    if classified:
        return plot_classified_raster(a, ax=ax, figsize=figsize, **kwargs)
    else:
        return plot_raster(a, ax=ax, figsize=figsize, **kwargs)


@add_method("get_cartopy_projection", xr.core.dataarray.DataArray)
def xarray_get_cartopy_projection(self=None):
    if self.rio.crs is None:
        raise ValueError("No projection is set on the DataArray")

    crs = self.rio.crs

    if crs.to_epsg() == 4326:
        return ccrs.PlateCarree()
    elif crs.to_epsg() == 3857:
        return ccrs.Mercator()
    elif crs.to_epsg() is not None:
        return ccrs.epsg(crs.to_epsg())
    else:
        return ccrs.Projection(crs)


@add_method("plot_raster", xr.core.dataarray.DataArray)
def xarray_plot_raster(self=None, *, ax=None, **kwargs):
    """
    Plot data from an DataArray. This function is exposed as ``plot_raster`` in a ``DataArray``.

    Parameters
    ----------
    self : DataArray
        This is not supposed to be used directly, but as the self parameter when called from a DataArray directly
    classified : bool
        Switch between `plot_raster` and `plot_classified_raster`
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
    resampling: Resampling method, optional
            See rasterio.warp.reproject for more details.
    kwargs
        kwargs going to the plot_raster() command

    Returns
    -------
    (:obj:`~matplotlib.axes.Axes`, legend)
        Axes and legend. The legend depends on the `legend` parameter and can be None.
    """
    if self is None:
        return plot_raster(ax=ax, **kwargs)
    elif isinstance(self, xr.DataArray):
        return _plot_combined_raster(classified=False, ax=ax, da=self, **kwargs)
    else:
        raise TypeError("This method does not support positional arguments")


@add_method("plot_classified_raster", xr.core.dataarray.DataArray)
def xarray_plot_classified_raster(self=None, *, ax=None, **kwargs):
    """
    Plot data from an DataArray. This function is exposed as ``plot_classified_raster`` in a ``DataArray``.

    Parameters
    ----------
    self : DataArray
        This is not supposed to be used directly, but as the self parameter when called from a DataArray directly
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
    resampling: Resampling method, optional
        See rasterio.warp.reproject for more details.
    kwargs
        kwargs going to the plot_classified_raster() command

    Returns
    -------
    (:obj:`~matplotlib.axes.Axes`, legend)
        Axes and legend. The legend depends on the `legend` parameter and can be None.
    """
    if self is None:
        return plot_raster(ax=ax, **kwargs)
    elif isinstance(self, xr.DataArray):
        return _plot_combined_raster(classified=True, ax=ax, da=self, **kwargs)
    else:
        raise TypeError("This method does not support positional arguments")


@add_method("plot_world", xr.core.dataarray.DataArray)
def xarray_plot_world(self=None, projection=None, ax=None, extent="global", **kwargs):
    """
    Plots the outer bounds of a DataArray on a world map.

    Parameters
    ----------
    self : DataArray
        This is not supposed to be used directly, but as the self parameter when called from a DataArray directly
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

    bounds = self.rio._internal_bounds()
    gdf = bounds_to_polygons([bounds])
    gdf.crs = self.rio.crs
    gdf.plot(ax=ax, edgecolor="green", facecolor="none", transform=self.get_cartopy_projection(), zorder=2)

    return ax


@add_method("plot_file", xr.core.dataarray.DataArray)
def xarray_plot_file(self=None, **kwargs):
    """
    Plots a map of the outer bounds of DataArray.

    Parameters
    ----------
    self : DataArray
        This is not supposed to be used directly, but as the self parameter when called from a DataArray directly
    kwargs
        arguments for the :func:`geomappy.xarray_plot_world` function

    Returns
    -------
    GeoAxis
    """
    ax = self.plot_world(extent=self.rio._internal_bounds(), projection=self.get_cartopy_projection(),
                         extent_projection=self.get_cartopy_projection(), **kwargs)

    return ax

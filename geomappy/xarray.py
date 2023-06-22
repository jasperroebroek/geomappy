from typing import Optional, Union, Tuple

import cartopy.crs as ccrs
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import pyplot as plt

from geomappy.basemap import basemap as basemap_function, add_gridlines, add_ticks
from geomappy.raster import plot_classified_raster, plot_raster
from geomappy.types import Number, OptionalLegend
from geomappy.utils import add_method, change_between_bounds_and_extent
from geomappy.world import plot_world


def _plot_combined_raster(classified: bool, *, da: xr.DataArray, basemap: bool = True,
                          figsize: Tuple[int, int] = (8, 8), ax: Optional[plt.Axes] = None,
                          lines: Union[Number, Tuple[Number, Number], Tuple[Tuple[Number], Tuple[Number]]] = 30,
                          ticks: Union[Number, Tuple[Number, Number], Tuple[Tuple[Number], Tuple[Number]]] = 30,
                          fontsize: int = 10, **kwargs) -> Tuple[plt.Axes, OptionalLegend]:
    if da.ndim == 3 and da.shape[0] == 1:
        da = da[0]

    if da.ndim != 2:
        raise IndexError("Only 2 or 3 dimensional DataArrays are accepted")

    if isinstance(ax, GeoAxes):
        basemap = True

    if basemap:
        projection = da.get_cartopy_projection()
        ax = basemap_function(ax=ax, projection=projection, figsize=figsize)
        ax.set_extent(da.get_extent(), crs=projection)
        ax.coastlines()
        add_gridlines(ax, lines)
        add_ticks(ax, ticks, fontsize=fontsize)
        kwargs['extent'] = da.get_extent()
        kwargs['transform'] = projection

    a = da.to_masked_array()

    if classified:
        return plot_classified_raster(a, ax=ax, figsize=figsize, **kwargs)
    else:
        return plot_raster(a, ax=ax, figsize=figsize, **kwargs)


@add_method("get_cartopy_projection", xr.core.dataarray.DataArray)
def xarray_get_cartopy_projection(self):
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


@add_method("get_extent", xr.core.dataarray.DataArray)
def xarray_get_extent(self):
    return change_between_bounds_and_extent(self.rio._internal_bounds())


@add_method("plot_raster", xr.core.dataarray.DataArray)
def xarray_plot_raster(self, *, ax: Optional[plt.Axes] = None, **kwargs):
    """
    Plot data from an DataArray. This function is exposed as ``plot_raster`` in a ``DataArray``.

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
        kwargs going to the plot_classified_raster() command

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
def xarray_plot_classified_raster(self, *, ax: Optional[plt.Axes] = None, **kwargs):
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
def xarray_plot_world(self, ax: Optional[plt.Axes] = None, extent: Union[Tuple[int, int, int, int], str] = 'global',
                      projection: ccrs.Projection = ccrs.PlateCarree(),
                      extent_projection: ccrs.Projection = ccrs.PlateCarree(), **kwargs) -> GeoAxes:
    """
    Plots the outer bounds of a DataArray on a world map.

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
    bounds = self.rio._internal_bounds()
    plot_world(bounds, ax=ax, extent=extent, projection=projection, extent_projection=extent_projection,
               bounds_projection=self.get_cartopy_projection(), **kwargs)


@add_method("plot_file", xr.core.dataarray.DataArray)
def xarray_plot_file(self, **kwargs) -> GeoAxes:
    """
    Plots a map of the outer bounds of DataArray.

    Parameters
    ----------
    kwargs
        arguments for the :func:`geomappy.xarray_plot_world` function

    Returns
    -------
    GeoAxes
    """
    return self.plot_world(extent=self.get_extent(), projection=self.get_cartopy_projection(),
                           extent_projection=self.get_cartopy_projection(), **kwargs)

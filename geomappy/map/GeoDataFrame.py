import geopandas as gpd
from functools import wraps
import cartopy.crs as ccrs
import numpy as np

from ..plotting import plot_shapes as _plot_shapes
from ..plotting import plot_classified_shapes as _plot_classified_shapes
from ..plotting import basemap as basemap_function


def add_method(*cls):
    """
    Decorator to add functions to existing classes

    Parameters
    ----------
    cls
        class or classes that the function that the wrapper is placed around will be added to

    Notes
    -----
    https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        for c in cls:
            setattr(c, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator


def _plot_combined_shapes(classified, *, df=None, values=None, basemap=False, figsize=(10, 10), ax=None, log=False,
                          data_epsg=None, plot_epsg=None, xticks=30, yticks=30, resolution="110m", fontsize=10,
                          basemap_kwargs=None, bounds=None, **kwargs):
    """
    Plot data from a GeoDataFrame. Classified or not, depends on the first parameter.

    Parameters
    ----------
    classified : bool
        Switch between `plot_classified_shapes` and `plot_shapes`
    basemap : bool, optional
        plot a basemap behind the data
    figsize : tuple, optional
        matplotlib figsize parameter
    ax : Axes, optional
        matplotlib axes where plot is drawn on
    log : bool, optional
        Plot the colors on a log scale if `classified` is False and only a single layer is selected.
    data_epsg : int, optional
        EPGS code that represents the data. Normally this should be presented within the dataframe itself and this does
        not need to be set.
    plot_epsg : int, optional
        EPSG code that will be used to render the plot, the default is the projection of the data itself.
    xticks : float or list, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used.
    yticks : float or list, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used.
    resolution : {"110m", "50m", "10m"} , optional
        coastline resolution
    fontsize : float/tuple, optional
        fontsize for both the lon/lat ticks and the ticks on the colorbar if one number, if a list of is passed it
        represents the basemap fontsize and the colorbar fontsize.
    basemap_kwargs : dict, optional
        kwargs going to the basemap command
    bounds : list or string, optional
        extent of the plot, if not provided it will plot the extent belonging to dataframe. If the string "global" is
        set the global extent will be used.
    **kwargs
        kwargs going to the plot_map() command

    Returns
    -------
    (:obj:`~matplotlib.axes.Axes` or GeoAxis, legend)
    """
    if isinstance(df, type(None)):
        raise TypeError("Internal error: df not received")

    if isinstance(bounds, type(None)):
        extent = df.total_bounds.tolist()
    elif bounds == 'global':
        extent = [-180, -90, 180, 90]
    else:
        extent = bounds

    if not classified and log and not isinstance(values, type(None)):
        if isinstance(values, str):
            df.loc[:, values] = df.loc[:, values]
        else:
            values = np.log(values)

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
        if 'resolution' not in basemap_kwargs:
            basemap_kwargs.update({'resolution': resolution})

        if isinstance(data_epsg, type(None)):
            crs = df.crs
            if "init" in crs and 'epsg' in crs['init']:
                init = crs['init']
                data_epsg = int(init[init.find(":") + 1:])
            else:
                raise ValueError("EPSG code was not found in the DataFrame crs object, please pass directly")

        if isinstance(plot_epsg, type(None)):
            plot_epsg = data_epsg

        if data_epsg == 4326:
            transform = ccrs.PlateCarree()
        else:
            transform = ccrs.epsg(data_epsg)

        ax = basemap_function(*extent, ax=ax, epsg=plot_epsg, figsize=figsize, **basemap_kwargs)
        kwargs.update({'transform': transform})

    if classified:
        return _plot_classified_shapes(df=df, values=values, ax=ax, figsize=figsize, **kwargs)
    else:
        return _plot_shapes(df=df, values=values, ax=ax, figsize=figsize, **kwargs)


@add_method(gpd.GeoDataFrame, gpd.GeoSeries)
def plot_shapes(self=None, *, ax=None, **kwargs):
    """
    Wrapper around `_plot_shapes`. It redirects to `_plot_combined_shapes` with parameter `classified` = False
    """
    if isinstance(self, type(None)):
        return _plot_shapes(ax=ax, **kwargs)
    elif isinstance(self, gpd.GeoDataFrame):
        return _plot_combined_shapes(classified=False, ax=ax, df=self, **kwargs)
    elif isinstance(self, gpd.GeoSeries):
        return _plot_combined_shapes(classified=False, ax=ax, df=gpd.GeoDataFrame(self), **kwargs)
    else:
        raise TypeError("This method does not support positional arguments")


@add_method(gpd.GeoDataFrame, gpd.GeoSeries)
def plot_classified_shapes(self=None, *, ax=None, **kwargs):
    """
    Wrapper around `_plot_classified_shapes`. It redirects to `_plot_combined_shapes` with parameter
    `classified` = False
    """
    if isinstance(self, type(None)):
        return _plot_classified_shapes(ax=ax, **kwargs)
    elif isinstance(self, gpd.GeoDataFrame):
        return _plot_combined_shapes(classified=True, ax=ax, df=self, **kwargs)
    elif isinstance(self, gpd.GeoSeries):
        return _plot_combined_shapes(classified=True, ax=ax, df=gpd.GeoDataFrame(self), **kwargs)
    else:
        raise TypeError("This method does not support positional arguments")

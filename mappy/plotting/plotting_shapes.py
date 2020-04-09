import cartopy
from geopandas.plotting import plot_polygon_collection, plot_linestring_collection, plot_point_collection
from matplotlib.colors import Colormap, Normalize
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
import cartopy.crs as ccrs

from .colors import add_colorbar
from .misc import _determine_cmap_boundaries


def plot_shapes(lat=None, lon=None, values=None, s=None, df=None, bins=None, bin_labels=None, cmap=None, vmin=None,
                vmax=None, legend="colorbar", clip_legend=False, ax=None, figsize=(10, 10), legend_kwargs=None,
                aspect=30, pad_fraction=0.6, linewidth=0, **kwargs):
    """
    Plot shapes in a continuous fashion

    Parameters
    ----------
    lat, lon : array-like
        Lattitude and Longitude
    values : array-like or str
        Values at each pair of lattitude and longitude entries, or if `df` is set, it is the name of the column that
        is used to plot the data. The default string value is "values".
    s : array-like, optional
        Size values for each pair of lattitude and longitude entries. If `df` is set, the column named 's' will be used
        if present. This only works with point data.
    df : GeoDataFrame, optional
        GeoDataFrame with columns values and s as described above.
    bins : array-like, optional
        List of bins that will be used to create a BoundaryNorm instance to discretise the plotting. This does not work
        in conjunction with vmin and vmax. Bins in that case will take the upper hand.  Alternatively a 'norm' parameter
        can be passed on in the have outside control on the behaviour. This list should contain at least two numbers,
        as otherwise the extends can't be drawn.
    bin_labels : array-like, optional
        This parameter can be used to override the labels on the colorbar. Should have the same length as bins.
    cmap : matplotlib.cmap or str, optional
        Matplotlib cmap instance or string that will be recognized by matplotlib
    vmin, vmax : float, optional
        vmin and vmax parameters for plt.imshow(). This does have no effect in conjunction with bins being provided.
    legend : {'colorbar', 'legend', False}, optional
        Legend type that will be plotted. The 'legend' type will only work if bins are specified.
    clip_legend : bool, optional
        Clip the legend to the minimum and maximum of bins are provided. If False the colormap will remain intact over
        the whole provided bins, which potentially lowers contrast a lot.
    ax : matplotlib.Axes, optional
        Axes object. If not provided it will be created on the fly.
    figsize : tuple, optional
        Matplotlib figsize parameter. Default is (10,10)
    legend_kwargs : dict, optional
        Extra parameters for the colorbar call
    aspect : float, optional
        aspact ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    linewidth : numeric, optional
        width of the line around the shapes
    **kwargs
        Keyword arguments for the geopandas plotting functions: plot_point_collection, plot_polygon_collection and
        plot_linestring_collection

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Returns
    -------
    Axes
    """
    if isinstance(ax, type(None)):
        f, ax = plt.subplots(figsize=figsize)

    if isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot):
        if 'transform' not in kwargs:
            kwargs.update({'transform': ccrs.PlateCarree()})

    if isinstance(cmap, type(None)):
        cmap = plt.cm.get_cmap("viridis")
    elif isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    elif not isinstance(cmap, Colormap):
        raise TypeError("cmap not recognized")

    if isinstance(df, type(None)):
        lon = np.array(lon)
        lat = np.array(lat)
        df = gpd.GeoDataFrame({'geometry': [Point(lon[i], lat[i]) for i in range(len(lon))],
                               'values': values,
                               's': s})
        values = "values"
        if isinstance(s, type(None)):
            markersize = None
        else:
            markersize = df.loc[:, "s"]
    else:
        if isinstance(values, type(None)):
            values = "values"
        if 's' in df.columns:
            markersize = df.loc[:, "s"]
        else:
            markersize = None

    if isinstance(bins, type(None)):
        minimum = df.loc[:, values].min()
        maximum = df.loc[:, values].max()
        if isinstance(vmin, type(None)):
            vmin = minimum
        if isinstance(vmax, type(None)):
            vmax = maximum

        if minimum < vmin and maximum > vmax:
            extend = 'both'
        elif minimum < vmin and not maximum > vmax:
            extend = 'min'
        elif not minimum < vmin and maximum > vmax:
            extend = 'max'
        elif not minimum < vmin and not maximum > vmax:
            extend = 'neither'

        norm = Normalize(vmin, vmax)
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)

    else:
        cmap, norm, legend_patches, extend = _determine_cmap_boundaries(m=df.loc[:, values], bins=bins, cmap=cmap,
                                                                        clip_legend=clip_legend)
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)

        if legend == "legend":
            if isinstance(legend_kwargs, type(None)):
                legend_kwargs = {"facecolor": "white", "edgecolor": "lightgrey", 'loc': 0}
            ax.legend(handles=legend_patches, **legend_kwargs)

    colors = pd.Series(cmap(norm(df.loc[:, values].values)).tolist())

    geom_types = df.geometry.type
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray((geom_types == "LineString") | (geom_types == "MultiLineString"))
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = df.geometry[poly_idx]
    if not polys.empty:
        plot_polygon_collection(ax, polys, color=colors[poly_idx], linewidth=linewidth, **kwargs)

    # plot all LineStrings and MultiLineString components in same collection
    lines = df.geometry[line_idx]
    if not lines.empty:
        plot_linestring_collection(ax, lines, color=colors[line_idx], linewidth=linewidth, **kwargs)

    # plot all Points in the same collection
    points = df.geometry[point_idx]
    if not points.empty:
        if isinstance(markersize, np.ndarray):
            markersize = markersize[point_idx]
        plot_point_collection(ax, points,  color=colors[point_idx], markersize=markersize, linewidth=linewidth, **kwargs)

    if isinstance(legend_kwargs, type(None)):
        legend_kwargs = {}

    if 'extend' not in legend_kwargs:
        legend_kwargs.update({'extend': extend})

    if legend == "colorbar":
        cbar = add_colorbar(im=sm, ax=ax, aspect=aspect, pad_fraction=pad_fraction, **legend_kwargs)
        if not isinstance(bins, type(None)):
            if isinstance(bin_labels, type(None)):
                bin_labels = bins
            if 'position' in legend_kwargs and legend_kwargs['position'] == 'bottom':
                cbar.ax.set_xticklabels(bin_labels)
            else:
                cbar.ax.set_yticklabels(bin_labels)

    return ax


def plot_classified_shapes(**kwargs):
    raise NotImplementedError("This will be implemented in the future")

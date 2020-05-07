import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas.plotting import plot_polygon_collection, plot_linestring_collection, plot_point_collection
from matplotlib.colors import Colormap, ListedColormap, Normalize
from mpl_toolkits import axes_grid1

from .colors import add_colorbar, cmap_random, create_colorbar_axes, cmap_discrete
from .misc import _determine_cmap_boundaries_discrete, _create_geometry_values_and_sizes, \
    _determine_cmap_boundaries_continuous, cbar_decorator
from ..ndarray_functions import nanunique, nandigitize
from ..plotting import legend_patches as lp


def _plot_geometries(ax, df, colors, linewidth, markersize, **kwargs):
    """
    internal plotting function for geometries, called by plot_shapes and plot_classified_shapes

    Parameters
    ----------
    ax : matplotlib.Axes
        axes to plot on
    df : GeoDataFrame
        geodataframe containing the geometries
    colors : pd.Series
        Series object containing the colors
    linewidth : numeric
        linewidth of the geometries
    markersize : pd.Series
        size of points in `df`
    **kwargs
        Keyword arguments for the geopandas plotting functions: plot_point_collection, plot_polygon_collection and
        plot_linestring_collection
    """
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
        plot_point_collection(ax, points, color=colors[point_idx], markersize=markersize, linewidth=linewidth, **kwargs)


def plot_shapes(lat=None, lon=None, values=None, s=None, df=None, bins=None, bin_labels=None, cmap=None, vmin=None,
                vmax=None, legend="colorbar", clip_legend=False, ax=None, figsize=(10, 10), legend_ax=None,
                legend_kwargs=None, fontsize=None, aspect=30, pad_fraction=0.6, linewidth=0, force_equal_figsize=None,
                nan_color="White", **kwargs):
    """
    Plot shapes in a continuous fashion

    Parameters
    ----------
    lat, lon : array-like
        Latitude and Longitude
    values : array-like or numeric or str
        Values at each pair of latitude and longitude entries if list like. A single numeric value will be cast to all
        geometries. If `df` is set a string can be passed to values which will be interpreted as the name of the column
        holding the values (the default string if None is set is "values).
    s : array-like, optional
        Size values for each pair of latitude and longitude entries if list like. A single numeric value will be cast
        to all geometries. If `df` is set a string will be interpreted as the name of the column holding the sizes. If
        None is set no sizes will be set.
    df : GeoDataFrame, optional
        Optional GeoDataframe which can be used to plot different shapes than points.
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
    legend_ax : `matplotlib.Axes`, optional
        Axes object that the legend will be drawn on
    legend_kwargs : dict, optional
        Extra parameters to create and decorate the colorbar or the call to `plt.legend` if `legend` == "legend"
        For the colorbar creation: shrink, position and extend (which would override the internal behaviour)
        For the colorbar decorator see `cbar_decorate`.
    fontsize : float, optional
        Fontsize of the legend
    aspect : float, optional
        aspact ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    linewidth : numeric, optional
        width of the line around the shapes
    force_equal_figsize : bool, optional
        when plotting with a colorbar the figure is going be slightly smaller than when you are using `legend` or non
        at all. This parameter can be used to force equal sizes, meaning that the version with a `legend` is going to
        be slightly reduced.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'white'
    **kwargs
        kwargs for the _plot_geometries function

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Returns
    -------
    (Axes or GeoAxes, legend)
    legend depends on the `legend` parameter
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

    geometry, values, markersize = _create_geometry_values_and_sizes(lat=lat, lon=lon, values=values, s=s, df=df)

    if not isinstance(bins, type(None)) and len(bins) == 1:
        nan_mask = np.isnan(values)
        values = (values > bins[0]).astype(float)
        values[nan_mask] = np.nan
        ax, legend_ = plot_classified_shapes(df=geometry, values=values, s=markersize, linewidth=linewidth,
                                             colors=['lightgrey', 'red'], labels=[f'< {bins[0]}', f'> {bins[0]}'],
                                             ax=ax, legend_ax=legend_ax, legend=legend, legend_kwargs=legend_kwargs,
                                             aspect=aspect, pad_fraction=pad_fraction, force_equal_figsize=False,
                                             nan_color=nan_color, **kwargs)

    elif np.issubdtype(values.dtype, np.bool_):
        if isinstance(legend_kwargs, type(None)):
            if legend == "colorbar" or not legend:
                legend_kwargs = {}
            elif legend == "legend":
                legend_kwargs = {"facecolor": "white", "edgecolor": "lightgrey", 'loc': 0}

        if 'fontsize' not in legend_kwargs and not isinstance(fontsize, type(None)):
            legend_kwargs['fontsize'] = fontsize

        if isinstance(bins, type(None)):
            norm, extend = _determine_cmap_boundaries_continuous(m=values, vmin=vmin, vmax=vmax)
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)

        else:
            cmap, norm, legend_patches, extend = _determine_cmap_boundaries_discrete(m=values, bins=bins, cmap=cmap,
                                                                                     clip_legend=clip_legend)
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)

            if legend == "legend":
                if isinstance(legend_ax, type(None)):
                    legend_ax = ax
                legend_ = legend_ax.legend(handles=legend_patches, **legend_kwargs)

        colors = pd.Series(cmap(norm(values)).tolist())
        colors[pd.isna(values)] = nan_color

        _plot_geometries(ax, geometry, colors, linewidth, markersize, **kwargs)

        if legend == "colorbar":
            cbar = add_colorbar(im=sm, ax=ax, cax=legend_ax, aspect=aspect, pad_fraction=pad_fraction,
                                shrink=legend_kwargs.pop("shrink", 1), extend=legend_kwargs.pop("extend", extend),
                                position=legend_kwargs.pop("position", "right"))
            cbar_decorator(cbar, ticks=bins, ticklabels=bin_labels, **legend_kwargs)

        if legend == "colorbar":
            legend_ = cbar
        elif not legend:
            legend_ = None

    else:
        # Boolean values
        ax, legend_ = plot_classified_shapes(df=geometry, values=values.astype(int), s=markersize,
                                             linewidth=linewidth, colors=['lightgrey', 'red'],
                                             labels=[f'< {bins[0]}', f'> {bins[0]}'], ax=ax, legend_ax=legend_ax,
                                             legend=legend, legend_kwargs=legend_kwargs, aspect=aspect,
                                             pad_fraction=pad_fraction, force_equal_figsize=False, nan_color=nan_color,
                                             **kwargs)

    if force_equal_figsize and legend != 'colorbar':
        create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction,
                             position=legend_kwargs.get("position", "right")).axis("off")

    return ax, legend_


def plot_classified_shapes(lat=None, lon=None, values=None, s=None, df=None, bins=None, colors=None, cmap="tab10",
                           labels=None, legend="legend", clip_legend=False, ax=None, figsize=(10, 10),
                           suppress_warnings=False, legend_ax=None, legend_kwargs=None, fontsize=None, aspect=30,
                           pad_fraction=0.6, linewidth=0, force_equal_figsize=False, nan_color="White", **kwargs):
    """
    Plot shapes with discrete classes or index

    Parameters
    ----------
    lat, lon : array-like
        Latitude and Longitude
    values : array-like or numeric or str
        Values at each pair of latitude and longitude entries if list like. A single numeric value will be cast to all
        geometries. If `df` is set a string can be passed to values which will be interpreted as the name of the column
        holding the values (the default string if None is set is "values).
    s : array-like, optional
        Size values for each pair of latitude and longitude entries if list like. A single numeric value will be cast
        to all geometries. If `df` is set a string will be interpreted as the name of the column holding the sizes. If
        None is set no sizes will be set.
    df : GeoDataFrame, optional
        Optional GeoDataframe which can be used to plot different shapes than points.
    bins : list, optional
        list of either bins as used in np.digitize or unique values corresponding to `colors` and `labels`. By default
        this parameter is not necessary, the unique values are taken from the input map
    colors : list, optional
        List of colors in a format understandable by matplotlib. By default random colors are taken
    cmap : matplotlib cmap or str
        Can be used to set a colormap when no colors are provided.
    labels : list, optional
        list of labels for the different classes. By default the unique values are taken as labels
    legend : {'legend', 'colorbar', False}, optional
        Presence and type of legend. 'Legend' wil insert patches, 'colorbar' wil insert a colorbar and False will
        prevent any legend to be printed.
    clip_legend : bool, optional
        remove the items from the legend that don't occur on the map but are passed in
    ax : axes, optional
        matplotlib axes to plot the map on. If not given it is created on the fly. A cartopty GeoAxis can be provided.
    figsize : tuple, optional
        Matplotlib figsize parameter. Default is (10,10)
    suppress_warnings : bool, optional
        By default 10 classes is the maximum that can be plotted. If set to True this maximum is removed
    legend_ax : `matplotlib.Axes`, optional
        Axes object that the legend will be drawn on
    legend_kwargs : dict, optional
        Extra parameters to create and decorate the colorbar or the call to `plt.legend` if `legend` == "legend"
        For the colorbar creation: shrink, position and extend (which would override the internal behaviour)
        For the colorbar decorator see `cbar_decorate`.
    fontsize : float, optional
        Fontsize of the legend
    aspect : float, optional
        aspact ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    linewidth : numeric, optional
        width of the line around the shapes
    force_equal_figsize : bool, optional
        when plotting with a colorbar the figure is going be slightly smaller than when you are using `legend` or non
        at all. This parameter can be used to force equal sizes, meaning that the version with a `legend` is going to
        be slightly reduced.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'white'
    **kwargs : dict, optional
        kwargs for the _plot_geometries function

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Returns
    -------
    (Axes or GeoAxes, legend)
    legend depends on the `legend` parameter
    """
    if isinstance(ax, type(None)):
        f, ax = plt.subplots(figsize=figsize)
    elif isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
        if "transform" not in kwargs:
            kwargs["transform"] = ccrs.PlateCarree()

    geometry, values, markersize = _create_geometry_values_and_sizes(lat=lat, lon=lon, values=values, s=s, df=df)

    if isinstance(bins, type(None)):
        data = values[~np.isnan(values)]
        bins = np.unique(data)
    else:
        bins = np.array(bins)
        bins.sort()

    if len(bins) > 10 and not suppress_warnings:
        raise ValueError("Number of bins above 10, this creates issues with visibility")

    if not isinstance(colors, type(None)):
        if len(bins) != len(colors):
            raise IndexError(f"length of bins and colors don't match\nbins: {len(bins)}\ncolors: {len(colors)}")
    else:
        colors = cmap_discrete(cmap=cmap, n=len(list), return_type="list")

    if not isinstance(labels, type(None)):
        if len(bins) != len(labels):
            raise IndexError("length of bins and labels don't match")
    else:
        labels = list(bins)

    colors = np.array(colors)
    labels = np.array(labels)

    m_binned = nandigitize(values, bins=bins, right=True)

    m_binned_unique = nanunique(m_binned).astype(int)
    if (~np.all(m_binned_unique == np.linspace(0, m_binned_unique.max(), num=m_binned_unique.size)) or \
        len(m_binned_unique) != len(colors)) and clip_legend:
        colors = colors[m_binned_unique]
        labels = labels[m_binned_unique]
        m_binned = nandigitize(m_binned, bins=m_binned_unique, right=True)
        bins = m_binned_unique

    cmap = ListedColormap(colors)
    norm = Normalize(vmin=0, vmax=bins.size - 1)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    legend_patches = lp(colors=colors, labels=labels, edgecolor='lightgrey')

    plotting_colors = pd.Series(cmap(norm(m_binned)).tolist())
    plotting_colors[pd.isna(values)] = nan_color
    _plot_geometries(ax, geometry, plotting_colors, linewidth, markersize, **kwargs)

    # Legend
    if isinstance(legend_kwargs, type(None)):
        if legend == "colorbar" or not legend:
            legend_kwargs = {}
        elif legend == "legend":
            legend_kwargs = {"facecolor": "white", "edgecolor": "lightgrey", 'loc': 0}

    if 'fontsize' not in legend_kwargs and not isinstance(fontsize, type(None)):
        legend_kwargs['fontsize'] = fontsize

    if legend == "legend":
        if isinstance(legend_ax, type(None)):
            legend_ax = ax
        legend_ = legend_ax.legend(handles=legend_patches, **legend_kwargs)
    elif legend == "colorbar":
        cbar = add_colorbar(im=sm, ax=ax, cax=legend_ax, shrink=legend_kwargs.pop("shrink", 1),
                            position=legend_kwargs.pop("position", "right"), aspect=aspect, pad_fraction=pad_fraction)

        boundaries = cbar._boundaries
        tick_locations = [(boundaries[i]-boundaries[i-1])/2+boundaries[i-1]
                          for i in range(1, len(boundaries))]

        cbar_decorator(cbar, ticks=tick_locations, ticklabels=labels, **legend_kwargs)

    if force_equal_figsize and legend != 'colorbar':
        create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction,
                             position=legend_kwargs.get("position", "right")).axis("off")

    if legend == "colorbar":
        legend_ = cbar
    elif not legend:
        legend_ = None

    return ax, legend_

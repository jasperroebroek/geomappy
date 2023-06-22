from typing import Union, Optional, Tuple, Dict

import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from geopandas import GeoDataFrame, GeoSeries
from geopandas.plotting import _plot_polygon_collection, _plot_linestring_collection, _plot_point_collection
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, Colormap
from shapely import Point

from geomappy.axes_decoration import prepare_axes
from geomappy.classified import parse_classified_plot_params
from geomappy.legends import add_legend
from geomappy.scalar import parse_scalar_plot_params
from geomappy.types import Number, Color, OptionalLegend


def _create_geometries_and_values_from_lat_lon(lat: Optional[np.ndarray] = None,
                                               lon: Optional[np.ndarray] = None,
                                               values: Optional[np.ndarray] = None,
                                               s: Optional[Union[Number, np.ndarray]] = None) \
        -> Tuple[GeoSeries, np.ndarray, Optional[np.ndarray]]:
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    values = np.asarray(values)

    if lon.size != lat.size:
        raise IndexError("Mismatch in length of `lat` and `lon`")

    if s is None:
        markersize = None
    else:
        markersize = np.asarray(s)

    geometry = gpd.GeoSeries([Point(lon[i], lat[i]) for i in range(len(lon))])
    return geometry, values, markersize


def _create_geometries_and_values_from_gdf(values: Optional[Union[str, Number, np.ndarray]] = None,
                                           s: Optional[Union[Number, np.ndarray]] = None,
                                           df: Optional[Union[GeoDataFrame, GeoSeries]] = None) \
        -> Tuple[GeoSeries, np.ndarray, Optional[np.ndarray]]:
    geometry = df['geometry']
    if isinstance(values, str):
        if values not in df.columns:
            raise ValueError("values are not present in dataframe")
        values = df.loc[:, values].values
    else:
        values = np.array(values).flatten()

    if s is None:
        markersize = None
    elif isinstance(s, str):
        markersize = None
        if s in df.columns:
            markersize = df.loc[:, s].values
    else:
        markersize = np.array(s).flatten()

    return geometry, values, markersize


def _create_geometry_values_and_sizes(lat: Optional[np.ndarray] = None,
                                      lon: Optional[np.ndarray] = None,
                                      values: Optional[Union[str, Number, np.ndarray]] = None,
                                      s: Optional[Union[Number, np.ndarray]] = None,
                                      df: Optional[Union[GeoDataFrame, GeoSeries]] = None) \
        -> Tuple[GeoSeries, np.ma.MaskedArray, Optional[np.ndarray]]:
    """
    Function that deals with the input data for `plot_shapes` and `plot_classified_shapes`. Lat and Lon will be used
    if `df` is not given. `Values` and `s` will be cast to the length of the geometries, or set to 1 and None
    respectively if not provided.

    Parameters
    ----------
    lat, lon : array-like
        Latitude and Longitude
    values : array-like or numeric or str
        Values at each geometry. A single numeric value will be cast to all geometries. If `df` is set a string can
        be passed to values which will be interpreted as the name of the column holding the values.
    s : array-like or str, optional
        Size for each geometry. A single numeric value will be cast to all geometries. If `df` is set a string will
        be interpreted as the name of the column holding the sizes. If None is set no sizes will be set.
    df : GeoDataFrame, optional
        GeoDataFrame containing the geometries that will be used for plotting in a 'geometry' column.

    Returns
    -------
    geometry, values, markersize
    markersize can be None if None is passed in as `s`
    """
    if df is None:
        geometry, values, markersize = _create_geometries_and_values_from_lat_lon(lat, lon, values, s)
    else:
        geometry, values, markersize = _create_geometries_and_values_from_gdf(values, s, df)

    if values.size == 1:
        if values[0] is None or values[0] == np.nan:
            values = np.array(1)
        values = values.repeat(geometry.size)

    if values.size != geometry.size:
        raise IndexError("Mismatch length sizes and geometries")

    if markersize is not None:
        if markersize.size == 1:
            if markersize[0] is None:
                markersize = None
            else:
                markersize = markersize.repeat(geometry.size)

        if markersize.size != geometry.size:
            raise IndexError("Mismatch length of `s` and coordindates")

    values = np.ma.fix_invalid(values)

    return geometry, values, markersize


def _plot_geometries(ax: plt.Axes, geometries: GeoSeries, colors: np.ndarray, linewidth: float,
                     markersize: Union[float, np.ndarray], **kwargs) -> None:
    """
    internal plotting function for geometries, called by plot_shapes and plot_classified_shapes

    Parameters
    ----------
    ax : matplotlib.Axes
        axes to plot on
    geometries : GeoSeries
        geoseries containing the geometries
    colors : np.ndarray
        Series object containing the colors
    linewidth : numeric
        linewidth of the geometries
    markersize : np.ndarray or numeric
        size of points in `df`
    **kwargs
        Keyword arguments for the geopandas plotting functions: plot_point_collection, plot_polygon_collection and
        plot_linestring_collection

    Notes
    -----
    Setting either ``facecolor`` or ``edgecolor`` does the same as in geopandas. It overwrites the behaviour of this
    function.
    """
    geom_types = geometries.type
    poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
    line_idx = np.asarray((geom_types == "LineString") | (geom_types == "MultiLineString"))
    point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

    facecolor = kwargs.pop('facecolor', colors)
    edgecolor = kwargs.pop('edgecolor', colors)

    if not isinstance(facecolor, np.ndarray):
        facecolor = np.asarray(facecolor).repeat(geometries.shape[0])
    if not isinstance(edgecolor, np.ndarray):
        edgecolor = np.asarray(edgecolor).repeat(geometries.shape[0])

    # plot all Polygons and all MultiPolygon components in the same collection
    polys = geometries[poly_idx]
    if not polys.empty:
        _plot_polygon_collection(ax, polys, facecolor=facecolor[poly_idx], edgecolor=edgecolor[poly_idx],
                                 linewidth=linewidth, **kwargs)

    # plot all LineStrings and MultiLineString components in same collection
    lines = geometries[line_idx]
    if not lines.empty:
        _plot_linestring_collection(ax, lines, facecolor=facecolor[line_idx], edgecolor=edgecolor[line_idx],
                                    linewidth=linewidth, **kwargs)

    # plot all Points in the same collection
    points = geometries[point_idx]
    if not points.empty:
        if isinstance(markersize, np.ndarray):
            markersize = markersize[point_idx]
        _plot_point_collection(ax, points, facecolor=facecolor[point_idx], edgecolor=edgecolor[point_idx],
                               markersize=markersize, linewidth=linewidth, **kwargs)


def plot_classified_shapes(lat: Optional[np.ndarray] = None,
                           lon: Optional[np.ndarray] = None,
                           values: Optional[Union[str, Number, np.ndarray]] = None,
                           s: Optional[Union[Number, np.ndarray]] = None,
                           df: Optional[Union[GeoDataFrame, GeoSeries]] = None,
                           levels: Optional[Tuple[Number]] = None,
                           colors: Optional[Tuple[Color]] = None,
                           cmap: Union[str, Colormap] = "Set1",
                           labels: Optional[Tuple[str]] = None,
                           legend: Optional[str] = "colorbar",
                           ax: Optional[plt.Axes] = None,
                           figsize: Optional[Tuple[int, int]] = None,
                           suppress_warnings: bool = False,
                           legend_kw: Optional[Dict] = None,
                           linewidth: Number = 1,
                           nan_color: Optional[Color] = None, **kwargs) -> Tuple[plt.Axes, OptionalLegend]:
    """
    Plot shapes with discrete classes or index

    Parameters
    ----------
    lat, lon : array-like
        Latitude and Longitude
    values : array-like or numeric or str
        Values at each pair of latitude and longitude entries if list like. A single numeric value will be cast to all
        geometries. If `df` is set, a string can be passed to values which will be interpreted as the name of the column
        holding the values.
    s : array-like, optional
        Size values for each pair of latitude and longitude entries if list like. A single numeric value will be cast
        to all geometries. If `df` is set, a string will be interpreted as the name of the column holding the sizes. If
        None is set no sizes will be set.
    df : GeoDataFrame, optional
        Optional GeoDataframe which can be used to plot different shapes than points.
    levels : list, optional
        list of either bins as used in np.digitize or unique values corresponding to `colors` and `labels`. By default
        this parameter is not necessary, the unique values are taken from the input map
    colors : list, optional
        List of colors in a format understandable by matplotlib. By default colors will be taken from cmap
    cmap : matplotlib cmap or str
        Can be used to set a colormap when no colors are provided. The default is 'Set1'
    labels : list, optional
        list of labels for the different classes. By default the unique values are taken as labels
    legend : {'legend', 'colorbar', None}, optional
        Presence and type of legend. 'Legend' wil insert patches, 'colorbar' wil insert a colorbar and None will
        prevent any legend to be printed.
    ax : axes, optional
        matplotlib axes to plot the map on. If not given it is created on the fly. A cartopty GeoAxis can be provided.
    figsize : tuple, optional
        Matplotlib figsize parameter.
    suppress_warnings : bool, optional
        By default 9 classes is the maximum that can be plotted. If set to True this maximum is removed
    linewidth : numeric, optional
        width of the line around the shapes
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'white'
    **kwargs : dict, optional
        kwargs for the `_plot_geometries` function

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Setting either ``facecolor`` or ``edgecolor`` does the same as in geopandas. It overwrite the behaviour of this
    function.

    Returns
    -------
    (:obj:`~matplotlib.axes.Axes`, legend)
        Axes and legend. The legend depends on the `legend` parameter and can be None.
    """
    if values is None:
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = "lightgrey"
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = "black"
        legend = None

    if isinstance(ax, GeoAxes):
        if 'transform' not in kwargs:
            kwargs["transform"] = ccrs.PlateCarree()

    geometry, values, markersize = _create_geometry_values_and_sizes(lat, lon, values, s, df)
    cmap, norm = parse_classified_plot_params(values, levels=levels, colors=colors, cmap=cmap, nan_color=nan_color,
                                              suppress_warnings=suppress_warnings)

    colors = cmap(norm(values))
    ax = prepare_axes(ax, figsize)
    _plot_geometries(ax, geometry, colors, linewidth, markersize, **kwargs)

    if legend_kw is None:
        legend_kw = {}

    if labels is None and levels is None:
        labels = np.unique(values)
    elif labels is None:
        labels = levels

    l = add_legend('classified', legend, ax=ax, labels=labels, norm=norm, cmap=cmap, **legend_kw)

    return ax, l


def plot_shapes(lat: Optional[np.ndarray] = None,
                lon: Optional[np.ndarray] = None,
                values: Optional[Union[str, Number, np.ndarray]] = None,
                s: Optional[Union[Number, np.ndarray]] = None,
                df: Optional[Union[GeoDataFrame, GeoSeries]] = None,
                bins: Optional[Tuple[Number]] = None,
                cmap: Optional[Colormap] = None,
                norm: Optional[Normalize] = None,
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                ax: Optional[plt.Axes] = None,
                legend: Optional[str] = "colorbar",
                figsize: Optional[Tuple[int, int]] = None,
                legend_kw: Optional[Dict] = None,
                linewidth: float = 1,
                nan_color: Optional[Color] = None, **kwargs) -> Tuple[plt.Axes, OptionalLegend]:
    """
    Plot shapes in a continuous fashion

    Parameters
    ----------
    lat, lon : array-like
        Latitude and Longitude
    values : array-like or numeric or str
        Values at each pair of latitude and longitude entries if list like. A single numeric value will be cast to all
        geometries. If `df` is set, a string can be passed to values which will be interpreted as the name of the column
        holding the values.
    s : array-like, optional
        Size values for each pair of latitude and longitude entries if list like. A single numeric value will be cast
        to all geometries. If `df` is set, a string will be interpreted as the name of the column holding the sizes. If
        None is set no sizes will be set.
    df : GeoDataFrame, optional
        Optional GeoDataframe which can be used to plot different shapes than points. A geometry column is expected
    bins : array-like, optional
        List of bins that will be used to create a BoundaryNorm instance to discretise the plotting. This does not work
        in conjunction with vmin and vmax. Bins in that case will take the upper hand.
    cmap : matplotlib.cmap or str, optional
        Matplotlib cmap instance or string that will be recognized by matplotlib
    norm : matplotlib.Normalize, optional
        Optional normalizer. Should not be provided together with bins.
    vmin, vmax : float, optional
        vmin and vmax parameters for plt.imshow(). This does have no effect in conjunction with bins being provided.
    legend : {'colorbar', 'legend', False}, optional
        Legend type that will be plotted. The 'legend' type will only work if bins are specified.
    ax : matplotlib.Axes, optional
        Axes object. If not provided it will be created on the fly.
    figsize : tuple, optional
        Matplotlib figsize parameter. Default is (10,10)
    legend_kw : dict, optional
        Extra parameters to create and decorate the colorbar or the call to `plt.legend` if `legend` == "legend"
        For the colorbar creation: shrink, position and extend (which would override the internal behaviour)
        For the colorbar decorator see `cbar_decorate`.
    linewidth : numeric, optional
        width of the line around the shapes
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'white'
    **kwargs
        kwargs for the _plot_geometries function

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Setting either ``facecolor`` or ``edgecolor`` does the same as in geopandas. It overwrite the behaviour of this
    function.

    Returns
    -------
    (:obj:`~matplotlib.axes.Axes`, legend)
        Axes and legend. The legend depends on the `legend` parameter and can be None.
    """
    if values is None:
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = "lightgrey"
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = "black"
        legend = None

    if isinstance(ax, GeoAxes):
        if 'transform' not in kwargs:
            kwargs["transform"] = ccrs.PlateCarree()

    geometry, values, markersize = _create_geometry_values_and_sizes(lat, lon, values, s, df)

    if bins is not None and len(bins) == 1:
        # If only one bin is present the data will be converted to a boolean array
        values = values > bins[0]

    if np.issubdtype(values.dtype, np.bool_):
        plot_classified_shapes(lat, lon, values, s, df, labels=("False", "True"), colors=("Lightgrey", "Red"), ax=ax,
                               figsize=figsize, legend=legend, legend_kw=legend_kw, linewidth=linewidth,
                               nan_color=nan_color, **kwargs)

    cmap, norm = parse_scalar_plot_params(values, cmap=cmap, bins=bins, vmin=vmin, vmax=vmax, norm=norm,
                                          nan_color=nan_color)
    colors = cmap(norm(values))
    ax = prepare_axes(ax, figsize)
    _plot_geometries(ax, geometry, colors, linewidth, markersize, **kwargs)

    if legend_kw is None:
        legend_kw = {}
    l = add_legend('scalar', legend, ax=ax, norm=norm, cmap=cmap, **legend_kw)

    return ax, l

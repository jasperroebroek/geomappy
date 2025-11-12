from collections.abc import Callable

import geopandas as gpd
import matplotlib.axes
import matplotlib.collections
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from geopandas.plotting import (
    _plot_linestring_collection,
    _plot_point_collection,
    _plot_polygon_collection,
)
from matplotlib import pyplot as plt
from matplotlib.colorizer import Colorizer
from matplotlib.colors import Colormap, Normalize
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike
from shapely.geometry import Point

from geomappy.colorizer import create_classified_colorizer, create_scalar_colorizer
from geomappy.legends import get_legend_creator
from geomappy.types import ExtendType, LegendCreator, LegendType
from geomappy.utils import check_increasing_and_unique, determine_extend, expand, get_data_range, parse_levels


def _create_geometries_and_values_from_lat_lon(
    values: ArrayLike | str | float | None,
    lat: ArrayLike | None = None,
    lon: ArrayLike | None = None,
    s: ArrayLike | str | float | None = None,
) -> tuple[GeoSeries, np.ndarray | None, np.ndarray | None]:
    if isinstance(values, str):
        raise TypeError('When no dataframe is provided, `values` need to be either numeric or an array')
    if isinstance(s, str):
        raise TypeError('When no dataframe is provided, `s` need to be either numeric or an array')
    if lat is None or lon is None:
        raise ValueError('When no dataframe is provided, lat and lon need to be provided')

    lon = np.asarray(lon).flatten()
    lat = np.asarray(lat).flatten()
    if lon.size != lat.size:
        raise IndexError('Mismatch in length of `lat` and `lon`')

    values = None if values is None else np.asarray(values).flatten()
    markersize = None if s is None else np.asarray(s).flatten()

    geometry = gpd.GeoSeries([Point(x, y) for x, y in zip(lon, lat)])
    return geometry, values, markersize


def _create_geometries_and_values_from_gdf(
    df: GeoDataFrame | GeoSeries,
    values: ArrayLike | str | float | None,
    s: ArrayLike | str | float | None = None,
) -> tuple[GeoSeries, np.ndarray | None, np.ndarray | None]:
    geometry = df['geometry']

    if isinstance(values, str):
        values = df[values].values
    values = None if values is None else np.asarray(values).flatten()

    match s:
        case None:
            markersize = None
        case str():
            markersize = df[s].values
        case _:
            markersize = np.asarray(s).flatten()

    return geometry, values, markersize


def _create_geometry_values_and_sizes(
    lat: ArrayLike | None = None,
    lon: ArrayLike | None = None,
    values: ArrayLike | str | float | None = None,
    s: ArrayLike | str | float | None = None,
    df: GeoDataFrame | GeoSeries | None = None,
) -> tuple[GeoSeries, np.ndarray | None, np.ndarray | None]:
    """
    Prepare geometry, value, and size arrays for shape plotting functions.

    Uses `lat` and `lon` if no `df` is provided. `values` and `s` are expanded
    to match the length of the geometries.

    Parameters
    ----------
    lat, lon : array_like, optional
        Latitude and longitude values.
    values : array_like, numeric, or str, optional
        Values to associate with each geometry. If a single number is provided, it is
        broadcast to all geometries. If `df` is provided and `values` is a string,
        it will be interpreted as the column name.
    s : array_like, numeric, or str, optional
        Marker sizes. Same broadcasting rules as `values`.
    df : :class:`~geopandas.GeoDataFrame` or :class:`~geopandas.GeoSeries`, optional
        DataFrame/Series containing the geometries. Must have a `geometry` column
        if a DataFrame is provided.

    Returns
    -------
    geometry : :class:`~geopandas.GeoSeries`
        Geometries to plot.
    values : :class:`numpy.ndarray` or None
        Numeric values associated with each geometry, masked for NaNs.
    markersize : :class:`numpy.ndarray` or None
        Marker sizes for each geometry.
    """
    geometry, values, markersize = (
        _create_geometries_and_values_from_lat_lon(values=values, lat=lat, lon=lon, s=s)
        if df is None
        else _create_geometries_and_values_from_gdf(df=df, values=values, s=s)
    )

    values = expand(values, geometry.size)
    values = None if values is None else np.ma.masked_invalid(values)

    markersize = expand(markersize, geometry.size)

    return geometry, values, markersize


def _plot_collection(
    plot_func: Callable,
    *,
    ax: matplotlib.axes.Axes,
    geometries: GeoSeries,
    idx: np.ndarray,
    colorizer: Colorizer,
    linewidth: np.ndarray | None,
    values: np.ndarray | None,
    markersize: np.ndarray | None,
    facecolor: np.ndarray | None,
    edgecolor: np.ndarray | None,
    **kwargs,
) -> matplotlib.collections.Collection:
    """
    Plot a subset of geometries using the provided plotting function.

    Parameters
    ----------
    plot_func : callable
        One of the geopandas internal plotting functions:
        `_plot_point_collection`, `_plot_linestring_collection`, or `_plot_polygon_collection`.
    ax : :class:`~matplotlib.axes.Axes`
        Axes to plot on.
    geometries : :class:`~geopandas.GeoSeries`
        Full set of geometries.
    idx : :class:`numpy.ndarray`
        Boolean index selecting which geometries to plot.
    colorizer : :class:`~matplotlib.colorizer.Colorizer`
        Colorizer object for mapping values to colors.
    linewidth : :class:`numpy.ndarray` or None
        Line widths for geometries.
    values : :class:`numpy.ndarray` or None
        Values used for color mapping.
    markersize : :class:`numpy.ndarray` or None
        Sizes for points (only used with point collections).
    facecolor, edgecolor : :class:`numpy.ndarray` or None
        Face and edge colors for geometries.
    **kwargs
        Additional keyword arguments for the plotting function.

    Returns
    -------
    :class:`~matplotlib.collections.Collection`
        Matplotlib collection object representing the plotted geometries.
    """
    geoms = geometries[idx]
    facecolor = facecolor[idx] if isinstance(facecolor, np.ndarray) else facecolor
    edgecolor = edgecolor[idx] if isinstance(edgecolor, np.ndarray) else edgecolor
    values = values[idx] if values is not None else None
    if markersize is not None:
        kwargs['markersize'] = markersize[idx]

    return plot_func(
        ax,
        geoms,
        values=values,
        facecolor=facecolor,
        edgecolor=edgecolor,
        cmap=colorizer.cmap if values is not None else None,
        norm=colorizer.norm if values is not None else None,
        linewidth=linewidth,
        **kwargs,
    )


def _plot_geometries(
    ax: matplotlib.axes.Axes,
    geometries: GeoSeries,
    colorizer: Colorizer,
    linewidth: float,
    values: np.ndarray | None,
    markersize: float | np.ndarray | None,
    **kwargs,
) -> list[matplotlib.collections.Collection]:
    """
    Internal plotting function for geometries, called by `plot_shapes` and `plot_classified_shapes`.

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to plot on.
    geometries : :class:`~geopandas.GeoSeries`
        GeoSeries containing the geometries.
    colorizer : :class:`~matplotlib.colorizer.Colorizer`
        Colorizer object to use for colorizing the geometries.
    linewidth : float
        Line width of the geometries.
    markersize : float or :class:`numpy.ndarray`, optional
        Size of points in `geometries`.
    **kwargs
        Additional keyword arguments passed to geopandas plotting functions
        (`_plot_point_collection`, `_plot_polygon_collection`, `_plot_linestring_collection`).

    Returns
    -------
    list of :class:`~matplotlib.collections.Collection`
        Collections of the plotted geometries.

    Notes
    -----
    Setting `facecolor` or `edgecolor` in `kwargs` overrides the colorizer behavior.
    """
    geom_types = geometries.type
    poly_idx = np.asarray((geom_types == 'Polygon') | (geom_types == 'MultiPolygon'))
    line_idx = np.asarray((geom_types == 'LineString') | (geom_types == 'MultiLineString'))
    point_idx = np.asarray((geom_types == 'Point') | (geom_types == 'MultiPoint'))

    facecolor = kwargs.pop('facecolor', None)
    edgecolor = kwargs.pop('edgecolor', None)

    if not isinstance(facecolor, np.ndarray) and facecolor is not None:
        facecolor = np.asarray(facecolor).repeat(geometries.shape[0])
    if not isinstance(edgecolor, np.ndarray) and edgecolor is not None and edgecolor != 'face':
        edgecolor = np.asarray(edgecolor).repeat(geometries.shape[0])

    c: list[matplotlib.collections.Collection] = []

    for idx, plot_func in zip(
        (point_idx, line_idx, poly_idx),
        (_plot_point_collection, _plot_linestring_collection, _plot_polygon_collection),
    ):
        c.append(
            _plot_collection(
                plot_func,
                ax=ax,
                geometries=geometries,
                values=values,
                idx=idx,
                colorizer=colorizer,
                linewidth=linewidth,
                markersize=markersize if plot_func == _plot_point_collection else None,
                facecolor=facecolor,
                edgecolor=edgecolor,
                **kwargs,
            ),
        )

    return c


def plot_classified_shapes(
    df: GeoDataFrame | GeoSeries | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    values: str | float | np.ndarray | None = None,
    s: float | np.ndarray | None = None,
    levels: ArrayLike | None = None,
    colors: ArrayLike | None = None,
    cmap: str | Colormap = 'Set1',
    nan_color: ColorType | None = None,
    labels: ArrayLike | None = None,
    ax: matplotlib.axes.Axes | None = None,
    legend: str | LegendCreator | None = 'colorbar',
    legend_ax: matplotlib.axes.Axes | None = None,
    linewidth: float = 1,
    **kwargs,
) -> tuple[list[matplotlib.collections.Collection], LegendType | None]:
    """
    Plot discrete-class shapes (choropleth style).

    Parameters
    ----------
    df : :class:`~geopandas.GeoDataFrame` or :class:`~geopandas.GeoSeries`, optional
        GeoDataFrame/GeoSeries containing the geometries.
    lat, lon : array_like, optional
        Latitude and longitude values when `df` is not provided.
    values : array_like, numeric, or str
        Values used for classification. If `df` is given and `values` is a string, it
        refers to a column name.
    s : array_like, optional
        Size of each geometry marker.
    levels : array_like, optional
        Classification levels. If None, inferred from `values`.
    colors : array_like, optional
        List of colors corresponding to levels. Overrides `cmap`.
    cmap : :class:`~matplotlib.colors.Colormap` or str, optional
        Colormap used if `colors` are not provided.
    nan_color : :class:`~matplotlib.typing.ColorType`, optional
        Color for NaN values.
    labels : array_like, optional
        Labels for the classes.
    ax : :class:`~matplotlib.axes.Axes`, optional
        Axes for plotting. If None, created automatically.
    legend : {'legend', 'colorbar', None} or :class:`~geomappy.types.LegendCreator`, optional
        Legend type to display.
    legend_ax : :class:`~matplotlib.axes.Axes`, optional
        Axes for the legend.
    linewidth : float, optional
        Line width for geometry borders.
    **kwargs
        Additional keyword arguments passed to `_plot_geometries`.
        `facecolor` is **not supported** for classified shapes.

    Returns
    -------
    collections : list of :class:`~matplotlib.collections.Collection`
    legend : :class:`~matplotlib.legend.Legend` or :class:`~matplotlib.colorbar.Colorbar` or None

    Notes
    -----
    When providing a GeoAxes in `ax`, provide `extent` if the axes and data do not perfectly overlap.
    If plotting with a different projection, provide `transform`.
    """
    if 'facecolor' in kwargs:
        raise ValueError('facecolor is not supported for classified shapes')
    if values is None:
        raise ValueError('values must be provided for classified shapes')

    geometry, values, markersize = _create_geometry_values_and_sizes(lat, lon, values, s, df)

    levels = parse_levels(values, levels)
    labels = labels or levels

    colorizer = create_classified_colorizer(
        levels=levels,
        colors=colors,
        cmap=cmap,
        nan_color=nan_color,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    collections = _plot_geometries(
        ax,
        geometry,
        colorizer,
        linewidth,
        values,
        markersize,
        **kwargs,
    )

    legend_creator = get_legend_creator('classified', legend)
    l = legend_creator(ax=ax, ca=collections[0], legend_ax=legend_ax, labels=labels)

    return collections, l


def plot_shapes(
    df: GeoDataFrame | GeoSeries | None = None,
    lat: ArrayLike | None = None,
    lon: ArrayLike | None = None,
    values: ArrayLike | str | float | None = None,
    s: ArrayLike | str | float | None = None,
    bins: tuple[float] | None = None,
    cmap: Colormap | None = None,
    nan_color: ColorType | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    extend: ExtendType | None = None,
    legend: str | LegendCreator | None = 'colorbar',
    ax: matplotlib.axes.Axes | None = None,
    legend_ax: matplotlib.axes.Axes | None = None,
    linewidth: float = 1,
    **kwargs,
) -> tuple[list[matplotlib.collections.Collection], LegendType | None]:
    """
    Plot shapes with continuous numeric values (scalar choropleth).

    Parameters
    ----------
    df : :class:`~geopandas.GeoDataFrame` or :class:`~geopandas.GeoSeries`, optional
        DataFrame or Series containing the geometries. If None, `lat` and `lon` must be provided.
    lat, lon : array_like, optional
        Latitude and longitude values if `df` is not provided.
    values : array_like, numeric, or str, optional
        Values to map to colors. Can be broadcast if scalar. If `df` is provided and `values` is a string,
        it will be interpreted as a column name.
    s : array_like, numeric, or str, optional
        Marker sizes, broadcast if scalar. Strings are interpreted as column names in `df`.
    bins : array_like, optional
        Bins for discretizing the values using a :class:`~matplotlib.colors.BoundaryNorm`. Overrides `vmin` and `vmax`.
    cmap : :class:`~matplotlib.colors.Colormap` or str, optional
        Colormap for the scalar values.
    nan_color : :class:`~matplotlib.typing.ColorType`, optional
        Color for NaN values.
    norm : :class:`~matplotlib.colors.Normalize`, optional
        Normalizer for scalar values. Not compatible with `bins`.
    vmin, vmax : float, optional
        Limits for color mapping. Ignored if `bins` or `norm` are provided.
    extend : {'neither', 'min', 'max', 'both'}, optional
        Extend of the colorbar. Determined automatically if None.
    legend : {'colorbar', 'legend', None} or :class:`~geomappy.types.LegendCreator`, optional
        Type of legend to display.
    ax : :class:`~matplotlib.axes.Axes`, optional
        Axes to plot on. Created if None.
    legend_ax : :class:`~matplotlib.axes.Axes`, optional
        Axes for the legend. Created if None.
    linewidth : float, optional
        Line width of geometry borders.
    **kwargs
        Additional keyword arguments for `_plot_geometries`, including `facecolor` and `edgecolor`.

    Returns
    -------
    collections : list of :class:`~matplotlib.collections.Collection`
    legend : :class:`~matplotlib.legend.Legend` or :class:`~matplotlib.colorbar.Colorbar` or None

    Notes
    -----
    When using a GeoAxes, provide `extent` if data and axes do not perfectly overlap.
    If the plotting projection differs from the data, provide `transform`.
    """
    if values is None:
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = 'lightgrey'
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'black'
        legend = None

    geometry, values, markersize = _create_geometry_values_and_sizes(lat, lon, values, s, df)

    if bins is not None:
        bins = np.asarray(bins).flatten()
        check_increasing_and_unique(bins, 'bins')
        if len(bins) == 1:
            values = values > bins[0]

    if values is not None and np.issubdtype(values.dtype, np.bool_):
        return plot_classified_shapes(
            df=df,
            lat=lat,
            lon=lon,
            values=values,
            s=s,
            labels=('False', 'True'),
            colors=('Lightgrey', 'Red'),
            nan_color=nan_color,
            ax=ax,
            legend=legend,
            linewidth=linewidth,
            **kwargs,
        )

    if values is None:
        extend = 'neither'
    else:
        extend = extend or determine_extend(
            get_data_range(values),
            vmin,
            vmax,
            norm,
            bins,
        )

    colorizer = create_scalar_colorizer(
        bins=bins,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        extend=extend,
        cmap=cmap,
        nan_color=nan_color,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    collections = _plot_geometries(
        ax,
        geometry,
        colorizer,
        linewidth,
        values,
        markersize,
        **kwargs,
    )

    legend_creator = get_legend_creator('scalar', legend)
    l = legend_creator(ax=ax, ca=collections[0], legend_ax=legend_ax, labels=None)

    return collections, l

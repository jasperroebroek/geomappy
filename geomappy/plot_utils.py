from collections.abc import Sequence

import matplotlib.axes
from cartopy import crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import Gridliner
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.typing import ColorType
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rasterio.coords import BoundingBox

from geomappy.bounds import bounds_to_polygons
from geomappy.types import GridSpacer
from geomappy.utils import calculate_horizontal_locations, calculate_vertical_locations


def create_colorbar_axes(
    ax: matplotlib.axes.Axes,
    *,
    location: str = 'right',
    width: float = 0.02,
    pad: float = 0.03,
    shrink: float = 1.0,
) -> matplotlib.axes.Axes:
    """
    Create an axes suitable for a colorbar next to a given axes.

    Uses `make_axes_locatable` to place the colorbar next to the parent axes.
    If `shrink` is less than 1, the colorbar is centered within the allocated space.

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        The parent axes to attach the colorbar to.
    location : str, default 'right'
        Side of the axes to place the colorbar. Options are 'right', 'left', 'top', or 'bottom'.
    width : float, default 0.02
        Width (for vertical) or height (for horizontal) of the colorbar as a fraction of the parent axes.
    pad : float, default 0.03
        Space between the parent axes and the colorbar as a fraction of the parent axes.
    shrink : float, default 1.0
        Fraction of the colorbar length to display. Values less than 1 shrink the bar while keeping it centered.

    Returns
    -------
    :class:`~matplotlib.axes.Axes`
    """
    orientation = 'vertical' if location in ('right', 'left') else 'horizontal'

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        location,
        size=f'{width * 100:.2f}%',
        pad=f'{pad * 100:.2f}%',
        axes_class=matplotlib.axes.Axes,
    )

    if shrink == 1:
        return cax

    cax.set_axis_off()
    if orientation == 'vertical':
        bbox = [0.0, (1.0 - shrink) / 2.0, 1.0, shrink]
    else:
        bbox = [(1.0 - shrink) / 2.0, 0.0, shrink, 1.0]

    inset_cax = inset_axes(
        cax,
        width='100%',
        height='100%',
        bbox_to_anchor=bbox,
        bbox_transform=cax.transAxes,
        loc='lower left',
        borderpad=0,
        axes_class=matplotlib.axes.Axes,
    )

    return inset_cax


def legend_patches(
    colors: Sequence[ColorType],
    labels: Sequence[str],
    *,
    legend_type: str = 'patch',
    edgecolor: str = 'lightgrey',
) -> list[Patch | Line2D]:
    """
    Create legend handles for categorical data.

    Can create solid patches or line handles depending on `legend_type`.

    Parameters
    ----------
    colors : sequence of color-like
        Colors for each legend element.
    labels : sequence of str
        Labels corresponding to each color.
    legend_type : str, optional
        'patch' for solid patches or a valid line style ('-', '--', ':', etc.) for line handles.
    edgecolor : str, optional
        Edge color of patches or line markers.

    Returns
    -------
    list of :class:`~matplotlib.patches.Patch` or :class:`~matplotlib.lines.Line2D`
        List of legend handles.
    """
    if len(colors) != len(labels):
        raise ValueError(f"Length of labels and colors don't match:\n{labels}\n{colors}")

    if legend_type == 'patch':
        return [Patch(facecolor=color, label=label, edgecolor=edgecolor) for color, label in zip(colors, labels)]

    return [
        Line2D(
            [0],
            [0],
            color=color,
            label=label,
            linestyle=legend_type,
            markeredgecolor=edgecolor,
        )
        for color, label in zip(colors, labels)
    ]


def add_gridlines(
    ax: GeoAxes,
    lines: GridSpacer,
    *,
    color: str = 'grey',
    linestyle: str = '--',
    alpha: float = 0.5,
    n_steps: int = 300,
    linewidth: float = 1,
    crs: ccrs.Projection = ccrs.PlateCarree(),
    **kwargs,
) -> Gridliner:
    """
    Add gridlines to a map axes.

    Returns a Gridliner object that can be further styled.

    Parameters
    ----------
    ax : :class:`cartopy.mpl.geoaxes.GeoAxes`
        Map axes to add gridlines to.
    lines : int, tuple of ints, tuple of tuple of ints
        Interval(s) or positions for the gridlines in degrees.
        Single int is applied to both axes, tuple of two ints sets separate x and y intervals,
        or tuple of explicit positions.
    color : str, optional
        Color of the gridlines. Default is grey.
    linestyle : str, optional
        Line style for the gridlines. Default is dashed '--'.
    alpha : float, optional
        Opacity of the gridlines. Default is 0.5.
    n_steps : int, optional
        Number of interpolation steps for the gridlines. Default is 300.
    linewidth : float, optional
        Width of the gridlines. Default is 1.
    crs : :class:`cartopy.crs.Projection`, optional
        Coordinate reference system of the provided gridline positions. Default is PlateCarree.
    **kwargs
        Additional keyword arguments passed to `ax.gridlines`.

    Returns
    -------
    :class:`cartopy.mpl.gridliner.Gridliner`
        The created gridliner object.
    """
    if isinstance(lines, int):
        lines = lines, lines

    xlines = calculate_horizontal_locations(lines[0])
    ylines = calculate_vertical_locations(lines[1])

    g = ax.gridlines(
        draw_labels=False,
        color=color,
        linestyle=linestyle,
        crs=crs,
        alpha=alpha,
        auto_inline=True,
        linewidth=linewidth,
        **kwargs,
    )
    g.xlocator = mticker.FixedLocator(xlines)
    g.ylocator = mticker.FixedLocator(ylines)
    g.n_steps = n_steps
    return g


def add_ticks(
    ax: GeoAxes,
    ticks: GridSpacer,
    *,
    formatter: mticker.Formatter | tuple[mticker.Formatter, mticker.Formatter] | None = None,
    fontsize: int = 10,
    crs: ccrs.Projection = ccrs.PlateCarree(),
    **kwargs,
) -> Gridliner:
    """
    Add tick labels to a map axes and return a Gridliner object for further styling.

    Parameters
    ----------
    ax : :class:`cartopy.mpl.geoaxes.GeoAxes`
        Map axes to add tick labels to.
    ticks : int, tuple of ints, tuple of tuple of ints
        Interval(s) or positions for ticks in degrees.
        Single int for both axes, tuple of two ints for separate x and y intervals,
        or tuple of explicit positions.
    formatter : :class:`matplotlib.ticker.Formatter`, or tuple of two, optional
        Formatter(s) for x and y tick labels. Default uses LongitudeFormatter and LatitudeFormatter.
    fontsize : int, optional
        Font size of the tick labels. Default is 10.
    crs : :class:`cartopy.crs.Projection`, optional
        CRS of the tick positions. Default is PlateCarree.
    **kwargs
        Additional keyword arguments passed to `ax.gridlines`.

    Returns
    -------
    :class:`cartopy.mpl.gridliner.Gridliner`
        The created Gridliner object.
    """
    if isinstance(ticks, int):
        ticks = ticks, ticks

    xticks = calculate_horizontal_locations(ticks[0])
    yticks = calculate_vertical_locations(ticks[1])

    if formatter is None:
        xtick_formatter = LongitudeFormatter()
        ytick_formatter = LatitudeFormatter()
    else:
        if isinstance(formatter, mticker.Formatter):
            formatter = formatter, formatter
        xtick_formatter = formatter[0]
        ytick_formatter = formatter[1]

    draw_labels = kwargs.pop('draw_labels', True)
    g = ax.gridlines(draw_labels=draw_labels, alpha=0, crs=crs, **kwargs)

    g.xlocator = mticker.FixedLocator(xticks)
    g.ylocator = mticker.FixedLocator(yticks)

    g.xformatter = xtick_formatter
    g.yformatter = ytick_formatter
    g.xlabel_style = {'size': fontsize}
    g.ylabel_style = {'size': fontsize}
    g.top_labels = False
    g.right_labels = False

    return g


def plot_world(
    bounds: tuple[float, float, float, float],
    bounds_projection: ccrs.Projection = ccrs.PlateCarree(),
) -> GeoAxes:
    """
    Plot a world map with specified bounding box highlighted.

    Parameters
    ----------
    bounds : tuple of four floats
        Bounding box to highlight: (minx, miny, maxx, maxy).
    bounds_projection : :class:`cartopy.crs.Projection`, optional
        Projection of the bounding box coordinates. Default is PlateCarree.

    Returns
    -------
    :class:`cartopy.mpl.geoaxes.GeoAxes`
        The map axes containing the world map and bounding box.
    """
    f, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()

    add_gridlines(ax, 30)
    add_ticks(ax, 30)

    gdf = bounds_to_polygons((BoundingBox(*bounds),))
    gdf.plot(ax=ax, edgecolor='red', facecolor='none', zorder=2, transform=bounds_projection)
    ax.set_global()

    return ax

from typing import Union, Tuple

import cartopy  # type: ignore
import cartopy.crs as ccrs  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mticker  # type: ignore
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes  # type: ignore
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # type: ignore


class ProjectCustomExtent(ccrs.Projection):
    """
    Creating a custom extent for a given epsg code, if the hardcoded values do not suffice
    """

    def __init__(self, epsg: Union[str, int], extent: Tuple[int, int, int, int], *args, **kwargs):
        super(ccrs.Projection, self).__init__(f"EPSG:{epsg}")
        xmin, xmax, ymin, ymax = extent
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @property
    def boundary(self):
        coords = ((self.x_limits[0], self.y_limits[0]),
                  (self.x_limits[0], self.y_limits[1]),
                  (self.x_limits[1], self.y_limits[1]),
                  (self.x_limits[1], self.y_limits[0]))

        return ccrs.sgeom.LineString(coords)

    @property
    def bounds(self):
        xlim = self.x_limits
        ylim = self.y_limits
        return xlim[0], xlim[1], ylim[0], ylim[1]

    @property
    def threshold(self):
        return 1e5

    @property
    def x_limits(self):
        return self.xmin, self.xmax

    @property
    def y_limits(self):
        return self.ymin, self.ymax


def calculate_horizontal_locations(v: Union[float, Tuple[float]]):
    if isinstance(v, (float, int)):
        return np.linspace(-180, 180, int(360 / v + 1))
    return np.asarray(v)


def calculate_vertical_locations(v: Union[float, Tuple[float]]):
    if isinstance(v, (float, int)):
        return np.linspace(-90, 90, int(180 / v + 1))
    return np.asarray(v)


def add_gridlines(ax: GeoAxes, lines: Union[float, Tuple[float, float], Tuple[Tuple[float], Tuple[float]]], *,
                  color: str = "grey", linestyle: str = "--", alpha: float = 0.5, n_steps: int = 300,
                  linewidth: float = 1, crs: ccrs.Projection = ccrs.PlateCarree(), **kwargs) -> None:
    """Add gridlines to a basemap. Return a Gridliner object that can be further modified

    Parameters
    ----------
    ax: GeoAxes
        Basemap
    lines: int, tuple of ints, tuple of tuple of ints
        Interval between gridlines in degrees. One int works for x and y-axes, while a tuple allows to separate them.
        It also accepts a tuple of values where the gridlines need to be placed.
    n_steps: int, optional
        Interpolation steps of the gridlines
    color: str, optional
        Color of the gridlines, the default is grey
    linestyle : str, optional
        matplotlib linestyle for the gridlines.
    alpha : float, optional
        opacity of the gridlines
    linewidth : float, optional
        Linewidth specifier for gridlines
    crs: ccrs.Projection, optional
        Projection of the coordinates provided in lines. Default is lat-lon.
    **kwargs
        keyword arguments for ax.gridlines
    """
    # Determine ticks and gridlines
    if isinstance(lines, (float, int)):
        lines = lines, lines

    xlines = calculate_horizontal_locations(lines[0])
    ylines = calculate_vertical_locations(lines[1])

    g = ax.gridlines(draw_labels=False, color=color, linestyle=linestyle, crs=crs, alpha=alpha, auto_inline=True,
                     linewidth=linewidth, **kwargs)
    g.xlocator = mticker.FixedLocator(xlines)
    g.ylocator = mticker.FixedLocator(ylines)
    g.n_steps = n_steps
    return g


def add_ticks(ax: GeoAxes,
              ticks: Union[float, Tuple[float, float], Tuple[Tuple[float], Tuple[float]]], *,
              formatter: Union[mticker.Formatter, Tuple[mticker.Formatter, mticker.Formatter], None] = None,
              fontsize: int = 10, crs: ccrs.Projection = ccrs.PlateCarree(), **kwargs) -> None:
    """Helper function creating labels. Returns a Gridliner object that can be further modified for styling

    Parameters
    ----------
    ax: GeoAxes
        Basemap
    ticks: int, tuple of ints, tuple of tuple of ints
        Interval between ticks in degrees. One int works for x and y-axes, while a tuple allows to separate them.
        It also accepts a tuple of values where the ticks need to be placed.
    formatter: mticker.Formatter or tuple of mticker.Formatter, optional
        Formatters for x and y-labels, or both at the same time.
        The default is LongitudeFormatter and LattitudeFormatter
    fontsize: int
        Fontsize
    crs: ccrs.Projection, optional
        Projection of the coordinates provided in ticks.
    **kwargs
        keyword arguments for ax.gridlines
    """

    # Determine ticks and gridlines
    if isinstance(ticks, (float, int)):
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

    g = ax.gridlines(draw_labels=True, alpha=0, crs=crs, **kwargs)

    g.xlocator = mticker.FixedLocator(xticks)
    g.ylocator = mticker.FixedLocator(yticks)

    g.xformatter = xtick_formatter
    g.yformatter = ytick_formatter
    g.xlabel_style = {'size': fontsize}
    g.ylabel_style = {'size': fontsize}
    g.top_labels = False
    g.right_labels = False

    return g


def basemap(epsg=4326, projection=None, ax=None, figsize=(8, 8)):
    """
    Creating a basemap for geographical maps

    Parameters
    ----------
    epsg : int or str, optional
        EPSG code of the GeoAxes. Is ignored if 'projection' is provided.
    projection : `ccrs.projection`, optional
        Cartopy projection object for plotting.
    ax : `plt.axes` or GeoAxes
        If a regular matplotlib axis is provided here, it gets replaced by a GeoAxes. If a GeoAxis is inserted, it is
        retained.
    figsize : tuple, optional
        Matplotlib figsize parameter

    Returns
    -------
    Cartopy GeoAxes
    """
    if projection is None:
        if epsg is None:
            raise TypeError("Both projection and EPSG code are not provided")
        elif epsg == 4326:
            projection = ccrs.PlateCarree()
        elif epsg == 3857:
            projection = ccrs.Mercator()
        else:
            projection = ccrs.epsg(epsg)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection=projection)
    elif not isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot):
        position = ax.get_position(original=False)
        ax.figure.delaxes(ax)
        ax = ax.figure.add_subplot(position=position, projection=projection)

    return ax

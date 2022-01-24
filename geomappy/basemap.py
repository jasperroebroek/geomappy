import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


class ProjectCustomExtent(ccrs.Projection):
    """
    Creating a custom extent for a given epsg code, if the hardcoded values do not suffice
    """
    def __init__(self, epsg=28992, extent=[-300000, 500000, -100000, 800000]):
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


def basemap(extent=None, epsg=4326, projection=None, extent_projection=None, ax=None, figsize=(10, 10),
            resolution="110m", coastlines=True, earth_image=False, land=None, ocean=None, xticks=30, yticks=30,
            xtick_formatter=None, ytick_formatter=None, labels=True, xlabel_location="bottom", ylabel_location="left",
            grid=True, xlines=None, ylines=None, grid_color="grey", grid_linestyle="--", grid_alpha=0.5,
            grid_linewidth=None, border_linewidth=None, coastline_linewidth=None, linewidth=1, n_steps=300,
            fontsize=10, xlabel_style=None, ylabel_style=None, padding=5):
    """
    Creating a basemap for geographical maps

    Parameters
    ----------
    extent : list, optional
        A list with the four coordinates (x0, y0, x1, y1), or the string "global", which will set the plot
        to it's natural maximum extent.
    epsg : int or str, optional
        EPSG code of the GeoAxes. Is ignored if 'projection' is provided.
    projection : `ccrs.projection`, optional
        Cartopy projection object for plotting.
    extent_projection : `ccrs.Projection`, optional
        Cartopy projection that define the units of the extent. By default lat-lon are expected.
    ax : `plt.axes` or GeoAxes
        If a regular matplotlib axis is provided here, it gets replaced by a GeoAxes. If a GeoAxis is inserted, it is
        retained.
    figsize : tuple, optional
        Matplotlib figsize parameter
    resolution : {"110m", "50m", "10m"} , optional
        Coastline resolution
    coastlines : bool, optional
        Switch to plot the coastlines
    earth_image : bool, optional
        Plot a background on the map, the default is False.
    land : str, optional
        Color of the landmass. None is the default, which results in the landmass not to be colored
    ocean : bool, optional
        Color of the oceans. None is the default, which results in the oceans not to be colored
    xticks, yticks : float or list, optional
        Parameter that describes the distance between two xticks in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a labels gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used. These parameters are used for the labels and gridlines are set to the same
        rules if xlines/ylines are not provided.
    xtick_formatter, ytick_formatter : matplotlib.formatter, optional
        Latitude and Longitude formatters. Using the the cartopy one as default.
    labels : bool, optional
        Switch to draw labels
    xlabel_location : {"bottom", "top", "both"}, optional
        Location of the xlabels
    ylabel_location : {"left", "right", "both"}, optional
        Location of the ylabels
    grid : bool, optional
        Switch for gridlines
    xlines, ylines : float or list, optional
        Parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a line gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used. These parameters are used for the gridlines only, for the labels see xticks
        and yticks. If these parameters are not set xticks and yticks will be used instead.
    grid_color : color, optional
        Color of the gridlines, the default is grey
    grid_linestyle : str, optional
        matplotlib linestyle for the gridlines.
    grid_alpha : float, optional
        opacity of the gridlines
    grid_linewidth : float, optional
        linewidth specifier for gridlines. If not specified, the value will be taken from 'linewidth'.
    border_linewidth : float, optional
        linewidth specifier for the borders around the plot. If not specified, the value will be taken from 'linewidth'.
    coastline_linewidth : float, optional
        linewidth specifier for the coastlines, If not specified, the value will be taken from 'linewidth'.
    linewidth : float, optional
        Default linewidth parameter for the gridlines, coastlines and border around the plot. Its value is used in case
        the others are not specifically specified.
    n_steps : int, optional
        The number of discrete steps when plotting the gridlines. For circular gridlines this can be too low.
    fontsize : float, optional
        Label fontsize
    xlabel_style, ylabel_style : dict, optional
        Dictionaries containing the properties of the labels on x and y axis. Size provided here will be overwritten
        by the parameter fontsize.
    padding : (int, list), optional
        Padding of labels. Int is mapped to both axes, list is interpreted as (xpadding, ypadding)

    Returns
    -------
    Cartopy GeoAxes

    Notes
    -----
    - Labels on projections that are not a rectangle, when plotted, are not handled in this function. This needs to be
      done outside the function and needs a cartopy version over 0.18.
    - Labels are not checked for overlap, so if they do you need to specify the exact labels that you want
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
        position = ax.get_position(original=True)
        ax.figure.delaxes(ax)
        ax = ax.figure.add_subplot(position=position, projection=projection)

    ax.tick_params(axis='both', which='both', length=0)

    # Set extent
    if isinstance(extent, str) and extent == "global":
        ax.set_global()

    elif extent is not None:
        extent = [extent[0], extent[2], extent[1], extent[3]]

        if extent_projection is None:
            extent_projection = ccrs.PlateCarree()

        ax.set_extent(extent, crs=extent_projection)

    if not isinstance(linewidth, (float, int)):
        raise TypeError("Linewidth should be numeric")
    if coastline_linewidth is None:
        coastline_linewidth = linewidth
    if border_linewidth is None:
        border_linewidth = linewidth
    if grid_linewidth is None:
        grid_linewidth = linewidth

    if coastlines:
        ax.coastlines(resolution=resolution, linewidth=coastline_linewidth)

    if earth_image:
        ax.stock_img()
    if land is not None:
        ax.add_feature(cf.LAND, color=land)
    if ocean is not None:
        ax.add_feature(cf.OCEAN, color=ocean)

    # Determine ticks and gridlines
    if isinstance(xticks, (float, int)):
        xtick_locations = np.linspace(-180, 180, int(360 / xticks + 1))
    else:
        xtick_locations = np.array(xticks)

    if xlines is None:
        if len(xtick_locations) == 0:
            xline_locations = [-180, 180]
        else:
            xline_locations = xtick_locations
    elif isinstance(xlines, (float, int)):
        xline_locations = np.linspace(-180, 180, int(360 / xlines + 1))
    else:
        xline_locations = xlines

    if isinstance(yticks, (float, int)):
        ytick_locations = np.linspace(-90, 90, int(180 / yticks + 1))
    else:
        ytick_locations = np.array(yticks)

    if ylines is None:
        if len(ytick_locations) == 0:
            yline_locations = [-90, 90]
        else:
            yline_locations = ytick_locations
    elif isinstance(ylines, (float, int)):
        yline_locations = np.linspace(-180, 180, int(360 / ylines + 1))
    else:
        yline_locations = ylines

    # Draw grid
    if grid:
        g = ax.gridlines(draw_labels=False, color=grid_color, linestyle=grid_linestyle, crs=ccrs.PlateCarree(),
                         linewidth=grid_linewidth, alpha=grid_alpha, auto_inline=True)
        g.xlocator = mticker.FixedLocator(xline_locations)
        g.ylocator = mticker.FixedLocator(yline_locations)
        g.n_steps = n_steps

    # Determine label locations
    labels_top = (xlabel_location == "top" or xlabel_location == "both")
    labels_bottom = (xlabel_location == "bottom" or xlabel_location == "both")
    labels_left = (ylabel_location == "left" or ylabel_location == "both")
    labels_right = (ylabel_location == "right" or ylabel_location == "both")

    if labels:
        if xtick_formatter is None:
            xtick_formatter = LongitudeFormatter()
        if ytick_formatter is None:
            ytick_formatter = LatitudeFormatter()

        g = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), alpha=0)
        g.top_labels = labels_top
        g.bottom_labels = labels_bottom
        g.left_labels = labels_left
        g.right_labels = labels_right

        g.xlocator = mticker.FixedLocator(xtick_locations)
        g.ylocator = mticker.FixedLocator(ytick_locations)
        g.xformatter = xtick_formatter
        g.yformatter = ytick_formatter

        if xlabel_style is None:
            xlabel_style = {}
        if ylabel_style is None:
            ylabel_style = {}

        xlabel_style.update({'size': fontsize})
        ylabel_style.update({'size': fontsize})

        g.xlabel_style = xlabel_style
        g.ylabel_style = ylabel_style

        if isinstance(padding, (float, int)):
            g.xpadding = padding
            g.ypadding = padding
        else:
            g.xpadding = padding[0]
            g.ypadding = padding[1]

    ax.spines['geo'].set_linewidth(border_linewidth)
    ax.figure.canvas.draw()

    return ax

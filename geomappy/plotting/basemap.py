from copy import copy

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import shapely
import shapely.geometry as sgeom
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from packaging import version

"""
inspiration source:
https://nbviewer.jupyter.org/gist/ajdawson/dd536f786741e987ae4e
"""


def _clone_geoaxes(ax):
    extent = ax.get_extent()
    ax = ax.figure.add_subplot(projection=ax.projection, zorder=-1, label="xtick_new_" + str(np.random.rand()))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.outline_patch.set_linewidth(0)
    return ax


def basemap_xticks(ax, ticks, side='bottom', add=True, fontsize=10, formatter=None):
    """
    Calculate and insert xticks on a GeoAxis

    Parameters
    ----------
    ax : ``GeoAxes``
        GeoAxes that the xticks need to be inserted in
    ticks : list
        locations of the ticks in PlateCarree coordinates
    side : {"top", "bottom"}
        side of the axes that gets the ticks
    add : bool, optional
        if both bottom and top are different and both need to be added, one (or both) of the calls to this function
        should have `add` = True
    fontsize : float, optional
        fontsize of the labels
    formatter : matplotlib.tickformatter, optional
        The formatter for the labels. The default is the default cartopy LongitudeFormatter
    """
    # tick_extractor (pick the first coordinate)
    te = lambda xy: xy[0]
    # line_constructor (create line with fixed x-coordinates and variabel y coordinates)
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2] - 1, b[3] + 1, n))).T
    xticks, xticklabels = _basemap_ticks(ax, ticks, side, lc, te)

    # Insert and format the ticks
    if add:
        ax = _clone_geoaxes(ax)
    ax.tick_params(axis='both', which='both', length=0, labelsize=fontsize)
    if side == "top":
        ax.xaxis.tick_top()

    if isinstance(formatter, type(None)):
        formatter = LongitudeFormatter()

    if version.parse(cartopy.__version__) < version.parse("0.18"):
        ax_temp = plt.gcf().add_subplot(projection=ccrs.PlateCarree())
        ax_temp.set_xticks(xticklabels, crs=ccrs.PlateCarree())
        ax_temp.xaxis.set_major_formatter(formatter)
        plt.draw()
        xticklabels_formatted = [item.get_text() for item in ax_temp.get_xticklabels()]
        ax_temp.figure.delaxes(ax_temp)
    else:
        formatter.set_locs(xticklabels)
        xticklabels_formatted = [formatter(value) for value in xticklabels]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels_formatted)


def basemap_yticks(ax, ticks, side='left', add=True, fontsize=10, formatter=None):
    """
    Calculate and insert yticks on a GeoAxis

    Parameters
    ----------
    ax : ``GeoAxes``
        GeoAxes that the yticks need to be inserted in
    ticks : list
        locations of the ticks in PlateCarree coordinates
    side : {"left", "right"}
        side of the axes that gets the ticks
    add : bool, optional
        if both bottom and top are different and both need to be added, one (or both) of the calls to this function
        should have 'add' = True
    fontsize : float, optional
        fontsize of the labels
    formatter : matplotlib.tickformatter, optional
        The formatter for the labels. The default is the default cartopy LatitudeFormatter
    """
    # tick_extractor (pick the second coordinate)
    te = lambda xy: xy[1]
    # line_constructor (create line with fixed y-coordinates and variabel x coordinates)
    lc = lambda t, n, b: np.vstack((np.linspace(b[0] - 1, b[1] + 1, n), np.zeros(n) + t)).T
    yticks, yticklabels = _basemap_ticks(ax, ticks, side, lc, te)

    # Insert and format the ticks
    if add:
        ax = _clone_geoaxes(ax)
    ax.tick_params(axis='both', which='both', length=0, labelsize=fontsize)
    if side == "right":
        ax.yaxis.tick_right()

    if isinstance(formatter, type(None)):
        formatter = LatitudeFormatter()

    if version.parse(cartopy.__version__) < version.parse("0.18"):
        ax_temp = plt.gcf().add_subplot(projection=ccrs.PlateCarree())
        ax_temp.set_yticks(yticklabels, crs=ccrs.PlateCarree())
        ax_temp.yaxis.set_major_formatter(formatter)
        plt.draw()
        yticklabels_formatted = [item.get_text() for item in ax_temp.get_yticklabels()]
        ax_temp.figure.delaxes(ax_temp)
    else:
        formatter.set_locs(yticklabels)
        yticklabels_formatted = [formatter(value) for value in yticklabels]

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels_formatted)


def _basemap_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """
    Get the tick locations and labels for an axis of an unsupported projection.

    Parameters
    ----------
    ax : geoaxis
        geoaxis that the ticks are inserted into
    ticks : list
        tick locations in PlateCarree coordinates
    tick_location : {"left", "right", "bottom", "top"}
        the side of the axis that the ticks are inserted into
    line_constructor : function
        function that creates a numpy version of the line from either top to bottom or left to right with a certain
        amount of steps
    tick_extractor : function
        function that extracts either the first or second coordinate from the created lines

    Returns
    -------
    (ticks, ticklabels)
    """
    # the border of the axis
    minx, maxx, miny, maxy = ax.get_extent()
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)], }
    axis = sgeom.LineString(points[tick_location])
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    # placeholder for the tick locations
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        # list of projected points on the line
        projected_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        # extract only x and y
        xyt = projected_xyz[..., :2]
        # convert this again to a LineString
        ls = sgeom.LineString(xyt.tolist())
        try:
            # location where the line intersects the axis
            locs = axis.intersection(ls)
        except shapely.errors.TopologicalError:
            # this sometimes walks into errors for some reason, but this doesn't seem to be a problem
            locs = None
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels


class ProjectCustomExtent(ccrs.Projection):
    """
    Creating a custom extent for a given epsg code, if the hardcoded values do not suffice
    """

    def __init__(self, epsg=28992, extent=[-300000, 500000, -100000, 800000]):
        xmin, xmax, ymin, ymax = extent

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        super().__init__(ccrs.epsg(epsg).proj4_params)

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


def basemap(x0=-180, y0=-90, x1=180, y1=90, epsg=4326, projection=None, ax=None, figsize=(10, 10), resolution="110m",
            coastlines=True, earth_image=False, land=False, ocean=False, xticks=30, yticks=30, xtick_formatter=None,
            ytick_formatter=None, labels=True, xlabel_location="bottom", ylabel_location="left", grid=True, xlines=None,
            ylines=None, grid_color="grey", grid_linestyle="--", grid_alpha=0.5, grid_linewidth=None,
            border_linewidth=None, coastline_linewidth=None, linewidth=1, n_steps=300, fontsize=10):
    """
    Parameters
    ----------
    x0, x1, y0, y1 : float, optional
        Latitude and Longitude of the corners
    epsg : int or str, optional
        EPSG code of the GeoAxes
    projection : `ccrs.projection`
        cartopy projection object. If provided it overwrites the epsg code.
    ax : `plt.axes` or GeoAxes
        if a regular matplotlib axis is provided here, it gets replaced by a geoaxis. If a geoaxis is inserted, it
        just perform all the operations on that.
    figsize : tuple, optional
        matplotlib figsize parameter
    resolution : {"110m", "50m", "10m"} , optional
        coastline resolution
    coastlines : bool, optional
        switch to plot the coastlines
    earth_image : bool, optional
        plot a background on the map, the default is False
    land : bool, optional
        Switch to color the landmass. The default color is lightgrey, but a color might be used here that overwrites
        that default.
    ocean : bool, optional
        Switch to color the ocean. The default color is lightblue, but a color might be used here that overwrites
        that default.
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
        the number of discrete steps when plotting the gridlines. For circular gridlines this can be too low.
    fontsize : float, optional
        label fontsize

    Returns
    -------
    Cartopy geoaxis

    Notes
    -----
    # Labels on projections that are not a rectangle, when plotted, are not handled in this function. This needs to be
      done outside the function and needs a cartopy version over 0.18.
    # Labels are not checked for overlap, so if they do you need to specify the exact labels that you want
    # If you are plotting a very small area and xticks and yticks are very small, it can take a long time to do,
      in which case it might be beneficial to put the labels yourself.
    """
    if isinstance(projection, type(None)):
        if isinstance(epsg, type(None)):
            raise TypeError("Both projection and EPSG code are not provided")
        else:
            if epsg == 4326:
                projection = ccrs.PlateCarree()
            elif epsg == 3857:
                projection = ccrs.Mercator()
            else:
                projection = ccrs.epsg(epsg)

    if isinstance(ax, type(None)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection=projection)
    elif not isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot):
        position = ax.get_position(original=True)
        ax.figure.delaxes(ax)
        ax = plt.gcf().add_subplot(position=position, projection=projection)

    # Set extent
    extent = list(ax.get_extent(crs=ccrs.PlateCarree()))
    if extent[0] < x0:
        extent[0] = x0
    if extent[1] > x1:
        extent[1] = x1
    if extent[2] < y0:
        extent[2] = y0
    if extent[3] > y1:
        extent[3] = y1
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if not isinstance(linewidth, (float, int)):
        raise TypeError("Linewidth should be numeric")
    if isinstance(coastline_linewidth, type(None)):
        coastline_linewidth = linewidth
    if isinstance(border_linewidth, type(None)):
        border_linewidth = linewidth
    if isinstance(grid_linewidth, type(None)):
        grid_linewidth = linewidth

    if coastlines:
        ax.coastlines(resolution=resolution, linewidth=coastline_linewidth)

    if earth_image:
        ax.stock_img()
    if isinstance(land, bool) and land == True:
        if isinstance(land, bool):
            land = 'lightgrey'
        ax.add_feature(cf.LAND, color=land)
    if isinstance(ocean, bool) and ocean == True:
        if isinstance(ocean, bool):
            ocean = 'lightblue'
        ax.add_feature(cf.OCEAN, color=ocean)

    # Determine ticks and gridlines
    if isinstance(xticks, (float, int)):
        xtick_locations = np.linspace(-180, 180, int(360 / xticks + 1))
    else:
        xtick_locations = np.array(xticks)
    
    if isinstance(xlines, type(None)):
        if len(xtick_locations) == 0:
            xline_locations = [-180,180]
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
        
    if isinstance(ylines, type(None)):
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
                         linewidth=grid_linewidth, alpha=grid_alpha)
        g.xlocator = mticker.FixedLocator(xline_locations)
        g.ylocator = mticker.FixedLocator(yline_locations)
        g.n_steps = n_steps

    # Determine label locations
    xlabels_top = (xlabel_location == "top" or xlabel_location == "both")
    xlabels_bottom = (xlabel_location == "bottom" or xlabel_location == "both")
    ylabels_left = (ylabel_location == "left" or ylabel_location == "both")
    ylabels_right = (ylabel_location == "right" or ylabel_location == "both")

    if labels:
        if isinstance(xtick_formatter, type(None)):
            xtick_formatter = LongitudeFormatter()
        if isinstance(ytick_formatter, type(None)):
            ytick_formatter = LatitudeFormatter()

        if isinstance(ax.projection, ccrs.Mercator):
            g = ax.gridlines(draw_labels=True, color=grid_color, linestyle=grid_linestyle, crs=ccrs.PlateCarree(),
                             linewidth=grid_linewidth, alpha=0)
            g.xlabels_top = xlabels_top
            g.xlabels_bottom = xlabels_bottom
            g.ylabels_left = ylabels_left
            g.ylabels_right = ylabels_right

            g.xlocator = mticker.FixedLocator(xtick_locations)
            g.ylocator = mticker.FixedLocator(ytick_locations)
            g.xformatter = LONGITUDE_FORMATTER
            g.yformatter = LATITUDE_FORMATTER

        elif isinstance(ax.projection, ccrs.PlateCarree):
            xtick_locations = xtick_locations[np.logical_and(xtick_locations >= x0, xtick_locations <= x1)]
            ytick_locations = ytick_locations[np.logical_and(ytick_locations >= y0, ytick_locations <= y1)]

            if xlabel_location == "bottom":
                ax.set_xticks(xtick_locations, crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(xtick_formatter)
            elif xlabel_location == "top":
                ax.xaxis.tick_top()
                ax.set_xticks(xtick_locations, crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(xtick_formatter)
            else:
                ax.set_xticks(xtick_locations, crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(xtick_formatter)
                ax_new = _clone_geoaxes(ax)
                ax_new.xaxis.tick_top()
                ax_new.set_xticks(xtick_locations, crs=ccrs.PlateCarree())
                ax_new.xaxis.set_major_formatter(xtick_formatter)

            if ylabel_location == "left":
                ax.set_yticks(ytick_locations, crs=ccrs.PlateCarree())
                ax.yaxis.set_major_formatter(ytick_formatter)
            elif ylabel_location == "right":
                ax.yaxis.tick_right()
                ax.set_yticks(ytick_locations, crs=ccrs.PlateCarree())
                ax.yaxis.set_major_formatter(ytick_formatter)
            else:
                ax.set_yticks(ytick_locations, crs=ccrs.PlateCarree())
                ax.yaxis.set_major_formatter(ytick_formatter)
                ax_new = _clone_geoaxes(ax)
                ax_new.yaxis.tick_top()
                ax_new.set_yticks(ytick_locations, crs=ccrs.PlateCarree())
                ax_new.yaxis.set_major_formatter(ytick_formatter)

        else:
            if len(xtick_locations) > 0:
                if xlabels_bottom:
                    basemap_xticks(ax, list(xtick_locations), add=False, side='bottom', formatter=xtick_formatter)
                if xlabels_top:
                    basemap_xticks(ax, list(xtick_locations), add=True, side='top', formatter=xtick_formatter)

            if len(ytick_locations) > 0:
                if ylabels_left:
                    basemap_yticks(ax, list(ytick_locations), add=False, side='left', formatter=ytick_formatter)
                if ylabels_right:
                    basemap_yticks(ax, list(ytick_locations), add=True, side='right', formatter=ytick_formatter)

    ax.outline_patch.set_linewidth(border_linewidth)
    ax.tick_params(axis='both', which='both', length=0, labelsize=fontsize)

    return ax


# TESTS
if __name__ == "__main__":
    basemap()
    plt.show()

    basemap(x0=0)
    plt.tight_layout()
    plt.show()

    basemap(epsg=3857)
    plt.show()

    ax = basemap(y0=0, epsg=3857)
    plt.show()

    # Trying out different projections
    basemap(epsg=3035, resolution="10m", grid=True)
    plt.show()

    basemap(epsg=3035, resolution="10m", grid=True, xticks=5, yticks=5)
    plt.show()

    basemap(epsg=3857, resolution="10m", grid=True)
    plt.show()

    basemap(epsg=5643, resolution="10m", grid=True, xticks=5, yticks=5)
    plt.show()

    # Add labels on four axes
    ax = basemap(epsg=5643, resolution="10m", grid=True, xticks=2, yticks=2, xlabel_location='both',
                 ylabel_location='both')
    plt.show()

    # Add labels on three axes
    ax = basemap(epsg=3035, resolution="10m", grid=True, xticks=5, yticks=5, ylabel_location='both')
    plt.show()

    # Changing fontsize
    ax = basemap(epsg=3035, resolution="10m", grid=True, xticks=5, yticks=5, fontsize=6)
    basemap_yticks(ax, list(np.linspace(-90, 90, (180 // 5 + 1))), side="right", fontsize=6)
    plt.show()

    # Smaller extent and labels on both sides
    ax = basemap(x0=0, epsg=3035, resolution="10m", grid=True, xticks=5, yticks=10, ylabel_location="both")
    plt.show()

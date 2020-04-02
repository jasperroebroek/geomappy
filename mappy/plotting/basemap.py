import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import shapely
import shapely.geometry as sgeom
from copy import copy

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


def basemap_xticks(ax, ticks, precision=0, side='bottom', add=True, fontsize=10):
    """
    Calculate and insert xticks on a GeoAxis

    Parameters
    ----------
    ax : geoaxis
        geoaxis that the xticks need to be inserted in
    ticks : list
        locations of the ticks in PlateCarree coordinates
    precision : int, optional
        precision of the coordinates that are displayed. Default is 0.
    side : {"top", "bottom"}
        side of the axes that gets the ticks
    add : bool, optional
        if both bottom and top are different and both need to be added, one (or both) of the calls to this function
        should have 'add' = True
    fontsize : float, optional
        fontsize of the labels
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
    ax.set_xticks(xticks)
    xticklabels_formatted = []
    # Format the labels
    for label in xticklabels:
        if int(label) == -180 or int(label) == 180:
            xticklabels_formatted.append(str(180) + u'\u00B0')
        elif int(label) == 0:
            xticklabels_formatted.append(str(0) + u'\u00B0')
        else:
            if label < 0:
                hemisphere = u'\u00B0' + "W"
            elif label > 0:
                hemisphere = u'\u00B0' + "E"
            else:
                hemisphere = u'\u00B0'

            if precision == 0:
                xticklabels_formatted.append(str(int(np.abs(np.round(label)))) + hemisphere)
            else:
                xticklabels_formatted.append(str(np.abs(np.round(label, precision))) + hemisphere)
    ax.set_xticklabels(xticklabels_formatted)


def basemap_yticks(ax, ticks, precision=0, side='left', add=True, fontsize=10):
    """
    Calculate and insert yticks on a GeoAxis

    Parameters
    ----------
    ax : geoaxis
        geoaxis that the yticks need to be inserted in
    ticks : list
        locations of the ticks in PlateCarree coordinates
    precision : int, optional
        precision of the coordinates that are displayed. Default is 0.
    side : {"left", "right"}
        side of the axes that gets the ticks
    add : bool, optional
        if both bottom and top are different and both need to be added, one (or both) of the calls to this function
        should have 'add' = True
    fontsize : float, optional
        fontsize of the labels
    """
    # tick_extractor (pick the second coordinate)
    te = lambda xy: xy[1]
    # line_constructor (create line with fixed y-coordinates and variabel x coordinates)
    lc = lambda t, n, b: np.vstack((np.linspace(b[0] - 1, b[1] + 1, n), np.zeros(n) + t)).T
    yticks, yticklabels = _basemap_ticks(ax, ticks, side, lc, te)

    # Insert and format the ticks
    # Insert and format the ticks
    if add:
        ax = _clone_geoaxes(ax)
        ax.tick_params(axis='both', which='both', length=0, labelsize=fontsize)
    if side == "right":
        ax.yaxis.tick_right()
    ax.set_yticks(yticks)
    yticklabels_formatted = []
    # Format the labels
    for label in yticklabels:
        if int(label) == -90 or int(label) == 90:
            yticklabels_formatted.append(str(90) + u'\u00B0')
        elif int(label) == 0:
            yticklabels_formatted.append(str(0) + u'\u00B0')
        else:
            if label < 0:
                hemisphere = u'\u00B0' + "S"
            elif label > 0:
                hemisphere = u'\u00B0' + "N"
            else:
                hemisphere = u'\u00B0'

            if precision == 0:
                yticklabels_formatted.append(str(int(np.abs(np.round(label)))) + hemisphere)
            else:
                yticklabels_formatted.append(str(np.abs(np.round(label, precision))) + hemisphere)
    ax.set_yticklabels(yticklabels_formatted)


def _basemap_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of an unsupported projection.
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
    Creating a custom extent for a given epsg code, if the hard coded values do not suffice
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


def basemap(x0=-180, x1=180, y0=-90, y1=90, epsg=4326, projection=None, ax=None, figsize=(10, 10), resolution="110m",
            coastlines=True, earth_image=False, land=False, ocean=False, yticks=30, xticks=30, grid=True, n_steps=300,
            linewidth=1, grid_linewidth=None, border_linewidth=None, coastline_linewidth=None, grid_alpha=0.5,
            fontsize=10):
    """
    Parameters
    ----------
    x0 : float, optional
        most western latitude
    x1 : float, optional
        most eastern latitude
    y0 : float, optional
        most southern longitude
    y1 : float, optional
        most northern longitude
    epsg : int or str, optional
        EPSG code of the geoaxes
    projection : .
        cartopy projection object. If provided it overwrites the epsg code.
    ax : plt.axes
        if a regular matplotlib axis is provided here, it gets replaced by a geoaxis
    figsize : tuple, optional
        matplotlib figsize parameter
    resolution : {"110m", "50m", "10m"} , optional
        coastline resolution
    coastlines : bool, optional
        switch to plot the coastlines
    earth_image : bool, optional
        plot a background on the map, the default is False
    land : bool, optional
        switch to color the landmass lightgrey
    ocean : bool, optional
        switch to color the ocean lightblue
    yticks : float or list, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used.
    xticks : float or list, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
        coordinates in the list are used.
    grid : bool, optional
        switch for gridlines and ticks
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
    grid_alpha : float, optional
        opacity of the gridlines
    fontsize : float, optional
        label fontsize

    Returns
    -------
    Cartopy geoaxis
    
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

    if not isinstance(ax, type(None)):
        position = ax.get_position(original=True)
        ax.remove()
        ax = plt.gcf().add_subplot(position=position, projection=projection)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection=projection)

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
    if land:
        ax.add_feature(cf.LAND, color="lightgrey")
    if ocean:
        ax.add_feature(cf.OCEAN, color="lightblue")

    if isinstance(xticks, (float, int)):
        xtick_locations = np.linspace(-180, 180, int(360 / xticks + 1))
    else:
        xtick_locations = xticks

    if isinstance(yticks, (float, int)):
        ytick_locations = np.linspace(-90, 90, int(180 / yticks + 1))
    else:
        ytick_locations = yticks

    if grid:
        if isinstance(ax.projection, ccrs.PlateCarree):
            g = ax.gridlines(draw_labels=False, color="gray", linestyle="--", crs=ccrs.PlateCarree(),
                             linewidth=grid_linewidth, alpha=grid_alpha)
            g.xlocator = mticker.FixedLocator(xtick_locations)
            g.ylocator = mticker.FixedLocator(ytick_locations)
            g.n_steps = 30

            ax.set_xticks(xtick_locations, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.set_yticks(ytick_locations, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(LatitudeFormatter())

        elif isinstance(ax.projection, ccrs.Mercator):
            # todo; figure out how to display the 90 degree mercator marks
            g = ax.gridlines(draw_labels=True, color="gray", linestyle="--", crs=ccrs.PlateCarree(),
                             linewidth=grid_linewidth, alpha=grid_alpha)
            g.xlocator = mticker.FixedLocator(xtick_locations)
            g.ylocator = mticker.FixedLocator(ytick_locations)
            g.n_steps = 30

            g.xlabels_top = False
            g.ylabels_right = False
            g.xlocator = mticker.FixedLocator(xtick_locations)
            g.ylocator = mticker.FixedLocator(ytick_locations)
            g.xformatter = LONGITUDE_FORMATTER
            g.yformatter = LATITUDE_FORMATTER

        else:
            g = ax.gridlines(draw_labels=False, color="gray", linestyle="--", crs=ccrs.PlateCarree(),
                             linewidth=grid_linewidth, alpha=grid_alpha)
            g.xlocator = mticker.FixedLocator(xtick_locations)
            g.ylocator = mticker.FixedLocator(ytick_locations)
            g.n_steps = n_steps

            basemap_xticks(ax, list(xtick_locations), add=False)
            basemap_yticks(ax, list(ytick_locations), add=False)

    ax.outline_patch.set_linewidth(border_linewidth)
    ax.tick_params(axis='both', which='both', length=0, labelsize=fontsize)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    return ax


# TESTS
if __name__ == "__main__":
    pass
    # basemap()
    # plt.show()
    #
    # basemap(x0=0)
    # plt.show()
    #
    # basemap(epsg=3857)
    # plt.show()
    #
    # ax = basemap(y0=0, epsg=3857)
    # plt.show()
    #
    # # Trying out different projections
    # basemap(epsg=3035, resolution="10m", grid=True)
    # plt.show()
    #
    # basemap(epsg=3035, resolution="10m", grid=True, xticks=5, yticks=5)
    # plt.show()
    #
    # basemap(epsg=3857, resolution="10m", grid=True)
    # plt.show()
    #
    # basemap(epsg=5643, resolution="10m", grid=True, xticks=5, yticks=5)
    # plt.show()
    #
    # # Add labels on four axes
    # ax = basemap(epsg=5643, resolution="10m", grid=True, xticks=2, yticks=2)
    # basemap_xticks(ax, list(np.linspace(-180, 180, (360 // 2 + 1))), side="top")
    # basemap_yticks(ax, list(np.linspace(-90, 90, (180 // 2 + 1))), side="right")
    # plt.show()
    #
    # Add labels on three axes
    # ax = basemap(epsg=3035, resolution="10m", grid=True, xticks=5, yticks=5)
    # basemap_yticks(ax, list(np.linspace(-90, 90, (180 // 5 + 1))), side="right")
    # plt.show()
    #
    # # Changing fontsize
    # ax = basemap(epsg=3035, resolution="10m", grid=True, xticks=5, yticks=5, fontsize=6)
    # basemap_yticks(ax, list(np.linspace(-90, 90, (180 // 5 + 1))), side="right", fontsize=6)
    # plt.show()

    # Smaller extent and labels on both sides
    # ax = basemap(x0=0, epsg=3035, resolution="10m", grid=True, xticks=5, yticks=10)
    # basemap_yticks(ax, list(np.linspace(-90, 90, (180 // 5 + 1))), side="right")
    # plt.tight_layout()
    # plt.show()

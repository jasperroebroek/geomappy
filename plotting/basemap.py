import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import shapely.geometry as sgeom
from copy import copy
import shapely

"""
inspiration source:
https://nbviewer.jupyter.org/gist/ajdawson/dd536f786741e987ae4e
"""


def custom_xticks(ax, ticks, precision=0, side='bottom', add=True, fontsize=10):
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
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2]-1, b[3]+1, n))).T
    xticks, xticklabels = _custom_ticks(ax, ticks, side, lc, te)

    # Insert and format the ticks
    if add:
        ax = plt.gcf().add_subplot(projection=ax.projection, zorder=-1, label="xticknew")
        ax.tick_params(axis='both', which='both', length=0, labelsize=fontsize)
    if side == "top":
        ax.xaxis.tick_top()
    ax.set_xticks(xticks)
    xticklabels_formatted = []
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


def custom_yticks(ax, ticks, precision=0, side='left', add=True, fontsize=12):
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
    lc = lambda t, n, b: np.vstack((np.linspace(b[0]-1, b[1]+1, n), np.zeros(n) + t)).T
    yticks, yticklabels = _custom_ticks(ax, ticks, side, lc, te)

    # Insert and format the ticks
    if add:
        ax = plt.gcf().add_subplot(projection=ax.projection, zorder=-1, label="yticknew")
        ax.tick_params(axis='both', which='both', length=0, labelsize=fontsize)
    if side == "right":
        ax.yaxis.tick_right()
    ax.set_yticks(yticks)
    yticklabels_formatted = []
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


def _custom_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
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
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    minx, miny, maxx, maxy = outline_patch.bounds
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


def basemap(x0=-180, x1=180, y0=-90, y1=90, epsg=4326, projection=None, ax=None, figsize=(10, 10), resolution="110m",
            coastlines=True, earth_image=False, land=False, ocean=False, yticks=30, xticks=30, grid=True, n_steps=300,
            grid_linewidth=1, border_linewidth=1, coastline_linewidth=1, grid_alpha=0.5, fontsize=10):
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
    yticks : float, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn.
    xticks : float, optional
        parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
        means that every 30 degrees a gridline gets drawn.
    grid : bool, optional
        switch for gridlines and ticks
    grid_linewidth : float, optional
        linewidth specifier for gridlines
    border_linewidth : float, optional
        linewidth specifier for the borders around the plot
    coastline_linewidth : float, optional
        linewidth specifier for the coastlines
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

    if coastlines:
        ax.coastlines(resolution=resolution, linewidth=coastline_linewidth)

    if earth_image:
        ax.stock_img()
    if land:
        ax.add_feature(cf.LAND, color="lightgrey")
    if ocean:
        ax.add_feature(cf.OCEAN, color="lightblue")

    xtick_locations = np.linspace(-180, 180, int(360 / xticks + 1))
    ytick_locations = np.linspace(-90, 90, int(180 / yticks + 1))

    if grid:
        if ax.projection == ccrs.PlateCarree():
            g = ax.gridlines(draw_labels=False, color="gray", linestyle="--", crs=ccrs.PlateCarree(),
                             linewidth=grid_linewidth, alpha=grid_alpha)
            g.xlocator = mticker.FixedLocator(xtick_locations)
            g.ylocator = mticker.FixedLocator(ytick_locations)
            g.n_steps = 30

            ax.set_xticks(xtick_locations, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.set_yticks(ytick_locations, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(LatitudeFormatter())

        elif ax.projection == ccrs.Mercator():
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

            custom_xticks(ax, list(xtick_locations))
            custom_yticks(ax, list(ytick_locations))

    ax.outline_patch.set_linewidth(border_linewidth)
    ax.tick_params(axis='both', which='both', length=0, labelsize=fontsize)

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
    return ax


# TESTS
if __name__ == "__main__":
    basemap()
    basemap(x0=0)
    basemap(y0=0, epsg=3857)

    # Trying out different projections
    basemap(epsg=3035, resolution="10m", grid=True)
    basemap(epsg=3035, resolution="10m", grid=True, xticks=5, yticks=5)
    basemap(epsg=3857, resolution="10m", grid=True)
    basemap(epsg=5643, resolution="10m", grid=True, xticks=5, yticks=5)

    # Add labels on four axes
    ax = basemap(epsg=5643, resolution="10m", grid=True, xticks=2, yticks=2)
    custom_xticks(ax, list(np.linspace(-180, 180, (360//2+1))), side="top", add=True)
    custom_yticks(ax, list(np.linspace(-90, 90, (180//2+1))), side="right", add=True)
    plt.show()

    # Add labels on three axes
    ax = basemap(epsg=3035, resolution="10m", grid=True, xticks=5, yticks=5)
    custom_yticks(ax, list(np.linspace(-90, 90, (180//5+1))), side="right", add=True)
    plt.show()

    # Changing fontsize
    ax = basemap(epsg=3035, resolution="10m", grid=True, xticks=5, yticks=5, fontsize=6)
    custom_yticks(ax, list(np.linspace(-90, 90, (180 // 5 + 1))), side="right", add=True, fontsize=6)
    plt.show()


import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.patches import Patch
from .colors import legend_patches as lp
from shapely.geometry import Point


def _determine_cmap_boundaries_discrete(m, bins, cmap, clip_legend=False):
    """
    Function that creates the BoundaryNorm instance and an adjusted Colormap to segregate the data that will be plotted
    in bins. It is called from `plot_maps` and `plot_shapes`.

    Parameters
    ----------
    m : array_like
        Data
    bins : array_like
        Bins in which the data will be segragated
    cmap : `matplotlib.colors.Colormap` instance
        Colormap used for plotting
    clip_legend : bool, optional
        Remove the values from `bins` that fall outside the range found in `m`

    Returns
    -------
    cmap, norm, legend_patches, extend

    """
    m = np.array(m)

    if 'float' in m.dtype.name:
        data = m[~np.isnan(m)]
    else:
        data = m.flatten()

    bins = np.array(bins)
    bins.sort()

    vmin = data.min()
    vmax = data.max()
    boundaries = bins.copy()

    if clip_legend:
        bins = bins[np.logical_and(bins >= vmin, bins <= vmax)]

    if vmin < bins[0]:
        boundaries = np.hstack([vmin, boundaries])
        extend_min = True
        labels = [f"< {bins[0]}"]
    else:
        extend_min = False
        labels = [f"{bins[0]} - {bins[1]}"]

    labels = labels + [f"{bins[i - 1]} - {bins[i]}" for i in range(1 + (not extend_min), len(bins))]

    if vmax > bins[-1]:
        boundaries = np.hstack([boundaries, vmax])
        extend_max = True
        labels = labels + [f"> {bins[-1]}"]
    else:
        extend_max = False

    if extend_min and extend_max:
        extend = "both"
    elif not extend_min and not extend_max:
        extend = "neither"
    elif not extend_min and extend_max:
        extend = "max"
    elif extend_min and not extend_max:
        extend = "min"

    colors = cmap(np.linspace(0, 1, boundaries.size - 1))
    cmap = ListedColormap(colors)

    legend_patches = lp(colors=colors, labels=labels, edgecolor='lightgrey')

    end = -1 if extend_max else None
    cmap_cbar = ListedColormap(colors[int(extend_min):end, :])
    cmap_cbar.set_under(cmap(0))
    cmap_cbar.set_over(cmap(cmap.N))
    cmap_cbar.set_bad("White")
    norm = BoundaryNorm(bins, len(bins) - 1)

    return cmap_cbar, norm, legend_patches, extend


def _determine_cmap_boundaries_continuous(m, vmin, vmax):
    """
    Function that creates the BoundaryNorm instance and an adjusted Colormap to segregate the data that will be plotted
    in bins. It is called from `plot_maps` and `plot_shapes`.

    Parameters
    ----------
    m : array_like
        Data
    vmin, vmax : float, optional
        vmin and vmax parameters for plt.imshow().

    Returns
    -------
    norm, extend
    """
    if m.dtype == np.float:
        data = m[~np.isnan(m)]
    else:
        data = m
    minimum = data.min()
    maximum = data.max()

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

    return norm, extend


def _create_geometry_values_and_sizes(lat=None, lon=None, values=None, s=None, df=None):
    """
    Function that manages values from either lists or a geodataframe. It is used for `plot_shapes` and
    `plot_classified_shapes`.

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

    Returns
    -------
    geometry, values, markersize
    markersize can be None if None is passed in as `s`
    """
    if isinstance(df, type(None)):
        lon = np.array(lon).flatten()
        lat = np.array(lat).flatten()
        values = np.array(values).flatten()
        if lon.size != lat.size:
            raise IndexError("Mismatch in length of `lat` and `lon`")

        if isinstance(s, type(None)):
            markersize = None
        else:
            markersize = np.array(s).flatten()
        geometry = np.array([Point(lon[i], lat[i]) for i in range(len(lon))])

    else:
        geometry = df.loc[:, 'geometry']
        if isinstance(values, type(None)):
            values = "values"
        if isinstance(values, str):
            if values in df.columns:
                values = df.loc[:, values].values
            else:
                values = np.array([None])
        else:
            values = np.array(values).flatten()

        if isinstance(s, type(None)):
            s = "s"
        if isinstance(s, str):
            if s in df.columns:
                markersize = df.loc[:, s]
            else:
                markersize = None
        else:
            markersize = np.array(s).flatten()

    if values.size == 1:
        if isinstance(values[0], type(None)):
            values = np.array(1)
        values = values.repeat(geometry.size)
    elif values.size != geometry.size:
        raise IndexError("Mismatch length sizes and geometries")

    if not isinstance(markersize, type(None)):
        if markersize.size == 1:
            markersize = markersize.repeat(geometry.size)
        elif markersize.size != geometry.size:
            raise IndexError("Mismatch length of `s` and coordindates")

    return gpd.GeoDataFrame(geometry=geometry), values, markersize


def cbar_decorator(cbar, ticks=None, ticklabels=None, title="", label="", tick_params=None, title_font=None,
                   label_font=None, fontsize=None):
    if not isinstance(ticks, type(None)):
        cbar.set_ticks(ticks)
        if not isinstance(ticklabels, type(None)):
            cbar.set_ticklabels(ticklabels)

    if isinstance(tick_params, type(None)):
        tick_params = {}
    if isinstance(title_font, type(None)):
        title_font = {}
    if isinstance(label_font, type(None)):
        label_font = {}

    if not isinstance(fontsize, type(None)):
        tick_params.update({'labelsize': fontsize})
        title_font.update({"fontsize": fontsize})
        label_font.update({"fontsize": fontsize})

    cbar.ax.set_title(title, **title_font)
    cbar.ax.tick_params(**tick_params)
    cbar.set_label(label, **label_font)

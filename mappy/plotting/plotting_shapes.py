import cartopy
from matplotlib.colors import Colormap, Normalize
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
import cartopy.crs as ccrs

from .colors import add_colorbar
from .misc import _determine_cmap_boundaries


def plot_shapes(lat=None, lon=None, values=None, s=None, df=None, bins=None, cmap=None, vmin=None, vmax=None,
                legend="colorbar", clip_legend=False, ax=None, figsize=(10, 10), legend_kwargs=None, aspect=30,
                pad_fraction=0.6, **kwargs):
    """
    Plot shapes in a continuous fashion

    Parameters
    ----------
    lat, lon : array-like
        Lattitude and Longitude
    values : array-like
        Values at each pair of lattitude and longitude entries
    s : array-like, optional
        Size values for each pair of lattitude and longitude entries
    df : GeoDataFrame, optional
        GeoDataFrame with columns values and s as described above. The size parameter only works when dealing with
        points
    bins : array-like, optional
        todo; currently fails for one bin
        List of bins that will be used to create a BoundaryNorm instance to discretise the plotting. This does not work
        in conjunction with vmin and vmax. Bins in that case will take the upper hand.  Alternatively a 'norm' parameter
        can be passed on in the have outside control on the behaviour.
    cmap : matplotlib.cmap or str, optional
        Matplotlib cmap instance or string the will be recognized by matplotlib
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
    legend_kwargs : dict, optional
        Extra parameters for the colorbar call
    aspect : float, optional
        aspact ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    kwargs : dict, optional
        Keyword arguments for plt.imshow()

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Returns
    -------
    Axes
    """
    if isinstance(ax, type(None)):
        f, ax = plt.subplots(figsize=figsize)

    if isinstance(cmap, type(None)):
        cmap = plt.cm.get_cmap("viridis")
    elif isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    elif not isinstance(cmap, Colormap):
        raise TypeError("cmap not recognized")

    if isinstance(df, type(None)):
        lon = np.array(lon)
        lat = np.array(lat)
        df = gpd.GeoDataFrame({'geometry': [Point(lon[i], lat[i]) for i in range(len(lon))],
                               'values': values,
                               's': s})
        if isinstance(s, type(None)):
            markersize = None
        else:
            markersize = "s"
    else:
        if 's' in df.columns:
            markersize = "s"
        else:
            markersize = None

    if isinstance(bins, type(None)):
        if isinstance(vmin, type(None)):
            vmin = df.loc[:,'values'].min()
        if isinstance(vmax, type(None)):
            vmax = df.loc[:, 'values'].max()
        df.plot(column='values', vmin=vmin, vmax=vmax, markersize=markersize, linewidth=0, cmap=cmap,
                transform=ccrs.PlateCarree(), ax=ax, **kwargs)
        im = matplotlib.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {}
        if legend == "colorbar":
            add_colorbar(im, ax=ax, aspect=aspect, pad_fraction=pad_fraction, **legend_kwargs)
    else:
        cmap, norm, legend_patches, extend = _determine_cmap_boundaries(m=values, bins=bins, cmap=cmap,
                                                                        clip_legend=clip_legend)
        df.plot(column='values', norm=norm, markersize=markersize, linewidth=0, cmap=cmap,
                transform=ccrs.PlateCarree(), ax=ax, **kwargs)
        im = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        if legend == "legend":
            if isinstance(legend_kwargs, type(None)):
                legend_kwargs = {"facecolor": "white", "edgecolor": "lightgrey", 'loc': 0}
            ax.legend(handles=legend_patches, **legend_kwargs)
        elif legend == "colorbar":
            if isinstance(legend_kwargs, type(None)):
                legend_kwargs = {}
            cbar = add_colorbar(im=im, ax=ax, extend=extend, aspect=aspect, pad_fraction=pad_fraction, **legend_kwargs)

    return ax
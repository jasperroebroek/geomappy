#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cartopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Colormap, Normalize

from geomappy.plotting.misc import _determine_cmap_boundaries_continuous, _determine_cmap_boundaries_discrete, \
    cbar_decorator
from .colors import create_colorbar_axes, cmap_discrete
from ..ndarray_functions.nan_functions import nanunique, nandigitize
from ..plotting import add_colorbar, cmap_2d
from ..plotting import legend_patches as lp


def plot_map(m, bins=None, bin_labels=None, cmap=None, vmin=None, vmax=None, legend="colorbar", clip_legend=False,
             ax=None, figsize=(10, 10), legend_ax=None, legend_kwargs=None, fontsize=None, aspect=30, pad_fraction=0.6,
             force_equal_figsize=False, nan_color="White", **kwargs):
    """
    Plot rasters in a continuous fashion

    Parameters
    ----------
    m : array-like
        Input array. Needs to be either 2D or 3D if the third axis contains RGB(A) information
    bins : array-like, optional
        List of bins that will be used to create a BoundaryNorm instance to discretise the plotting. This does not work
        in conjunction with vmin and vmax. Bins in that case will take the upper hand.  Alternatively a 'norm'
        parameter can be passed on in the have outside control on the behaviour.
    bin_labels : array-like, optional
        This parameter can be used to override the labels on the colorbar. Should have the same length as bins.
    cmap : matplotlib.cmap or str, optional
        Matplotlib cmap instance or string the will be recognized by matplotlib
    vmin, vmax : float, optional
        vmin and vmax parameters for plt.imshow(). This does have no effect in conjunction with bins being provided.
    legend : {'colorbar', 'legend', False}, optional
        Legend type that will be plotted. The 'legend' type will only work if bins are specified.
    clip_legend : bool, optional
        Clip the legend to the minimum and maximum of bins are provided. If False the colormap will remain intact over
        the whole provided bins, which potentially lowers contrast a lot, but works great for having a common colorbar
        for several plots.
    ax : `matplotlib.Axes`, optional
        Axes object. If not provided it will be created on the fly.
    figsize : tuple, optional
        Matplotlib figsize parameter. Default is (10, 10)
    legend_ax : `matplotlib.Axes`, optional
        Axes object that the legend will be drawn on
    legend_kwargs : dict, optional
        Extra parameters to create and decorate the colorbar or the call to `plt.legend` if `legend` == "legend"
        For the colorbar creation: shrink, position and extend (which would override the internal behaviour)
        For the colorbar decorator see `cbar_decorate`.
    fontsize : float, optional
        Fontsize of the legend
    aspect : float, optional
        aspect ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    force_equal_figsize : bool, optional
        when plotting with a colorbar the figure is going be slightly smaller than when you are using `legend` or non
        at all. This parameter can be used to force equal sizes, meaning that the version with a `legend` is going to
        be slightly reduced.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'white'
    **kwargs
        Keyword arguments for plt.imshow()

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the `extent` of the data should be provided
    if there is not a perfect overlap. If provided to this parameter it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different than the data projection.

    Returns
    -------
    (Axes or GeoAxes, legend)
    legend depends on the `legend` parameter
    """
    if m.ndim not in (2, 3):
        raise ValueError("Input data needs to be 2D or present RGB(A) values on the third axis.")
    if m.ndim == 3 and (m.shape[-1] not in (3, 4) or np.issubdtype(m.dtype, np.bool_)):
        raise ValueError(f"3D arrays are only acceptable if presenting RGB(A) information. Shape: {m.shape}, "
                         f"dtype: {m.dtype}")

    if isinstance(ax, type(None)):
        f, ax = plt.subplots(figsize=figsize)
    elif isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
        if "extent" not in kwargs:
            kwargs["extent"] = ax.get_extent()
        if "transform" not in kwargs:
            kwargs["transform"] = ax.projection

    if isinstance(cmap, type(None)):
        cmap = plt.cm.get_cmap("viridis")
    elif isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    elif not isinstance(cmap, Colormap):
        raise TypeError("cmap not recognized")

    if isinstance(legend_kwargs, type(None)):
        if legend == "colorbar" or not legend:
            legend_kwargs = {}
        elif legend == "legend":
            legend_kwargs = {"facecolor": "white", "edgecolor": "lightgrey", 'loc': 0}

    if 'fontsize' not in legend_kwargs and not isinstance(fontsize, type(None)):
        legend_kwargs['fontsize'] = fontsize

    if not isinstance(bins, type(None)) and len(bins) == 1:
        nan_mask = np.isnan(m)
        m = (m > bins[0]).astype(float)
        m[nan_mask] = np.nan
        ax, legend_ = plot_classified_map(m, colors=['lightgrey', 'red'], bins=[0, 1],
                                          labels=[f'< {bins[0]}', f'> {bins[0]}'], ax=ax, legend_ax=legend_ax,
                                          legend=legend, legend_kwargs=legend_kwargs, aspect=aspect,
                                          pad_fraction=pad_fraction, force_equal_figsize=False, nan_color=nan_color,
                                          **kwargs)

    elif not np.issubdtype(m.dtype, np.bool_) and m.ndim == 2:
        if isinstance(bins, type(None)):
            norm, extend = _determine_cmap_boundaries_continuous(m=m, vmin=vmin, vmax=vmax)
            cmap.set_bad(nan_color)
            im = ax.imshow(m, norm=norm, origin='upper', cmap=cmap, **kwargs)

        else:
            cmap, norm, legend_patches, extend = _determine_cmap_boundaries_discrete(m=m, bins=bins, cmap=cmap,
                                                                                     clip_legend=clip_legend)
            cmap.set_bad(nan_color)
            im = ax.imshow(m, norm=norm, cmap=cmap, origin='upper', **kwargs)

            if legend == "legend":
                if isinstance(legend_ax, type(None)):
                    legend_ax = ax
                legend_ = legend_ax.legend(handles=legend_patches, **legend_kwargs)

        if legend == "colorbar":
            cbar = add_colorbar(im=im, ax=ax, cax=legend_ax, aspect=aspect, pad_fraction=pad_fraction,
                                extend=legend_kwargs.pop('extend', extend), shrink=legend_kwargs.pop('shrink', 1),
                                position=legend_kwargs.pop("position", "right"))
            cbar_decorator(cbar, ticks=bins, ticklabels=bin_labels, **legend_kwargs)

        if legend == "colorbar":
            legend_ = cbar
        elif not legend:
            legend_ = None

    elif m.ndim == 3:
        ax.imshow(m, origin='upper', **kwargs)
        legend_ = None

    else:
        # Case of boolean data
        ax, legend_ = plot_classified_map(m.astype(int), bins=[0, 1], colors=['lightgrey', 'red'],
                                          labels=['False', 'True'], ax=ax, legend_ax=legend_ax, legend=legend,
                                          legend_kwargs=legend_kwargs, aspect=aspect, pad_fraction=pad_fraction,
                                          force_equal_figsize=False, **kwargs)

    if force_equal_figsize and legend != 'colorbar':
        create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction,
                             position=legend_kwargs.get("position", "right")).axis("off")

    return ax, legend_


def plot_classified_map(m, bins=None, colors=None, cmap="tab10", labels=None, legend="legend", clip_legend=False,
                        ax=None, figsize=(10, 10), suppress_warnings=False, legend_ax=None,  legend_kwargs=None,
                        fontsize=None, aspect=30, pad_fraction=0.6, force_equal_figsize=False, nan_color="White",
                        **kwargs):
    """
    Plot a map with discrete classes or index

    Parameters
    ----------
    m : array
        input map
    bins : list, optional
        list of either bins as used in np.digitize . By default this parameter is not necessary, the unique values are
        taken from the input map
    colors : list, optional
        List of colors in a format understandable by matplotlib. By default colors will be obtained from `cmap`
    cmap : matplotlib cmap or str, optional
        Can be used to set a colormap when no colors are provided.
    labels : list, optional
        list of labels for the different classes. By default the unique values are taken as labels
    legend : {'legend', 'colorbar', False}, optional
        Presence and type of legend. 'Legend' wil insert patches, 'colorbar' wil insert a colorbar and False will
        prevent any legend to be printed.
    clip_legend : bool, optional
        remove the items from the legend that don't occur on the map but are passed in
    ax : axes, optional
        matplotlib axes to plot the map on. If not given it is created on the fly. A cartopty GeoAxis can be provided.
    figsize : tuple, optional
        Matplotlib figsize parameter. Default is (10, 10)
    suppress_warnings : bool, optional
        By default 10 classes is the maximum that can be plotted. If set to True this maximum is removed
    legend_ax : `matplotlib.Axes`, optional
        Axes object that the legend will be drawn on
    legend_kwargs : dict, optional
        Extra parameters to create and decorate the colorbar or the call to `plt.legend` if `legend` == "legend"
        For the colorbar creation: shrink, position and extend (which would override the internal behaviour)
        For the colorbar decorator see `cbar_decorate`.
    fontsize : float, optional
        Fontsize of the legend
    aspect : float, optional
        aspact ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    force_equal_figsize : bool, optional
        when plotting with a colorbar the figure is going be slightly smaller than when you are using `legend` or non
        at all. This parameter can be used to force equal sizes, meaning that the version with a `legend` is going to
        be slightly reduced.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'white'
    **kwargs : dict, optional
        kwargs for the plt.imshow command

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Returns
    -------
    (Axes or GeoAxes, legend)
    legend depends on the `legend` parameter
    """
    if isinstance(ax, type(None)):
        f, ax = plt.subplots(figsize=figsize)
    elif isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
        if "extent" not in kwargs:
            kwargs["extent"] = ax.get_extent()
        if "transform" not in kwargs:
            kwargs["transform"] = ax.projection

    if isinstance(bins, type(None)):
        data = m[~np.isnan(m)]
        bins = np.unique(data)
    else:
        bins = np.array(bins)
        bins.sort()

    if len(bins) > 10 and not suppress_warnings:
        raise ValueError("Number of bins above 10, this creates issues with visibility")

    if not isinstance(colors, type(None)):
        if len(bins) != len(colors):
            raise IndexError(f"length of bins and colors don't match\nbins: {len(bins)}\ncolors: {len(colors)}")
    else:
        colors = cmap_discrete(cmap=cmap, n=len(list), return_type="list")

    if not isinstance(labels, type(None)):
        if len(bins) != len(labels):
            raise IndexError("length of bins and labels don't match")
    else:
        labels = list(bins)

    colors = np.array(colors)
    labels = np.array(labels)

    m_binned = nandigitize(m, bins=bins, right=True)

    m_binned_unique = nanunique(m_binned).astype(int)
    if (~np.all(m_binned_unique == np.linspace(0, m_binned_unique.max(), num=m_binned_unique.size)) or \
        len(m_binned_unique) != len(colors)) and clip_legend:
        colors = colors[m_binned_unique]
        labels = labels[m_binned_unique]
        m_binned = nandigitize(m_binned, bins=m_binned_unique, right=True)
        bins = m_binned_unique

    cmap = ListedColormap(colors)
    cmap.set_bad(nan_color)
    norm = Normalize(vmin=0, vmax=bins.size - 1)

    legend_patches = lp(colors=colors, labels=labels, edgecolor='lightgrey')

    im = ax.imshow(m_binned, cmap=cmap, origin='upper', norm=norm, **kwargs)

    # Legend
    if isinstance(legend_kwargs, type(None)):
        if legend == "colorbar" or not legend:
            legend_kwargs = {}
        elif legend == "legend":
            legend_kwargs = {"facecolor": "white", "edgecolor": "lightgrey", 'loc': 0}

    if 'fontsize' not in legend_kwargs and not isinstance(fontsize, type(None)):
        legend_kwargs['fontsize'] = fontsize

    if legend == "legend":
        if isinstance(legend_ax, type(None)):
            legend_ax = ax
        legend_ = legend_ax.legend(handles=legend_patches, **legend_kwargs)

    elif legend == "colorbar":
        cbar = add_colorbar(im=im, ax=ax, cax=legend_ax, shrink=legend_kwargs.pop("shrink", 1),
                            position=legend_kwargs.pop("position", "right"), aspect=aspect, pad_fraction=pad_fraction)

        boundaries = cbar._boundaries
        tick_locations = [(boundaries[i] - boundaries[i - 1]) / 2 + boundaries[i - 1]
                          for i in range(1, len(boundaries))]

        cbar_decorator(cbar, ticks=tick_locations, ticklabels=labels, **legend_kwargs)

    if force_equal_figsize and legend != 'colorbar':
        create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction,
                             position=legend_kwargs.get("position", "right")).axis("off")

    if legend == "colorbar":
        legend_ = cbar
    elif not legend:
        legend_ = None

    return ax, legend_


def create_map_cmap_2d(map1, map2, bin1=None, bin2=None, vmin1=None, vmax1=None, vmin2=None, vmax2=None, cmap=None,
                       diverging=False, plotting=False, figsize=(10, 10), ax=None, silent=False, show=False):
    """
    Combine two maps with a two dimensional legend.
    
    Parameters
    ----------
    map1, map2 : array
        two identically shaped arrays
    bin1, bin2 : list, optional
        bins used for digitization with raster_functions.misc.nandigitize().
    vmin1, vmax1, vmin2, vmax2
        min and max of both input maps used for plotting (like vmin and vmax in imshow). Ff None, which is the default,
        they will be calculated from the data. These values are only considered when no bin is passed.
    cmap : array, optional
        3D array providing bins for map1 and map2 and RGB colors on the third axis. This can be created with the cmap_2d
        interface. If not provided it will be created on the fly. If the map is already classified by np.digitize it
        will be recognized and that specific axis will be split in the number of bins as intended. If not digitized
        1000 bins will be created, which will be a smooth scale.
    diverging : bool, optional
        Use a diverging color scale if cmap is not provided. This means that the center of the 2D color legend will  be
        white instead of grey to notify divergence. Default is False.
    plotting : bool, optional
        plotting the created result
    show : bool, optional
        Execute the plt.plot() command
    ax : Axis, optional
        matplotlib axes where the plot should be plotted to. If not given a new
        figure is created, which is the default behaviour.
    figsize : list, optional
        Figsize parameter for matplotlib. Default is (10,10)
    silent : bool, optional
        Return the created map, which is a 3D numpy array with on the last axis the
        RGB color components (0-1).
        
    Returns
    -------
    map : np.ndarray
        numpy array containing the newly created map. Dimensionality depends on 'cmap' parameter. If none is passed a
        RGB legend is created, resulting in a 3D array
    """
    if not isinstance(bin1, type(None)):
        map1 = nandigitize(map1, bins=bin1)
    if not isinstance(bin2, type(None)):
        map2 = nandigitize(map2, bins=bin2)

    if map1.shape != map2.shape:
        raise ValueError("maps are not of equal shape")
    if type(plotting) != bool:
        raise TypeError("plotting is a boolean variable")
    if type(diverging) != bool:
        raise TypeError("diverging is a boolean variable")

    # copy maps to prevent overwriting problems because a memory locations is send
    # to the function instead of an array.
    axis0 = map1.copy()
    axis1 = map2.copy()

    # create a nan mask that will be used to overwrite the final map with nans
    mask = np.logical_or(np.isnan(axis0), np.isnan(axis1))

    # create lists with the unique values in both maps (nan values filtered out)
    axis0_unique = np.unique(axis0[~np.isnan(axis0)])
    axis1_unique = np.unique(axis1[~np.isnan(axis1)])

    # Routine to check if input maps are already classified or not.
    # Output is a list of two boolean values. A map is classified if the unique
    # values of that map overlap with a linspace instance with the same min,max and
    # length as the unique values.
    # for example:
    #       unique_values = [0,1,2]
    #       np.linspace(unique_values[0], unique_values[-1], num = unique_values.size)
    #       > [0,1,2]
    # these two lists are equal and therefore the map can be seen as classified.
    classified = [np.allclose(np.linspace(axis0_unique[0], axis0_unique[-1],
                                          num=axis0_unique.size), axis0_unique),
                  np.allclose(np.linspace(axis1_unique[0], axis1_unique[-1],
                                          num=axis1_unique.size), axis1_unique)]

    # If cmap is not provided it is created here. 
    if isinstance(cmap, type(None)):
        print("cmap:")
        l = [axis0_unique.size if classified[0] else 1000,
             axis1_unique.size if classified[1] else 1000]
        cmap = cmap_2d(shape=l, plotting=True, diverging=diverging)

    old_settings = np.seterr(all='ignore')  # silence all numpy warnings

    # Routines to create a vector containing the bins for both input maps
    # called axisx_space.
    if classified[0]:
        # if axis0_unique.size != cmap.shape[0]:
        #    raise IndexError("Shape of cmap and the first map don't match")
        axis0_space = list(axis0_unique)
    else:
        if type(vmin1) != type(None):
            axis0[axis0 < vmin1] = vmin1
        if type(vmax1) != type(None):
            axis0[axis0 > vmax1] = vmax1
        axis0_space = np.linspace(np.nanmin(axis0), np.nanmax(axis0),
                                  num=cmap.shape[0])

    if classified[1]:
        # if axis1_unique.size != cmap.shape[1]:
        #    raise IndexError("Shape of cmap and the second map don't match")
        axis1_space = list(axis1_unique)
    else:
        if not isinstance(vmin2, type(None)):
            axis1[axis1 < vmin2] = vmin2
        if type(vmax2) != type(None):
            axis1[axis1 > vmax2] = vmax2
        axis1_space = np.linspace(np.nanmin(axis1), np.nanmax(axis1),
                                  num=cmap.shape[1])

    np.seterr(**old_settings)  # reset to default

    # remove nans with temporary value that will be replaced based on the mask variable
    axis0[np.isnan(axis0)] = 0
    axis1[np.isnan(axis1)] = 0

    # digitize both maps
    axis0_digitized = np.digitize(axis0, axis0_space) - 1
    axis1_digitized = np.digitize(axis1, axis1_space) - 1

    # combine maps with cmap
    digitized_map = cmap[axis0_digitized, axis1_digitized]
    # set the nan values to white

    if cmap.ndim == 2:
        digitized_map[mask] = -1
    if cmap.ndim == 3:
        digitized_map[mask, :] = np.array((1, 1, 1))

    # plotting routine
    if plotting:
        if type(ax) == type(None):
            f, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(digitized_map)
        ax.set_xticks([])
        ax.set_yticks([])
        if show:
            plt.show()

    if not silent:
        return digitized_map
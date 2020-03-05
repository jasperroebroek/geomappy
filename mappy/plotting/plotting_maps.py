#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import geopandas as gpd
import pandas as pd
import cartopy
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from mpl_toolkits import axes_grid1
from ..plotting import add_colorbar, cmap_2d, cmap_random, cmap_discrete
from ..ndarray_functions.nan_functions import nanunique, nandigitize


def plot_map(m, bins=None, cmap=None, vmin=None, vmax=None, legend="colorbar", clip_legend=False, ax=None,
             figsize=(10, 10), legend_kwargs=None, aspect=30, pad_fraction=0.6, **kwargs):
    """
    Plot rasters in a continuous fashion

    Parameters
    ----------
    m : array-like
        Input array. Needs to be either 2D or 3D if the third axis contains RGB(A) information
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

    Returns
    -------
    Axes
    """
    if m.ndim not in (2, 3):
        raise ValueError("Input data needs to be 2D or present RGB(A) values on the third axis.")
    if m.ndim == 3 and (m.shape[-1] not in (3, 4) or m.dtype != "bool_"):
        raise ValueError("3D arrays are only acceptable if presenting RGB(A) information")

    if isinstance(ax, type(None)):
        f, ax = plt.subplots(figsize=figsize)

    if isinstance(cmap, type(None)):
        cmap = plt.cm.get_cmap("viridis")
    elif isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    # in case of a boolean array no colorbar is shown
    if m.dtype != "bool_" and m.ndim == 2:
        if isinstance(bins, type(None)):
            im = ax.imshow(m, vmin=vmin, vmax=vmax, origin='upper', **kwargs)
            if isinstance(legend_kwargs, type(None)):
                legend_kwargs = {}
            if legend=="colorbar":
                add_colorbar(im, aspect=aspect, pad_fraction=pad_fraction, **legend_kwargs)
        else:
            data = m[~np.isnan(m)]
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

            labels = labels + [f"{bins[i - 1]} - {bins[i]}" for i in range(2, len(bins))]

            if vmax > bins[-1]:
                boundaries = np.hstack([boundaries, vmax])
                extend_max = True
                labels = labels + [f"> {bins[-1]}"]
            else:
                extend_max = False

            colors = cmap(np.linspace(0, 1, boundaries.size - 1))
            cmap = ListedColormap(colors)

            legend_patches = [Patch(facecolor=icolor, label=label, edgecolor="lightgrey")
                              for icolor, label in zip(colors, labels)]

            end = -1 if extend_max else None
            cmap_cbar = ListedColormap(colors[int(extend_min):end, :])
            cmap_cbar.set_under(cmap(0))
            cmap_cbar.set_over(cmap(cmap.N))
            norm = BoundaryNorm(bins, len(bins) - 1)
            im = ax.imshow(m, norm=norm, cmap=cmap_cbar, origin='upper', **kwargs)

            if legend == "legend":
                if isinstance(legend_kwargs, type(None)):
                    legend_kwargs = {"facecolor": "white", "edgecolor": "lightgrey", 'loc': 0}
                ax.legend(handles=legend_patches, **legend_kwargs)
            elif legend == "colorbar":
                if isinstance(legend_kwargs, type(None)):
                    legend_kwargs = {}
                if extend_min and extend_max:
                    extend = "both"
                elif not extend_min and not extend_max:
                    extend = "neither"
                elif not extend_min and extend_max:
                    extend = "max"
                elif extend_min and not extend_max:
                    extend = "min"
                cbar = add_colorbar(im=im, extend=extend, aspect=aspect, pad_fraction=pad_fraction, **legend_kwargs)

    elif m.ndim == 3:
        ax.imshow(m, origin='upper', **kwargs)
    else:
        if len(np.unique(m)) == 2:
            plot_classified_map(m.astype(int), colors=['lightgrey', 'red'], labels=['False', 'True'], ax=ax,
                                legend=legend, legend_kwargs=legend_kwargs, **kwargs)
        elif np.isin(True, m):
            plot_classified_map(m.astype(int), colors=['lightgrey'], labels=['True'], ax=ax, legend=legend,
                                legend_kwargs=legend_kwargs, **kwargs)
        else:
            plot_classified_map(m.astype(int), colors=['lightgrey'], labels=['False'], ax=ax, legend=legend,
                                legend_kwargs=legend_kwargs, **kwargs)

    return ax


def plot_world(points=None, box_bounds=False, figsize=(10, 10)):
    """
    Plotting the world. Points to plot should be given in a pandas dataframe with Lat and Lon column. Bounds object of
    a rasterio file can be plotted with box_bounds parameter.

    # todo; check this function and if it still useful

    Parameters
    ----------
    points
        1: [DataFrame]
            Need to contain the columns "Lat" and "Lon"
        2: [GeoDataFrame]
            Contains column with shapely Point geometry
        3: [list : Point]
            Contains shapely Points
    box_bounds : rasterio.bounds
        bounds from rasterio profile
    figsize : [tuple : ints]
        Two ints, passed to the plotting functions based on matplotlib
    """
    # read a small background world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # plot the background map
    t = world.plot(figsize=figsize, color="white", edgecolor='black')

    # plot the points if any
    # !! would be nice to implement a possibility of passing a list/dict
    if type(points) != type(None):
        if type(points) == pd.DataFrame:
            # add shapely geometry column (Point)
            points['geometry'] = points.apply(lambda z: Point(z.Lon, z.Lat), axis=1)
            # convert to geopandas
            points = gpd.GeoDataFrame(points)

        if type(points) == gpd.GeoDataFrame:
            pass
        if type(points) == list:
            assert type(points[0]) == Point, "List doesn't contain points"
            points = gpd.GeoDataFrame(data={"geometry": points})

        # plot the points on top of the background map
        points.plot(ax=t, color="red")

    # plot a box on the map
    if np.any(box_bounds != False):
        # create a list with the x coordinates of the corners
        box_x = [box_bounds[0], box_bounds[2], box_bounds[2], box_bounds[0]]
        # create a list with the y coordinates of the corners
        box_y = [box_bounds[3], box_bounds[3], box_bounds[1], box_bounds[1]]
        # create a shapely Polygon from those lists
        bounding_box = Polygon(zip(box_x, box_y))
        # create a geopandas dataframe with one entry
        # !! check if this step can be skipped
        gdf = gpd.GeoDataFrame(index=[0], geometry=[bounding_box])
        # plot the box on top of the background map
        gdf.plot(ax=t, edgecolor="blue", facecolor="none")

    t.set_ylim([-91, 91])
    t.set_xlim([-181, 181])
    t.set_axis_off()

    plt.show()


def plot_classified_map_old(m, bins=None, colors=None, labels=None, legend="legend", clip_legend=False, ax=None,
                        suppress_warnings=False, mode="continues", cmap=None, transform=None, extent=None,
                        legend_title_pad=10, legend_kwargs=None):
    """
    Plot a map with discrete colors

    Parameters
    ----------
    m : array
        input map
    bins : list, optional
        list of either bins as used in np.digitize or unique values corresponding to `colors` and `labels`. By default
        this parameter is not necessary, the unique values are taken from the input map
    colors : list, optional
        list of colors in a format understandable by matplotlib. By default random colors are taken
    labels : list, optional
        list of labels for the different classes. By default the unique values are taken as labels
    legend : ['legend', 'colorbar', False], optional
        Presence and type of legend. 'Legend' wil insert patches, 'colorbar' wil insert a colorbar and False will
        prevent any legend to be printed.
    clip_legend : bool, optional
        # todo; clip either legend or colormap in continuous mode
        remove the items from the legend that don't occur on the map but are passed in
    ax : axes, optional
        matplotlib axes to plot the map on. If not given it is created on the fly. Possible to pass a tuple of axes,
        the first containing a Basemap instance the second the axis to plot the legend on.
    suppress_warnings : bool, optional
        By default 15 classes is the maximum that can be plotted. If set to True this maximum is removed
    mode : ['continues', 'classes', 'index']
        'Continues' is used for mostly floating point rasters that are digitized
        'Classes' is used for individual values that can not directly be used as indices
        'Index' for raters that already contain the exact index for colors and labels lists
    cmap :
        parameter directly passed to matplotlib functions. Is used instead of colors when mode=='continues'. If specific
        colors are required a ListedColormap can be passed in.
    transform : cartopy transform object, optional
        if the axis that is passed in is a geoaxis, this transform might be used to transform the data to the right
        projection. For it to be able to work, provide the extent as well
    extent : list, optional
        should be provided with transform
    legend_title_pad : float, optional
        pad between colorbar and title of the colorbar
    legend_kwargs : dict, optional
        kwargs passed into either the legend or colorbar function

    Returns
    -------
    Similar to ax parameter. If return_image is True, the image instance will be added to the list that is returned.
    """
    if isinstance(ax, type(None)):
        f, ax = plt.subplots(figsize=(10, 10))

    if mode not in ['continues', 'classes', 'index']:
        raise ValueError("mode not recognized")

    if isinstance(cmap, type(None)):
        cmap = "viridis"

    data = m[~np.isnan(m)]

    if isinstance(bins, type(None)):
        bins = np.unique(data)
        mode = 'classes'
    else:
        bins = np.array(bins)
        bins.sort()

    if len(bins) > 15 and not suppress_warnings:
        raise ValueError("Number of bins above 15, this creates issues with visibility")

    if mode == 'continues':
        colors = np.array(colors)
        labels = np.array(labels)

        minimum = data.min()
        maximum = data.max()

        if clip_legend:
            bins = bins[np.logical_and(bins >= minimum, bins <= maximum)]

        if minimum < bins[0]:
            bins = np.hstack([minimum, bins])
            labels = [f"<{bins[1]}"]
        else:
            labels = [f"{bins[0]} - {bins[1]}"]

        labels = labels + [f"{bins[i - 1]} - {bins[i]}" for i in range(2, len(bins))]

        if maximum > bins[-1]:
            bins = np.hstack([bins, maximum])
            labels = labels + [f">{bins[-2]}"]

        m_binned = nandigitize(m, bins=bins, right=False) - 1

        if isinstance(cmap, ListedColormap):
            colors = cmap(np.linspace(0, 1, bins.size - 1))
        else:
            colors = cmap_discrete(bins.size - 1, cmap=cmap, return_type='list')

    else:
        if not isinstance(colors, type(None)):
            if len(bins) != len(colors):
                raise IndexError(f"length of bins and colors don't match\nbins: {len(bins)}\ncolors: {len(colors)}")
        else:
            colors = cmap_random(len(bins), verbose=False, return_type="rgb", color_type="pastel")

        if not isinstance(labels, type(None)):
            if len(bins) != len(labels):
                raise IndexError("length of bins and labels don't match")
        else:
            labels = list(bins)

        colors = np.array(colors)
        labels = np.array(labels)

        if mode == 'index':
            m_binned = m
        elif mode == 'classes':
            m_binned = nandigitize(m, bins=bins, right=True)

        m_binned_unique = nanunique(m_binned).astype(int)
        if (~np.all(m_binned_unique == np.linspace(0, m_binned_unique.max(), num=m_binned_unique.size)) or \
            len(m_binned_unique) != len(colors)) and clip_legend:
            colors = colors[m_binned_unique]
            labels = labels[m_binned_unique]
            m_binned = nandigitize(m_binned, m_binned_unique)

    cmap = ListedColormap(colors)

    legend_patches = [Patch(facecolor=icolor, label=label, edgecolor="lightgrey")
                      for icolor, label in zip(colors, labels)]

    # PLOTTING
    if isinstance(transform, type(None)):
        if clip_legend:
            im = ax.imshow(m_binned, cmap=cmap, origin='upper')
        else:
            im = ax.imshow(m_binned, cmap=cmap, origin='upper', vmin=0, vmax=bins.size - 1)
    else:
        if clip_legend:
            im = ax.imshow(m_binned, cmap=cmap, origin='upper', transform=transform, extent=extent)
        else:
            im = ax.imshow(m_binned, cmap=cmap, origin='upper', vmin=0, vmax=bins.size - 1, transform=transform,
                           extent=extent)

    if legend == "legend":
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {"facecolor": "white", "edgecolor": "lightgrey", 'loc': 0}
        ax.legend(handles=legend_patches, **legend_kwargs)
    elif legend == "colorbar":
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {}
        aspect = legend_kwargs.pop('aspect', 20)
        pad_fraction = legend_kwargs.pop('pad_fraction', 0.5)
        title = legend_kwargs.pop("title", "")

        divider = axes_grid1.make_axes_locatable(im.axes)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1. / aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad, axes_class=plt.Axes)
        plt.sca(current_ax)
        cbar = im.axes.figure.colorbar(im, cax=cax, **legend_kwargs)

        N = colors.shape[0]
        step = (cbar.vmax - cbar.vmin) / N
        margin = step / 2

        if mode == 'continues':
            cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, num=N + 1))
            cbar.ax.set_yticklabels(bins)
        else:
            cbar.set_ticks(np.linspace(cbar.vmin + margin, cbar.vmax - margin, num=N))
            cbar.ax.set_yticklabels(labels)

        cbar.ax.set_title(title, pad=legend_title_pad)

    return ax


def plot_classified_map(m, bins=None, colors=None, labels=None, legend="legend", clip_legend=False, ax=None,
                        suppress_warnings=False, mode="classes", transform=None, legend_kwargs=None,
                        legend_title_pad=10, **kwargs):
    """
    Plot a map with discrete classes or index

    Parameters
    ----------
    m : array
        input map
    bins : list, optional
        list of either bins as used in np.digitize or unique values corresponding to `colors` and `labels`. By default
        this parameter is not necessary, the unique values are taken from the input map
    colors : list, optional
        list of colors in a format understandable by matplotlib. By default random colors are taken
    labels : list, optional
        list of labels for the different classes. By default the unique values are taken as labels
    legend : ['legend', 'colorbar', False], optional
        Presence and type of legend. 'Legend' wil insert patches, 'colorbar' wil insert a colorbar and False will
        prevent any legend to be printed.
    clip_legend : bool, optional
        remove the items from the legend that don't occur on the map but are passed in
    ax : axes, optional
        matplotlib axes to plot the map on. If not given it is created on the fly. A cartopty GeoAxis can be provided.
    suppress_warnings : bool, optional
        By default 15 classes is the maximum that can be plotted. If set to True this maximum is removed
    mode : {'classes', 'index'}
        'Classes' is used for individual values that can not directly be used as indices
        'Index' for raters that already contain the exact index for colors and labels lists
    cmap : Colormap
        parameter directly passed to matplotlib functions. Is used instead of colors when mode=='continues'. If specific
        colors are required a ListedColormap can be passed in.
    transform : list, optional
        When providing a cartopy GeoAxes in 'ax' parameter, the transform of the data can be provided here.
    legend_kwargs : dict, optional
        kwargs passed into either the legend or colorbar function
    legend_title_pad : float, optional
        pad between colorbar and title of the colorbar
    **kwargs : dict, optional
        kwargs for the plt.imshow command

    Returns
    -------
    Axes
    """
    if isinstance(ax, type(None)):
        f, ax = plt.subplots(figsize=(10, 10))

    if mode not in ('classes', 'index'):
        raise ValueError("mode not recognized")

    if isinstance(bins, type(None)):
        data = m[~np.isnan(m)]
        bins = np.unique(data)
        mode = 'classes'
    else:
        bins = np.array(bins)
        bins.sort()

    if len(bins) > 15 and not suppress_warnings:
        raise ValueError("Number of bins above 15, this creates issues with visibility")

    if not isinstance(colors, type(None)):
        if len(bins) != len(colors):
            raise IndexError(f"length of bins and colors don't match\nbins: {len(bins)}\ncolors: {len(colors)}")
    else:
        colors = cmap_random(len(bins), verbose=False, return_type="rgb", color_type="pastel")

    if not isinstance(labels, type(None)):
        if len(bins) != len(labels):
            raise IndexError("length of bins and labels don't match")
    else:
        labels = list(bins)

    colors = np.array(colors)
    labels = np.array(labels)

    if mode == 'index':
        m_binned = m
    elif mode == 'classes':
        m_binned = nandigitize(m, bins=bins, right=True)

    m_binned_unique = nanunique(m_binned).astype(int)
    if (~np.all(m_binned_unique == np.linspace(0, m_binned_unique.max(), num=m_binned_unique.size)) or \
        len(m_binned_unique) != len(colors)) and clip_legend:
        colors = colors[m_binned_unique]
        labels = labels[m_binned_unique]
        m_binned = nandigitize(m_binned, m_binned_unique)

    cmap = ListedColormap(colors)

    legend_patches = [Patch(facecolor=icolor, label=label, edgecolor="lightgrey")
                      for icolor, label in zip(colors, labels)]

    # Plotting
    if clip_legend:
        vmin=None
        vmax=None
    else:
        vmin=0
        vmax=bins.size-1

    if isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
        if isinstance(transform, type(None)):
            transform = ax.projection
        if isinstance(kwargs, dict):
            if 'extent' in kwargs:
                extent = kwargs['extent']
            else:
                extent = ax.get_extent()
        else:
            extent = ax.get_extent()

        kwargs.update({'transform': transform,
                       'extent': extent})

    im = ax.imshow(m_binned, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax, **kwargs)


    # Legend
    if legend == "legend":
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {"facecolor": "white", "edgecolor": "lightgrey", 'loc': 0}
        ax.legend(handles=legend_patches, **legend_kwargs)
    elif legend == "colorbar":
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {}
        title = legend_kwargs.pop("title", "")

        cbar = add_colorbar(im=im, **legend_kwargs)

        N = colors.shape[0]
        step = (cbar.vmax - cbar.vmin) / N
        margin = step / 2

        cbar.set_ticks(np.linspace(cbar.vmin + margin, cbar.vmax - margin, num=N))
        cbar.ax.set_yticklabels(labels)
        cbar.ax.set_title(title, pad=legend_title_pad)

    return ax


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


if __name__ == "__main__":
    from thesis import *
    wtd = Map("../data/wtd.tif")[0]

    plot_map(wtd, [0, 1, 2, 3, 6, 100], cmap=plt.cm.get_cmap("Greens"), legend="colorbar")
    plt.show()

    plot_map(wtd, np.logspace(-4, 4), cmap=plt.cm.get_cmap("Greens"), legend="colorbar",
             legend_kwargs={'format': "%.5f"}, clip_legend=True)
    plt.show()

    plot_classified_map_old(wtd, [1, 2, 3, 6, 100], cmap=plt.cm.get_cmap("Greens"), legend="colorbar")
    plt.show()

    a = np.random.randint(0,3,900).reshape(30,30)-1
    bins = [-1,0,1,2]
    colors = ['Red', 'Green', 'Yellow', 'Purple']
    mode = 'classes'

    plot_classified_map_old(a, bins=bins, colors=colors, mode=mode, legend='colorbar', clip_legend=True)
    plt.show()
    plot_classified_map(a, bins=bins, colors=colors, mode=mode, legend='colorbar', clip_legend=True)
    plt.show()


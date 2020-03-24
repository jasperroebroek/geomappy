#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib.cm import ScalarMappable
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
from matplotlib import colors, colorbar
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array
import numpy as np
from ..ndarray_functions.misc import grid_from_corners
import colorsys
from matplotlib.colorbar import ColorbarBase


def plot_colors(c, figsize=(10, 1), ticks=False, **kwargs):
    """
    Plot a horizontal colorbar to inspect colors

    Parameters
    ----------
    c : array-like or Colormap instance
        Iterable containing colors. A numpy ndarray with 3 dimensions will be interpreted as RBA(A).
    figsize : tuple, optional
        Matplotlib figsize parameter
    ticks : bool, optional
        Add ticks to the figure
    show : bool, optional
        Execute the plt.show() command
    """
    plt.rcParams['savefig.pad_inches'] = 0

    if isinstance(c, (list, tuple, np.ndarray)):
        c = np.array(c)
        N = c.shape[0]
        cmap = LinearSegmentedColormap.from_list('plot', c, N=N)
    elif isinstance(c, colors.Colormap):
        N = c.N
        cmap = c

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    bounds = np.linspace(0, 1, N + 1)
    if N < 256:
        norm = colors.BoundaryNorm(bounds, N)
    else:
        norm = None

    cb = ColorbarBase(ax, cmap=cmap, norm=norm, spacing='proportional', ticks=None,
                      boundaries=bounds, format='%1i', orientation=u'horizontal', **kwargs)
    ax.patch.set_edgecolor('black')
    if not ticks:
        plt.tick_params(axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)  # labels along the bottom edge are off
    else:
        cb.set_ticks(np.linspace(1/(2*N), 1-1/(2*N), N))
        cb.set_ticklabels(np.arange(N))
    plt.tight_layout()


def cmap_2d(shape=(1000, 1000), v=None, alpha=0, plotting=False, show=False, diverging=False, diverging_alpha=0.5,
            silent=False, rotate=0, flip=False, ax=None):
    """
    Creates a 2 dimensional legend
    
    Parameters
    ----------
    shape : list
        Shape of the returned array
    v : list, optional
        If provided it will override the corner colors.
        It needs to exist of a list of four colors (convertible by to_rgba_array).
    alpha : float, optional
        Value between 0 and 1. For details check Ryan's paper. Default is 0.
    plotting : bool, optional
        Plotting cmap. Default is False
    show : bool, optional
        Execute the plt.show() command
        todo; deprecate
    diverging : bool, optional
        Apply whitening kernel causing the centre of the cmap to be white if v is left to None. Default is False.
    diverging_alpha : float, optional
        The central RGB components are raised with this number. The default is 0.5, which is also the maximum, and will
        lead to white as the central colors. The minimum is -0.5 which will lead to black as the central color.
    silent : bool, optional
        Prevent returning the array, for testing purposes. Default is False.
    rotate : int, optional
        Rotate the created array clockwise. The default is zero. Options are 1, 2 or 3 which will lead to 90, 180 or 270
        degrees rotation.
    flip : bool, optional
        Flip the array left to right. Default is False.
    ax : Axes, optional
        matplotlib Axes to plot on. If not provided it is created on the fly.
    
    Returns
    -------
    cmap : array
        The created two dimensional legend. Array of dimensions (*shape, 3). 
        RGB configuration
    """

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha needs to be between 0 and 1")
    if type(shape) not in (list, tuple):
        raise TypeError("shape is expected in the form of a list or tuple")
    else:
        if len(shape) != 2:
            raise ValueError("shape needs to have length 2")
        else:
            if shape[0] < 2 or shape[1] < 2:
                raise ValueError("shape needs to exist out of positive integers above 1")
            if type(shape[0]) != int or type(shape[1]) != int:
                raise TypeError("shape needs to contain two ints")

    if type(plotting) != bool:
        raise TypeError("plotting needs to be a boolean")
    if type(show) != bool:
        raise TypeError("show needs to be a boolean")
    if type(diverging) != bool:
        raise TypeError("diverging needs to be a boolean")
    if type(silent) != bool:
        raise TypeError("silent needs to be a boolean")

    if type(rotate) != int:
        raise TypeError("rotate should be an integer")
    if type(flip) != bool:
        raise TypeError("flip should be a boolean")

    if type(v) == type(None):
        v = [0, 0, 0, 0]
        v[0] = (0, 0.5, 1)  # left upper corner
        v[1] = (1 - alpha, 0, 1 - alpha)  # right upper corner
        v[2] = (1, 0.5, 0)  # lower right corner
        v[3] = (alpha, 1, alpha)  # lower left corner
        corner_colors = np.array(v)
        corner_colors = np.roll(corner_colors, rotate, axis=0)
        if flip:
            corner_colors = np.flipud(corner_colors)
    else:
        corner_colors = to_rgba_array(v)
    cmap = np.empty((*shape, 3))
    for i in range(3):
        cmap[:, :, i] = grid_from_corners(corner_colors[:, i], shape=shape,
                                          plotting=False)
    if diverging:
        x, y = np.mgrid[-1:1:shape[0] * 1j,
               -1:1:shape[1] * 1j]
        cmap += (-diverging_alpha * x ** 2 / 2 + -diverging_alpha * y ** 2 / 2 + diverging_alpha)[:, :, np.newaxis]

    cmap = np.maximum(0, cmap)
    cmap = np.minimum(1, cmap)
    if plotting:
        if type(ax) == type(None):
            f, ax = plt.subplots(1)
        ax.imshow(cmap, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"shape : {shape}, alpha : {alpha}, diverging : {diverging}")
        if show:
            plt.show()
    if not silent:
        return cmap


def cmap_discrete(n, cmap='hsv', return_type='cmap', plotting=False):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color as obtained from a matplotlib
    cmap instance as indicated with the 'cmap' parameter.

    Parameters
    ----------
    n : int
        Number of colors
    cmap : str or cmap instance, optional
        Name of cmap (or cmap itself), needs to be available in the matplotlib namespace
    return_type : str, optional
        'cmap' returns a linearly segmented cmap, 'list' returns a numpy array with the colors
    plotting : bool, optional
        plot the created cmap, the default is False

    Returns
    -------
    cmap
        type depends on return_type parameter
    """

    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap, n)
    else:
        cmap = LinearSegmentedColormap.from_list("new_cmap", cmap(np.linspace(0, 1, n)), n)

    if plotting:
        plot_colors(cmap)
    if return_type == 'cmap':
        return cmap
    elif return_type == 'list':
        return cmap(np.linspace(0, 1, n))


def cmap_from_borders(colors, n=256, return_type='cmap', plotting=False):
    """
    Creates a cmap based on two colors

    Parameters
    ----------
    colors : list
        List of two colors. Should be in a format accepted by matplotlib to_rgba_array
    n : int, optional
        Can be set if a discrete colormap is needed.  The default is 256 which is the standard for a smooth color
        gradient
    return_type : str, optional
        'cmap' returns a cmap object, 'list' returns an ndarray.
    plotting : bool, optional
        plot colors

    Returns
    -------
    cmap or list
        Behaviour depends on return_type parameter
    """
    colors = to_rgba_array(colors)[:, :-1]
    cmap = np.vstack([np.linspace(colors[0][i], colors[1][i], num=n) for i in range(3)]).T

    if plotting:
        plot_colors(cmap)

    if return_type == 'list':
        return cmap
    if return_type == 'cmap':
        return LinearSegmentedColormap.from_list("new_cmap", cmap, N=n)


def cmap_random(nlabels, color_type='pastel', first_color=None, last_color=None, verbose=False, return_type="cmap"):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks

    # todo; revisit

    Parameters
    ----------
    nlabels : int
        Number of labels (size of colormap)
    color_type : {"bright","pastel"}
        'bright' for strong colors, 'soft' for pastel colors, which is the default behaviour
    first_color : {"black", "white"}
        Option to set first color to black/ white or to take a random color
    last_color : {"black", "white"}
        Option to set last color to black/ white or to take a random color
    verbose : bool
        Prints the number of labels and shows the colormap. True or False
    return_type : {'cmap', 'rgb']
        Returning a matplotlib cmap or a list of RGB colors
        todo; change to 'list' of 'rgb'
    
    Returns
    -------
    cmap or list
        Behaviour depends on return_type parameter

    Source
    ------
    https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    
    """
    if color_type not in ('bright', 'pastel'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if color_type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color in ("black", "white"):
            if first_color == "black":
                randRGBcolors[0] = [0, 0, 0]
            if first_color == "white":
                randRGBcolors[0] = [1, 1, 1]
        if last_color in ("black", "white"):
            if last_color == "black":
                randRGBcolors[-1] = [0, 0, 0]
            if last_color == "white":
                randRGBcolors[-1] = [1, 1, 1]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if color_type == 'pastel':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color in ("black", "white"):
            if first_color == "black":
                randRGBcolors[0] = [0, 0, 0]
            if first_color == "white":
                randRGBcolors[0] = [1, 1, 1]
        if last_color in ("black", "white"):
            if last_color == "black":
                randRGBcolors[-1] = [0, 0, 0]
            if last_color == "white":
                randRGBcolors[-1] = [1, 1, 1]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                              boundaries=bounds, format='%1i', orientation=u'horizontal')

    if return_type == "cmap":
        return random_colormap
    if return_type == "rgb":
        return randRGBcolors


def add_colorbar(im=None, ax=None, aspect=20, pad_fraction=0.5, position="right", **kwargs):
    """
    Add colorbar to a plot

    Parameters
    ----------
    im : ScalarMappable, optional
        The source ScalarMappable. If not provided it will use the last image on the Axes
    ax : Axes, optional
        The Axes that the colorbar will be placed in. If not provided it will be assumed to be the last Axes that was
        used (plt.gca())
    aspect : float, optional
        The aspect ratio of the colorbar
    pad_fraction : float, otional
        The fraction of the height of the colorbar that the colorbar is removed from the image
    position : {"left", "right", "top", "bottom"}
        The position of the colorbar in respect to the image
    **kwargs : dict, optional
        Keyword arguments for the colorbar call

    Returns
    -------
    Colorbar

    Notes
    -----
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    if isinstance(im, type(None)):
        if isinstance(ax, type(None)):
            ax = plt.gca()
        im = ax.images[-1]
    else:
        if isinstance(ax, ScalarMappable):
            ax = plt.gca()
        elif isinstance(ax, type(None)):
            ax = im.axes

    orientation = "vertical" if position in ("right", "left") else "horizontal"

    divider = axes_grid1.make_axes_locatable(ax)
    width = axes_grid1.axes_size.AxesY(ax, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes(position=position, size=width, pad=pad, axes_class=plt.Axes)
    plt.sca(current_ax)
    return ax.figure.colorbar(im, orientation=orientation, cax=cax, **kwargs)


if __name__ == "__main__":
    plot_colors(plt.get_cmap("viridis", 20), ticks=True)
    plt.show()
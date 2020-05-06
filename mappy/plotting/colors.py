#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions here implement shortcuts to create different sort of colors/cmap instances for different scenarios. It
also contains a convenient function to add a colorbar that has the right size for the plots.
"""
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
from matplotlib import colors, colorbar
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array
from matplotlib.patches import Patch
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from mappy.ndarray_functions.misc import grid_from_corners
import colorsys
from matplotlib.colorbar import ColorbarBase


def plot_colors(c, figsize=(10, 1), ticks=False, **kwargs):
    """
    Plot a horizontal colorbar to inspect colors

    Parameters
    ----------
    c : array-like or Colormap instance
        Iterable containing colors. A :obj:`~numpy.ndarray` with 3 dimensions will be interpreted as RBA(A).
    figsize : tuple, optional
        Matplotlib figsize parameter
    ticks : bool, optional
        Add ticks to the figure
    **kwargs
        Parameters for `plt.ColorBase`
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


def cmap_2d(shape=(1000, 1000), v=None, alpha=0, plotting=False, diverging=False, diverging_alpha=0.5, rotate=0,
            flip=False, ax=None):
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
    diverging : bool, optional
        Apply whitening kernel causing the centre of the cmap to be white if v is left to None. Default is False.
    diverging_alpha : float, optional
        The central RGB components are raised with this number. The default is 0.5, which is also the maximum, and will
        lead to white as the central colors. The minimum is -0.5 which will lead to black as the central color.
    rotate : int, optional
        Rotate the created array clockwise. The default is zero. Options are 1, 2 or 3 which will lead to 90, 180 or 270
        degrees rotation.
    flip : bool, optional
        Flip the array left to right. Default is False.
    ax : `plt.Axes`, optional
        matplotlib Axes to plot on. If not provided it is created on the fly.
    
    Returns
    -------
    cmap : :obj:`~numpy.ndarray`
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
    if type(diverging) != bool:
        raise TypeError("diverging needs to be a boolean")
    if type(rotate) != int:
        raise TypeError("rotate should be an integer")
    if type(flip) != bool:
        raise TypeError("flip should be a boolean")

    if isinstance(v, type(None)):
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
        x, y = np.mgrid[-1:1:shape[0] * 1j, -1:1:shape[1] * 1j]
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

    return cmap


def cmap_discrete(cmap='hsv', n=256, return_type='cmap'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color as obtained from a matplotlib
    cmap instance as indicated with the 'cmap' parameter.

    Parameters
    ----------
    n : int
        Number of colors
    cmap : str or cmap instance, optional
        Name of cmap (or cmap itself). If `cmap` is a string it needs to be available in the matplotlib namespace
    return_type : str, optional
        'cmap' returns a linearly segmented cmap, 'list' returns a :obj:`~numpy.ndarray` array with the colors. This
        array will have shape (n, 4).

    Returns
    -------
    cmap
        type depends on return_type parameter
    """

    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap, n)
    else:
        cmap = LinearSegmentedColormap.from_list("new_cmap", cmap(np.linspace(0, 1, n)), n)

    if return_type == 'cmap':
        return cmap
    elif return_type == 'list':
        return cmap(np.linspace(0, 1, n))


def cmap_from_borders(colors=['white', 'black'], n=256, return_type='cmap'):
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

    Returns
    -------
    cmap or list
        Behaviour depends on return_type parameter
    """
    colors = to_rgba_array(colors)[:, :-1]
    cmap = np.vstack([np.linspace(colors[0][i], colors[1][i], num=n) for i in range(3)]).T

    if return_type == 'list':
        return cmap
    if return_type == 'cmap':
        return LinearSegmentedColormap.from_list("new_cmap", cmap, N=n)


def cmap_random(n, color_type='pastel', first_color=None, last_color=None, return_type="cmap"):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks

    Parameters
    ----------
    n : int
        Number of labels (size of colormap)
    color_type : {"bright","pastel"}
        'bright' for strong colors, 'soft' for pastel colors, which is the default behaviour
    first_color : str, optional
        Option to set first color if necessary
    last_color : str, optional
        Option to set last color if necessary
    return_type : str, optional
        'cmap' returns a cmap object, 'list' returns an ndarray.
    
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

    # Generate color map for bright colors, based on hsv
    if color_type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(n)]

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

    # Generate soft pastel colors, by limiting the RGB spectrum
    if color_type == 'pastel':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(n)]

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

    random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=n)

    if return_type == "cmap":
        return random_colormap
    if return_type == "list":
        return randRGBcolors


def legend_patches(colors, labels, type='patch', **kwargs):
    if len(colors) != len(labels):
        raise IndexError("Length of labels and colors don't match")

    if type == 'patch':
        return [Patch(facecolor=color, label=label, **kwargs)
                for color, label in zip(colors, labels)]
    else:
        return [Line2D([0], [0], color=color, label=label, linestyle=type, **kwargs)
                for color, label in zip(colors, labels)]


def create_colorbar_axes(ax, aspect=30, pad_fraction=0.6, position="right"):
    """
    Create an axes for the colorbar to be drawn on that has the same size as the figure

    Parameters
    ----------
    ax : Axes, optional
        The Axes that the colorbar will added to.
    aspect : float, optional
        The aspect ratio of the colorbar
    pad_fraction : float, optional
        The fraction of the height of the colorbar that the colorbar is removed from the image
    position : {"left", "right", "top", "bottom"}
        The position of the colorbar in respect to the image

    Returns
    -------
    Axes
    """
    divider = axes_grid1.make_axes_locatable(ax)
    width = axes_grid1.axes_size.AxesY(ax, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    ax = divider.append_axes(position=position, size=width, pad=pad, axes_class=plt.Axes)
    return ax

def add_colorbar(im=None, ax=None, cax=None, aspect=30, pad_fraction=0.6, position="right", shrink=1, **kwargs):
    """
    Add colorbar to a plot

    Parameters
    ----------
    im : ScalarMappable, optional
        The source ScalarMappable. If not provided it will use the last image on the Axes
    ax : Axes, optional
        The Axes that the colorbar will added to. If not provided it will be assumed to be the last Axes that was
        used (plt.gca())
    cax : Axes, optional
        The Axes instance that the colorbar will be drawn on. If not given it will be added internally.
    aspect : float, optional
        The aspect ratio of the colorbar
    pad_fraction : float, optional
        The fraction of the height of the colorbar that the colorbar is removed from the image
    position : {"left", "right", "top", "bottom"}
        The position of the colorbar in respect to the image
    shrink : float, optional
        float between 0 and 1, which is the fraction of the space it will cover. Does not work if `cax` is provided.
    **kwargs : dict, optional
        Keyword arguments for the colorbar call

    Returns
    -------
    Colorbar
    """
    if isinstance(im, type(None)):
        if isinstance(ax, type(None)):
            ax = plt.gca()
        im = ax.images[-1]
    else:
        if isinstance(ax, type(None)):
            if isinstance(im, ScalarMappable):
                ax = plt.gca()
            else:
                ax = im.axes

    orientation = "vertical" if position in ("right", "left") else "horizontal"
    if isinstance(cax, type(None)):
        cax = create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction, position=position)

        if shrink < 1:
            length = 1 / (aspect / shrink)
            pad = pad_fraction * length
            create_colorbar_axes(ax=ax, aspect=aspect/2, pad_fraction=pad_fraction, position=position).axis("off")

            if position == "left":
                bbox = [-pad - length, (1 - shrink)/2, length, shrink]
            elif position == "right":
                bbox = [1 + pad, (1 - shrink)/2, length, shrink]
            elif position == "bottom":
                bbox = [(1 - shrink)/2, -pad - length, shrink, length]
            elif position == "top":
                bbox = [(1 - shrink) / 2, 1 + pad, shrink, length]

            ip = ip = InsetPosition(ax, bbox)
            cax.set_axes_locator(ip)

    elif shrink < 1:
        raise ValueError("Shrink can only be set if no colorbar axes is provided")

    return ax.figure.colorbar(im, orientation=orientation, cax=cax, **kwargs)


if __name__ == "__main__":
    plot_colors(plt.get_cmap("viridis", 20), ticks=True)
    plt.show()

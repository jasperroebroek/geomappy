#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The functions here implement shortcuts to create different sort of colors/cmap instances for different scenarios. It
also contains a convenient function to add a colorbar that has the right size for the plots.
"""
import colorsys
from typing import Iterable, Union, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.colorbar import ColorbarBase  # type: ignore
from matplotlib.colors import (  # type: ignore
    Colormap, LinearSegmentedColormap, ListedColormap, to_rgba_array, BoundaryNorm, to_rgba
)  # type: ignore
from geomappy.types import Color
from geomappy.utils import _grid_from_corners


def plot_colors(c: Union[Color, Colormap], ticks: bool = False):
    """
    Plot a horizontal colorbar to inspect colors

    Parameters
    ----------
    c : array-like or Colormap instance
        Iterable containing matplotlib interpretable colors, matplotlib cmap, or str indicating the cmap.
    ticks : bool, optional
        Add ticks to the figure
    """
    if isinstance(c, str):
        c = plt.get_cmap(c)
    elif not isinstance(c, Colormap):
        c = ListedColormap(c)

    fig, ax = plt.subplots(figsize=(10, 1))

    bounds = np.linspace(0, 1, c.N + 1)
    norm = BoundaryNorm(bounds, c.N)

    cb = ColorbarBase(ax, cmap=c, norm=norm, spacing='proportional', ticks=None,
                      boundaries=bounds, format='%1i', orientation=u'horizontal')
    ax.patch.set_edgecolor('black')

    if not ticks:
        # Clear everything around the plot
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        offset = 1 / (2 * c.N)
        cb.set_ticks(list(np.linspace(offset, 1 - offset, c.N)), labels=["" for x in range(c.N)])
        ax.tick_params(width=3, length=8)

    plt.tight_layout()
    plt.show()


def cmap_2d(shape=(1000, 1000), v=None, alpha=0, plotting=False, diverging=False, diverging_alpha=0.5, rotate=0,
            flip=False, ax=None):
    """
    Creates a 2 dimensional legend

    # todo; it would be nice to convert this to a Norm and Cmap like combination

    Parameters
    ----------
    shape : list
        Shape of the returned array
    v : list, optional
        If provided it will override the corner colors.
        It needs to exist of a list of four colors (convertible by to_rgba_array).
    alpha : float, optional
        Value between 0 and 1. Default is 0.
    plotting : bool, optional
        Plotting cmap. Default is False
    diverging : bool, optional
        Apply whitening kernel causing the centre of the cmap to be white if v is left to None. Default is False.
    diverging_alpha : float, optional
        The central RGB components are raised with this float. The default is 0.5, which is also the maximum, and will
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

    Notes
    -----
    For reference see
    Teuling et al., 2010: "Bivariate colour maps for visualizing climate data"
    http://iacweb.ethz.ch/doc/publications/2153_ftp.pdf
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
        cmap[:, :, i] = _grid_from_corners(corner_colors[:, i], shape=shape)
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


def colors_discrete(cmap: Union[str, Colormap] = 'hsv', n: int = 256) -> np.ndarray:
    """
    Returns n sampled colors from Colormap

    Parameters
    ----------
    cmap : str or Colormap, optional
        Name of cmap (or cmap itself). If `cmap` is a string it needs to be available in the matplotlib namespace
    n : int
        float of colors
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    return cmap(np.linspace(0, 1, n))


def cmap_discrete(cmap: Union[str, Colormap] = 'hsv', n: int = 256) -> Colormap:
    """
    Returns a resampled colormap

    Parameters
    ----------
    cmap : str or Colormap, optional
        Name of cmap (or cmap itself). If `cmap` is a string it needs to be available in the matplotlib namespace
    n : int
        float of colors
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    return cmap.resampled(n)


def cmap_from_borders(colors: Iterable[Color], n: int = 256):
    """
    Creates a Colormap based on two colors

    Parameters
    ----------
    colors : list
        List of two colors, in any matplotlib accepted format
    n : int, optional
        Can be set if a discrete colormap is needed. The default is 256 which is the standard for a smooth color
        gradient
    """
    return LinearSegmentedColormap.from_list('', colors, n)


def colors_random(n: int,
                  color_type: str = 'pastel',
                  first_color: Optional[Color] = None,
                  last_color: Optional[Color] = None) -> np.ndarray:
    """
    Creates random RGBA colors

    Parameters
    ----------
    n : int
        float of labels (size of colormap)
    color_type : {"bright","pastel"}
        'bright' for strong colors, 'soft' for pastel colors, which is the default behaviour
    first_color : str, optional
        Option to set first color if necessary
    last_color : str, optional
        Option to set last color if necessary

    Source
    ------
    https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    """
    # Generate color map for bright colors, based on hsv
    if color_type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(n)]

        # Convert HSV list to RGB
        rand_rgb_colors = []
        for HSVcolor in randHSVcolors:
            rand_rgb_colors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]) + (1,))

    # Generate soft pastel colors, by limiting the RGB spectrum
    elif color_type == 'pastel':
        low = 0.6
        high = 0.95
        rand_rgb_colors = [(np.random.uniform(low=low, high=high),
                            np.random.uniform(low=low, high=high),
                            np.random.uniform(low=low, high=high),
                            1) for i in range(n)]

    else:
        raise ValueError('Please choose "bright" or "pastel" for type')

    if first_color is not None:
        rand_rgb_colors[0] = to_rgba(first_color)

    if last_color is not None:
        rand_rgb_colors[-1] = to_rgba(last_color)

    return np.asarray(rand_rgb_colors)


def cmap_random(n: int,
                color_type: str = 'pastel',
                first_color: Optional[Color] = None,
                last_color: Optional[Color] = None) -> Colormap:
    """
    Creates a random Colormap object

    Parameters
    ----------
    n : int
        float of labels (size of colormap)
    color_type : {"bright","pastel"}
        'bright' for strong colors, 'soft' for pastel colors, which is the default behaviour
    first_color : str, optional
        Option to set first color if necessary
    last_color : str, optional
        Option to set last color if necessary

    Source
    ------
    https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    """
    colors = colors_random(n, color_type, first_color, last_color)
    return LinearSegmentedColormap.from_list('random_cmap', colors, N=n)

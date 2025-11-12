"""
The functions here implement shortcuts to create different sort of colors/cmap instances for different scenarios. It
also contains a convenient function to add a colorbar that has the right size for the plots.
"""

import colorsys
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    to_rgba,
)
from matplotlib.typing import ColorType


def plot_colors(c: Sequence[ColorType] | Colormap, show_ticks: bool = False) -> None:
    """
    Plot a horizontal colorbar to inspect colors

    Parameters
    ----------
    c : array_like or Colormap instance
        Iterable containing matplotlib interpretable colors, matplotlib cmap, or str indicating the cmap.
    show_ticks : bool, optional
        Add ticks to the figure
    """
    if isinstance(c, str):
        c = plt.get_cmap(c)
    elif not isinstance(c, Colormap):
        c = ListedColormap(c)

    fig, ax = plt.subplots(figsize=(10, 1))

    bounds = np.linspace(0, 1, c.N + 1)
    norm = BoundaryNorm(bounds, c.N)

    cb = ColorbarBase(
        ax,
        cmap=c,
        norm=norm,
        spacing='proportional',
        ticks=None,
        boundaries=bounds,
        format='%1i',
        orientation='horizontal',
    )
    ax.patch.set_edgecolor('black')

    if not show_ticks:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        offset = 1 / (2 * c.N)
        cb.set_ticks(list(np.linspace(offset, 1 - offset, c.N)), labels=['' for x in range(c.N)])
        ax.tick_params(width=3, length=8)

    plt.tight_layout()
    plt.show()


def colors_discrete(cmap: str | Colormap = 'hsv', n: int = 256) -> np.ndarray:
    """
    Returns n sampled colors from Colormap

    Parameters
    ----------
    cmap : str or Colormap, optional
        Name of cmap (or cmap itself). If `cmap` is a string it needs to be available in the matplotlib namespace
    n : int
        Number of colors

    Returns
    -------
    np.ndarray
        Array of colors
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    return cmap(np.linspace(0, 1, n))


def cmap_discrete(cmap: str | Colormap = 'hsv', n: int = 256) -> Colormap:
    """
    Returns a resampled colormap

    Parameters
    ----------
    cmap : str or Colormap, optional
        Name of cmap (or cmap itself). If `cmap` is a string it needs to be available in the matplotlib namespace
    n : int
        Number of colors

    Returns
    -------
    Colormap
    """
    if isinstance(cmap, str):
        return plt.get_cmap(cmap, n)
    return cmap.resampled(n)


def cmap_from_borders(colors: Sequence[ColorType], n: int = 256):
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


def colors_random(
    n: int,
    color_type: Literal['bright', 'pastel'] = 'pastel',
    first_color: ColorType | None = None,
    last_color: ColorType | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Creates random RGBA colors

    Parameters
    ----------
    n : int
        Number of labels (size of colormap)
    color_type : {"bright","pastel"}
        'bright' for strong colors, 'soft' for pastel colors, which is the default behaviour
    first_color : Matplotlib color, optional
        Option to set first color if necessary
    last_color : Matplotlib color, optional
        Option to set last color if necessary
    seed : int, optional
        Option to set seed if necessary

    Returns
    -------
    :class:`numpy.ndarray`
        Array of colors

    Source
    ------
    https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
    """
    rs = np.random.default_rng(seed)

    # Generate color map for bright colors, based on hsv
    if color_type == 'bright':
        randHSVcolors = [
            (rs.uniform(low=0.0, high=1), rs.uniform(low=0.2, high=1), rs.uniform(low=0.9, high=1)) for i in range(n)
        ]

        # Convert HSV list to RGB
        rand_rgb_colors = []
        for HSVcolor in randHSVcolors:
            rand_rgb_colors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]) + (1,))

    # Generate soft pastel colors, by limiting the RGB spectrum
    elif color_type == 'pastel':
        low = 0.6
        high = 0.95
        rand_rgb_colors = [
            (rs.uniform(low=low, high=high), rs.uniform(low=low, high=high), rs.uniform(low=low, high=high), 1)
            for i in range(n)
        ]

    else:
        raise ValueError('Please choose "bright" or "pastel" for type')

    if first_color is not None:
        rand_rgb_colors[0] = to_rgba(first_color)

    if last_color is not None:
        rand_rgb_colors[-1] = to_rgba(last_color)

    return np.asarray(rand_rgb_colors)


def cmap_random(
    n: int,
    color_type: str = 'pastel',
    first_color: ColorType | None = None,
    last_color: ColorType | None = None,
    seed: int | None = None,
) -> Colormap:
    """
    Creates a random Colormap object

    Parameters
    ----------
    n : int
        Number of labels (size of colormap)
    color_type : {"bright","pastel"}
        'bright' for strong colors, 'soft' for pastel colors, which is the default behaviour
    first_color : Matplotlib color, optional
        Option to set first color if necessary
    last_color : Matplotlib color, optional
        Option to set last color if necessary
    seed : int, optional
        Option to set seed if necessary

    Returns
    -------
    Colormap
    """
    colors = colors_random(n, color_type, first_color, last_color, seed)
    return ListedColormap(colors, 'random_cmap')

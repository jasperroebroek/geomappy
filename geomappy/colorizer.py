import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colorizer import Colorizer
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike

from geomappy.colors import colors_discrete
from geomappy.types import ExtendType
from geomappy.utils import check_exclusive_limits, check_increasing_and_unique


def create_scalar_colorizer(
    bins: ArrayLike | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    extend: ExtendType = 'both',
    cmap: str | Colormap | None = None,
    nan_color: ColorType | None = None,
) -> Colorizer:
    """
    Configure a colormap and normalization for continuous scalar data.

    Parameters
    ----------
    bins : array_like, optional
        Bin edges for discrete colormap. Must be strictly increasing.
        Mutually exclusive with `norm`.
    norm : :class:`~matplotlib.colors.Normalize`, optional
        Custom normalizer. Mutually exclusive with `bins`.
    vmin : float, optional
        Minimum value for the colormap. Mutually exclusive with `bins` and `norm`.
    vmax : float, optional
        Maximum value for the colormap. Mutually exclusive with `bins` and `norm`.
    extend : {'neither', 'min', 'max', 'both'}, optional
        Colorbar extension type. Defaults to 'both'.
    cmap : str or :class:`~matplotlib.colors.Colormap`, optional
        Colormap name or instance.
    nan_color : :class:`~matplotlib.typing.ColorType`, optional
        Color for NaN or masked values.

    Returns
    -------
    colorizer : :class:`~matplotlib.colorizer.Colorizer`
        Configured colorizer object.
    """
    check_exclusive_limits(bins, norm, vmin, vmax)

    if norm is not None:
        lut = None
    elif bins is not None:
        bins = np.asarray(bins).flatten()
        check_increasing_and_unique(bins, 'bins')
        extension_count = (1 if extend in ('min', 'both') else 0) + (1 if extend in ('max', 'both') else 0)
        lut = len(bins) + extension_count - 1
        norm = BoundaryNorm(bins, lut, extend=extend)
    else:
        lut = None
        norm = Normalize(vmin, vmax)

    cmap = plt.get_cmap(cmap, lut)
    if nan_color is not None:
        cmap.set_bad(nan_color)
    cmap.colorbar_extend = extend

    return Colorizer(cmap=cmap, norm=norm)


def create_classified_colorizer(
    levels: ArrayLike,
    colors: ArrayLike | None = None,
    cmap: str | Colormap | None = 'Set1',
    nan_color: ColorType | None = None,
) -> Colorizer:
    """
    Configure a colormap and normalization for classified (categorical) data.

    Creates a discrete colormap with one color per class level. Boundaries
    are placed midway between consecutive levels.

    Parameters
    ----------
    levels : array_like
        Class levels to display. Must be strictly increasing.
    colors : array_like, optional
        Colors for each level. If None, sampled from `cmap`.
    cmap : str or :class:`~matplotlib.colors.Colormap`, optional
        Colormap to sample colors from if `colors` is None. Defaults to 'Set1'.
    nan_color : :class:`~matplotlib.typing.ColorType`, optional
        Color for NaN or masked values.

    Returns
    -------
    colorizer : :class:`~matplotlib.colorizer.Colorizer`
        Configured colorizer object.

    Notes
    -----
    If more than 9 levels are provided, a warning is issued as plot visibility may be reduced.
    """
    levels = np.asarray(levels).flatten()
    check_increasing_and_unique(levels, 'levels')

    if len(levels) > 9:
        warnings.warn(
            f'Using {len(levels)} levels may reduce plot visibility. Consider using 9 or fewer levels.',
            UserWarning,
            stacklevel=2,
        )

    cmap = plt.get_cmap(cmap)
    colors = colors or colors_discrete(cmap=cmap, n=len(levels))

    if len(levels) == 1:
        boundaries = np.array([levels[0] - 0.5, levels[0] + 0.5])
    else:
        midpoints = (levels[1:] + levels[:-1]) / 2
        boundaries = np.concatenate(
            [
                [levels[0] - (midpoints[0] - levels[0])],
                midpoints,
                [levels[-1] + (levels[-1] - midpoints[-1])],
            ],
        )

    norm = BoundaryNorm(boundaries, len(levels))
    cmap_result = ListedColormap(colors)

    if nan_color is not None:
        cmap_result.set_bad(nan_color)

    return Colorizer(cmap=cmap_result, norm=norm)

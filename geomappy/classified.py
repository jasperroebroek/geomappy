from typing import Optional, Tuple, Iterable

import numpy as np
from matplotlib import pyplot as plt  # type: ignore
from matplotlib.colors import Colormap, BoundaryNorm, ListedColormap  # type: ignore

from geomappy.colors import colors_discrete
from geomappy.types import Color
from geomappy.utils import check_increasing_and_unique


def parse_classified_plot_params(m: np.ma.MaskedArray, *, levels: Optional[Iterable] = None,
                                 colors: Optional[Iterable] = None,
                                 cmap: Colormap = "Set1", nan_color: Optional[Color] = None,
                                 suppress_warnings: bool = False) -> Tuple[ListedColormap, BoundaryNorm, np.ndarray]:
    unique_values = np.unique(m)

    if levels is None:
        levels = unique_values
    levels = np.asarray(levels)

    check_increasing_and_unique(levels)

    if levels.size > 9 and not suppress_warnings:
        raise ValueError("float of levels above 9, this creates issues with visibility. This error can be suppressed"
                         "by setting suppress_warnings to True. Be aware that the default colormap will cause "
                         "issues")

    if cmap is None:
        cmap = plt.get_cmap()
    else:
        cmap = plt.get_cmap(cmap)

    if colors is None:
        colors = colors_discrete(cmap=cmap, n=levels.size)

    boundaries = np.hstack((levels[0] - 1, (levels[1:] + levels[:-1]) / 2, levels[-1] + 1))
    norm = BoundaryNorm(boundaries, len(levels))
    cmap = ListedColormap(colors)

    if nan_color is not None:
        cmap.set_bad(nan_color)

    return cmap, norm, levels

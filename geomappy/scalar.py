from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, Normalize, BoundaryNorm
from numpy.ma.core import MaskedConstant

from geomappy.types import Color, Number
from geomappy.utils import check_increasing_and_unique


def _define_extend(vmin: float, minimum: float, vmax: float, maximum: float) -> str:
    if minimum < vmin and maximum > vmax:
        extend = 'both'
    elif minimum < vmin and not maximum > vmax:
        extend = 'min'
    elif not minimum < vmin and maximum > vmax:
        extend = 'max'
    elif not minimum < vmin and not maximum > vmax:
        extend = 'neither'
    else:
        raise ValueError

    return extend


def parse_scalar_plot_params(m: np.ma.MaskedArray, *,
                             bins: Optional[Tuple[Number]] = None,
                             vmin: Optional[float] = None,
                             vmax: Optional[float] = None,
                             cmap: Optional[Colormap] = None,
                             norm: Optional[Normalize] = None,
                             nan_color: Optional[Color] = None) -> Tuple[Colormap, Normalize]:
    if bins is not None and norm is not None:
        raise ValueError("Bins and norm provided")

    if norm is not None:
        if norm.vmin is None:
            norm.vmin = vmin
        if norm.vmax is None:
            norm.vmax = vmax

    minimum = m.min()
    maximum = m.max()

    if MaskedConstant in (minimum, maximum):
        raise ValueError("Data only contains NaNs")

    if bins is not None:
        bins = np.asarray(bins)
        check_increasing_and_unique(bins)
        vmin = bins.min()
        vmax = bins.max()
    elif norm is not None:
        vmin = norm.vmin
        vmax = norm.vmax

    if vmin is None:
        vmin = minimum
    if vmax is None:
        vmax = maximum

    extend = _define_extend(vmin, minimum, vmax, maximum)

    color_offset = -1
    if extend in ('min', 'both'):
        color_offset += 1
    if extend in ('max', 'both'):
        color_offset += 1

    if bins is None:
        lut = None
        norm = Normalize(vmin, vmax)
    else:
        lut = bins.size + color_offset
        norm = BoundaryNorm(bins, lut, extend=extend)

    cmap = plt.get_cmap(cmap, lut)
    if nan_color is not None:
        cmap.set_bad(nan_color)

    return cmap, norm

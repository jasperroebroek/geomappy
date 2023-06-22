from typing import Optional, Tuple, Union

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import BoundaryNorm, Normalize, Colormap

from geomappy.axes_decoration import legend_patches, add_colorbar
from geomappy.types import Number


def no_legend(*args, **kwargs) -> None:
    pass


def add_legend_patches_scalar(ax: plt.Axes, norm: Normalize, cmap: Colormap, **kwargs) -> matplotlib.legend.Legend:
    if not isinstance(norm, BoundaryNorm):
        raise TypeError("Can only plot legend patches when norm is BoundaryNorm or bins are provided")

    im = ScalarMappable(norm, cmap)
    bins = norm.boundaries

    xs = (bins[1:] + bins[:-1]) / 2
    labels = [f"{bins[i - 1]} - {bins[i]}" for i in range(1, len(bins))]

    if norm.extend in ('min', 'both'):
        xs = np.hstack((bins.min() - 1, xs))
        labels = [f"< {bins[0]}"] + labels
    if norm.extend in ('max', 'both'):
        xs = np.hstack((xs, bins.max() + 1))
        labels += [f"> {bins[-1]}"]

    handles = legend_patches(cmap(norm(xs)), labels)
    return ax.legend(handles=handles, **kwargs)


def add_colorbar_scalar(ax: plt.Axes, norm: Normalize, cmap: Colormap, **kwargs) -> Colorbar:
    im = ScalarMappable(norm, cmap)
    return add_colorbar(ax=ax, im=im, **kwargs)


def add_legend_patches_classified(ax: plt.Axes, labels: Optional[Tuple[Union[Number, str]]], norm: BoundaryNorm,
                                  cmap: Colormap, **kwargs) -> matplotlib.legend.Legend:
    if not isinstance(norm, BoundaryNorm):
        TypeError("Can only plot classified patches with BoundaryNorm")

    bins = norm.boundaries[:-1]
    if labels is None:
        labels = bins
    colors = cmap(norm(bins))
    handles = legend_patches(colors, labels)
    return ax.legend(handles=handles, **kwargs)


def add_colorbar_classified(ax: plt.Axes, labels: Optional[Tuple[Union[Number, str]]], norm: BoundaryNorm,
                            cmap: Colormap, **kwargs) -> Colorbar:
    if not isinstance(norm, BoundaryNorm):
        TypeError("Can only plot classified colorbar with BoundaryNorm")

    im = ScalarMappable(norm, cmap)

    l = add_colorbar(ax=ax, im=im, extend='neither', **kwargs)
    xs = (norm.boundaries[:-1] + norm.boundaries[1:]) / 2
    if labels is None:
        labels = norm.boundaries[:-1]
    l.set_ticks(xs, labels=labels)
    return l


SCALAR_LEGENDS = {None: no_legend,
                  "legend": add_legend_patches_scalar,
                  "colorbar": add_colorbar_scalar}

CLASSIFIED_LEGENDS = {None: no_legend,
                      "legend": add_legend_patches_classified,
                      "colorbar": add_colorbar_classified}


def add_legend(t: str, legend: Optional[str], *, ax: plt.Axes, **kwargs):
    if t == 'scalar':
        d = SCALAR_LEGENDS
    elif t == 'classified':
        d = CLASSIFIED_LEGENDS
    else:
        raise ValueError

    return d.get(legend)(ax=ax, **kwargs)

import warnings

import matplotlib.axes
import matplotlib.legend
import numpy as np
from matplotlib.colorbar import Colorbar
from matplotlib.colorizer import ColorizingArtist
from matplotlib.colors import BoundaryNorm
from numpy.typing import ArrayLike

from geomappy.plot_utils import create_colorbar_axes, legend_patches
from geomappy.types import LegendCreator


def add_legend_patches_scalar(
    ax: matplotlib.axes.Axes,
    ca: ColorizingArtist,
    legend_ax: matplotlib.axes.Axes | None,
    labels: ArrayLike | None,
) -> matplotlib.legend.Legend | None:
    norm = ca.colorizer.norm

    if not isinstance(norm, BoundaryNorm):
        warnings.warn('Legend patches are only supported for BoundaryNorm (discrete bins).')
        return None

    bins = norm.boundaries

    xs = (bins[1:] + bins[:-1]) / 2
    if norm.extend in ('min', 'both'):
        xs = np.hstack((bins.min() - 1, xs))
    if norm.extend in ('max', 'both'):
        xs = np.hstack((xs, bins.max() + 1))

    if labels is None:
        labels = [f'{bins[i - 1]} - {bins[i]}' for i in range(1, len(bins))]
        if norm.extend in ('min', 'both'):
            labels = [f'< {bins[0]}'] + labels
        if norm.extend in ('max', 'both'):
            labels += [f'> {bins[-1]}']

    handles = legend_patches(ca.colorizer.to_rgba(xs), labels)
    legend_ax = legend_ax or ax

    return legend_ax.legend(handles=handles)


def add_colorbar_scalar(
    ax: matplotlib.axes.Axes,
    ca: ColorizingArtist,
    legend_ax: matplotlib.axes.Axes | None,
    labels: ArrayLike | None,
) -> Colorbar | None:
    legend_ax = legend_ax or create_colorbar_axes(ax)
    return ax.figure.colorbar(ca, cax=legend_ax)


def add_legend_patches_classified(
    ax: matplotlib.axes.Axes,
    ca: ColorizingArtist,
    legend_ax: matplotlib.axes.Axes | None,
    labels: ArrayLike | None,
) -> matplotlib.legend.Legend | None:
    norm = ca.colorizer.norm

    if not isinstance(norm, BoundaryNorm):
        raise
        warnings.warn('Legend patches are only supported for BoundaryNorm (discrete bins).')
        return None

    bins = norm.boundaries[:-1]
    labels = labels if labels is not None else bins

    colors = ca.colorizer.to_rgba(bins)
    handles = legend_patches(colors, labels)
    legend_ax = legend_ax or ax

    return legend_ax.legend(handles=handles)


def add_colorbar_classified(
    ax: matplotlib.axes.Axes,
    ca: ColorizingArtist,
    legend_ax: matplotlib.axes.Axes | None,
    labels: ArrayLike | None,
) -> Colorbar | None:
    norm = ca.colorizer.norm

    if not isinstance(norm, BoundaryNorm):
        warnings.warn('Legend patches are only supported for BoundaryNorm (discrete bins).')
        return None

    legend_ax = legend_ax or create_colorbar_axes(ax)
    cbar = ax.figure.colorbar(ca, cax=legend_ax, extend='neither')

    xs = (norm.boundaries[:-1] + norm.boundaries[1:]) / 2
    if labels is None:
        labels = norm.boundaries[:-1]
    cbar.set_ticks(xs, labels=labels)

    return cbar


def no_legend_creator(
    ax: matplotlib.axes.Axes,
    ca: ColorizingArtist,
    legend_ax: matplotlib.axes.Axes | None,
    labels: ArrayLike | None,
) -> None:
    return None


def get_legend_creator(
    kind: str,
    legend_type: str | LegendCreator | None,
) -> LegendCreator:
    if callable(legend_type):
        return legend_type
    if legend_type is None:
        return no_legend_creator

    match (kind, legend_type):
        case ('scalar', 'legend'):
            return add_legend_patches_scalar
        case ('scalar', 'colorbar'):
            return add_colorbar_scalar
        case ('classified', 'legend'):
            return add_legend_patches_classified
        case ('classified', 'colorbar'):
            return add_colorbar_classified
        case _:
            raise ValueError(f'Invalid legend combination: {kind!r}, {legend_type!r}')

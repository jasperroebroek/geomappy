import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.image import AxesImage
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike

from geomappy.colorizer import create_classified_colorizer, create_scalar_colorizer
from geomappy.legends import get_legend_creator
from geomappy.types import ExtendType, LegendCreator, LegendType
from geomappy.utils import check_increasing_and_unique, determine_extend, get_data_range, parse_levels


def plot_classified_raster(
    m: ArrayLike,
    *,
    levels: ArrayLike | None = None,
    colors: ArrayLike | None = None,
    labels: ArrayLike | None = None,
    cmap: str | Colormap = 'Set1',
    nan_color: ColorType | None = None,
    ax: matplotlib.axes.Axes | None = None,
    legend: str | LegendCreator | None = 'colorbar',
    legend_ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> tuple[AxesImage, LegendType | None]:
    """
    Plot a 2D raster with discrete classes.

    Parameters
    ----------
    m : array_like
        Input 2D array representing the raster data.
    levels : array_like, optional
        Bins for classification (as used by :func:`numpy.digitize`). Defaults to unique values in `m`.
    labels : array_like, optional
        Labels for the classes. Defaults to `levels`.
    colors : array_like, optional
        Colors for each class. Defaults are generated from `cmap`.
    cmap : str or :class:`matplotlib.colors.Colormap`, optional
        Colormap to use if `colors` is not provided. Default is `'Set1'`.
    nan_color : ColorType, optional
        Color for NaN values. Default is white.
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to draw the plot. If not provided, a new figure and axes are created.
    legend : {'colorbar', 'legend', None}, optional
        Type of legend to draw. Default is `'colorbar'`.
    legend_ax : :class:`matplotlib.axes.Axes`, optional
        Axes for the legend. If not provided, it is created automatically.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    im : :class:`matplotlib.image.AxesImage`
        The image object.
    legend : :class:`matplotlib.legend.Legend`, :class:`matplotlib.colorbar.Colorbar`, or None
        The legend object.

    Notes
    -----
    When plotting on a :class:`cartopy.mpl.geoaxes.GeoAxes`, the `extent` and `transform` should be provided in
    `kwargs` if the raster does not perfectly align with the projection.
    """
    m = np.atleast_2d(m)
    if m.ndim != 2:
        raise ValueError('Input data needs to be 2D if plotting classified data')

    levels = parse_levels(m, levels)
    labels = labels or levels

    colorizer = create_classified_colorizer(
        levels=levels,
        colors=colors,
        cmap=cmap,
        nan_color=nan_color,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.ma.masked_invalid(m), colorizer=colorizer, **kwargs)

    legend_creator = get_legend_creator('classified', legend)
    l = legend_creator(ax=ax, ca=im, legend_ax=legend_ax, labels=labels)

    return im, l


def plot_raster(
    m: ArrayLike,
    *,
    ax: matplotlib.axes.Axes | None = None,
    bins: ArrayLike | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    extend: ExtendType | None = None,
    cmap: str | Colormap | None = None,
    nan_color: ColorType | None = None,
    legend: str | LegendCreator | None = 'colorbar',
    legend_ax: matplotlib.axes.Axes | None = None,
    **kwargs,
) -> tuple[AxesImage, LegendType | None]:
    """
    Plot a 2D scalar or 3D RGB(A) raster.

    Parameters
    ----------
    m : array_like
        Input array. 2D for scalar values, 3D for RGB(A) data in the last dimension.
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to draw the plot. If not provided, a new figure and axes are created.
    bins : array_like, optional
        Bins for discretizing continuous data. Mutually exclusive with `norm` and `vmin/vmax`.
    norm : :class:`matplotlib.colors.Normalize`, optional
        Normalizer for continuous data. Ignored if `bins` are provided.
    vmin, vmax : float, optional
        Min and max values for the colormap. Ignored if `bins` or `norm` are provided.
    extend : {'neither', 'min', 'max', 'both'}, optional
        Colorbar extensions. Determined automatically if None.
    cmap : str or :class:`matplotlib.colors.Colormap`, optional
        Colormap to use for scalar data.
    nan_color : ColorType, optional
        Color for NaN values. Default is white.
    legend : {'colorbar', 'legend', None}, optional
        Type of legend to draw. Default is `'colorbar'`.
    legend_ax : :class:`matplotlib.axes.Axes`, optional
        Axes for the legend. Created automatically if not provided.
    **kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    im : :class:`matplotlib.image.AxesImage`
        The image object.
    legend : :class:`matplotlib.legend.Legend`, :class:`matplotlib.colorbar.Colorbar`, or None
        The legend object.

    Notes
    -----
    When plotting on a :class:`cartopy.mpl.geoaxes.GeoAxes`, provide `extent` and `transform` in `kwargs` if the raster
    does not perfectly align with the projection. For boolean or 2-class data, the function internally calls
    :func:`plot_classified_raster`.
    """
    m = np.atleast_2d(m)
    if m.ndim not in (2, 3):
        raise ValueError('Input data needs to be 2D or present RGB(A) values on the third axis.')
    if m.ndim == 3 and (m.shape[-1] not in (3, 4) or np.issubdtype(m.dtype, np.bool_)):
        raise ValueError(
            f'3D arrays are only acceptable if presenting RGB(A) information. It does not work with '
            f'boolean variables.\nShape: {m.shape} \ndtype: {m.dtype}',
        )

    if bins is not None:
        bins = np.asarray(bins).flatten()
        check_increasing_and_unique(bins, 'bins')
        if len(bins) == 1:
            m = m > bins[0]

    if np.issubdtype(m.dtype, np.bool_):
        return plot_classified_raster(
            m,
            colors=('Lightgrey', 'Red'),
            legend=legend,
            labels=('False', 'True'),
            ax=ax,
            nan_color=nan_color,
            **kwargs,
        )

    extend = extend or determine_extend(get_data_range(m), vmin, vmax, norm, bins)

    colorizer = (
        None
        if m.ndim == 3
        else create_scalar_colorizer(
            bins=bins,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            extend=extend,
            cmap=cmap,
            nan_color=nan_color,
        )
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.ma.masked_invalid(m), colorizer=colorizer, **kwargs)

    if m.ndim == 3:
        return im, None

    legend_creator = get_legend_creator('scalar', legend)
    l = legend_creator(ax=ax, ca=im, legend_ax=legend_ax, labels=None)

    return im, l

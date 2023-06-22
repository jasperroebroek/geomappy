from typing import Tuple, Optional, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, Normalize

from geomappy.axes_decoration import prepare_axes
from geomappy.classified import parse_classified_plot_params
from geomappy.legends import add_legend
from geomappy.scalar import parse_scalar_plot_params
from geomappy.types import Color, Number, OptionalLegend


def plot_classified_raster(m: np.ndarray, *, levels: Optional[Tuple[Number]] = None,
                           colors: Optional[Tuple[Color]] = None, labels: Optional[Tuple[Union[Number, str]]] = None,
                           cmap: Colormap = "Set1", nan_color: Optional[Color] = None, suppress_warnings: bool = False,
                           ax: Optional[plt.Axes] = None, figsize: Optional[Tuple[int, int]] = None,
                           legend: Optional[str] = "colorbar", legend_kw: Optional[Dict] = None,
                           **kwargs) -> Tuple[plt.Axes, OptionalLegend]:
    """"Plot a classified raster

    Parameters
    ----------

    m : array-like
        Input array. Needs to be 2D
    levels : array-like, optional
        List of bins as used in np.digitize . By default this parameter is not necessary, the unique values are
        taken from the input data
    labels: array-like, optional
        list of labels for the different classes. By default the unique values are taken as labels
    colors: tuple of colors, optional
        List of colors in a format understandable by matplotlib. By default colors will be obtained from `cmap`
    cmap : matplotlib.Colormap or str, optional
        Can be used to set a colormap when no colors are provided. `Set1` is the default.
    suppress_warnings: bool, optional
        By default 9 classes is the maximum that can be plotted. If set to True this maximum is removed.
    ax : `matplotlib.Axes`, optional
        Axes object. If not provided it will be created on the fly.
    figsize : tuple, optional
        Matplotlib figsize parameter.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'White'
    legend: Legend, optional
        Legend type
    legend_kw: dict, optional
        Arguments for either plt.colorbar or plt.legend
    **kwargs
        Keyword arguments for plt.imshow()

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the `extent` of the data should be provided
    if there is not a perfect overlap. If provided to this parameter it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different than the data projection.

    Returns
    -------

    """
    if m.ndim != 2:
        raise ValueError("Input data needs to be 2D if plotting classified data")

    if not isinstance(m, np.ma.MaskedArray):
        m = np.ma.fix_invalid(m)

    cmap, norm = parse_classified_plot_params(m, levels=levels, colors=colors, cmap=cmap, nan_color=nan_color,
                                              suppress_warnings=suppress_warnings)

    ax = prepare_axes(ax, figsize)
    ax.imshow(m, cmap=cmap, norm=norm, **kwargs)

    if legend_kw is None:
        legend_kw = {}

    if labels is None and levels is None:
        labels = np.unique(m)
    elif labels is None:
        labels = levels

    l = add_legend('classified', legend, ax=ax, labels=labels, norm=norm, cmap=cmap, **legend_kw)

    return ax, l


def plot_raster(m: np.ndarray, *, bins: Optional[Tuple[Number]] = None, cmap: Optional[Colormap] = None,
                norm: Optional[Normalize] = None, ax: Optional[plt.Axes] = None, vmin: Optional[float] = None,
                vmax: Optional[float] = None, figsize: Optional[Tuple[int, int]] = None,
                nan_color: Optional[Color] = None, legend: Optional[str] = "colorbar",
                legend_kw: Optional[Dict] = None,
                **kwargs) -> Tuple[plt.Axes, OptionalLegend]:
    """
    Plot a scalar raster

    Parameters
    ----------
    m : array-like
        Input array. Needs to be either 2D or 3D if the third axis contains RGB(A) information
    bins : array-like, optional
        List of bins that will be used to create a BoundaryNorm instance to discretise the plotting. This does not work
        in conjunction with vmin and vmax. Bins in that case will take the upper hand.
    cmap : matplotlib.Colormap or str, optional
        Matplotlib cmap instance or string the will be recognized by matplotlib
    vmin, vmax : float, optional
        vmin and vmax parameters for plt.imshow(). This does have no effect when bins are provided or a provided norm
        already has these values set.
    norm : matplotlib.Normalize, optional
        Optional normalizer. Should not be provided together with bins.
    ax : `matplotlib.Axes`, optional
        Axes object. If not provided it will be created on the fly.
    figsize : tuple, optional
        Matplotlib figsize parameter.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'White'
    legend: Legend, optional
        Legend type
    legend_kw: dict, optional
        Arguments for either plt.colorbar or plt.legend
    **kwargs
        Keyword arguments for plt.imshow()

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the `extent` of the data should be provided
    if there is not a perfect overlap. If provided to this parameter it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different than the data projection.

    Returns
    -------

    """
    if m.ndim not in (2, 3):
        raise ValueError("Input data needs to be 2D or present RGB(A) values on the third axis.")
    if m.ndim == 3 and (m.shape[-1] not in (3, 4) or np.issubdtype(m.dtype, np.bool_)):
        raise ValueError(f"3D arrays are only acceptable if presenting RGB(A) information. It does not work with "
                         f"boolean variables.\nShape: {m.shape} \ndtype: {m.dtype}")

    if not isinstance(m, np.ma.MaskedArray):
        m = np.ma.fix_invalid(m)

    if bins is not None and len(bins) == 1:
        m = m > bins[0]

    if np.issubdtype(m.dtype, np.bool_):
        return plot_classified_raster(m, colors=("Lightgrey", "Red"), legend=legend, labels=["False", "True"], ax=ax,
                                      figsize=figsize, legend_kw=legend_kw, nan_color=nan_color, **kwargs)

    if m.ndim == 3:
        cmap = None
        norm = None
    else:
        cmap, norm = parse_scalar_plot_params(m, cmap=cmap, bins=bins, vmin=vmin, vmax=vmax, norm=norm,
                                              nan_color=nan_color)

    ax = prepare_axes(ax, figsize)
    ax.imshow(m, cmap=cmap, norm=norm, **kwargs)

    if legend_kw is None:
        legend_kw = {}
    l = add_legend('scalar', legend, ax=ax, norm=norm, cmap=cmap, **legend_kw)

    return ax, l

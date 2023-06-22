from typing import Tuple, Optional, Union

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from geomappy.types import Number


def prepare_axes(ax: Optional[plt.Axes] = None, figsize: Tuple[int, int] = (10, 10)) -> plt.Axes:
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)

    return ax


def create_colorbar_axes(ax: plt.Axes, aspect: Number = 30, pad_fraction: float = 0.6, position="right"):
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


def add_colorbar(im: Optional[Union[ScalarMappable, AxesImage]] = None, ax: Optional[plt.Axes] = None,
                 legend_ax: Optional[plt.Axes] = None, aspect: Number = 25, pad_fraction: float = 0.7,
                 position: str = "right", shrink=1, **kwargs):
    """
    Add colorbar to a plot

    Parameters
    ----------
    im : ScalarMappable, optional
        The source ScalarMappable. If not provided it will use the last image on the Axes
    ax : Axes, optional
        The Axes that the colorbar will added to. If not provided it will be assumed to be the last Axes that was
        used (plt.gca())
    legend_ax : Axes, optional
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
    if im is None:
        if ax is None:
            ax = plt.gca()
        im = ax.images[-1]
    else:
        if ax is None:
            if isinstance(im, ScalarMappable):
                ax = plt.gca()
            else:
                ax = im.axes

    orientation = "vertical" if position in ("right", "left") else "horizontal"
    if legend_ax is None:
        legend_ax = create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction, position=position)

        if shrink < 1:
            length = 1 / (aspect / shrink)
            pad = pad_fraction * length
            create_colorbar_axes(ax=ax, aspect=aspect / 2, pad_fraction=0, position=position).axis("off")

            if position == "left":
                bbox = [-pad - length, (1 - shrink) / 2, length, shrink]
            elif position == "right":
                bbox = [1 + pad, (1 - shrink) / 2, length, shrink]
            elif position == "bottom":
                bbox = [(1 - shrink) / 2, -pad - length, shrink, length]
            elif position == "top":
                bbox = [(1 - shrink) / 2, 1 + pad, shrink, length]
            else:
                raise ValueError("Position needs to be one of {left, right, bottom, top}")

            ip = InsetPosition(ax, bbox)
            legend_ax.set_axes_locator(ip)

    elif shrink < 1:
        raise ValueError("Shrink can only be set if no colorbar axes is provided")

    return ax.figure.colorbar(im, orientation=orientation, cax=legend_ax, **kwargs)


def legend_patches(colors, labels, type='patch', edgecolor="lightgrey", **kwargs):
    if len(colors) != len(labels):
        raise IndexError(f"Length of labels and colors don't match:\n"
                         f"{labels}\n{colors}")

    if type == 'patch':
        return [Patch(facecolor=color, label=label, edgecolor=edgecolor, **kwargs)
                for color, label in zip(colors, labels)]
    else:
        return [Line2D([0], [0], color=color, label=label, linestyle=type, markeredgecolor=edgecolor,
                       **kwargs)
                for color, label in zip(colors, labels)]

import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from .colors import legend_patches as lp


def _determine_cmap_boundaries(m, bins, cmap, clip_legend=False):
    """
    Function that creates the BoundaryNorm instance and an adjusted Colormap to segregate the data that will be plotted
    in bins. It is called from `plot_maps` and `plot_shapes`.

    Parameters
    ----------
    m : array_like
        Data
    bins : array_like
        Bins in which the data will be segragated
    cmap : `matplotlib.colors.Colormap` instance
        Colormap used for plotting
    clip_legend : bool, optional
        Remove the values from `bins` that fall outside the range found in `m`

    Returns
    -------
    cmap, norm, legend_patches, extend

    """
    m = np.array(m)

    if 'float' in m.dtype.name:
        data = m[~np.isnan(m)]
    else:
        data = m.flatten()

    bins = np.array(bins)
    bins.sort()

    vmin = data.min()
    vmax = data.max()
    boundaries = bins.copy()

    if clip_legend:
        bins = bins[np.logical_and(bins >= vmin, bins <= vmax)]

    if vmin < bins[0]:
        boundaries = np.hstack([vmin, boundaries])
        extend_min = True
        labels = [f"< {bins[0]}"]
    else:
        extend_min = False
        labels = [f"{bins[0]} - {bins[1]}"]

    labels = labels + [f"{bins[i - 1]} - {bins[i]}" for i in range(2, len(bins))]

    if vmax > bins[-1]:
        boundaries = np.hstack([boundaries, vmax])
        extend_max = True
        labels = labels + [f"> {bins[-1]}"]
    else:
        extend_max = False

    if extend_min and extend_max:
        extend = "both"
    elif not extend_min and not extend_max:
        extend = "neither"
    elif not extend_min and extend_max:
        extend = "max"
    elif extend_min and not extend_max:
        extend = "min"

    colors = cmap(np.linspace(0, 1, boundaries.size - 1))
    cmap = ListedColormap(colors)

    legend_patches = lp(colors=colors, labels=labels, edgecolor='lightgrey')

    end = -1 if extend_max else None
    cmap_cbar = ListedColormap(colors[int(extend_min):end, :])
    cmap_cbar.set_under(cmap(0))
    cmap_cbar.set_over(cmap(cmap.N))
    norm = BoundaryNorm(bins, len(bins) - 1)

    return cmap_cbar, norm, legend_patches, extend

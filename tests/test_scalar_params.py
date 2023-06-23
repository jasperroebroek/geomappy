import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm

from geomappy.scalar import parse_scalar_plot_params


def test_scalar_params():
    a = np.random.rand(10, 10)
    cmap, norm = parse_scalar_plot_params(np.ma.fix_invalid(a))
    assert cmap.name == 'viridis'
    assert isinstance(norm, Normalize)
    assert norm.vmin == a.min()
    assert norm.vmax == a.max()
    assert cmap == plt.get_cmap("viridis")


def test_scalar_params_vmin_vmax():
    a = np.ma.fix_invalid(np.random.rand(10, 10))
    cmap, norm = parse_scalar_plot_params(a, vmin=a.min() - 1, vmax=a.max() + 1)
    assert norm.vmin != a.min()
    assert norm.vmax != a.max()
    assert cmap == plt.get_cmap("viridis")


def test_scalar_params_bins():
    a = np.ma.fix_invalid(np.random.rand(10, 10))
    bins = (0.2, 0.4)
    cmap, norm = parse_scalar_plot_params(a, bins=bins)
    assert isinstance(norm, BoundaryNorm)
    assert np.array_equal(norm.boundaries, bins)
    assert cmap != plt.get_cmap("viridis")


def test_scalar_params_norm():
    # return norm untouched if provided
    a = np.ma.fix_invalid(np.random.rand(10, 10))
    bins = (0.2, 0.4)
    bnorm = BoundaryNorm(bins, ncolors=3)
    cmap, norm = parse_scalar_plot_params(a, norm=bnorm)
    assert norm == bnorm
    assert cmap == plt.get_cmap("viridis")

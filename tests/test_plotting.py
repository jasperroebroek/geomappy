import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.testing.compare
import numpy as np
import pytest
from matplotlib.colors import ListedColormap, from_levels_and_colors

import geomappy as mp


def imshow(*args, **kwargs):
    f, ax = plt.subplots()
    ax.imshow(*args, **kwargs)
    return ax


def compare_images(ax1, ax2, tol=0):
    try:
        os.mkdir("_test_figures")
    except FileExistsError:
        pass

    ax1.get_figure().savefig("_test_figures/ax_geomappy.png")
    ax2.get_figure().savefig("_test_figures/ax_matplotlib.png")
    r = matplotlib.testing.compare.compare_images("_test_figures/ax_geomappy.png", "_test_figures/ax_matplotlib.png",
                                                  tol=tol)

    shutil.rmtree("_test_figures")

    if r is not None:
        return False

    return True


def test_figure_base():
    # 3D image data
    x = np.random.rand(10, 10, 3)
    ax1, l = mp.plot_raster(x, legend=None)
    ax2 = imshow(x)
    assert compare_images(ax1, ax2)

    # 2D image data
    x = np.random.rand(10, 10)
    ax1, l = mp.plot_raster(x, legend=None)
    ax2 = imshow(x)
    assert compare_images(ax1, ax2)


def test_figure_vmax():
    x = np.random.rand(10, 10)
    ax1, l = mp.plot_raster(x, vmax=0.5, legend=None)
    ax2 = imshow(x, vmax=0.5)
    assert compare_images(ax1, ax2)


def test_figure_vmin():
    x = np.random.rand(10, 10)
    ax1, l = mp.plot_raster(x, vmin=0.5, legend=None)
    ax2 = imshow(x, vmin=0.5)
    assert compare_images(ax1, ax2)


@pytest.mark.parametrize(
    "bins",
    [(0.5, 0.8), (-1, 0, 0.5), (-1, 2), (0.5, 2), np.linspace(-2, 2, 100)],
)
def test_figure_bins(bins):
    x = np.random.rand(10, 10)

    ax1, leg = mp.plot_raster(x, bins=bins, legend=None)

    x_d = np.digitize(x, bins=bins, right=True)
    lut = len(bins) + 1
    if len(bins) not in x_d:
        lut -= 1
    if 0 not in x_d:
        x_d -= 1
        lut -= 1
    ax2 = imshow(x_d, cmap=plt.get_cmap(lut=lut), vmin=0, vmax=lut - 1)

    assert compare_images(ax1, ax2)


def test_figure_binary():
    x = np.random.rand(10, 10)
    ax1, l = mp.plot_raster(x, bins=(0.5,), legend=None)
    ax2 = imshow(x > 0.5, cmap=ListedColormap(["Lightgrey", "Red"]))
    assert compare_images(ax1, ax2)


def test_figure_classified():
    x = np.ones((10, 10))
    x[4, 4] = 2
    x[5, 5] = 5
    ax1, l = mp.plot_classified_raster(x, levels=(1, 2, 5), colors=('Red', "Green", "Blue"), legend=None)
    cmap, norm = from_levels_and_colors((1, 2, 5, 6), ['Red', "Green", "Blue"])
    ax2 = imshow(x, cmap=cmap, norm=norm)
    assert compare_images(ax1, ax2)


@pytest.mark.parametrize(
    "m",
    [2, 3, 10, 30, 300],
)
def test_figure_classified_large(m):
    x = np.random.RandomState(0).randint(0, m, (100, 100))

    ax1, leg = mp.plot_classified_raster(x, levels=np.arange(0, m), cmap="jet", suppress_warnings=True, legend=None)
    # plt.show()

    # get discrete colormap
    cmap = plt.get_cmap('jet', m)
    ax2 = imshow(x, cmap=cmap, vmin=0, vmax=m - 1)
    # plt.show()

    assert compare_images(ax1, ax2, tol=0.8)


def test_figure_legend_patches():
    # 3D image data
    # no legend should be plotted
    x = np.random.rand(10, 10, 3)
    with pytest.raises(TypeError):
        ax1, l = mp.plot_raster(x, legend="legend")

    # 2D image data, without bins
    # no legend should be plotted
    x = np.random.rand(10, 10)
    with pytest.raises(TypeError):
        ax1, l = mp.plot_raster(x, legend="legend")

    # 2D image data, with bins
    # no legend should be plotted, which should raise an error when comparing images
    x = np.random.rand(10, 10)
    ax1, l = mp.plot_raster(x, bins=(0.5,), legend="legend")
    ax2 = imshow(x > 0.5, cmap=ListedColormap(["Lightgrey", "Red"]))
    assert not compare_images(ax1, ax2)


def test_figure_legend_and_colorbar_difference():
    x = np.random.rand(10, 10)
    ax1, l = mp.plot_raster(x, bins=(0.5,), legend="legend")
    ax2, l = mp.plot_raster(x, bins=(0.5,), legend="colorbar")
    assert not compare_images(ax1, ax2)

    ax1, l = mp.plot_raster(x, bins=(0.5, 0.8), legend="legend")
    ax2, l = mp.plot_raster(x, bins=(0.5, 0.8), legend="colorbar")
    assert not compare_images(ax1, ax2)

from matplotlib.colors import ListedColormap, BoundaryNorm, from_levels_and_colors
import geomappy as mp
import matplotlib.pyplot as plt
import matplotlib.testing.compare
import numpy as np
import os
import shutil
import pytest


def imshow(*args, **kwargs):
    f, ax = plt.subplots()
    ax.imshow(*args, **kwargs)
    return ax


def compare_images(ax1, ax2):
    try:
        os.mkdir("_test_figures")
    except FileExistsError:
        pass

    ax1.get_figure().savefig("_test_figures/ax_geomappy.png")
    ax2.get_figure().savefig("_test_figures/ax_matplotlib.png")
    r = matplotlib.testing.compare.compare_images("_test_figures/ax_geomappy.png", "_test_figures/ax_matplotlib.png", tol=0)

    shutil.rmtree("_test_figures")

    if r is not None:
        raise ValueError("Images not equal")


def test_figure_base():
    # 3D image data
    x = np.random.rand(10, 10, 3)
    ax1, cbar = mp.plot_map(x)
    ax2 = imshow(x)
    compare_images(ax1, ax2)

    # 2D image data
    x = np.random.rand(10, 10)
    ax1, cbar = mp.plot_map(x, legend=None)
    ax2 = imshow(x)
    compare_images(ax1, ax2)


def test_figure_vmax():
    x = np.random.rand(10, 10)
    ax1, cbar = mp.plot_map(x, vmax=0.5, legend=None)
    ax2 = imshow(x, vmax=0.5)
    compare_images(ax1, ax2)


def test_figure_vmin():
    x = np.random.rand(10, 10)
    ax1, cbar = mp.plot_map(x, vmin=0.5, legend=None)
    ax2 = imshow(x, vmin=0.5)
    compare_images(ax1, ax2)


def test_figure_binary():
    x = np.random.rand(10, 10)
    ax1, cbar = mp.plot_map(x, bins=[0.5], legend=None)
    ax2 = imshow(x > 0.5, cmap=ListedColormap(["Lightgrey", "Red"]))
    compare_images(ax1, ax2)


def test_figure_classified():
    x = np.ones((10, 10))
    x[4, 4] = 2
    x[5, 5] = 5
    ax1, cbar = mp.plot_classified_map(x, [1, 2, 5], ['Red', "Green", "Blue"], legend=None)
    cmap, norm = from_levels_and_colors([1, 2, 5, 6], ['Red', "Green", "Blue"])
    ax2 = imshow(x, cmap=cmap, norm=norm)
    compare_images(ax1, ax2)


def test_figure_legend_patches():
    # 3D image data
    # no legend should be plotted
    x = np.random.rand(10, 10, 3)
    ax1, cbar = mp.plot_map(x, legend='legend')
    ax2 = imshow(x)
    compare_images(ax1, ax2)

    # 2D image data, without bins
    # no legend should be plotted
    x = np.random.rand(10, 10)
    ax1, cbar = mp.plot_map(x, legend='legend')
    ax2 = imshow(x)
    compare_images(ax1, ax2)

    # 2D image data, with bins
    # no legend should be plotted, which should raise an error when comparing images
    x = np.random.rand(10, 10)
    ax1, cbar = mp.plot_map(x, bins=[0.5], legend='legend')
    ax2 = imshow(x > 0.5, cmap=ListedColormap(["Lightgrey", "Red"]))
    with pytest.raises(ValueError):
        compare_images(ax1, ax2)


def test_figure_colorbar():
    # 2D image data, with bins
    # Legend and colorbar should be different
    x = np.random.rand(10, 10)
    ax1, cbar = mp.plot_map(x, bins=[0.5], legend='legend')
    ax2, cbar = mp.plot_map(x, bins=[0.5], legend='colorbar')
    with pytest.raises(ValueError):
        compare_images(ax1, ax2)

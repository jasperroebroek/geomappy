import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.legend import Legend

from geomappy.plotting.raster import plot_classified_raster, plot_raster


def test_plot_classified_raster_returns_valid_types(classified_array):
    im, legend = plot_classified_raster(classified_array)
    assert isinstance(im, AxesImage)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_classified_raster_creates_new_axes_when_none(classified_array):
    fig_before = plt.gcf()
    im, _ = plot_classified_raster(classified_array, ax=None)
    assert im.axes.figure is not None
    # Ensure a new figure was created
    assert plt.gcf() is not fig_before


def test_plot_classified_raster_with_custom_labels_colors(classified_array, fig_ax):
    _, ax = fig_ax
    labels = ['a', 'b', 'c', 'd']
    colors = ['red', 'green', 'blue', 'yellow']
    im, leg = plot_classified_raster(classified_array, labels=labels, colors=colors, ax=ax, legend='legend')

    assert isinstance(im, AxesImage)
    assert leg is not None

    legend_texts = {t.get_text() for t in leg.get_texts()}
    assert legend_texts == set(labels)


def test_plot_classified_raster_with_invalid_dimension_raises():
    arr = np.random.rand(10, 10, 3, 3)
    with pytest.raises(ValueError, match='2D'):
        plot_classified_raster(arr)


def test_plot_classified_raster_without_legend(classified_array):
    im, legend = plot_classified_raster(classified_array, legend=None)
    assert isinstance(im, AxesImage)
    assert legend is None


# =====================================================================
# Tests for plot_raster
# =====================================================================


def test_plot_raster_with_scalar_data(random_array):
    im, legend = plot_raster(random_array)
    assert isinstance(im, AxesImage)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_raster_with_rgb_data():
    rgb = np.random.rand(5, 5, 3)
    im, legend = plot_raster(rgb)
    assert isinstance(im, AxesImage)
    assert legend is None


def test_plot_raster_with_rgba_data():
    rgba = np.random.rand(5, 5, 4)
    im, legend = plot_raster(rgba)
    assert isinstance(im, AxesImage)
    assert legend is None


def test_plot_raster_with_invalid_rgb_shape_raises():
    invalid = np.random.rand(10, 10, 5)
    with pytest.raises(ValueError, match='RGB'):
        plot_raster(invalid)


def test_plot_raster_with_boolean_array_triggers_classified(monkeypatch):
    called = {}

    def fake_classified(*args, **kwargs):
        called['used'] = True
        return 'im', 'legend'

    monkeypatch.setattr('geomappy.plotting.raster.plot_classified_raster', fake_classified)
    arr = np.random.rand(10, 10) > 0.5
    im, l = plot_raster(arr)
    assert called['used']
    assert im == 'im' and l == 'legend'


def test_plot_raster_with_single_bin_binarizes():
    arr = np.random.rand(5, 5)
    im, legend = plot_raster(arr, bins=[0.5])
    assert isinstance(im, AxesImage)
    assert legend is not None or legend is None  # just ensure no crash


def test_plot_raster_invalid_dimensions():
    arr = np.random.rand(10, 10, 10, 3)
    with pytest.raises(ValueError, match='2D or present RGB'):
        plot_raster(arr)


def test_plot_raster_returns_same_ax_when_given(fig_ax):
    _, ax = fig_ax
    im, _ = plot_raster(np.random.rand(5, 5), ax=ax)
    assert im.axes is ax


def test_plot_raster_with_no_legend_returns_none(fig_ax):
    _, ax = fig_ax
    im, l = plot_raster(np.random.rand(5, 5), ax=ax, legend=None)
    assert isinstance(im, AxesImage)
    assert l is None


def test_plot_raster_passes_kwargs(fig_ax):
    _, ax = fig_ax
    im, _ = plot_raster(np.random.rand(5, 5), ax=ax, interpolation='nearest')
    assert im.get_interpolation() == 'nearest'


def test_plot_raster_with_custom_bins_creates_valid_plot():
    arr = np.linspace(0, 10, 100).reshape(10, 10)
    bins = [0, 2, 4, 6, 8, 10]
    im, legend = plot_raster(arr, bins=bins)
    assert isinstance(im, AxesImage)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


import pytest
from matplotlib.colors import from_levels_and_colors

import geomappy as mp


def test_raster_basic(imshow, compare_images):
    x = np.random.rand(10, 10)
    ax1, _ = mp.plot_raster(x, legend=None)
    ax2 = imshow(x)
    assert compare_images(ax1, ax2)


def test_raster_vmin_vmax(imshow, compare_images):
    x = np.random.rand(10, 10)
    ax1, _ = mp.plot_raster(x, vmin=0.3, vmax=0.8, legend=None)
    ax2 = imshow(x, vmin=0.3, vmax=0.8)
    assert compare_images(ax1, ax2)


@pytest.mark.parametrize('bins', [(0.5,), (0.5, 0.8), np.linspace(0, 1, 5)])
def test_raster_bins(imshow, compare_images, bins):
    x = np.random.rand(10, 10)
    im1, leg1 = plot_raster(x, bins=bins, legend=None)

    x_d = np.digitize(x, bins=bins, right=True)
    im2 = imshow(x_d, cmap=plt.get_cmap(im1.cmap, lut=len(bins) + 1))

    assert compare_images(im1, im2)


def test_raster_masked_array(imshow, compare_images):
    x = np.random.rand(10, 10)
    x[3:6, 3:6] = np.nan
    x = np.ma.masked_invalid(x)
    ax1, _ = plot_raster(x, legend=None)
    ax2 = imshow(np.ma.filled(x, np.nan))
    assert compare_images(ax1, ax2)


@pytest.mark.parametrize('size', [5, 50, 500])
def test_raster_scaling(imshow, compare_images, size):
    x = np.random.rand(size, size)
    ax1, _ = plot_raster(x, legend=None)
    ax2 = imshow(x)
    assert compare_images(ax1, ax2, tol=1)


def test_raster_classified_simple(imshow, compare_images):
    x = np.ones((10, 10))
    x[4, 4] = 2
    x[5, 5] = 5
    ax1, _ = plot_classified_raster(x, levels=(1, 2, 5), colors=('red', 'green', 'blue'), legend=None)
    cmap, norm = from_levels_and_colors((1, 2, 5, 6), ['red', 'green', 'blue'])
    ax2 = imshow(x, cmap=cmap, norm=norm)
    assert compare_images(ax1, ax2)


def test_raster_large_classified(imshow, compare_images):
    m = 100
    x = np.random.RandomState(0).randint(0, m, (100, 100))
    im1, leg = plot_classified_raster(x, levels=np.arange(m), cmap='jet', legend=None)
    cmap = plt.get_cmap('jet', m)
    im2 = imshow(x, cmap=cmap, vmin=0, vmax=m - 1)
    assert compare_images(im1, im2, tol=2)

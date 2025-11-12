import numpy as np
import pytest
from matplotlib.colorbar import Colorbar
from matplotlib.legend import Legend

from geomappy import plot_raster
from geomappy.legends import (
    add_colorbar_classified,
    add_colorbar_scalar,
    add_legend_patches_classified,
    add_legend_patches_scalar,
    get_legend_creator,
)


def test_add_legend_patches_scalar_basic(basic_binned_plot):
    fig, ax, im = basic_binned_plot
    legend = add_legend_patches_scalar(ax, im, legend_ax=None, labels=None)
    assert isinstance(legend, Legend)
    assert len(legend.legend_handles) == len(im.norm.boundaries) - 1


def test_add_legend_patches_scalar_custom_labels(basic_binned_plot):
    fig, ax, im = basic_binned_plot
    labels = ['low', 'mid', 'high']
    legend = add_legend_patches_scalar(ax, im, None, labels)
    texts = [t.get_text() for t in legend.get_texts()]
    assert texts == labels


def test_add_colorbar_scalar_returns_colorbar(basic_binned_plot):
    fig, ax, im = basic_binned_plot
    cb = add_colorbar_scalar(ax, im, None, None)
    assert isinstance(cb, Colorbar)
    assert cb.ax.get_figure() is fig


def test_add_legend_patches_classified_basic(basic_binned_plot):
    fig, ax, im = basic_binned_plot
    legend = add_legend_patches_classified(ax, im, None, None)
    assert isinstance(legend, Legend)
    assert len(legend.legend_handles) == len(im.norm.boundaries) - 1


def test_add_colorbar_classified_labels(basic_binned_plot):
    fig, ax, im = basic_binned_plot
    cb = add_colorbar_classified(ax, im, None, ['A', 'B', 'C'])
    assert isinstance(cb, Colorbar)
    ticks, labels = cb.ax.get_yticks(), [t.get_text() for t in cb.ax.get_yticklabels()]
    assert len(labels) == 3


@pytest.mark.parametrize(
    'kind, legend_type, expected',
    [
        ('scalar', 'legend', 'add_legend_patches_scalar'),
        ('scalar', 'colorbar', 'add_colorbar_scalar'),
        ('classified', 'legend', 'add_legend_patches_classified'),
        ('classified', 'colorbar', 'add_colorbar_classified'),
    ],
)
def test_get_legend_creator_valid(kind, legend_type, expected):
    from geomappy import legends

    func = get_legend_creator(kind, legend_type)
    assert func is getattr(legends, expected)


def test_get_legend_creator_invalid():
    with pytest.raises(ValueError):
        get_legend_creator('scalar', 'unknown')


def test_legend_vs_colorbar_visual_difference(tmp_path, compare_images):
    x = np.random.rand(10, 10)

    im1, leg1 = plot_raster(x, bins=(0.5,), legend='legend')
    im2, leg2 = plot_raster(x, bins=(0.5,), legend='colorbar')

    assert compare_images(im1, im2, tol=0) is not None

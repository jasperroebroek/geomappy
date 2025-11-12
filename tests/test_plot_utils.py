import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from geomappy.plot_utils import (
    add_gridlines,
    add_ticks,
    create_colorbar_axes,
    legend_patches,
)


@pytest.mark.parametrize('location', ['right', 'left', 'top', 'bottom'])
def test_create_colorbar_axes_locations(basic_plot, location):
    fig, ax, im = basic_plot
    cax = create_colorbar_axes(ax, location=location)
    assert isinstance(cax, plt.Axes)
    assert cax.get_figure() is fig
    assert cax not in fig.axes[:1]


@pytest.mark.parametrize('shrink', [1.0, 0.8, 0.5])
def test_create_colorbar_axes_shrink_changes_extent(basic_plot, shrink):
    fig, ax, _ = basic_plot
    cax = create_colorbar_axes(ax, location='right', shrink=shrink)
    bbox = cax.get_position().bounds
    assert isinstance(bbox, tuple)
    if shrink < 1:
        assert bbox[3] < 1.0


def test_legend_patches_creates_patches():
    colors = ['red', 'blue', 'green']
    labels = ['A', 'B', 'C']
    patches = legend_patches(colors, labels)
    assert all(isinstance(p, Patch) for p in patches)
    assert len(patches) == len(labels)
    assert patches[0].get_facecolor() is not None


def test_legend_patches_creates_lines():
    colors = ['red', 'blue']
    labels = ['X', 'Y']
    lines = legend_patches(colors, labels, legend_type='--')
    assert all(isinstance(l, Line2D) for l in lines)
    assert lines[0].get_linestyle() == '--'


def test_legend_patches_mismatched_input_raises():
    with pytest.raises(ValueError):
        legend_patches(['red', 'blue'], ['A'])


@pytest.mark.parametrize('lines', (10, (15, 20), ((0, 60, 120, 180), (-30, 0, 30, 60))))
def test_add_gridlines_creates_fixedlocators(fig_geoaxes, lines):
    fig, ax = fig_geoaxes
    g = add_gridlines(ax, lines, color='red', linestyle=':')
    assert hasattr(g, 'xlocator')
    assert hasattr(g, 'ylocator')
    assert hasattr(g, 'n_steps')
    assert g.n_steps == 300


def test_add_gridlines_returns_valid_gridliner(fig_geoaxes):
    fig, ax = fig_geoaxes
    g = add_gridlines(ax, (30, 15))
    assert hasattr(g, 'xlocator')
    assert hasattr(g, 'ylocator')
    assert g.n_steps == 300


@pytest.mark.parametrize('ticks', (20, (10, 20), ((-180, -90, 0, 90, 180), (-45, 0, 45))))
def test_add_ticks_creates_labels(fig_geoaxes, ticks):
    fig, ax = fig_geoaxes
    g = add_ticks(ax, ticks)
    assert hasattr(g, 'xformatter')
    assert hasattr(g, 'yformatter')
    assert hasattr(g, 'xlabel_style')
    assert hasattr(g, 'ylabel_style')
    assert g.xlabel_style['size'] == 10
    assert not g.top_labels
    assert not g.right_labels


def test_add_ticks_custom_formatter(fig_geoaxes):
    fig, ax = fig_geoaxes
    f = mticker.StrMethodFormatter('{x:.1f}')
    g = add_ticks(ax, (30, 30), formatter=f, fontsize=8)
    assert g.xformatter is f
    assert g.xlabel_style['size'] == 8


def test_add_ticks_draw_labels_kwarg(fig_geoaxes):
    fig, ax = fig_geoaxes
    g = add_ticks(ax, (10, 10), draw_labels=False)
    assert hasattr(g, 'xlocator')
    assert hasattr(g, 'ylocator')

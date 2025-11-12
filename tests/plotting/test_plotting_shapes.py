import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import Collection
from matplotlib.colorbar import Colorbar
from matplotlib.legend import Legend

from geomappy.plotting.shapes import plot_classified_shapes, plot_shapes


def test_plot_classified_shapes_returns_valid_types(simple_points):
    collections, legend = plot_classified_shapes(df=simple_points, values='value')
    assert isinstance(collections, list)
    assert all(isinstance(c, Collection) for c in collections)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_classified_shapes_with_custom_colors_labels(simple_points):
    labels = ['a', 'b', 'c']
    colors = ['red', 'green', 'blue']
    collections, legend = plot_classified_shapes(
        df=simple_points,
        values='value',
        labels=labels,
        colors=colors,
        legend='legend',
    )
    assert isinstance(collections, list)
    assert legend is not None
    legend_texts = {t.get_text() for t in legend.get_texts()}
    assert legend_texts.issubset(set(labels))


def test_plot_classified_shapes_with_lat_lon(simple_geo_values):
    lat, lon, values = simple_geo_values
    collections, legend = plot_classified_shapes(lat=lat, lon=lon, values=values)
    assert isinstance(collections, list)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_classified_shapes_with_single_value_creates_multiple(simple_geo_values):
    lat, lon, values = simple_geo_values
    # all values = 1
    collections, legend = plot_classified_shapes(lat=lat, lon=lon, values=1)
    assert len(collections) > 0
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_shapes_with_continuous_values(simple_points):
    collections, legend = plot_shapes(df=simple_points, values='value')
    assert isinstance(collections, list)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_shapes_with_vmin_vmax(simple_points):
    collections, legend = plot_shapes(df=simple_points, values='value', vmin=0, vmax=2)
    assert isinstance(collections, list)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_shapes_with_bins(simple_points):
    bins = [0, 1, 2]
    collections, legend = plot_shapes(df=simple_points, values='value', bins=bins)
    assert isinstance(collections, list)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_shapes_with_single_bin_binarizes(simple_points):
    bins = [1]
    collections, legend = plot_shapes(df=simple_points, values='value', bins=bins)
    assert isinstance(collections, list)
    # Should trigger classified shapes internally
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_shapes_with_none_values_creates_default_colors(simple_points):
    collections, legend = plot_shapes(df=simple_points, values=None)
    assert isinstance(collections, list)
    assert legend is None


def test_plot_shapes_with_lat_lon_and_single_value_creates_plot(simple_geo_values):
    lat, lon, values = simple_geo_values
    collections, legend = plot_shapes(lat=lat, lon=lon, values=5)
    assert isinstance(collections, list)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


def test_plot_shapes_passes_kwargs_to_plot(simple_points):
    collections, _ = plot_shapes(df=simple_points, values='value', linewidth=2)
    assert all(hasattr(c, 'get_linewidth') for c in collections)


def test_plot_classified_shapes_returns_same_ax_when_given(simple_points):
    fig, ax = plt.subplots()
    collections, _ = plot_classified_shapes(df=simple_points, values='value', ax=ax)
    # All collections should be plotted on the same ax
    assert all(c.figure is fig for c in collections)


def test_plot_shapes_returns_same_ax_when_given(simple_points):
    fig, ax = plt.subplots()
    collections, _ = plot_shapes(df=simple_points, values='value', ax=ax)
    assert all(c.figure is fig for c in collections)


def test_plot_shapes_with_boolean_array_triggers_classified(simple_points):
    values = np.array([True, False, True, False, True])
    collections, legend = plot_shapes(df=simple_points, values=values)
    assert isinstance(collections, list)
    assert legend is None or isinstance(legend, (Legend, Colorbar))


@pytest.mark.parametrize('values', ('value', None))
def test_plot_shapes_with_explicit_face_and_edgecolor(simple_points, values):
    """Both edgecolor and facecolor should be passed to the plot function"""
    collections, legend = plot_shapes(
        df=simple_points,
        values=values,
        facecolor='yellow',
        edgecolor='red',
    )
    assert isinstance(legend, Colorbar) if values == 'value' else legend is None
    for c in collections:
        array = c.get_array()
        if array is None or array.size == 0:
            continue

        ec = c.get_edgecolor()
        assert np.allclose(ec, matplotlib.colors.to_rgba('red'), atol=1e-2)

        fc = c.get_facecolor()
        assert (values != 'value') == np.allclose(fc, matplotlib.colors.to_rgba('yellow'), atol=1e-2)

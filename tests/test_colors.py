"""
pytest tests for color_utils.py
"""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

from geomappy.colors import (
    cmap_discrete,
    cmap_from_borders,
    cmap_random,
    colors_discrete,
    colors_random,
    plot_colors,
)


def test_colors_discrete():
    # Test with string cmap
    colors = colors_discrete('viridis', 10)
    assert isinstance(colors, np.ndarray)
    assert colors.shape == (10, 4)

    # Test with Colormap object
    cmap = plt.get_cmap('plasma')
    colors = colors_discrete(cmap, 5)
    assert isinstance(colors, np.ndarray)
    assert colors.shape == (5, 4)


def test_cmap_discrete():
    # Test with string cmap
    cmap = cmap_discrete('viridis', 10)
    assert isinstance(cmap, Colormap)
    assert cmap.N == 10

    # Test with Colormap object
    cmap_obj = plt.get_cmap('plasma')
    cmap = cmap_discrete(cmap_obj, 5)
    assert isinstance(cmap, Colormap)
    assert cmap.N == 5


def test_cmap_from_borders():
    """Test cmap_from_borders with multiple colors."""
    colors = ['red', 'green', 'blue']
    cmap = cmap_from_borders(colors, 256)
    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.N == 256

    # Test with two colors
    cmap = cmap_from_borders(['red', 'blue'], 10)
    assert cmap.N == 10


def test_colors_random():
    """Test colors_random with pastel and bright, and first/last color."""
    # Test pastel
    colors = colors_random(10, 'pastel', seed=42)
    assert isinstance(colors, np.ndarray)
    assert colors.shape == (10, 4)
    assert np.all(colors[:, 3] == 1)  # alpha should be 1

    # Test bright
    colors = colors_random(5, 'bright', seed=42)
    assert colors.shape == (5, 4)

    # Test with first and last color
    colors = colors_random(3, 'pastel', first_color='red', last_color='blue', seed=42)
    assert colors.shape == (3, 4)
    assert np.allclose(colors[0], plt.cm.colors.to_rgba('red'))
    assert np.allclose(colors[-1], plt.cm.colors.to_rgba('blue'))

    # Test invalid color_type
    with pytest.raises(ValueError):
        colors_random(2, 'invalid')


def test_cmap_random():
    """Test cmap_random with pastel and bright, and first/last color."""
    cmap = cmap_random(10, 'pastel', seed=42)
    assert isinstance(cmap, ListedColormap)
    assert cmap.N == 10

    cmap = cmap_random(5, 'bright', first_color='red', last_color='blue', seed=42)
    assert isinstance(cmap, ListedColormap)
    assert cmap.N == 5
    assert np.allclose(cmap(0), plt.cm.colors.to_rgba('red'))
    assert np.allclose(cmap(cmap.N - 1), plt.cm.colors.to_rgba('blue'))


@patch('matplotlib.pyplot.show')
def test_plot_colors(mock_show):
    """Test plot_colors with list, string, and Colormap input."""
    colors = ['red', 'green', 'blue']
    plot_colors(colors, show_ticks=True)
    mock_show.assert_called()

    plot_colors('viridis')
    mock_show.assert_called()

    plot_colors(plt.get_cmap('plasma'))
    mock_show.assert_called()

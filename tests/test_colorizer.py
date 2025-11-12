import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

from geomappy.colorizer import (
    create_classified_colorizer,
    create_scalar_colorizer,
)
from geomappy.utils import determine_extend, get_data_range, parse_levels


@pytest.mark.parametrize(
    'vmin,data_min,vmax,data_max,expected',
    [
        (0, 0, 1, 1, 'neither'),
        (0, -1, 1, 1, 'min'),
        (0, 0, 1, 2, 'max'),
        (0, -1, 1, 2, 'both'),
    ],
)
def test_determine_extend(vmin, data_min, vmax, data_max, expected):
    result = determine_extend((data_min, data_max), vmin=vmin, vmax=vmax)
    assert result == expected


def test_get_data_range_normal():
    m = np.array([[1, 2, 3], [4, 5, 6]])
    assert get_data_range(m) == (1, 6)


def test_get_data_range_masked():
    m = np.ma.array([1, 2, 3, np.ma.masked])
    result = get_data_range(m)
    assert result == (1, 3)


def test_get_data_range_all_masked():
    m = np.ma.array([np.ma.masked, np.ma.masked])
    with pytest.raises(ValueError, match='Data contains only masked'):
        get_data_range(m)


def test_get_data_range_all_nan():
    m = np.array([np.nan, np.nan])
    with pytest.raises(ValueError, match='NaN'):
        get_data_range(m)


def test_scalar_colorizer_basic_continuous():
    m = np.linspace(0, 1, 10)
    colorizer = create_scalar_colorizer(m, cmap='viridis')
    assert isinstance(colorizer.norm, Normalize)
    assert colorizer.norm.vmin == pytest.approx(0)
    assert colorizer.norm.vmax == pytest.approx(1)
    assert colorizer.cmap.name == 'viridis'


def test_scalar_colorizer_with_bins():
    bins = np.linspace(0, 1, 6)
    colorizer = create_scalar_colorizer(bins=bins, cmap='plasma')
    assert isinstance(colorizer.norm, BoundaryNorm)
    assert colorizer.cmap.name == 'plasma'
    assert np.allclose(colorizer.norm.boundaries, bins)


def test_scalar_colorizer_extend_auto():
    colorizer = create_scalar_colorizer(vmin=2, vmax=8, cmap='magma')
    assert isinstance(colorizer.norm, Normalize)
    assert colorizer.norm.vmin == 2
    assert colorizer.norm.vmax == 8


def test_scalar_colorizer_with_norm_and_bins_conflict():
    with pytest.raises(ValueError, match='exclusive'):
        create_scalar_colorizer(bins=[0, 1, 2], norm=Normalize())


def test_scalar_colorizer_with_nan_color():
    colorizer_black = create_scalar_colorizer(nan_color='black')
    colorizer_white = create_scalar_colorizer(nan_color='white')

    np.testing.assert_array_equal(colorizer_black.cmap(np.nan), (0, 0, 0, 1))
    np.testing.assert_array_equal(colorizer_white.cmap(np.nan), (1, 1, 1, 1))


def test_scalar_colorizer_with_masked_array():
    colorizer = create_scalar_colorizer(cmap='inferno')
    assert isinstance(colorizer.norm, Normalize)
    assert colorizer.cmap.name == 'inferno'


def test_scalar_colorizer_bins_extend_behavior():
    bins = np.linspace(0, 5, 6)
    colorizer = create_scalar_colorizer(bins=bins, extend='both', cmap='cividis')
    norm = colorizer.norm
    assert isinstance(norm, BoundaryNorm)
    assert norm.extend == 'both'
    assert colorizer.cmap.name == 'cividis'


def test_classified_colorizer_basic():
    m = np.array([1, 2, 3, 2, 1])
    levels = parse_levels(m)
    colorizer = create_classified_colorizer(levels)
    assert isinstance(colorizer.cmap, ListedColormap)
    assert isinstance(colorizer.norm, BoundaryNorm)
    assert np.allclose(colorizer.norm.boundaries, [0.5, 1.5, 2.5, 3.5])


def test_classified_colorizer_with_custom_levels_and_colors():
    levels = [1, 2, 3]
    colors = ['red', 'green', 'blue']
    colorizer = create_classified_colorizer(levels=levels, colors=colors)
    assert colorizer.cmap.colors == colors


def test_classified_colorizer_warns_many_levels():
    m = np.arange(15)
    with pytest.warns(UserWarning, match='may reduce plot visibility'):
        create_classified_colorizer(m)


def test_classified_colorizer_nan_color():
    colorizer = create_classified_colorizer(levels=[1, 2], nan_color='black')
    assert hasattr(colorizer.cmap, 'set_bad')
    assert np.array_equal(colorizer.cmap(np.ma.masked_invalid([np.nan])), ((0, 0, 0, 1),))


def test_classified_colorizer_single_level():
    colorizer = create_classified_colorizer([1])
    assert isinstance(colorizer.norm, BoundaryNorm)
    assert len(colorizer.norm.boundaries) == 2


def test_classified_colorizer_non_increasing_levels():
    with pytest.raises(ValueError, match='sorted'):
        create_classified_colorizer(levels=[1, 1, 2])


def test_scalar_colorizer_plot_smoke(tmp_path):
    m = np.random.rand(10, 10)
    colorizer = create_scalar_colorizer()
    fig, ax = plt.subplots()
    im = ax.imshow(m, cmap=colorizer.cmap, norm=colorizer.norm)
    fig.colorbar(im)
    fig.savefig(tmp_path / 'out.png')
    plt.close(fig)

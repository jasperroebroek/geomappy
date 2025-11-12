import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.testing.compare
import numpy as np
import pytest
from matplotlib.colorizer import Colorizer
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from shapely import Point


@pytest.fixture
def fig_ax():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


@pytest.fixture
def fig_geoaxes():
    """Provide a simple Cartopy GeoAxes for map plotting tests."""
    fig = plt.figure(figsize=(4, 3))
    ax = plt.axes(projection=ccrs.PlateCarree())
    yield fig, ax
    plt.close(fig)


@pytest.fixture
def simple_colorizer():
    cmap = ListedColormap(['blue', 'green', 'red'])
    boundaries = np.array([0, 1, 2, 3])
    norm = BoundaryNorm(boundaries, cmap.N, extend='neither')
    return Colorizer(cmap=cmap, norm=norm)


@pytest.fixture
def basic_plot():
    """Provide a simple figure and Axes with an image for testing."""
    fig, ax = plt.subplots(figsize=(4, 3))
    data = [[0, 1], [2, 3]]
    im = ax.imshow(data, cmap='viridis', norm=Normalize(vmin=0, vmax=3))
    yield fig, ax, im
    plt.close(fig)


@pytest.fixture
def basic_binned_plot(simple_colorizer):
    fig, ax = plt.subplots(figsize=(4, 3))
    data = [[0, 1], [2, 3]]
    im = ax.imshow(data, colorizer=simple_colorizer)
    yield fig, ax, im
    plt.close(fig)


@pytest.fixture
def random_array():
    return np.random.rand(10, 10)


@pytest.fixture
def classified_array():
    arr = np.array(
        [
            [0, 1, 1, 2],
            [2, 2, 3, 3],
            [3, 0, 1, 2],
            [1, 0, 0, 3],
        ],
        dtype=float,
    )
    return arr


@pytest.fixture
def simple_points():
    """Returns a simple GeoDataFrame of points with values."""
    df = gpd.GeoDataFrame(
        {
            'geometry': [Point(x, y) for x, y in zip(range(5), range(5))],
            'value': [0, 1, 2, 1, 0],
            'size': [10, 20, 30, 40, 50],
        },
    )
    return df


@pytest.fixture
def simple_geo_values():
    """Returns simple scalar values for lat/lon plotting."""
    lat = np.arange(5)
    lon = np.arange(5)
    values = np.array([0, 1, 2, 1, 0])
    return lat, lon, values


@pytest.fixture
def imshow():
    def _imshow(*args, **kwargs):
        f, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(*args, **kwargs)
        return ax

    return _imshow


@pytest.fixture
def compare_images(tmp_path):
    def _compare_images(a1, a2, tol=0):
        f1 = tmp_path / 'a1.png'
        f2 = tmp_path / 'a2.png'

        a1.figure.savefig(f1)
        a2.figure.savefig(f2)

        r = matplotlib.testing.compare.compare_images(str(f1), str(f2), tol=tol)
        return r is None

    return _compare_images

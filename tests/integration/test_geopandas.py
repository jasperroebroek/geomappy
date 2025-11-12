import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colorbar import Colorbar

from geomappy import gpd_plot_classified_shapes, gpd_plot_shapes
from geomappy.plotting import plot_shapes


@pytest.fixture
def simple_gdf():
    """Create a simple GeoDataFrame with points and values"""
    gdf = gpd.GeoDataFrame(
        {
            'value': [1, 2, 3, 4],
            'geometry': gpd.points_from_xy([0, 1, 2, 3], [0, 1, 2, 3]),
        },
        crs='EPSG:4326',
    )
    return gdf


def test_get_extent_and_projection(simple_gdf):
    """Test the GeoDataFrame extension methods"""
    ext = simple_gdf.get_extent()
    assert len(ext) == 4
    proj = simple_gdf.get_cartopy_projection()
    assert hasattr(proj, 'proj4_init')


@pytest.mark.parametrize('plot_fun', [gpd_plot_classified_shapes, gpd_plot_shapes])
def test_plot_combined_shapes_returns_axes_and_legend(simple_gdf, plot_fun):
    col, legend = plot_fun(simple_gdf, values='value')
    assert isinstance(col[0].axes, GeoAxes)
    assert isinstance(legend, Colorbar)


@pytest.mark.parametrize('cartopy', [True, False])
def test_plot_combined_shapes_with_cartopy(simple_gdf, cartopy):
    col, legend = simple_gdf.plot_shapes(values='value', cartopy=cartopy, ax=None)
    assert isinstance(col[0].axes, GeoAxes) == cartopy


def test_plot_shapes_with_series_and_geo_series(simple_gdf):
    """Check plotting with a GeoSeries"""
    geo_series = simple_gdf.geometry
    col, legend = geo_series.plot_shapes(values=np.random.rand(geo_series.size))
    assert isinstance(col[0].axes, GeoAxes)
    assert isinstance(legend, Colorbar)


def test_invalid_type_raises():
    with pytest.raises(TypeError):
        plot_shapes(123)


def test_plot_shapes_and_classified_shapes_same_ax(simple_gdf):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    col1, _ = simple_gdf.plot_shapes(values='value', ax=ax)
    col2, _ = simple_gdf.plot_classified_shapes(values='value', ax=ax)
    assert col1[0].axes == col2[0].axes == ax


def test_plot_shapes_with_face_and_edgecolor(simple_gdf):
    col, legend = simple_gdf.plot_shapes(values=None, facecolor='yellow', edgecolor='red')
    assert legend is None
    for c in col:
        fc = c.get_facecolor()
        ec = c.get_edgecolor()
        if fc.size > 0:
            assert np.allclose(fc, matplotlib.colors.to_rgba('yellow'), atol=1e-2)
        if ec.size > 0:
            assert np.allclose(ec, matplotlib.colors.to_rgba('red'), atol=1e-2)

import cartopy.crs as ccrs
import numpy as np
import pytest
import rioxarray as rxr  # noqa
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colorbar import Colorbar

from geomappy.integration.rioxarray import (
    da_get_cartopy_projection,
    da_get_extent,
    da_plot_classified_raster,
    da_plot_raster,
)


@pytest.fixture
def simple_da():
    shape = (4, 4)
    data = np.arange(np.prod(shape)).reshape(shape)
    da = xr.DataArray(data, dims=('y', 'x'))
    da = da.assign_coords(x=np.linspace(-180, 180, shape[1]), y=np.linspace(90, -90, shape[0]))
    da = da.rio.write_crs('EPSG:4326')
    return da


def test_get_cartopy_projection(simple_da):
    proj = da_get_cartopy_projection(simple_da)
    assert isinstance(proj, ccrs.PlateCarree)


def test_get_extent(simple_da):
    ext = da_get_extent(simple_da)
    assert ext == (
        simple_da.x.min().item(),
        simple_da.x.max().item(),
        simple_da.y.min().item(),
        simple_da.y.max().item(),
    )


@pytest.mark.parametrize('classified', [False, True])
def test_plot_combined_raster_returns_axes_and_legend(simple_da, classified):
    im, legend = da_plot_classified_raster(simple_da) if classified else da_plot_raster(simple_da)
    assert isinstance(im.axes, GeoAxes)
    assert isinstance(legend, Colorbar)


@pytest.mark.parametrize('cartopy', (True, False))
def test_plot_combined_raster_with_cartopy(simple_da, cartopy):
    im, legend = da_plot_raster(simple_da, cartopy=cartopy, ax=None)
    assert isinstance(im.axes, GeoAxes) == cartopy


def test_invalid_ndarray_raises(simple_da):
    da = simple_da.copy()
    da = da.expand_dims({'time': [1, 2, 3, 4, 5, 6]}, axis=-1)  # shape (4, 4, 6)
    with pytest.raises(IndexError):
        da_plot_raster(da)


def test_invalid_type_raises():
    with pytest.raises(TypeError):
        da_plot_raster(123)  # not a DataArray

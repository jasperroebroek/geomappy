from rasterio.coords import BoundingBox
import geomappy as mp
import numpy as np
import os
import joblib
from test_raster_utils import TestRaster

path = os.getcwd()
if path.split("/")[-1] == 'geomappy':
    prefix = ""
elif path.split("/")[-1] == 'tests':
    prefix = "../"
x = mp.Raster(f"{prefix}data/wtd.tif")
y = mp.Raster(f"{prefix}data/tree_height.asc", fill_value=0)
z = mp.Raster(f"{prefix}data/temp_and_precipitation.nc")


def test_bounds():
    default_bounds = BoundingBox(left=109.999999342, bottom=-44.99999854499999, right=155.000000419, top=-8.999999499)

    # Directly from rasterio object (through __getattr__)
    assert x.bounds == default_bounds

    # Creating a BoundaryBox object from a window that spans the whole file
    assert x.get_bounds(0) == default_bounds

    # Test bounds of window being maintained
    assert np.allclose(tuple(x.get_bounds((110, -45, 120, -35))), (110, -45, 120, -35), atol=0.01)


def test_shape():
    default_shape = (4320, 5400)

    # Directly from rasterio object (through __getattr__)
    assert x.shape == default_shape
    assert (x.height, x.width) == default_shape

    # From window
    assert x.get_shape(0) == default_shape

    # From numpy window
    assert x.idx[:100, :100].shape == (100, 100)

    # From geographic window
    assert x.geo[(110, -45, 120.001, -35)].shape == (1200, 1200)


def test_window_size():
    x.set_window_size(5)
    assert x.get_shape(0) == (4324, 5404)
    assert x.get_shape(None) == (4320, 5400)
    x.set_window_size(1)


def test_tiles():
    x.set_tiles(5)

    assert x.c_tiles == 4
    assert len(x._tiles) == 5
    assert x._tiles[4] is None

    # Check the right bounds
    assert np.allclose(np.asarray(x.get_bounds(1)) - np.asarray(x.get_bounds(0)), (22.5, 0, 22.5, 0))
    assert np.allclose(np.asarray(x.get_bounds(3)) - np.asarray(x.get_bounds(2)), (22.5, 0, 22.5, 0))

    assert np.allclose(np.asarray(x.get_bounds(0)) - np.asarray(x.get_bounds(2)), (0, 18, 0, 18))
    assert np.allclose(np.asarray(x.get_bounds(1)) - np.asarray(x.get_bounds(3)), (0, 18, 0, 18))

    # Check the right shape
    assert np.allclose(x.get_shape(0), np.asarray(x.shape) / 2)

    x.set_tiles(1)


def test_iter():
    x.set_tiles(200)

    assert list(x.iter(progress_bar=False)) == list(range(x.c_tiles))
    assert [i for i in x] == list(range(x.c_tiles))

    x.set_tiles(1)


def test_memmap():
    v1 = x.generate_memmap()
    v2 = x.values
    assert np.allclose(v1, v2, equal_nan=True)
    x.close_memmap()


def test_state_reader():
    state = x.__getstate__()
    x.__setstate__(state)
    assert x.__getstate__() == state

    x.__setstate__((x.location, 1, (2, 2), True, x.profile))
    assert not x.__getstate__() == state
    x.__setstate__(state)


def test_state_writer():
    with TestRaster():
        wr = mp.Raster("_test_rasters/wtd.tif", mode='w', profile=x.profile)
        wr.idx[:] = x.idx[:]
        assert wr._fp.mode == 'w+'
        wr.close(verbose=False)
        wr.__setstate__(wr.__getstate__())
        assert wr._fp.mode == 'r+'
        assert np.allclose(wr.read(), x.values, equal_nan=True)
        wr.close(verbose=False)


def test_params_reader():
    params = x.params
    x.params = params
    assert x.params == params

    x.params = (5, (2, 2), True)
    assert not x.params == params
    x.params = params


def test_idxlocator():
    values = x.values
    assert np.allclose(x.idx[:], values, equal_nan=True)
    assert np.allclose(x.idx[:, :], values, equal_nan=True)
    assert np.allclose(x.idx[0], values[0], equal_nan=True)
    assert np.allclose(x.idx[2000, 2000], values[2000, 2000], equal_nan=True)
    assert np.allclose(x.idx[:, 2000].squeeze(), values[:, 2000], equal_nan=True)
    assert np.allclose(x.idx[:, :, 1], values, equal_nan=True)
    assert np.allclose(x.idx[:, :, (1,)], values, equal_nan=True)
    assert x.idx[:, :, (1, 1)].shape == (2, 4320, 5400)


def test_geolocator():
    values = x.values
    bounds = (110, -45, 120.001, -35)
    assert np.allclose(x.geo[bounds], x[bounds], equal_nan=True)
    assert np.allclose(x.geo[bounds, 1], x[bounds], equal_nan=True)
    assert np.allclose(x.geo[bounds, (1, )], x[bounds], equal_nan=True)
    assert np.allclose(x.geo[bounds], values[-1200:, :1200], equal_nan=True)
    assert x.geo[bounds, (1, 1)].shape == (2, 1200, 1200)


def test_tilelocator():
    x.set_tiles(16)
    assert np.allclose(x.iloc[0], x.values[:1080, :1350], equal_nan=True)
    assert np.allclose(x.iloc[0], x[0], equal_nan=True)
    assert np.allclose(x.iloc[0, 1], x[0], equal_nan=True)
    assert np.allclose(x.iloc[0, (1,)], x[0], equal_nan=True)
    assert x.iloc[0, (1, 1)].shape == (2, 1080, 1350)
    x.set_tiles(1)


def test_raster_set():
    assert isinstance(z, mp.raster._set.RasterReaderSet)
    assert z.t2m.bounds == z.bounds['t2m']
    assert z.t2m.transform == z.transform['t2m']
    assert z.t2m.shape == z.shape['t2m']
    assert z.t2m.shape == (21, 81, 111)
    assert z.t2m.values.shape == (21, 81, 111)
    assert z.t2m.iloc[0, 1].shape == (81, 111)
    assert z.t2m.iloc[0, (1, 2)].shape == (2, 81, 111)
    assert z.t2m.shape == z.values['t2m'].shape
    assert np.allclose(z.values['t2m'], z.geo['t2m'][z.t2m.bounds], equal_nan=True)


def test_pickling():
    with TestRaster():
        joblib.dump(x, "_test_rasters/test.pickle")
        x_new = joblib.load("_test_rasters/test.pickle")
        # Should yield True when compared
        assert x == x_new
        # Test if comparing behaviour works correctly
        assert x != y

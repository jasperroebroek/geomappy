import geomappy as mp
import numpy as np
import os
import pytest

from test_raster_utils import TestRaster

path = os.getcwd()
if path.split("/")[-1] == 'geomappy':
    prefix = ""
elif path.split("/")[-1] == 'tests':
    prefix = "../"
x = mp.Raster(f"{prefix}data/wtd.tif")
y = mp.Raster(f"{prefix}data/tree_height.asc", fill_value=0)
z = mp.Raster(f"{prefix}data/temp_and_precipitation.nc").tp


@pytest.mark.parametrize("tiles,reduce,parallel", [(1, False, False), (25, False, False), (1, True, False),
                                                   (4, True, False), (1, False, True), (25, False, True),
                                                   (1, True, True), (4, True, True)])
def test_focal_correlation(tiles, reduce, parallel):
    with TestRaster():
        x.set_tiles(tiles)
        y.set_tiles(tiles)
        loc = "_test_rasters/test_focal_mean.tif"
        x.focal_correlation(y, window_size=5, output_file=loc, overwrite=True, progress_bar=False, reduce=reduce,
                            parallel=parallel)
        t = mp.Raster(loc)
        c1 = t.values
        t.close(verbose=False)

        ind_inner = np.s_[2:-2, 2:-2]
        c2 = mp.focal_statistics.correlate_maps(x.values, y.values, window_size=5, reduce=reduce)
        assert np.allclose(c1[ind_inner], c2[ind_inner], equal_nan=True)
        x.set_tiles(1)
        y.set_tiles(1)


@pytest.mark.parametrize("reduce", [True, False])
def test_tile_focal_correlation(reduce):
    with TestRaster():
        mp.Raster.set_tiles((8, 8))
        if reduce:
            mp.Raster.set_window_size(1)
        else:
            mp.Raster.set_window_size(5)

        loc = "_test_rasters/test_focal_correlation.tif"
        x.focal_correlation(y, ind=32, window_size=5, output_file=loc, overwrite=True, progress_bar=False, reduce=reduce)
        t = mp.Raster(loc)
        c1 = t.values
        t.close(verbose=False)

        c2 = mp.focal_statistics.correlate_maps(x.iloc[32], y.iloc[32], window_size=5, reduce=reduce)
        assert np.allclose(c1[x.ind_inner], c2[x.ind_inner], equal_nan=True)

    mp.Raster.set_window_size(1)
    mp.Raster.set_tiles(1)


def test_focal_correlation_manual():
    with TestRaster():
        mp.Raster.set_tiles(25)
        mp.Raster.set_window_size(5)
        loc = "_test_rasters/test_focal_mean_manual.tif"
        profile = x.profile
        profile['dtype'] = np.float64
        f = mp.Raster(loc, mode='w', profile=profile, tiles=x.tiles, overwrite=True,
                      force_equal_tiles=x._force_equal_tiles, window_size=x.window_size)
        for i in x:
            f[i] = mp.focal_statistics.correlate_maps(x.iloc[i], y.iloc[i], window_size=5, reduce=False)
        f.close(verbose=False)
        f = mp.Raster(loc)
        c1 = f[0]
        f.close(verbose=False)
        c2 = mp.focal_statistics.correlate_maps(x.values, y.values, window_size=5, reduce=False)

        assert np.allclose(c1[x.ind_inner], c2[x.ind_inner], equal_nan=True)

        mp.Raster.set_tiles(1)
        mp.Raster.set_window_size(1)

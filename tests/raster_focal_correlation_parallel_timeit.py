import geomappy as mp
import time

x = mp.Raster("data/wtd.tif")
y = mp.Raster("data/tree_height.asc", fill_value=0)
mp.Raster.set_tiles(12)
mp.Raster.set_window_size(5)

%timeit x.correlate(y, output_file='test.tif', overwrite=True, parallel=True)
%timeit x.correlate(y, output_file='test.tif', overwrite=True, parallel=False)

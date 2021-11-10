import geomappy as mp

x = mp.Raster("data/wtd.tif")
x.tiles = 12
x.window_size = 5

%timeit x.focal_mean(output_file='test.tif', overwrite=True, parallel=True)
%timeit x.focal_mean(output_file='test.tif', overwrite=True, parallel=False)

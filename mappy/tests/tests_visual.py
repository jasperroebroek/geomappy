import mappy as mp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib

WTD = mp.Map("data/wtd.tif")
TH = mp.Map("data/tree_height.asc")
mp.Map.set_tiles((2,2))

loc = "test.tif"
WTD.focal_mean(ind=0, window_size=10, output_file=loc, overwrite=True, reduce=True)

c = mp.Map(loc)

print(np.allclose(c[0], mp.focal_mean(WTD[0], window_size=10, reduce=True), equal_nan=True))

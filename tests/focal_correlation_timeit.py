import geomappy as mp
from geomappy.focal_statistics.focal_correlation import correlate_maps as correlate_maps_njit
from geomappy.focal_statistics.c_focal_correlation import correlate_maps as correlate_maps_cython
from geomappy.focal_statistics.focal_correlation import correlate_maps_base as correlate_maps_numpy

import matplotlib.pyplot as plt
import numpy as np

map1 = mp.Raster("data/wtd.tif").values
map2 = mp.Raster("data/tree_height.asc", fill_value=0).values
print("data loaded")

# %timeit correlate_maps_cython(map1, map2, window_size=5, fraction_accepted=0.25, verbose=True, reduce=False)
# %timeit correlate_maps_cython(map1, map2, window_size=5, fraction_accepted=0.25, verbose=True, reduce=True)

for f in [0, 0.25, 1]:
    for reduce in [False, True]:
        c1 = correlate_maps_njit(map1, map2, window_size=5, fraction_accepted=f, verbose=True, reduce=reduce)
        c2 = correlate_maps_cython(map1, map2, window_size=5, fraction_accepted=f, verbose=True, reduce=reduce)
        if not reduce:
            c3 = correlate_maps_numpy(map1, map2, window_size=5, fraction_accepted=f, verbose=True)

        # mp.plot_map(c1)
        # plt.show()

        # mp.plot_map(c2)
        # plt.show()

        # mp.plot_map(np.isclose(c1, c2, equal_nan=True))
        # plt.show()

        print(f"-> {np.allclose(c1, c2, equal_nan=True)}")

mp.Raster.close()

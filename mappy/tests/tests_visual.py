import mappy as mp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib

WTD = mp.Map("data/wtd.tif")

WTD.plot()
plt.show()

mp.plot_map(WTD[0], legend_kwargs={'shrink': 0.9})
plt.show()

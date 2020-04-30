import mappy as mp
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib

WTD = mp.Map("data/wtd.tif")

ax = mp.plot_map(WTD[0], legend=False)
x = mp.add_colorbar(ax=ax, shrink=0.9, position='right', aspect=50)
plt.show()

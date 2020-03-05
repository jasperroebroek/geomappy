from mappy import *
import matplotlib.pyplot as plt
import numpy as np

m = Map("/perm/mo/mojr/mappy/data/wtd.tif")

m.plot(basemap=False, legend="colorbar", basemap_kwargs={'xticks':10, 'yticks':10})
plt.show()

m.plot(basemap=True, bins=[1,50], legend="colorbar",
       basemap_kwargs={'xticks':10, 'yticks':10, 'resolution':'10m', 'coastline_linewidth':0.5})
plt.show()

a = np.random.randint(0,5,900).reshape(30,30)
plot_classified_map(a, legend='colorbar', bins=[0,1,2,3,4], colors=cmap_discrete(5, return_type='list'))
plt.show()
plot_classified_map_old(a, legend='colorbar', bins=[0,1,2,3,4], mode='classes', colors=cmap_discrete(5, return_type='list'))
plt.show()


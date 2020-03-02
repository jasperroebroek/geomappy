from thesis import *

m = Map("wtd.tif")
m.plot(basemap=True, colorbar=True, xticks=10, yticks=10)


# m.plot_classified(bins=[0, 1, 2, 5, 10, 200, 1000], legend='colorbar', basemap=True,
#                   basemap_kwargs={'coastline_linewidth':0.2, 'grid_linewidth':0.4, 'border_linewidth':0.6,
#                                   'fontsize':8, 'xticks':10, 'yticks':10})
# m.plot(basemap=True, colorbar=True).set_global()
#
# m2 = Map("/perm/mo/mojr/GloFAS_reanalysis_2015/data/CEMS_ECMWF_dis24_20150101_glofas_v2.1.nc")
# m2.epsg = 4326
# ax = m2.plot([-180, -90, 0, 0], basemap=True, colorbar=True, coastline_linewidth=0.2, grid_linewidth=0.4,
#              border_linewidth=0.6, fontsize=8)
# plt.show()

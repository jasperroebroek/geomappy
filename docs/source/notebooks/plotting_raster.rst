.. currentmodule:: geomappy
.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../../source/notebooks/plotting_raster.ipynb>`_.

.. _plotting_raster:


Plotting choropleth rasters
===========================

.. code:: python

    import rioxarray as rxr
    import matplotlib.pyplot as plt
    import geomappy as mp
    import numpy as np
    import os
    os.chdir("../../../")

.. code:: python

    r = rxr.open_rasterio("data/wtd.tif")
    a = r.values[0]
    a[a == -999.9] = np.nan

A contains a 2D raster of water table depth (Fan et al., 2017). To
visualise this one can simply use matplotlib directly.

.. code:: python

    plt.imshow(a, cmap="Blues_r", vmax=100)
    plt.colorbar()
    plt.show()



.. image:: plotting_raster_files/plotting_raster_4_0.png


At its simplests, ``geomappy`` does exactly this (although some esthetic
differences can be seen):

.. code:: python

    mp.plot_raster(a, cmap="Blues_r", vmax=100)
    plt.show()



.. image:: plotting_raster_files/plotting_raster_6_0.png


The biggest difference in workflow comes with the binning of the data.
This gets handled internally instead of having to specify this with
numpy outisde the plotting interface.

.. code:: python

    mp.plot_raster(a, bins=[0,0.1,0.5,1,2,5,10,25], cmap="Blues_r")
    plt.show()



.. image:: plotting_raster_files/plotting_raster_8_0.png


In this case, the colorbar can be converted into a true legend (figure
size is expanded with ``figsize`` to fit the legend):

.. code:: python

    mp.plot_raster(a, bins=[0,0.1,0.5,1,2,5,10,25], cmap="Blues_r", legend='legend', figsize=(10, 10))
    plt.show()



.. image:: plotting_raster_files/plotting_raster_10_0.png


With a basemap
--------------

The functionality described above and in the section on basemaps can be
applied here. First the bounds need to be extracted from the raster.

.. code:: python

    bounds = r.rio.bounds()
    extent = bounds[0], bounds[2], bounds[1], bounds[3]
    extent




.. parsed-literal::

    (109.999999342, 155.000000419, -44.999998545, -8.999999499)



Then a basemap needs to be created

.. code:: python

    ax = mp.basemap(figsize=(6, 6))
    ax.coastlines()
    ax.set_extent(extent)
    mp.add_ticks(ax, 10)
    mp.add_gridlines(ax, 10)




.. parsed-literal::

    <cartopy.mpl.gridliner.Gridliner at 0x183c5ea60>




.. image:: plotting_raster_files/plotting_raster_15_1.png


Then this GeoAxes object needs to be passed to the plotting function.

.. code:: python

    ax = mp.basemap(figsize=(6, 6))
    ax.coastlines()
    ax.set_extent(extent)
    mp.add_ticks(ax, 10)
    mp.add_gridlines(ax, 10)
    mp.plot_raster(a, ax=ax, cmap="Blues_r", bins=[0,0.1,0.5,1,2,5,10,25], extent=extent)
    plt.show()



.. image:: plotting_raster_files/plotting_raster_17_0.png




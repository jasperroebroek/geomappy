.. currentmodule:: geomappy
.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../../source/notebooks/rioxarray_integration.ipynb>`_.

.. _rioxarray_integration:


Integration of Geomappy into Rioxarray
======================================

.. code:: python

    import rioxarray as rxr
    import matplotlib.pyplot as plt
    import geomappy as mp
    import numpy as np
    import os
    import pyproj
    import cartopy.crs as ccrs
    from rasterio.enums import Resampling

.. code:: python

    os.chdir("../../../")

A 2D raster of water table depth (Fan et al., 2017).

.. code:: python

    r1 = rxr.open_rasterio("data/wtd.tif", masked=True)

Monthly mean discharges from 2019 from GloFAS

.. code:: python

    r2 = rxr.open_rasterio("data/dis_2019_monthlymeans_cropped_complete.nc", masked=True, decode_times=False).dis24[0]
    r2 = r2.rio.write_crs(ccrs.epsg(3035))


.. parsed-literal::

    /Users/jroebroek/opt/anaconda3/envs/geomappy/lib/python3.9/site-packages/rasterio/__init__.py:304: NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
      dataset = DatasetReader(path, driver=driver, sharing=sharing, **kwargs)
    /Users/jroebroek/opt/anaconda3/envs/geomappy/lib/python3.9/site-packages/rioxarray/_io.py:1111: NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
      warnings.warn(str(rio_warning.message), type(rio_warning.message))  # type: ignore


Outline on a world map
----------------------

The first file covers Australia, while the second file covers Europe.
Both have different projections. To see this at work, geomappy
integrates two functions to explore the data: ``plot_world`` and
``plot_file``.

.. code:: python

    r1.plot_world(figsize=(8, 8))
    plt.show()



.. image:: rioxarray_integration_files/rioxarray_integration_9_0.png


.. code:: python

    r2.plot_world(figsize=(8, 8))
    plt.show()



.. image:: rioxarray_integration_files/rioxarray_integration_10_0.png


Here you can see that different data projections cause different shapes.

To focus specifically on the area of the file itself geomappy integrates
the ``plot_file`` function, using the projection of the data.

.. code:: python

    r1.plot_file()
    r2.plot_file()
    plt.show()



.. image:: rioxarray_integration_files/rioxarray_integration_13_0.png



.. image:: rioxarray_integration_files/rioxarray_integration_13_1.png


Plotting the data
-----------------

The geomappy plotting functionality (``plot_raster``) is directly
integrated into rioxarray by loading geomappy. This results in the same
figure as seen before:

.. code:: python

    r1.plot_raster(basemap=False)
    plt.show()



.. image:: rioxarray_integration_files/rioxarray_integration_16_0.png


Including legends, bins and a cmap:

.. code:: python

    r1.plot_raster(bins=[0, 0.1, 0.5, 1, 2, 5, 10, 25], legend="legend", cmap="Blues_r", basemap=False)
    plt.show()



.. image:: rioxarray_integration_files/rioxarray_integration_18_0.png


Plotting the same image on a basemap from within the DataArray is much
easier though, by taking advantage of the internal projection
representation.

.. code:: python

    r1.plot_raster(bins=[0, 0.1, 0.5, 1, 2, 5, 10, 25], cmap="Blues_r", basemap=True, ticks=10, lines=10)
    plt.show()



.. image:: rioxarray_integration_files/rioxarray_integration_20_0.png


Plotting on different projections
---------------------------------

plotting on different projections is done with the projection keyword
(if basemap is activated)

.. code:: python

    r2.rio.reproject(dst_crs=ccrs.Mercator()).plot_raster(bins=[0,1,2,3,5,10,100,1000], cmap="Blues", ticks=10, lines=10)
    plt.show()



.. image:: rioxarray_integration_files/rioxarray_integration_23_0.png




.. geomappy documentation master file, created by
   sphinx-quickstart on Wed Dec  1 14:19:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********************
Creating maps in Python
***********************

This package aims to provide a quick interface to create publication quality maps. It combines the plotting of both
raster and polygon data on a basemap (based on ``cartopy``) and integrates its functionality into ``geopandas`` and
``rioxarray``. It is build in the matplotlib ecosystem, so all usual functionality is available here too. Only
choropleth plotting functionality is supported (to make plotting of shapes and rasters as similar as possible). See
for example the ``geoplot`` package for different plotting options for shapes. Both packages can play together
quite nicely, but terminology is not always identical. Projections are taken care of under the hood, but can at any
time be passed as a separate parameter, accepting all cartopy projections.

The package provides:

- A convenience :func:`geomappy.basemap` function
- Functions to plot rasters and shapes in an identical interface. They are both separated in the plotting of continuous and discrete data.
- Both plotting and basemap functionality is directly loaded into ``rioxarray`` and ``geopandas``

*************
Documentation
*************

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Methods

   Basemap
   =======
   notebooks/basemap

   Plotting functions
   ==================
   notebooks/plotting_raster
   notebooks/plotting_classified_raster
   notebooks/plotting_shapes
   notebooks/plotting_classified_shapes

   Integration
   ===========
   notebooks/rioxarray_integration
   notebooks/geopandas_integration

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   whats_new
   api


License
=======

geomappy is published under a MIT license.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

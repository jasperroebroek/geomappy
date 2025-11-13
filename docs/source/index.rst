.. geomappy documentation master file, created by sphinx-quickstart.
   This file contains the root `toctree` directive.

========
Geomappy
========

This package provides a streamlined interface for creating publication-quality maps. It supports plotting both raster
and vector (polygon) data on a basemap using cartopy and integrates with ``geopandas`` and ``rioxarray``. Built on top
of the ``matplotlib`` ecosystem, it retains access to all standard matplotlib functionality. Currently, only
choropleth-style plotting is supported, which ensures a consistent interface for both raster and vector data. For
additional vector plotting options, see the ``geoplot`` package—both can be used together, though terminology may
differ. Projections are handled automatically but can also be explicitly specified, supporting any valid cartopy
projection.

Key features
------------
- Unified plotting interface for rasters and vector shapes.
- Automatic integration with ``geopandas`` and ``rioxarray``.
- Support for classified and continuous data.
- Full control of projections and Matplotlib styling.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/plotting_raster
   notebooks/plotting_classified_raster
   notebooks/plotting_shapes
   notebooks/plotting_classified_shapes

.. toctree::
   :maxdepth: 2
   :caption: Integration

   notebooks/rioxarray_integration
   notebooks/geopandas_integration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   whats_new
   api

License
=======
Geomappy is published under an MIT license.

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

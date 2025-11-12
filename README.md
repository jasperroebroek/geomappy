[![Documentation Status](https://readthedocs.org/projects/geomappy/badge/?version=latest)](https://geomappy.readthedocs.io/en/latest/?badge=latest)

This package provides a streamlined interface for creating publication-quality maps. It supports plotting both raster and vector (polygon) data on a basemap using ``cartopy`` and integrates seamlessly with ``geopandas`` and ``rioxarray``. Built on top of the ``matplotlib`` ecosystem, it retains access to all standard ``matplotlib`` functionality. Currently, only choropleth-style plotting is supported, which ensures a consistent interface for both raster and vector data. For additional vector plotting options, see the geoplot package—both can be used together, although terminology may differ. Map projections are handled automatically, but can also be explicitly specified, supporting any valid Cartopy projection.

Key features
------------
- Unified plotting interface for both raster and vector shapes. 
- Automatic integration with ``geopandas`` and ``rioxarray``. 
- Support for classified (discrete) and continuous data. 
- Full control over map projections, color mapping, and Matplotlib styling. 

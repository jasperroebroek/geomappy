import warnings

import matplotlib  # type: ignore

from geomappy.basemap import basemap, add_ticks, add_gridlines
from geomappy.geodataframe import gpd_plot_shapes, gpd_plot_classified_shapes, gpd_plot_world, gpd_plot_file
from geomappy.raster import plot_raster, plot_classified_raster
from geomappy.shapes import plot_shapes, plot_classified_shapes
from geomappy.xarray import xarray_plot_raster, xarray_plot_classified_raster, xarray_plot_world, xarray_plot_file

# Import 'show' and 'savefig' for convenience. This should be avoided in production code.
show = matplotlib.pyplot.show
savefig = matplotlib.pyplot.savefig

try:
    from sphinx.ext.autodoc.mock import _MockModule

    if not isinstance(matplotlib, _MockModule):
        warnings.filterwarnings("once", category=FutureWarning)
        warnings.filterwarnings("once", category=matplotlib.MatplotlibDeprecationWarning)
        matplotlib.rcParams["image.interpolation"] = 'none'
except ModuleNotFoundError:
    pass

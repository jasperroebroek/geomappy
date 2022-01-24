from sphinx.ext.autodoc.mock import _MockModule
from geomappy.geodataframe import gpd_plot_shapes, gpd_plot_classified_shapes, gpd_plot_world, gpd_plot_file
from geomappy.xarray import xarray_plot_raster, xarray_plot_classified_raster, xarray_plot_world, xarray_plot_file
from geomappy.basemap import basemap
from geomappy.plotting import plot_raster, plot_classified_raster, plot_shapes, plot_classified_shapes

import matplotlib
import warnings

# Import 'show' and 'savefig' for convenience. This should be avoided in production code.
show = matplotlib.pyplot.show
savefig = matplotlib.pyplot.savefig

if not isinstance(matplotlib, _MockModule):
    warnings.filterwarnings("once", category=FutureWarning)
    warnings.filterwarnings("once", category=matplotlib.MatplotlibDeprecationWarning)
    matplotlib.rcParams["image.interpolation"] = 'none'

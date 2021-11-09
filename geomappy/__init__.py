from .raster import Raster
from .geodataframe import gpd
from .plotting import plot_classified_map, plot_classified_shapes, plot_shapes, plot_map, Legend, LegendPatches, \
    Colorbar
from .basemap import basemap
from .colors import add_colorbar

import matplotlib

# Import 'show' and 'savefig' for convenience. This should be avoided in production code.
show = matplotlib.pyplot.show
savefig = matplotlib.pyplot.savefig

import warnings
warnings.filterwarnings("once", category=FutureWarning)
warnings.filterwarnings("once", category=matplotlib.MatplotlibDeprecationWarning)
matplotlib.rcParams["image.interpolation"] = 'none'

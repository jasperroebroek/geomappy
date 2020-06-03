from .map import Map
from .geodataframe import gpd
from .plotting import plot_classified_map, plot_classified_shapes, plot_shapes, plot_map, Legend, LegendPatches, \
    Colorbar
from .basemap import basemap
from .colors import add_colorbar

import matplotlib

import warnings
warnings.filterwarnings("once", category=FutureWarning)
warnings.filterwarnings("once", category=matplotlib.MatplotlibDeprecationWarning)
matplotlib.rcParams["image.interpolation"] = 'none'

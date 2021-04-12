try:
    from .c_focal_correlation import correlate_maps
except ModuleNotFoundError:
    from .focal_correlation import correlate_maps

from .focal_correlation import correlate_maps_base
from .focal_statistics import focal_std, focal_min, focal_mean, focal_max, focal_statistics, focal_majority

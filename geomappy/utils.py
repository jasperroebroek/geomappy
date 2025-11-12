import numbers
from functools import wraps

import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike
from pyproj import CRS

from geomappy.types import ExtendType


def add_method(name, *cls):
    """
    Decorator to add functions/objects to existing classes

    Parameters
    ----------
    name : str
        function name when implemented
    cls : type or tuple of types
        class or classes that the function that the decorator is applied to will be added to

    Notes
    -----
    https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        for c in cls:
            setattr(c, name, wrapper)
        return func

    return decorator


def check_increasing_and_unique(v: np.ndarray, name: str) -> None:
    vs = np.sort(np.unique(v))
    if vs.size == 0:
        raise ValueError(f'{name} are empty')
    if not np.array_equal(v, vs):
        raise ValueError(f'{name} are not sorted, contain double entries or contain NaN values: {v}')


def calculate_horizontal_locations(v: float | tuple[float]):
    if isinstance(v, numbers.Real):
        return np.linspace(-180, 180, int(360 / v + 1))
    return np.asarray(v)


def calculate_vertical_locations(v: float | tuple[float]):
    if isinstance(v, numbers.Real):
        return np.linspace(-90, 90, int(180 / v + 1))
    return np.asarray(v)


def get_data_range(m: np.ndarray) -> tuple[float, float]:
    data_min = np.ma.min(m) if np.ma.isMaskedArray(m) else np.nanmin(m)
    data_max = np.ma.max(m) if np.ma.isMaskedArray(m) else np.nanmax(m)

    if np.ma.is_masked(data_min) or np.ma.is_masked(data_max):
        raise ValueError('Data contains only masked values')
    if np.isnan(data_min) or np.isnan(data_max):
        raise ValueError('Data contains only NaN values')

    return data_min, data_max


def check_exclusive_limits(
    bins: ArrayLike | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    if (
        sum(
            (
                bins is not None,
                norm is not None,
                vmin is not None or vmax is not None,
            ),
        )
        > 1
    ):
        raise ValueError('bins, norm and vmin/vmax are mutually exclusive')


def determine_extend(
    data_range: tuple[float, float],
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    bins: ArrayLike | None = None,
) -> ExtendType:
    check_exclusive_limits(bins, norm, vmin, vmax)

    data_min, data_max = data_range

    if bins is not None:
        vmin = bins[0]
        vmax = bins[-1]
    if norm is not None:
        vmin = norm.vmin
        vmax = norm.vmax

    if vmin is None:
        vmin = data_min
    if vmax is None:
        vmax = data_max

    extends_below = data_min < vmin
    extends_above = data_max > vmax

    if extends_below and extends_above:
        return 'both'
    if extends_below:
        return 'min'
    if extends_above:
        return 'max'
    return 'neither'


def parse_levels(m: np.ndarray, levels: ArrayLike | None = None) -> np.ndarray[tuple[int], np.float64]:
    if levels is None:
        levels = np.unique(m.compressed() if np.ma.isMaskedArray(m) else m)
    levels = np.asarray(levels, dtype=np.float64).flatten()
    check_increasing_and_unique(levels, 'levels')
    return levels


def expand(v: ArrayLike | None, size: int) -> np.ndarray | None:
    if v is None:
        return None
    v = np.asarray(v).flatten()
    if v.size == 1:
        v = v.repeat(size)
    if v.size != size:
        raise IndexError('Length of `v` does not match length of `size`')
    return v


def get_cartopy_projection(crs: CRS) -> ccrs.CRS:
    if crs.to_epsg() == 4326:
        return ccrs.PlateCarree()
    crs = ccrs.epsg(crs.to_epsg())
    if crs is None:
        raise ValueError(
            f'Projection cannot be converted to Cartopy projection: {crs}. Set the projection manually in the plotting '
            f'functions',
        )
    return crs

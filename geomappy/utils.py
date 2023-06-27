#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities used in geomappy. Interesting functions to use outside the internal scope are
progress_bar and reproject_map_like
"""
from functools import wraps
from typing import Tuple, Union

import numpy as np
from numpy import ndarray
from rasterio.coords import BoundingBox  # type: ignore


def _grid_from_corners(v: Tuple[Union[float, ndarray], ...], shape: Tuple[int, int]):
    """
    function returns an linearly interpolated grid from values at the corners

    Parameters
    ----------
    v : tuple
        List of four numeric values that will be interpolated. Order of the values is clockwise; starting from the upper
        left corner.
    shape : tuple
        List of two integers, determining the shape of the array that will be returned

    Returns
    -------
    :obj:`~numpy.ndarray`
        Interpolated array
    """
    return np.linspace(np.linspace(v[0], v[1], shape[1]),
                       np.linspace(v[3], v[2], shape[1]), shape[0])


def add_method(name, *cls):
    """
    Decorator to add functions to existing classes

    Parameters
    ----------
    name : str
        function name when implemented
    cls : class or list of classes
        class or classes that the function that the wrapper is placed around will be added to

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
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still be used normally

    return decorator


def check_increasing_and_unique(v: np.ndarray) -> None:
    vs = np.sort(np.unique(v))
    if not np.array_equal(v, vs):
        raise ValueError("Levels are not sorted or contain double entries")


def change_between_bounds_and_extent(x: Union[BoundingBox, Tuple[float, float, float, float]]):
    return x[0], x[2], x[1], x[3]

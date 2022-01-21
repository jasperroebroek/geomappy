#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities used in geomappy. Interesting functions to use outside the internal scope are
progress_bar and reproject_map_like
"""
from functools import wraps
import numpy as np


def grid_from_corners(v, shape):
    """
    function returns an linearly interpolated grid from values at the corners

    Parameters
    ----------
    v : list
        List of four numeric values that will be interpolated. Order of the values is clockwise; starting from the upper
        left corner.
    shape : list
        List of two integers, determining the shape of the array that will be returned

    Returns
    -------
    :obj:`~numpy.ndarray`
        Interpolated array
    """
    if type(v) not in (tuple, list, np.ndarray):
        raise TypeError("corner values need to be supplied in a list")
    else:
        if len(v) != 4:
            raise ValueError("Four corner values need to be given")
    if type(shape) not in (tuple, list):
        raise TypeError("shape needs to be a list")
    else:
        if len(shape) != 2:
            raise ValueError("shape needs to contain 2 integers")
        if type(shape[0]) not in (float, int) or type(shape[1]) not in (float, int):
            raise TypeError("shape needs to be provided as integers")

    grid = np.linspace(np.linspace(v[0], v[1], shape[1]),
                       np.linspace(v[3], v[2], shape[1]), shape[0])
    return grid


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
        return func # returning func means func can still be used normally
    return decorator

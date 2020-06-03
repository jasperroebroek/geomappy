#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progress bar
"""
import os
import sys
from functools import wraps

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def progress_bar(s, frac=True, line=True, width=20, prefix="", silent=True):
    """
    Taking either a percentage or a fraction and prints a progress bar that updates the line every time the function
    is called.
    
    Parameters
    ----------
    s : float
        Fraction or percentage of progress.
    frac : bool, optional
        If true, s is taken as a fraction, otherwise s is taken as a percentage which is the default
    line : bool, optional
        If True a line of '#' is printed to visually check progress. If False only the percentage is printed.
    width : int, optional
        Width of the line of #. The default is 20
    prefix : str, optional
        String to be placed in front of the progress bar. The default is empty.
    silent : bool, optional
        If True this function doesn't return anything, which is the default. If false the print statement is returned to
        be able to add something before printing it.
        
    Returns
    -------        
    string : None or str
        the string that is outputted to the screen for the purpose of adding something after the string. It can be put
        before but one needs to take into account the '\\\\r' as the first character. Use string[1:] to remove it.
        If `silent` is set to True, nothing is returned. This behaviour is the default.

    Examples
    --------
        >>> for pct in range(0, 101):
                progress_bar(pct, frac = False)
                time.sleep(0.1)
        >>> |###           | 15%
            
        >>> for pct in range(0, 101):
                pct /= 100
                progress_bar(pct, line = False)
                time.sleep(0.1)
        >>> 15%
    """
    
    if frac:
        s *= 100
    s = int(s)
    width = int(width)
    
    f = 100//width
    w = 100//f

    # \r will put the cursor back at the beginning of the previous
    # print statement, if no carriage return has passed
    string = f"\r{str(prefix)}"
    
    if line:
        string += f"|{s//f * '#'}{(w - s//f) * ' '}| "
    
    string += f"{s:3}%"
    
    print(string, end="")
    
    if not silent:
        return string


def update_line(w_str):
    """
    Taking a string and overwrites (flushes) the previously printed line
    
    Parameters
    ----------
    w_str : str
        String to print
    """
    w_str = str(w_str)
    sys.stdout.write("\b" * len(w_str))
    sys.stdout.write(" " * len(w_str))
    sys.stdout.write("\b" * len(w_str))
    sys.stdout.write(w_str)
    sys.stdout.flush()


def reproject_map_like(input_map=None, ref_map=None, output_map=None, resampling=Resampling.bilinear, dtype=None):
    """
    Reprojecting a map like a reference map.

    Parameters
    ----------
    input_map : str
        Path to the input map
    ref_map : str
        todo; make to accept profile directly
        Path to the reference map where the profile is pulled from
    output_map : str
        Path to the location where the transformed map is written to.
    resampling : `rio.Resampling`
        resampling strategy
    dtype : `numpy.dtype`, optional
        export dtype

    Raises
    ------
    IOError
        reference map or input map not found or output_map already exists
    """
    # todo; make it possible to write several bands instead of just one
    # todo; make it accept Map
    if not os.path.isfile(ref_map):
        raise IOError("no reference map provided")
    if not os.path.isfile(input_map):
        raise IOError("no input map provided")
    if os.path.isfile(output_map):
        raise IOError("output file name already exists")

    with rio.open(ref_map) as ref_file:
        ref_transform = ref_file.transform
        ref_crs = ref_file.crs
        shape = ref_file.shape
        profile = ref_file.profile

    with rio.open(input_map) as src:
        current_transform = src.transform
        current_crs = src.crs
        nodata = src.profile['nodata']
        if isinstance(dtype, type(None)):
            dtype = src.profile['dtype']

        if isinstance(current_crs, type(None)):
            print("input map does not have a CRS. Therefore the crs of the reference map is assumed")
            current_crs = ref_crs

        new_map = np.ndarray(shape, dtype=dtype)
        print("start reprojecting")
        reproject(src.read(1), new_map,
                  src_transform=current_transform,
                  src_crs=current_crs,
                  dst_transform=ref_transform,
                  dst_crs=ref_crs,
                  src_nodata=nodata,
                  dst_nodata=nodata,
                  resampling=resampling)

    print("writing file")
    profile.update({"dtype": str(new_map.dtype),
                    "driver": "GTiff",
                    "count": 1,
                    "nodata": nodata})

    with rio.open(output_map, "w", **profile) as dst:
        dst.write(new_map, 1)
    print("reprojection completed")


def export_map_like(m, ref_map=None, output_map=None):
    """
    Exporting a map, taking the needed parameters from a reference_map

    Parameters
    ----------
    m : :obj:`~numpy.ndarray`
        2D input map
    ref_map : str
        Path of the reference map
        todo; make the direct insertion of a rasterio profile possible
    output_map : str
        Path where the data will be written to. If extension is not given it will be added (.tif)

    Raises
    ------
    TypeError
        if no input data is given
    ValueError
        if data is not 2D
    IOError
        if output_map location already exists
    """
    if type(m) != np.ndarray:
        raise TypeError("Input data should be a ndarray")
    if m.ndim != 2:
        raise ValueError("map should be a 2D ndarray")

    # reference map
    with rio.open(ref_map) as src:
        ref_transform = src.transform
        ref_crs = src.crs
        shape = (src.height, src.width)

    # todo; create functionality to overwrite the file if needed
    if os.path.isfile(output_map):
        raise IOError("Output location already exists")

    # if file extension is not given or is not tif, it will be added
    if output_map[-4:] != ".tif":
        output_map = output_map.strip() + ".tif"

    # create the new file
    with rio.open(output_map, "w", driver="GTiff", dtype=str(m.dtype), height=shape[0], width=shape[1],
                  crs=ref_crs, count=1, transform=ref_transform) as src:
        # todo; more than one layer
        src.write(m, 1)


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


def overlapping_arrays(m, preserve_input=True):
    """
    Taking two maps of the same shape and returns them with  all the cells that don't exist in the other set to np.nan

    Parameters
    ----------
    m: iterable of :obj:`~numpy.ndarray`
        list of a minimum of 2 identically shaped numpy arrays.
    preserve_input : bool, optional
        if set to True the data is copied before applying the mask to preserve the input arrays. If set to False the
        memory space of the input arrays will be used.

    Returns
    -------
    m : list of :obj:`~numpy.ndarray`

    Raises
    ------
    TypeError
        if arrays are unreadable
    """
    if len(m) < 2:
        raise IndexError("list needs to contain a minimum of two arrays")

    for a in m:
        if not isinstance(a, np.ndarray):
            raise TypeError("all entries in the list need to be ndarrays")

    if not np.all([a.shape == m[0].shape for a in m]):
        raise ValueError("arrays are not of the same size")

    for a in m:
        if a.dtype != np.float64:
            a = a.astype(np.float64)
        elif preserve_input:
            a = a.copy()

    valid_cells = np.logical_and.reduce([~np.isnan(a) for a in m])

    for a in m:
        a[~valid_cells] = np.nan

    return m


def nanunique(m, **kwargs):
    """
    Equivalent of np.unique while not touching NaN values. Returned ndarray has dtype float64

    Parameters
    ----------
    m : array
        input array
    **kwargs : dict, optional
        kwargs for np.unique function

    Returns
    -------
    :obj:`~numpy.ndarray`
        array has dtype np.float64

    """
    return np.unique(m[~np.isnan(m)], **kwargs)


def nandigitize(m, bins, **kwargs):
    """
    Equivalent of np.digitize while not touching NaN values. Returned ndarray has dtype float64.

    Parameters
    ----------
    m : :obj:`~numpy.ndarray`
        input array
    bins : list
        ascending list of values on which the input array `a` is digitized. Look at `numpy.digitize` documentation
    **kwargs : dict
        keyword arguments to be passed to `numpy.digitize`

    Returns
    -------
    :obj:`~numpy.ndarray`
        array has dtype np.float64, shape is the same as the input array.
    """
    mask = np.isnan(m)
    m_digitized = np.digitize(m.copy(), bins=bins, **kwargs).astype(np.float64)
    m_digitized[mask] = np.nan
    return m_digitized


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
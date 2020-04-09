#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:28:12 2019
@author: jroebroek
"""
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

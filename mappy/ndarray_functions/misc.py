#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:28:12 2019
@author: jroebroek
"""
import numpy as np
import matplotlib.pyplot as plt
# todo; circular references
# from plotting import add_colorbar


def grid_from_corners(v, shape, plotting=True, show=True):
    """
    function returns an linearly interpolated grid from values at the corners
    
    Parameters
    ----------
    v : list
        List of four numeric values that will be interpolated. Order of the values is clockwise; starting from the upper
        left corner.
    shape : list
        List of two integers, determining the shape of the array that will be returned
    plotting : bool, optional
        Plot the created array. Default is True.
    # todo; remove parameter
    show : bool, optional
        Execute the plt.show() command. Default is True.
    
    Returns
    -------
    :obj:`~numpy.ndarray`
        Interpolated array
    """

    if type(plotting) != bool:
        raise TypeError("plotting is a boolean variable")
    if type(show) != bool:
        raise TypeError("show is a boolean variable")
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
    if plotting:
        im = plt.imshow(grid, aspect="auto")
        from mappy.plotting import add_colorbar
        add_colorbar(im)
        if show:
            plt.show()

    return grid


def overlapping_arrays(map1, map2):
    """
    Taking two maps of the same shape and returns them with  all the cells that don't exist in the other set to np.nan

    Parameters
    ----------
    map1, map2 : :obj:`~numpy.ndarray`
        two identically shaped array objects

    Returns
    -------
    map1, map2 : :obj:`~numpy.ndarray`
        two identically shaped array objects of dtype np.float64

    Raises
    ------
    TypeError
        if arrays are unreadable
    ValueError
        if maps don't match or are not of dtype np.float64
    """
    if not isinstance(map1, np.ndarray):
        raise TypeError("Map1 not understood")
    if not isinstance(map2, np.ndarray):
        raise TypeError("Map2 not understood")

    if map1.shape != map2.shape:
        raise ValueError("Maps don't exist or are not of the same size")
    if map1.dtype != np.float64:
        raise ValueError("Can't perform on map1, not formatted as float64")
    if map2.dtype != np.float64:
        raise ValueError("Can't perform on map2, not formatted as float64")

    map1 = map1.copy()
    map2 = map2.copy()

    valid_cells = np.logical_and(~np.isnan(map1), ~np.isnan(map2))
    map1[~valid_cells] = np.nan
    map2[~valid_cells] = np.nan

    return map1, map2


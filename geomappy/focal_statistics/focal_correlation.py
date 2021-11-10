#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm to correlate two arrays (2D) with each other.
"""

import time
import warnings

import numpy as np
from numpy.lib.index_tricks import s_
from ..rolling import rolling_window, rolling_sum


def _correlate_maps_input_checks(map1, map2, window_size, fraction_accepted, reduce, verbose):
    """
    Input checks for correlate_maps. Check their docstring for input requirements.
    """
    if not isinstance(verbose, bool):
        raise TypeError("verbose is a boolean variable")

    if not isinstance(reduce, bool):
        raise TypeError("reduce is a boolean variable")

    if map1.ndim != 2:
        raise IndexError("Only two dimensional arrays are supported")
    if map2.ndim != 2:
        raise IndexError("Only two dimensional arrays are supported")
    if map1.shape != map2.shape:
        raise IndexError(f"Different shapes: {map1.shape}, {map2.shape}")

    if not isinstance(window_size, int):
        raise TypeError("window_size should be an integer")
    elif window_size < 2:
        raise ValueError("window_size should be uneven and bigger than or equal to 2")

    if np.any(np.array(map1.shape) < window_size):
        raise ValueError("window is bigger than the input array on at least one of the dimensions")

    if reduce:
        if ~np.all(np.array(map1.shape) % window_size == 0):
            raise ValueError("The reduce parameter only works when providing a window_size that divedes the input "
                             "exactly.")
    else:
        if window_size % 2 == 0:
            raise ValueError("window_size should be uneven if reduce is set to False")

    if not isinstance(fraction_accepted, (int, float)):
        raise TypeError("fraction_accepted should be numeric")
    elif fraction_accepted < 0 or fraction_accepted > 1:
        raise ValueError("fraction_accepted should be between 0 and 1")


def correlate_maps_base(map1, map2, *, window_size=5, fraction_accepted=0.7, reduce=False, verbose=False):
    """
    Takes two rasters and returning the local correlation between them. Correlation is calculated in a rolling window with
    the size `window_size`. If either of the input rasters contains NaN value on a location, the output raster will also
    have a NaN on that location. Fraction_accepted is used to define a minimum necessary fraction of values in a window for
    the calculation to be done.

    Parameters
    ----------
    a, b : array-like
        Input arrays that will be correlated. If not present in dtype `np.float64` it will be converted internally. They
        have exatly the same shape and have two dimensions.
    window_size : int, optional
        Size of the window used for the correlation calculations. It should be bigger than 1, the default is 5.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.
    reduce : bool, optional
        Reuse all cells exactly once by setting a stepsize of the same size as window_size. The resulting raster will have
        the shape: shape/window_size
    verbose ; bool, optional
        Times the correlation calculations

    Returns
    -------
    corr : :obj:`~numpy.ndarray`
        numpy array of the local correlation. If reduce is set to False, the output has the same shape as the input raster,
        while if reduce is True, the output is reduce by the window size: shape // window_size.
    """
    # Input checks
    _correlate_maps_input_checks(map1, map2, window_size, fraction_accepted, reduce, verbose)

    if verbose:
        print("testing validity of request")

    if reduce:
        raise NotImplementedError("Reduction option is currently not implemented for the numpy function. Compile the "
                                  "cython version to obtain this functionality")

    map1 = map1.astype(np.float64)
    map2 = map2.astype(np.float64)

    # overlapping the maps
    nans = np.logical_or(np.isnan(map1), np.isnan(map2))
    map1[nans] = np.nan
    map2[nans] = np.nan

    # parameters
    fringe = window_size // 2  # count of neighbouring cells in each window
    ind_inner = s_[fringe:-fringe, fringe:-fringe]

    # prepare storage for correlation maps
    corr = np.full(map1.shape, np.nan)

    # Start main calculation
    start = time.time()

    # create the windowed view on the data. These are views, no copies
    map1_view = rolling_window(map1, window_size)
    map2_view = rolling_window(map2, window_size)

    # boolean mask if values are present
    values = ~np.isnan(map1)

    # sum up the boolean mask in the windows to get the amount of values
    count_values = rolling_sum(values, window_size)

    # remove cases from count_values where the original cell was NaN and where there are too many NaNs present
    count_values[count_values < fraction_accepted * window_size ** 2] = 0
    count_values[~values[ind_inner]] = 0
    valid_cells = count_values > 0

    if valid_cells.sum() == 0:
        if verbose:
            print("- empty tile")
        return corr

    if verbose:
        print(f"- preparation: {time.time() - start}")

    # create a focal statistics mean map
    map1[nans] = 0
    map2[nans] = 0

    map1_sum = rolling_sum(map1, window_size)
    map1_mean = np.divide(map1_sum, count_values, where=valid_cells, out=map1_sum)

    map2_sum = rolling_sum(map2, window_size)
    map2_mean = np.divide(map2_sum, count_values, where=valid_cells, out=map2_sum)

    # add empty dimensions to make it possible to broadcast
    map1_mean = map1_mean[:, :, np.newaxis, np.newaxis]
    map2_mean = map2_mean[:, :, np.newaxis, np.newaxis]

    if verbose:
        print(f"- mean: {time.time() - start}")

    # subtract all values from the mean map, with a sampling mask to prevent nan operations. map1/2_dist will therefore
    # not contain any NaNs but only zero because of np.full is 0 initialisation
    sampling_mask = np.logical_and(valid_cells[:, :, np.newaxis, np.newaxis],
                                   rolling_window(values, window_size))
    shape = (*count_values.shape, window_size, window_size)

    map1_dist = np.subtract(map1_view, map1_mean, where=sampling_mask,
                            out=np.full(shape, 0, dtype=np.float64))
    map2_dist = np.subtract(map2_view, map2_mean, where=sampling_mask,
                            out=np.full(shape, 0, dtype=np.float64))

    # add empty dimensions (shape=1) to make it possible to broadcast
    map1_dist = map1_dist.reshape(*valid_cells.shape, window_size ** 2)
    map2_dist = map2_dist.reshape(*valid_cells.shape, window_size ** 2)

    if verbose:
        print(f"- distances: {time.time() - start}")

    # calculate the numerator and denominator of the correlation formula
    r_num = np.sum(map1_dist * map2_dist, axis=2)
    r_den = np.sqrt(np.sum(map1_dist ** 2, axis=2) * np.sum(map2_dist ** 2, axis=2),
                    where=valid_cells, out=corr[ind_inner])

    # insert the correlation (r_num/r_den) into the predefined correlation map
    corr_inner = np.divide(r_num, r_den, where=np.logical_and(valid_cells, r_den != 0), out=corr[ind_inner])

    corr_inner[np.where(r_den == 0)] = 0

    # End main calculation
    if verbose:
        print(f"- correlation: {time.time() - start}")

    return corr

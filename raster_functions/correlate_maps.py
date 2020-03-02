#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algorithm to correlate two arrays (2D) with each other. All implementations here should yield an equal result. For the
rest of the packages the correlate_maps functions is implemented, which is the most heavily tested one.
correlate_maps_simple is the easiest implementation and used as a baseline. The two *opt versions are optimised and
should run faster. This still requires some testing though.
"""

import numpy as np
from numpy.lib.index_tricks import s_
from scipy.stats import pearsonr

from ndarray_functions.rolling_functions import rolling_window, rolling_sum
from ndarray_functions.misc import overlapping_arrays
from raster_functions.focal_statistics import focal_statistics
import time


def correlate_maps(map1, map2, *, window_size=5, fraction_accepted=0.7, verbose=False):
    """
    Takes two maps and returning the local correlation between them with the same dimensions as the input maps. 
    Correlation calculated in a rolling window with the size `window_size`. If either of the input maps contains 
    a NaN value on a location, the output map will also have a NaN on that location.
    todo; make this behaviour optional
    
    Parameters
    ----------
    map1, map2 : array-like
        Input arrays that will be correlated. If not present in dtype `np.float64` it will be converted internally.
    window_size : int, optional
        Size of the window used for the correlation calculations. It should be bigger than 1, the default is 5.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.
    verbose : bool, optional
        Printing progress. False by default.
   
    Returns
    -------
    corr : :obj:`~numpy.ndarray`
        numpy array of the same shape as map1 and map2 with the local correlation
    
    Raises
    ------
    ValueError
        - When data is not 2D
        - When `window_size` is bigger than one of the dimensions of the input
          data
    """
    # todo; 3D

    if verbose:
        print("testing validity of request")

    if type(map1) != np.ndarray:
        raise TypeError("Map1 not a numpy array")
    if type(map2) != np.ndarray:
        raise TypeError("Map2 not a numpy array")

    if map1.ndim != 2:
        raise ValueError("This function only supports 2D input data")

    # check if the two maps overlap
    if map1.shape != map2.shape:
        raise ValueError(f"Different shapes: {map1.shape},{map2.shape}")

    window_size = int(window_size)
    if window_size < 3:
        raise ValueError("Window_size needs to be bigger than 1")
    if window_size % 2 == 0:
        raise ValueError("Window_size should be an uneven nuber")

    if ~np.all(np.array(map1.shape) >= window_size):
        raise ValueError("Window bigger than input array")

    if map1.dtype != np.float64:
        # print in log mode
        # print("map1 converted to float")
        map1 = map1.astype(np.float64)
    if map2.dtype != np.float64:
        # print in log mode
        # print("map2 converted to float")
        map2 = map2.astype(np.float64)

    # overlapping the maps
    map1, map2 = overlapping_arrays(map1, map2)

    if type(verbose) != bool:
        raise TypeError("Verbose has to be a boolean variable")

    # parameters
    fringe = window_size // 2  # count of neighbouring cells in each window
    ind_inner = s_[fringe:-fringe, fringe:-fringe]

    # prepare storage for correlation maps
    corr = np.full(map1.shape, np.nan)

    # Start main calculiation
    start = time.time()

    # create the windowed view on the data. These are views, no copies
    # todo; create a reduction option like in focal_statistics
    map1_view = rolling_window(map1, window_size)
    map2_view = rolling_window(map2, window_size)

    # boolean mask if values are present
    values = ~np.isnan(map1)

    # sum up the boolean mask in the windows to get the amount of values
    count_values = rolling_window(values, window_size).sum(axis=(2, 3))

    # remove cases from count_values where the original cell was NaN and
    # where there are too many NaNs present
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
    map1_mean = focal_statistics(map1, window_size=window_size, func="nanmean",
                                 count_values=count_values)[ind_inner]
    map2_mean = focal_statistics(map2, window_size=window_size, func="nanmean",
                                 count_values=count_values)[ind_inner]

    # add empty dimensions to make it possible to broadcast
    map1_mean = map1_mean[:, :, np.newaxis, np.newaxis]
    map2_mean = map2_mean[:, :, np.newaxis, np.newaxis]

    if verbose:
        print(f"- mean: {time.time() - start}")

    # subtract all values from the mean map, with a sampling mask to prevent
    # nan operations. map1/2_dist will therefore not contain any NaNs but only
    # zero because of np.full is 0 initialisation
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
                    where=valid_cells, out=np.full_like(r_num, 0, dtype="float64"))

    # insert the correlation (r_num/r_den) into the predefined correlation map
    # todo; creates a warning when deviding by 0, this can be prevented by adding
    #   the condition to valid_cells, but be sure to test it!
    corr_inner = np.divide(r_num, r_den, where=valid_cells, out=corr[ind_inner])

    # find places where all the values in a window caused a NaN
    # this happens when all values are equal, causing r_den to become 0
    # and the division causes problems
    corr_mask = np.logical_and(np.isnan(corr[ind_inner]), valid_cells)
    corr_inner[corr_mask] = 0

    # End main calculation
    if verbose:
        print(f"- correlation: {time.time() - start}")

    return corr


def correlate_maps_simple(map1, map2, window_size=5, fraction_accepted=0.7):
    """
    Takes two maps and returning the local correlation between them with the same dimensions as the input maps.
    Correlation calculated in a rolling window with the size `window_size`. If either of the input maps contains
    a NaN value on a location, the output map will also have a NaN on that location.

    Parameters
    ----------
    map1, map2 : array-like
        Input arrays that will be correlated. If not present in dtype `np.float64` it will be converted internally.
    window_size : int, optional
        Size of the window used for the correlation calculations. It should be bigger than 1, the default is 5.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.

    Returns
    -------
    corr : :obj:`~numpy.ndarray`
        numpy array of the same shape as map1 and map2 with the local correlation

    Raises
    ------
    ValueError
        - When data is not 2D
        - When `window_size` is bigger than one of the dimensions of the input
          data
    """

    map1, map2 = overlapping_arrays(map1.astype(np.float64), map2.astype(np.float64))
    fringe = window_size // 2
    corr = np.full(map1.shape, np.nan)
    for i in range(fringe, map1.shape[0] - fringe):
        for j in range(fringe, map1.shape[1] - fringe):
            ind = s_[i - fringe:i + fringe + 1, j - fringe:j + fringe + 1]

            if np.isnan(map1[i, j]):
                continue

            d1 = map1[ind].flatten()
            d2 = map2[ind].flatten()

            d1 = d1[~np.isnan(d1)]
            d2 = d2[~np.isnan(d2)]

            if d1.size < fraction_accepted * window_size ** 2:
                continue

            if np.all(d1 == d1[0]) or np.all(d2 == d2[0]):
                corr[i, j] = 0
                continue

            corr[i, j] = pearsonr(d1, d2)[0]

    return corr


def correlate_maps_opt(map1, map2, *, window_size=5, fraction_accepted=0.7, verbose=False, preserve_input=True):
    """
    Takes two maps and returning the local correlation between them with the same dimensions as the input maps.
    Correlation calculated in a rolling window with the size `window_size`. If either of the input maps contains
    a NaN value on a location, the output map will also have a NaN on that location. This is the first trial of
    optimisation, by not linking back to focal_statistics wrapper and optimising memory usage in performing inplace
    operations.

    Parameters
    ----------
    map1, map2 : array-like
        Input arrays that will be correlated. If not present in dtype `np.float64` it will be converted internally.
    window_size : int, optional
        Size of the window used for the correlation calculations. It should be bigger than 1, the default is 5.
    fraction_accepted : float, optional
        Fraction of the window that has to contain not-nans for the function to calculate the correlation. The default
        is 0.7.
    verbose : bool, optional
        Printing progress. False by default.

    Returns
    -------
    corr : :obj:`~numpy.ndarray`
        numpy array of the same shape as map1 and map2 with the local correlation

    Raises
    ------
    ValueError
        - When data is not 2D
        - When `window_size` is bigger than one of the dimensions of the input
          data
    """
    if verbose: print("testing validity of request")

    if type(map1) != np.ndarray:
        raise TypeError("Map1 not a numpy array")
    if type(map2) != np.ndarray:
        raise TypeError("Map2 not a numpy array")

    if map1.ndim != 2:
        raise ValueError("This function only supports 2D input data")
    # todo; This might be improved in the future. It might already work but some
    #   testing needs to happen first.

    # check if the two maps overlap
    if map1.shape != map2.shape:
        raise ValueError(f"Different shapes: {map1.shape},{map2.shape}")

    window_size = int(window_size)
    if window_size < 3:
        raise ValueError("Window_size needs to be bigger than 1")
    if window_size % 2 == 0:
        raise ValueError("Window_size should be an uneven nuber")

    if ~np.all(np.array(map1.shape) > window_size):
        raise ValueError("Window bigger than input array")

    if map1.dtype != np.float64:
        # print in log mode
        # print("map1 converted to float")
        map1 = map1.astype(np.float64)
    elif preserve_input:
        map1 = map1.copy()

    if map2.dtype != np.float64:
        # print in log mode
        # print("map2 converted to float")
        map2 = map2.astype(np.float64)
    elif preserve_input:
        map2 = map2.copy()

    # overlapping the maps
    nans = np.logical_or(np.isnan(map1), np.isnan(map2))
    map1[nans] = np.nan
    map2[nans] = np.nan

    if type(verbose) != bool:
        raise TypeError("Verbose has to be a boolean variable")

    # parameters
    fringe = window_size // 2  # count of neighbouring cells in each window
    ind_inner = s_[fringe:-fringe, fringe:-fringe]

    # prepare storage for correlation maps
    corr = np.full(map1.shape, np.nan)

    # Start main calculation
    start = time.time()

    # create the windowed view on the data. These are views, no copies
    # todo; create a reduction option like in focal_statistics
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

    # todo; this likely is referencing nan values because it is written to the same memory space
    corr_inner[np.where(r_den == 0)] = 0

    # End main calculation
    if verbose:
        print(f"- correlation: {time.time() - start}")

    return corr
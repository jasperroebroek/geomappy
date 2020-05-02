#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.lib.index_tricks import s_
import time
from ..ndarray_functions.rolling_functions import rolling_window, rolling_sum, rolling_mean


def _focal_majority(a, window_size, fraction_accepted, reduce, r, ind_inner, majority_mode):
    if a.dtype not in ('float32', 'float64'):
        a = a.astype(np.float64)

    values = np.unique(a)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return r

    count_values = rolling_sum(values, window_size, reduce=reduce)
    if not reduce:
        count_values[~values[ind_inner]] = 0
    count_values[count_values < fraction_accepted * (window_size ** 2)] = 0

    # highest digit corresponds to nan
    digitized_a = np.digitize(a, values, right=True)

    digitized_a_view = rolling_window(digitized_a, window_size=window_size, flatten=True, reduce=reduce)

    value_count = np.apply_along_axis(lambda p: np.bincount(p, minlength=len(values)+1), axis=2, arr=digitized_a_view)

    if majority_mode == "ascending":
        t = values[value_count[:, :, :-1].argmax(axis=2)]
    elif majority_mode == "descending":
        t = values[::-1][value_count[:, :, :-1][:, :, ::-1].argmax(axis=2)]
    elif majority_mode == "nan":
        t = values[value_count[:, :, :-1].argmax(axis=2)]
        occurrence_maximum = value_count[:, :, :-1].max(axis=2)
        mask_two_maxima = (value_count[:, :, :-1] == occurrence_maximum[:, :, np.newaxis]).sum(axis=2) > 1
        t[mask_two_maxima] = np.nan

    t[count_values == 0] = np.nan
    r[ind_inner] = t
    return r


def _focal_mean(a, window_size, fraction_accepted, reduce, r, ind_inner, values):
    count_values = rolling_sum(values, window_size, reduce=reduce)
    if not reduce:
        count_values[~values[ind_inner]] = 0
    count_values[count_values < fraction_accepted * (window_size ** 2)] = 0

    a[~values] = 0
    a_sum = rolling_sum(a, window_size, reduce=reduce)
    a_mean = np.divide(a_sum, count_values, out=r[ind_inner], where=(count_values > 0))

    return r


def _focal_nanmax(a, window_size, fraction_accepted, reduce, r, ind_inner, values):
    count_values = rolling_sum(values, window_size, reduce=reduce)
    if not reduce:
        count_values[~values[ind_inner]] = 0
    count_values[count_values < fraction_accepted * (window_size ** 2)] = 0

    a[np.isnan(a)] = -np.inf
    a_view = rolling_window(a, window_size, reduce=reduce)
    r[ind_inner] = np.max(a_view, axis=(2, 3), out=r[ind_inner])

    r[np.isinf(r)] = np.nan
    r[ind_inner][count_values == 0] = np.nan
    return r


def _focal_nanmin(a, window_size, fraction_accepted, reduce, r, ind_inner, values):
    count_values = rolling_sum(values, window_size, reduce=reduce)
    if not reduce:
        count_values[~values[ind_inner]] = 0
    count_values[count_values < fraction_accepted * (window_size ** 2)] = 0

    a[np.isnan(a)] = np.inf
    a_view = rolling_window(a, window_size, reduce=reduce)
    r[ind_inner] = np.min(a_view, axis=(2, 3), out=r[ind_inner])

    r[np.isinf(r)] = np.nan
    r[ind_inner][count_values == 0] = np.nan
    return r


def _focal_std(a, window_size, fraction_accepted, reduce, std_df, r, ind_inner, values):
    count_values = rolling_sum(values, window_size, reduce=reduce)
    if not reduce:
        count_values[~values[ind_inner]] = 0
    count_values[count_values < fraction_accepted * (window_size ** 2)] = 0
    valid_cells = count_values > 0

    a_mean = focal_statistics(a, window_size=window_size, func="mean", fraction_accepted=fraction_accepted,
                              reduce=reduce)[ind_inner]

    # add empty dimensions to make it possible to broadcast
    a_mean = a_mean[:, :, np.newaxis, np.newaxis]

    # subtract all values from the mean map, with a sampling mask to prevent
    # nan operations. map1_dist will therefore not contain any NaNs
    sampling_mask = np.logical_and(valid_cells[:, :, np.newaxis, np.newaxis],
                                   rolling_window(values, window_size, reduce=reduce))
    shape = (*count_values.shape, window_size, window_size)
    a_view = rolling_window(a, window_size, reduce=reduce)
    a_dist = np.subtract(a_view, a_mean, where=sampling_mask,
                         out=np.full(shape, 0, dtype="float64"))

    if std_df == 1:
        # denominator (count_values - 1), but no negative values allowed
        count_values = np.maximum(count_values - 1, [[0]])

    r[ind_inner] = np.sqrt(np.divide(np.sum(a_dist ** 2, axis=(2, 3)), count_values, where=valid_cells),
                           where=valid_cells, out=r[ind_inner])

    return r


def focal_statistics(a, window_size, *, func=None, fraction_accepted=0.7, verbose=False, std_df=1, reduce=False,
                     majority_mode="nan"):
    """
    Focal statistics wrapper.
    
    Parameters
    ----------
    a : :obj:`~numpy.ndarray`
        Input array (2D). If not np.float64 it will be converted internally, except when the majority function is called
        in which case the input array's dtype is preserved.
    window_size : int
        Window size for focal statistics. Should be bigger than 1.
    func : {"mean","min","max","std","majority"}
        Function to be applied on the windows
    fraction_accepted : float, optional
        Fraction of valid cells per window that is deemed acceptable
        0: all window are calculated if at least 1 value is present
        1: only windows completely filled with values are calculated
        0-1: fraction of acceptability
        Note that this parameter has no effect yet when calculating with 'nanmax' and 'nanmin'
    verbose : bool, optional
        Verbosity with timing. False by default
    std_df : {1,0}, optional
        Only for nanstd and std calculations. Degrees of freedom; meaning if the function is devided by the count of
        observations or the count of observations minus one. Should be 0 or 1.
    reduce : bool, optional
        The way in which the windowed array is created. If true, all values are used exactly once. If False (which is
        the default), values are reduced and the output array has the same shape as the input array, albeit with a
        border of nans where there are not enough values to calculate the cells.
    majority_mode : {"nan", "ascending", "descending"}, optional
        nan: when more than one class has the same score NaN will be assigned
        ascending: the first occurrence of the maximum count will be assigned
        descending: the last occurrence of the maximum count will be assigned.
        Parameter only used when the `func` is "majority".
        
    Returns
    -------
    :obj:`~numpy.ndarray`
        if `reduce` is False:
            numpy ndarray of the function applied to input array `a`. The shape will
            be the same as the input array. The border of the map will be np.nan, 
            because of the lack of data to calculate the border. In the future other 
            behaviours might be implemented. To obtain only the useful cells the 
            following might be done:
                
                >>> window_size = 5
                >>> fringe = window_size // 2
                >>> ind_inner = s_[fringe:-fringe, fringe:-fringe]
                >>> a = a[ind_inner]
                
            in which case a will only contain the cells for which all data was 
            present
        if `reduce` is True:
            numpy ndarray of the function applied on input array `a`. The shape
            will be the original shape divided by the `window_size`. Dimensions
            remain equal. No border of NaN values is present. 
            
    Raises
    ------
    KeyError
        Function not in the list of allowed functions
    ValueError
        If data is not a 2D array
    """
    list_of_functions = ["mean", "majority", "min", "max", "std"]
    if func not in list_of_functions:
        raise KeyError("Function not available")

    if a.dtype != np.float64 and func != "majority":
        # print("input array converted to float")
        a = a.astype(np.float64)

    if std_df not in (0, 1):
        raise ValueError("STD_DF wrongly defined")

    if not isinstance(verbose, bool):
        raise TypeError("verbose is a boolean variable")
    if not isinstance(reduce, bool):
        raise TypeError("reduce is a boolean variable")

    if a.ndim != 2:
        raise ValueError("Only 2D data is supported")

    if not isinstance(window_size, int):
        raise TypeError("window_size should be an integer")
    elif window_size < 2:
        raise ValueError("window_size should be uneven and bigger than or equal to 2")

    if np.any(np.array(a.shape) < window_size):
        raise IndexError("window is bigger than the input array on at least one of the dimensions")

    if reduce:
        if ~np.all(np.array(a.shape) % window_size == 0):
            raise ValueError("The reduce parameter only works when providing a window_size that divides the input "
                             "exactly.")
    else:
        if window_size % 2 == 0:
            raise ValueError("window_size should be uneven if reduce is set to False")

    if not isinstance(fraction_accepted, (int, float)):
        raise TypeError("fraction_accepted should be numeric")
    elif fraction_accepted < 0 or fraction_accepted > 1:
        raise ValueError("fraction_accepted should be between 0 and 1")

    fringe = window_size // 2

    # return array
    if not reduce:
        r = np.full(a.shape, np.nan)
        ind_inner = s_[fringe:-fringe, fringe:-fringe]
    else:
        shape = list(np.array(a.shape) // window_size)
        r = np.full(shape, np.nan)
        ind_inner = s_[:, :]

    nans = np.isnan(a)
    values = ~nans
    if values.sum() == 0:
        if verbose:
            print("- Empty array")
        return r

    start = time.time()

    if nans.sum() == 0:
        nan_flag = False
    else:
        nan_flag = True
        # todo; remove for numba
        a = a.copy()

    if func == "majority":
        r = _focal_majority(a, window_size, fraction_accepted, reduce, r, ind_inner, majority_mode)

    elif func == "mean":
        r = _focal_mean(a, window_size, fraction_accepted, reduce, r, ind_inner, values)

    elif func == "max":
        if nan_flag:
            r = _focal_nanmax(a, window_size, fraction_accepted, reduce, r, ind_inner, values)
        else:
            r[ind_inner] = rolling_window(a, window_size, reduce=reduce).max(axis=(2, 3))

    elif func == "min":
        if nan_flag:
            r = _focal_nanmin(a, window_size, fraction_accepted, reduce, r, ind_inner, values)
        else:
            r[ind_inner] = rolling_window(a, window_size, reduce=reduce).min(axis=(2, 3))

    elif func == "std":
        r = _focal_std(a, window_size, fraction_accepted, reduce, std_df, r, ind_inner, values)

    if verbose:
        print(f"- mean: {time.time() - start}")

    return r


def focal_min(a, window_size, **kwargs):
    return focal_statistics(a, window_size, func="min", **kwargs)


def focal_max(a, window_size, **kwargs):
    return focal_statistics(a, window_size, func="max", **kwargs)


def focal_mean(a, window_size, **kwargs):
    return focal_statistics(a, window_size, func="mean", **kwargs)


def focal_std(a, window_size, **kwargs):
    return focal_statistics(a, window_size, func="std", **kwargs)


def focal_majority(a, window_size, **kwargs):
    return focal_statistics(a, window_size, func="majority", **kwargs)


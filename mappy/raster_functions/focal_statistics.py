#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.lib.index_tricks import s_
import time
from ..ndarray_functions.rolling_functions import rolling_window


def focal_statistics(a, *, window_size=None, func=None, fraction_accepted=0.7, verbose=False, std_df=1, reduce=False,
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
    func : {"mean","min","max","std","nanmean","nanmin","nanmax","nanstd","majority"}
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
    # todo; use rolling_sum and rolling_mean for speadup of calculations. Be sure to check for consistency
    # todo; split this function up in several functions, this function may remain as a wrapper
    # todo; 3D

    list_of_functions = ["mean", "min", "max", "std", "nanmean", "nanmin", "nanmax", "nanstd", "majority",
                         "majority_old"]

    if func not in list_of_functions:
        raise KeyError("Function not available")

    if isinstance(a, type(None)):
        raise ValueError("No input array given")
    if isinstance(window_size, type(None)):
        raise ValueError("No window size given")

    if not isinstance(a, np.ndarray):
        a = np.array(a)

    if a.dtype != np.float64 and func != "majority":
        # print("input array converted to float")
        a = a.astype(np.float64)

    if not isinstance(window_size, int):
        window_size = int(window_size)

    if window_size % 2 == 0 and not reduce:
        raise ValueError("Window_size should be uneven")
    if window_size < 2:
        raise ValueError(f"Window_size too small. {window_size}")

    if a.ndim != 2:
        raise ValueError("This function only supports 2D data at the moment")

    if std_df not in (0, 1):
        raise ValueError("STD_DF wrongly defined")

    if ~np.all(np.array(a.shape) >= window_size):
        raise ValueError("Window bigger than input array")

    fringe = window_size // 2

    # return array
    if not reduce:
        r = np.full(a.shape, np.nan)
        ind_inner = s_[fringe:-fringe, fringe:-fringe]
    else:
        shape = []
        for dim in a.shape:
            if dim // window_size == dim / window_size:
                shape.append(dim // window_size)
            else:
                raise ValueError(
                    f"Can't reduce with current window_size; not all dimensions diviseble by {window_size}")
        r = np.full(shape, np.nan)
        ind_inner = s_[:, :]

    start = time.time()

    if func[:3] == "nan":
        a = a.copy()

        # find pixels with values
        values = ~np.isnan(a)

        if np.sum(values) == 0:
            if verbose:
                print("- Empty array")
            return r

        # Find the amount of true values in each window
        count_values = rolling_window(values, window_size, reduce=reduce).sum(axis=(2, 3))

        # Remove windows that have too many NaN values
        if not reduce:
            count_values[~values[ind_inner]] = 0

        count_values[count_values < fraction_accepted * (window_size ** 2)] = 0

    # FOCAL STATISTICS FUNCTIONS    
    a_view = rolling_window(a, window_size=window_size, reduce=reduce)

    if func == "mean":
        # return the mean, nan values not accepted and leading to weird behaviour
        r[ind_inner] = a_view.mean(axis=(2, 3))

    if func == "max":
        # return the max, nan values not accepted and leading to weird behaviour
        r[ind_inner] = a_view.max(axis=(2, 3))

    if func == "min":
        # return the min, nan values not accepted and leading to weird behaviour
        r[ind_inner] = a_view.min(axis=(2, 3))

    if func == "std":
        # return the min, nan values not accepted and leading to weird behaviour
        r[ind_inner] = a_view.std(axis=(2, 3), ddof=std_df)

    if func == "nanmean":
        # Return the mean. NaN values are filtered with the fraction_accepted parameter
        # remove the nans
        a[~values] = 0

        if verbose:
            print(f"- preparation: {time.time() - start}")

        # calculate the rolling sum
        a_sum = a_view.sum(axis=(2, 3))

        if verbose:
            print(f"- sum: {time.time() - start}")

        # devide by the amount of values to get the mean
        a_mean = np.divide(a_sum, count_values, out=r[ind_inner], where=(count_values > 0))

        if verbose:
            print(f"- mean: {time.time() - start}")

    if func == "nanstd":
        # create boolean array to mask the valid cells
        valid_cells = count_values > 0

        if verbose:
            print(f"- preparation: {time.time() - start}")

        a_mean = focal_statistics(a, window_size=window_size, func="nanmean",
                                  fraction_accepted=fraction_accepted,
                                  reduce=reduce)[ind_inner]

        # add empty dimensions to make it possible to broadcast
        a_mean = a_mean[:, :, np.newaxis, np.newaxis]

        if verbose:
            print(f"- mean: {time.time() - start}")

        # subtract all values from the mean map, with a sampling mask to prevent
        # nan operations. map1_dist will therefore not contain any NaNs
        sampling_mask = np.logical_and(valid_cells[:, :, np.newaxis, np.newaxis],
                                       rolling_window(values, window_size, reduce=reduce))
        shape = (*count_values.shape, window_size, window_size)
        a_dist = np.subtract(a_view, a_mean, where=sampling_mask,
                             out=np.full(shape, 0, dtype="float64"))

        if verbose:
            print(f"- distance: {time.time() - start}")

        if std_df == 1:
            # denominator (count_values - 1), but no negative values allowed
            count_values = np.maximum(count_values - 1, [[0]])

        # calculate the standard deviation with valid_cells as mask inserting 0 in the divide array and not writing to
        # the output_array in the np.sqrt function
        r[ind_inner] = np.sqrt(np.divide(np.sum(a_dist ** 2, axis=(2, 3)), count_values, where=valid_cells),
                               where=valid_cells, out=r[ind_inner])

        if verbose:
            print(f"- standard deviation: {time.time() - start}")

    if func == "nanmax":
        # TODO; implement accepted_fraction
        # mask the valid cells
        valid_cells = count_values > 0

        # replace the cells with nans to -inf, so this value will never
        # be chosen as the maximum value if there is any other value present
        a[np.isnan(a)] = -np.inf

        # insert the maximum value of each window in the output map
        r[ind_inner] = np.max(a_view, axis=(2, 3), out=r[ind_inner])

        # replace the np.inf values placed in the output map
        r[np.isinf(r)] = np.nan

        # remove values that have a lower amount data than acceptible 
        # calculated with the fraction_accepted parameter
        r[ind_inner][~valid_cells] = np.nan

        if verbose:
            print(f"- calculation: {time.time() - start}")

    if func == "nanmin":
        # TODO; implement accepted_fraction
        # mask the valid cells
        valid_cells = count_values > 0

        # replace the cells with nans to inf, so this value will never
        # be chosen as the minimum value if there is any other value present
        a[np.isnan(a)] = np.inf

        # insert the minimum value of each window in the output map
        r[ind_inner] = np.min(a_view, axis=(2, 3), out=r[ind_inner])

        # replace the np.inf values placed in the output map
        r[np.isinf(r)] = np.nan

        # remove values that have a lower amount data than acceptible 
        # calculated with the fraction_accepted parameter
        r[ind_inner][~valid_cells] = np.nan

        if verbose:
            print(f"- calculation: {time.time() - start}")

    if func == "majority":
        return focal_majority(a=a, window_size=window_size, fraction_accepted=fraction_accepted, reduce=reduce,
                              majority_mode=majority_mode)
    return r


def focal_majority(a, *, window_size=None, fraction_accepted=0.7, reduce=False, majority_mode="nan"):
    if a.dtype not in ('float32', 'float64'):
        a = a.astype(np.float64)

    if isinstance(window_size, type(None)):
        raise ValueError("No window size given")

    if not isinstance(a, np.ndarray):
        a = np.array(a)

    if not isinstance(window_size, int):
        window_size = int(window_size)

    if window_size % 2 == 0 and not reduce:
        raise ValueError("Window_size should be uneven")
    if window_size < 2:
        raise ValueError(f"Window_size too small. {window_size}")

    if a.ndim != 2:
        raise ValueError("This function only supports 2D data at the moment")

    if ~np.all(np.array(a.shape) >= window_size):
        raise ValueError("Window bigger than input array")

    # return array
    if not reduce:
        r = np.full(a.shape, np.nan)
        fringe = window_size // 2
        ind_inner = np.s_[fringe:-fringe, fringe:-fringe]
    else:
        shape = []
        for dim in a.shape:
            if dim // window_size == dim / window_size:
                shape.append(dim // window_size)
            else:
                raise ValueError(
                    f"Can't reduce with current window_size; not all dimensions divisible by {window_size}")
        r = np.full(shape, np.nan)
        ind_inner = np.s_[:, :]

    values = np.unique(a)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return r

    # todo; convert to rolling sum (when reduce option is available)
    count_values = rolling_window(~np.isnan(a), window_size=window_size, reduce=reduce).sum(axis=(2, 3))
    if not reduce:
        count_values[np.isnan(a[ind_inner])] = 0
    count_values[count_values < window_size ** 2 * fraction_accepted] = 0

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


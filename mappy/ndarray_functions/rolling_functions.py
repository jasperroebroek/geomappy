#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to create a windowed view on the input array. Like
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html but available for arrays with
unlimited dimensionality.
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.lib.index_tricks import s_


def rolling_window(a, window_size, *, flatten=False, reduce=False):
    """
    Takes a ndarray and returns a windowed version of that same array
    
    Parameters
    ----------
    a : :obj:`~numpy.ndarray`
        Input array
    window_size : int
        Size of the window that is applied over ndarray a. Should be bigger than 1.
    flatten : bool, optional
        Flag to flatten the windowed view to 1 dimension. The shape of the returned array if set to True will be:
            *reduce* == False:
                shape : [s - window_size + 1 for s in a.shape] + [window_size ^ a.ndim]
            *reduce* == True:
                shape : [s // window_size for s in a.shape] + [window_size ^ a.ndim]
        If set to False (which is the default) the shape of the window will not change and the data will be added in as
        many dimensions as the input array. The shape will be:
            *reduce* == False:
                shape : [s - window_size + 1 for s in a.shape] + [window_size] * a.ndim
            *reduce* == True:
                shape : [s // window_size for s in a.shape] + [window_size] * a.ndim
        False has the nice property of returning a view, not copying the data while if True is passed, all the data will
        be copied. This can be very slow and memory intensive for large arrays.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see *flatten*. If set to true, every entry is used exactly once. Creating  much
        smaller dimensions.
    
    Returns
    -------
    strided_a : :obj:`~numpy.ndarray`
        windowed array of the data
        
    Raises
    ------
    ValueError
        - window_size too bigger than on of the dimensions of the input array
        - if reduce is True, the window_size needs to be an exact divisor for all dimensions of the input array
    """
    # todo; create the possibility of passing a list to window_size to create
    #   different shapes of windows
    # todo; create possibility of passing a ndarray to window_size with booleans to select a specific region
    # todo; even numbers
    # todo; optionally, replace reduce with a step parameter

    if type(a) not in (np.ndarray, np.ma.array):
        a = np.array(a)
    if type(window_size) != int:
        window_size = int(window_size)
        
    if window_size <= 1:
        raise ValueError("Window_size should be bigger than 1") 
        
    if ~np.all(np.array(a.shape) >= window_size):
        raise ValueError("Window bigger than input array")

    if not isinstance(flatten, bool):
        raise TypeError("flatten needs to be a boolean variable")
    if not isinstance(reduce, bool):
        raise TypeError("'reduce' needs to be a boolean variable")

    shape = np.array(a.shape)
    strides = np.array(a.strides)

    if reduce:
        if not np.array_equal(shape//window_size, shape/window_size):
            raise ValueError("not all dimensions are divisible by window_size")

        output_shape = np.r_[shape//window_size, [window_size] * a.ndim]
        output_strides = np.r_[strides * window_size, strides]

    else:
        output_shape = np.r_[shape - window_size + 1, [window_size] * a.ndim]
        output_strides = np.r_[strides, strides]
    
    # create view on the data with new shape and strides
    strided_a = as_strided(a, shape=output_shape, strides=output_strides)
    
    if flatten:
        strided_a = strided_a.reshape((*output_shape[:-a.ndim], -1))
    
    return strided_a


def rolling_mean(a, window_size, reduce=False):
    """
    Takes an ndarray and returns the rolling mean. Not suitable for arrays with NaN values.
    
    Parameters
    ----------
    a : :obj:`~numpy.ndarray`
        input array
    window_size : int
        size of the window that is applied over a. Should be bigger than 1.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see *flatten*. If set to true, every entry is used exactly once. Creating  much
        smaller dimensions.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling mean over array `a`. Shape is slightly smaller than input array
        to buffer the window size on the edges:
            shape : [s - window_size + 1 for s in a.shape]
    """
    # TODO; implement reduce
    return rolling_sum(a, window_size, reduce=reduce)/np.power(window_size, a.ndim)


def rolling_sum(a, window_size, reduce=False):
    """
    Takes an ndarray and returns the rolling sum. Not suitable for arrays with NaN values.

    Parameters
    ----------
    a : :obj:`~numpy.ndarray`
        input array
    window_size : int
        size of the window that is applied over a. Should be bigger than 1.
    reduce : bool, optional
        Reuse data if set to False (which is the default) in which case an array will be returned with dimensions that
        are close to the original; see *flatten*. If set to true, every entry is used exactly once. Creating  much
        smaller dimensions.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Rolling sum over array `a`. Shape is slightly smaller than input array
        to buffer the window size on the edges:
            shape : [s - window_size + 1 for s in a.shape]
    """
    if window_size < 2:
        raise ValueError("window_size should be bigger than 2")

    if ~np.all(np.array(a.shape) >= window_size):
        raise ValueError("Window bigger than input array")

    if a.ndim == 1 and not reduce:
        cumsum = np.cumsum(np.hstack((0, a)))
        return cumsum[window_size:] - cumsum[:-window_size]

    elif a.ndim == 2 and not reduce:
        if a.dtype == np.bool_:
            dtype = np.int
        else:
            dtype = a.dtype

        cumsum = np.zeros([x+1 for x in a.shape], dtype=dtype)
        cumsum[1:, 1:] = a

        for i in range(a.ndim):
            np.cumsum(cumsum[1:, 1:], axis=i, out=cumsum[1:, 1:])

        r = cumsum[window_size:, window_size:] + cumsum[:-window_size, :-window_size]
        np.subtract(r, cumsum[:-window_size, window_size:], out=r)
        np.subtract(r, cumsum[window_size:, :-window_size], out=r)

        return r

    elif a.ndim == 3 and not reduce:
        # todo; check if this is actually faster than the backup solution
        if a.dtype == np.bool_:
            dtype = np.int
        else:
            dtype = a.dtype

        cumsum = np.zeros([x+1 for x in a.shape], dtype=dtype)
        cumsum[(s_[1:],)*a.ndim] = a

        for i in range(a.ndim):
            np.cumsum(cumsum[(s_[1:],)*a.ndim], axis=i, out=cumsum[(s_[1:],)*a.ndim])

        s_sub = s_[:-window_size]
        s_add = s_[window_size:]

        r = cumsum[(s_add,)*a.ndim] - cumsum[(s_sub,)*a.ndim]

        for i in range(a.ndim):
            ind = [s_add]*a.ndim
            ind[i] = s_sub
            ind = tuple(ind)
            np.subtract(r, cumsum[ind], out=r)

        for i in range(a.ndim):
            ind = [s_sub]*a.ndim
            ind[i] = s_add
            ind = tuple(ind)
            np.add(r, cumsum[ind], out=r)

        return r

    else:
        # backup solution for ndarrays that are not covered in the logic above
        return rolling_window(a, window_size, reduce=reduce).sum(axis=tuple(range(a.ndim, 2*a.ndim)))

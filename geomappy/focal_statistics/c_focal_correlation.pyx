# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""
Algorithm to correlate two arrays (2D) with each other
"""

import time
import numpy as np
from .focal_correlation import _correlate_maps_input_checks, correlate_maps_base

cimport numpy as np
from libc.math cimport isnan, sqrt


cdef double[:, ::1] _correlate_maps(double[:, ::1] map1,
                                    double[:, ::1] map2,
                                    int window_size,
                                    float fraction_accepted,
                                    bint reduce,
                                    ):
    cdef:
        int start, stop_x, stop_y, step, shape_x, shape_y, fringe
        int p, q, i, j, x, y
        int[:] x_i, y_j
        double[:, ::1] corr
        double r_num, d1_mean, d2_mean, d1_sum, d2_sum, c1_dist, c2_dist, r_den_d1, r_den_d2
        double num_values, threshold, count_values, first_value1, first_value2
        bint all_equal_d1, all_equal_d2

    if reduce:
        shape_x = map1.shape[0] // window_size
        shape_y = map1.shape[1] // window_size
        fringe = 0
        start = 0
        stop_x = map1.shape[0]
        stop_y = map1.shape[1]
        step = window_size

    else:
        shape_x = map1.shape[0]
        shape_y = map1.shape[1]
        fringe = window_size // 2
        start = 0
        stop_x = map1.shape[0] - 2 * fringe
        stop_y = map1.shape[1] - 2 * fringe
        step = 1

    corr = np.full((shape_x, shape_y), np.nan, dtype=np.float64)
    x_i = np.arange(start, stop_x, step, dtype=np.int32)
    y_j = np.arange(start, stop_y, step, dtype=np.int32)

    num_values = window_size ** 2
    threshold = fraction_accepted * num_values

    for x in range(x_i.shape[0]):
        for y in range(y_j.shape[0]):
            i = x_i[x]
            j = y_j[y]

            if not reduce:
                if isnan(map1[i + fringe, j + fringe]) or isnan(map2[i + fringe, j + fringe]):
                    continue

            d1_sum = 0
            d2_sum = 0
            count_values = 0
            all_equal_d1 = True
            all_equal_d2 = True

            for p in range(window_size):
                for q in range(window_size):
                    if not isnan(map1[i + p, j + q]) and not isnan(map2[i + p, j + q]):
                        if count_values == 0:
                            first_value1 = map1[i + p, j + q]
                            first_value2 = map2[i + p, j + q]
                        d1_sum = d1_sum + map1[i + p, j + q]
                        d2_sum = d2_sum + map2[i + p, j + q]

                        if map1[i + p, j + q] != first_value1:
                            all_equal_d1 = False
                        if map2[i + p, j + q] != first_value2:
                            all_equal_d2 = False

                        count_values = count_values + 1

            if count_values < threshold:
                pass

            elif all_equal_d1 or all_equal_d2:
                corr[(i // step) + fringe, (j // step) + fringe] = 0

            else:
                d1_mean = d1_sum / count_values
                d2_mean = d2_sum / count_values

                r_num = 0
                r_den_d1 = 0
                r_den_d2 = 0

                for p in range(window_size):
                    for q in range(window_size):
                        if not isnan(map1[i + p, j + q]) and not isnan(map2[i + p, j + q]):
                            c1_dist = map1[i + p, j + q] - d1_mean
                            c2_dist = map2[i + p, j + q] - d2_mean

                            r_num = r_num + (c1_dist * c2_dist)
                            r_den_d1 = r_den_d1 + c1_dist ** 2
                            r_den_d2 = r_den_d2 + c2_dist ** 2

                corr[(i // step) + fringe, (j // step) + fringe] = r_num / sqrt(r_den_d1 * r_den_d2)

    return corr


def correlate_maps(map1, map2, window_size=5, fraction_accepted=0.7, reduce=False, verbose=False):
    start = time.time()
    map1 = np.asarray(map1, dtype=np.float64)
    map2 = np.asarray(map2, dtype=np.float64)

    # Input checks
    _correlate_maps_input_checks(map1, map2, window_size, fraction_accepted, reduce, verbose)

    corr = _correlate_maps(map1=map1, map2=map2, window_size=window_size,
                           fraction_accepted=fraction_accepted, reduce=reduce)

    if verbose:
        print(f"- correlation: {time.time() - start}")

    return np.asarray(corr)

correlate_maps.__doc__ = correlate_maps_base.__doc__

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def subplots(n=1, **kwargs):
    """
    Creating a flat array of subplots of length 'n'. 
    
    Parameters
    ----------
    n : int or tuple
        Count of subplots. Should be bigger than 1. If an iterable is given it gives the horizontal and vertical count
        of the plots.

    Other Parameters
    ----------------
    **kwargs : keyword arguments passed to :meth:`~matplotlib.pyplot.subplots`
    
    Returns
    -------
    f : :obj:`~matplotlib.figure.Figure`
    ax : list of :obj:`~matplotlib.axes.Axes`
        flat list of axes that can be looped over
    """
    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (10, 10)

    if not isinstance(n, (list, tuple, int)):
        raise TypeError("n should be an integer or a tuple of 2 integers")

    if type(n) == int:
        # columns
        i = int(np.sqrt(n))
        # rows
        j = int(n / i)

        # check if enough rows are inserted
        if i * j < n:
            j += 1

    if isinstance(n, (tuple, list)):
        if len(n) != 2:
            raise ValueError("List should be of length 2")
        if type(n[0]) != int or type(n[1]) != int:
            raise TypeError("List should contain integers")
        i = n[0]
        j = n[1]

    # construct the holders for the subplots
    f, ax = plt.subplots(j, i, **kwargs)

    # flatten the ax list
    ax = ax.flatten()

    return f, ax

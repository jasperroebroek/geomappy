#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import gamma, skewnorm, norm


def plot_normal(data, ax, xlim=False, nan=True, text=False):
    """
    Adding a normal distribution to the axis.
    
    Parameters
    ----------
    data : np.ndarray
        Data where the distribution will be drawn from
    ax : plt.axes
        Axis object where the distribution will be plotted on
    xlim : list, optional
        can be estimated from the data but can also be given as a tuple
    nan : bool, optional
        NaNs present in the data. If false it should speed up to processing
        but the function won't run if there are any present
    text : bool, optional
        Print the parameters for the distribution on top of the plot
    
    Returns
    -------
    list of floats
        mu : float
            mean of the data
        si : float
            sigma of the data
    """
    # todo; add input checks

    if not xlim:
        xlim = (np.min(data),np.max(data))
    x = np.linspace(*xlim, 1000)
    
    if nan:
        mu = np.nanmean(data)
        si = np.nanstd(data)
    else:
        mu = np.mean(data)
        si = np.std(data)
        
    ax.plot(x, norm.pdf(x,mu,si))
    if text:
        ax.text(0.75, 0.5, f"mu: {mu:.2f}\nsi:   {si:.2f}\n#: {data.size}",
                transform=ax.transAxes)
    return mu, si


def plot_gamma(data, ax, xlim=False, nan=False, text=False):
    """
    Adding a gamma distrubition to the axis.
    
    Parameters
    ----------
    data : np.ndarray
        Data where the distribution will be drawn from
    ax : plt.axes
        Axis object where the distribution will be plotted on
    xlim : list, optional
        can be estimated from the data but can also be given as a tuple
    nan : bool, optional
        NaNs present in the data. If false it should speed up to processing
        but the function won't run if there are any present
    text : bool, optional
        Print the parameters for the distribution on top of the plot
    
    Returns
    -------
    params : list of float
        parameter of the gamma distribution as obtained by scipy.stats.gamma.fit
    """
    # todo; add input checks

    if not xlim:
        xlim = (np.min(data),np.max(data))
    x = np.linspace(*xlim, 1000)
    if nan:
        data = data[~np.isnan(data)]
        
    params = gamma.fit(data)
    ax.plot(x,gamma.pdf(x,*params))
    # todo; sort this out
    # if text:
    #    ax.text(0.75,0.5,f"mu: {mu:.2f}\nsi:   {si:.2f}\n#: {data.size}",
    #            transform=ax.transAxes)
    return params


def plot_skewnormal(data, ax, xlim=False, nan=False, text=False):
    """
    Adding a skewnormal distrubition to the axis.
    
    Parameters
    ----------
    data : np.ndarray
        Data where the distribution will be drawn from
    ax : plt.axes
        Axis object where the distribution will be plotted on
    xlim : list, optional
        can be estimated from the data but can also be given as a tuple
    nan : bool, optional
        NaNs present in the data. If false it should speed up to processing
        but the function won't run if there are any present
    text : bool, optional
        Print the parameters for the distribution on top of the plot
    
    Returns
    -------
    params : list of float
        parameter of the gamma distribution as obtained by scipy.stats.skewnorm.fit
    """
    # todo; add input checks

    if not xlim:
        xlim = (np.min(data),np.max(data))
    x = np.linspace(*xlim, 1000)
    if nan:
        data = data[~np.isnan(data)]
    
    a, loc, scale = skewnorm.fit(data)
    ax.plot(x, skewnorm.pdf(x, a, loc, scale))
    if text:
        ax.text(0.05, 0.5, f"a: {a:.2f}\nloc:   {loc:.2f}\nscale: {scale:.2f}",
                transform=ax.transAxes)
    return a, loc, scale

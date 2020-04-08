import numpy as np


def nanunique(m, **kwargs):
    """
    Equivalent of np.unique while not touching NaN values. Returned ndarray has dtype float64

    Parameters
    ----------
    m : array
        input array
    **kwargs : dict, optional
        kwargs for np.unique function

    Returns
    -------
    :obj:`~numpy.ndarray`
        array has dtype np.float64

    """
    return np.unique(m[~np.isnan(m)], **kwargs)


def nandigitize(m, bins, **kwargs):
    """
    Equivalent of np.digitize while not touching NaN values. Returned ndarray has dtype float64.

    Parameters
    ----------
    m : :obj:`~numpy.ndarray`
        input array
    bins : list
        ascending list of values on which the input array `a` is digitized. Look at `numpy.digitize` documentation
    **kwargs : dict
        keyword arguments to be passed to `numpy.digitize`

    Returns
    -------
    :obj:`~numpy.ndarray`
        array has dtype np.float64, shape is the same as the input array.
    """
    # todo; optimise. Think about masked arrays.
    mask = np.isnan(m)
    m_digitized = np.digitize(m.copy(), bins=bins, **kwargs).astype(np.float64)
    m_digitized[mask] = np.nan
    return m_digitized

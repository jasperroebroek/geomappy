import geomappy as mp
import numpy as np
from scipy.stats import pearsonr
from geomappy.utils import overlapping_arrays
import pytest


def correlate_maps_simple(map1, map2, window_size=5, fraction_accepted=0.7):
    """
    Takes two maps and returning the local correlation between them with the same dimensions as the input maps.
    Correlation calculated in a rolling window with the size `window_size`. If either of the input maps contains
    a NaN value on a location, the output map will also have a NaN on that location. This is a simplified version of
    correlate_maps() in raster_functions with the purpose of testing. It is super slow, so don't throw large maps
    at it.

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

    """
    map1, map2 = overlapping_arrays([map1, map2])
    fringe = window_size // 2
    corr = np.full(map1.shape, np.nan)
    for i in range(fringe, map1.shape[0] - fringe):
        for j in range(fringe, map1.shape[1] - fringe):
            ind = np.s_[i - fringe:i + fringe + 1, j - fringe:j + fringe + 1]

            if np.isnan(map1[i, j]) or np.isnan(map2[i, j]):
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


def test_correlation_values():
    a = np.random.rand(5, 5)
    b = np.random.rand(5, 5)

    # Cython implementation
    assert np.allclose(pearsonr(a.flatten(), b.flatten())[0],
                       mp.focal_statistics.correlate_maps(a, b, window_size=5, reduce=True))
    # Numpy implementation
    assert np.allclose(pearsonr(a.flatten(), b.flatten())[0],
                       mp.focal_statistics.correlate_maps_base(a, b, window_size=5)[2, 2])
    # Local implementation
    assert np.allclose(pearsonr(a.flatten(), b.flatten())[0],
                       correlate_maps_simple(a, b, window_size=5)[2, 2])


def test_correlation_shape():
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)

    assert mp.focal_statistics.correlate_maps(a, b, window_size=3).shape == a.shape
    assert mp.focal_statistics.correlate_maps(a, b, window_size=10, reduce=True).shape == (1, 1)
    assert mp.focal_statistics.correlate_maps_base(a, b).shape == a.shape


def test_correlation_errors():
    for cm in [mp.focal_statistics.correlate_maps,
               mp.focal_statistics.correlate_maps_base]:

        with pytest.raises(TypeError):
            cm(np.random.rand(10, 10), np.random.rand(10, 10), window_size=5, verbose=1)

        with pytest.raises(TypeError):
            cm(np.random.rand(10, 10), np.random.rand(10, 10), window_size=5, reduce=1)

        # not 2D
        with pytest.raises(IndexError):
            a = np.random.rand(10, 10, 10)
            b = np.random.rand(10, 10, 10)
            cm(a, b)

        # different shapes
        with pytest.raises(IndexError):
            a = np.random.rand(10, 10)
            b = np.random.rand(10, 15)
            cm(a, b)

        with pytest.raises(TypeError):
            a = np.random.rand(10, 10)
            b = np.random.rand(10, 10)
            cm(a, b, window_size="5")

        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            b = np.random.rand(10, 10)
            cm(a, b, window_size=1)

        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            b = np.random.rand(10, 10)
            cm(a, b, window_size=11)

        # uneven window_size is not supported
        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            b = np.random.rand(10, 10)
            cm(a, b, window_size=4)

        # Not exactly divided in reduce mode
        with pytest.raises((NotImplementedError, ValueError)):
            a = np.random.rand(10, 10)
            b = np.random.rand(10, 10)
            cm(a, b, window_size=4, reduce=True)

        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            b = np.random.rand(10, 10)
            cm(a, b, fraction_accepted=-0.1)

        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            b = np.random.rand(10, 10)
            cm(a, b, fraction_accepted=1.1)


def test_nan_behaviour():
    for cm in [mp.focal_statistics.correlate_maps,
               mp.focal_statistics.correlate_maps_base,
               correlate_maps_simple]:

        a = np.random.rand(5, 5)
        b = np.random.rand(5, 5)
        a[2, 2] = np.nan
        assert np.allclose(cm(a, b), correlate_maps_simple(a, b), equal_nan=True)
        assert np.isnan(cm(a, b)[2, 2])

        a = np.random.rand(5, 5)
        b = np.random.rand(5, 5)
        a[1, 1] = np.nan
        assert np.allclose(cm(a, b), correlate_maps_simple(a, b), equal_nan=True)
        assert not np.isnan(cm(a, b)[2, 2])
        assert np.isnan(cm(a, b, fraction_accepted=1)[2, 2])
        assert not np.isnan(cm(a, b, fraction_accepted=0)[2, 2])


def test_correlation_dtype():
    for cm in [mp.focal_statistics.correlate_maps,
               mp.focal_statistics.correlate_maps_base,
               correlate_maps_simple]:

        a = np.random.rand(5, 5).astype(np.int)
        b = np.random.rand(5, 5).astype(np.int)
        assert cm(a, b).dtype == np.float64

        a = np.random.rand(5, 5).astype(np.float)
        b = np.random.rand(5, 5).astype(np.float)
        assert cm(a, b).dtype == np.float64

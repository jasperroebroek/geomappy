import unittest
import numpy as np
from numpy import s_
from mappy import overlapping_arrays, rolling_mean, rolling_sum, rolling_window, focal_statistics, Map
from scipy.stats import pearsonr
from mappy.raster_functions.correlate_maps import correlate_maps_njit as correlate_maps
from mappy.raster_functions.correlate_maps import correlate_maps_base


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


class TestRollingSum(unittest.TestCase):
    def test_1D(self):
        a = np.random.rand(100)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=1)))
        self.assertTrue(np.allclose(rolling_sum(a, 9), rolling_window(a, 9).sum(axis=1)))

    def test_2D(self):
        a = np.random.rand(20, 20)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=(2, 3))))
        self.assertTrue(np.allclose(rolling_sum(a, 9), rolling_window(a, 9).sum(axis=(2, 3))))

    def test_3D(self):
        a = np.random.rand(15, 15, 15)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=(3, 4, 5))))
        self.assertTrue(np.allclose(rolling_sum(a, 9), rolling_window(a, 9).sum(axis=(3, 4, 5))))

    def test_4D(self):
        a = np.random.rand(12, 12, 12, 12)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=(4, 5, 6, 7))))
        self.assertTrue(np.allclose(rolling_sum(a, 9), rolling_window(a, 9).sum(axis=(4, 5, 6, 7))))

    def test_5D(self):
        a = np.random.rand(10, 10, 10, 10, 10)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=(5, 6, 7, 8, 9))))

    def test_assumptions(self):
        with self.assertRaises(ValueError):
            # uneven window_size should raise a ValueError
            rolling_sum(np.array([1, 2, 3]), 4)
        with self.assertRaises(ValueError):
            # window_size bigger than dimensions of array should raise ValueError
            rolling_sum(np.array([1, 2, 3]), 5)


class TestCorrelateMapsNumba(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCorrelateMapsNumba, self).__init__(*args, **kwargs)
        self.map1 = np.random.rand(20, 20)
        self.map2 = np.random.rand(20, 20)

    def test_assumptions(self):
        with self.assertRaises((ValueError, IndexError)):
            # Only 2D is supported
            correlate_maps(np.random.rand(10, 10, 10), np.random.rand(10, 10, 10))
        with self.assertRaises((ValueError, IndexError)):
            # Only 2D is supported
            correlate_maps(np.random.rand(10), np.random.rand(10))
        with self.assertRaises(ValueError):
            # fraction accepted needs to be in range 0-1
            correlate_maps(self.map1, self.map2, fraction_accepted=-0.1)
        with self.assertRaises(ValueError):
            # fraction accepted needs to be in range 0-1
            correlate_maps(self.map1, self.map2, fraction_accepted=1.1)
        with self.assertRaises(ValueError):
            # window_size should be bigger than 1
            correlate_maps(self.map1, self.map2, window_size=1)
        with self.assertRaises(ValueError):
            # window_size can't be even
            correlate_maps(self.map1, self.map2, window_size=4)
        # window_size can't be even except when reduce=True
        correlate_maps(self.map1, self.map2, window_size=4, reduce=True)

    def test_correlation(self):
        # fraction_accepted 0.7 and window_size 5
        c1 = correlate_maps(self.map1, self.map2)
        c2 = correlate_maps_simple(self.map1, self.map2)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_fraction_accepted_0(self):
        c1 = correlate_maps(self.map1, self.map2, fraction_accepted=0)
        c2 = correlate_maps_simple(self.map1, self.map2, fraction_accepted=0)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_fraction_accepted_1(self):
        c1 = correlate_maps(self.map1, self.map2, fraction_accepted=1)
        c2 = correlate_maps_simple(self.map1, self.map2, fraction_accepted=1)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_window_size_15(self):
        c1 = correlate_maps(self.map1, self.map2, window_size=15)
        c2 = correlate_maps_simple(self.map1, self.map2, window_size=15)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_value(self):
        # test the value of the correlation against scipys implementation
        c1 = np.random.rand(5, 5)
        c2 = np.random.rand(5, 5)
        self.assertTrue(np.allclose(pearsonr(c1.flatten(), c2.flatten())[0],
                                    correlate_maps(c1, c2, window_size=5)[2, 2]))

    def test_reduce(self):
        # Test if the right shape comes out
        self.assertTrue(correlate_maps(self.map1, self.map2, window_size=4, reduce=True).shape == (5, 5))
        # Test if the right value comes out
        c1 = self.map1[:4, :4].flatten()
        c2 = self.map2[:4, :4].flatten()
        # print(correlate_maps(self.map1, self.map2, window_size=4, reduce=True))
        # print(pearsonr(c1, c2))
        self.assertTrue(np.allclose(correlate_maps(self.map1, self.map2, window_size=4, reduce=True)[0, 0],
                                    pearsonr(c1, c2)[0]))


class TestCorrelateMapsNumpy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCorrelateMapsNumpy, self).__init__(*args, **kwargs)
        self.map1 = np.random.rand(20, 20)
        self.map2 = np.random.rand(20, 20)

    def test_assumptions(self):
        with self.assertRaises((ValueError, IndexError)):
            # Only 2D is supported
            correlate_maps_base(np.random.rand(10, 10, 10), np.random.rand(10, 10, 10))
        with self.assertRaises((ValueError, IndexError)):
            # Only 2D is supported
            correlate_maps_base(np.random.rand(10), np.random.rand(10))
        with self.assertRaises(ValueError):
            # fraction accepted needs to be in range 0-1
            correlate_maps_base(self.map1, self.map2, fraction_accepted=-0.1)
        with self.assertRaises(ValueError):
            # fraction accepted needs to be in range 0-1
            correlate_maps_base(self.map1, self.map2, fraction_accepted=1.1)
        with self.assertRaises(ValueError):
            # window_size should be bigger than 1
            correlate_maps_base(self.map1, self.map2, window_size=1)
        with self.assertRaises(ValueError):
            # window_size can't be even
            correlate_maps_base(self.map1, self.map2, window_size=4)

    def test_correlation(self):
        # fraction_accepted 0.7 and window_size 5
        c1 = correlate_maps_base(self.map1, self.map2)
        c2 = correlate_maps_simple(self.map1, self.map2)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_fraction_accepted_0(self):
        c1 = correlate_maps_base(self.map1, self.map2, fraction_accepted=0)
        c2 = correlate_maps_simple(self.map1, self.map2, fraction_accepted=0)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_fraction_accepted_1(self):
        c1 = correlate_maps_base(self.map1, self.map2, fraction_accepted=1)
        c2 = correlate_maps_simple(self.map1, self.map2, fraction_accepted=1)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_window_size_15(self):
        c1 = correlate_maps_base(self.map1, self.map2, window_size=15)
        c2 = correlate_maps_simple(self.map1, self.map2, window_size=15)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_value(self):
        # test the value of the correlation against scipys implementation
        c1 = np.random.rand(5, 5)
        c2 = np.random.rand(5, 5)
        self.assertTrue(np.allclose(pearsonr(c1.flatten(), c2.flatten())[0],
                                    correlate_maps_base(c1, c2, window_size=5)[2, 2]))

    def test_reduce(self):
        # As this is not yet implemented, it should raise an error
        with self.assertRaises(NotImplementedError):
            correlate_maps_base(self.map1, self.map2, window_size=5, reduce=True)

class TestFocalStatistics(unittest.TestCase):
    def test_focalmean(self):
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).mean(axis=(2, 3)),
                                    focal_statistics(a, window_size=3, func="nanmean")[1:-1, 1:-1]))

    def test_focalmean_reduce(self):
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 5, reduce=True).mean(axis=(2, 3)),
                                    focal_statistics(a, window_size=5, func="nanmean", reduce=True)))

    def test_focalmean_reduce_fraction_accepted_0(self):
        a = np.full((5, 5), np.nan)
        # return nan when nothing is present
        self.assertTrue(np.allclose(focal_statistics(a, window_size=5, func="nanmean", reduce=True, fraction_accepted=0)
                                    , np.array([[np.nan]]), equal_nan=True))

        # return value if only one value is present
        a[3, 3] = 10
        self.assertTrue(np.allclose(focal_statistics(a, window_size=5, func="nanmean", reduce=True, fraction_accepted=0)
                                    , np.nanmean(rolling_window(a, 5, reduce=True))))

    def test_focalmean_reduce_fraction_accepted_1(self):
        a = np.random.rand(5, 5)
        # return value when all values are present
        self.assertTrue(np.allclose(focal_statistics(a, window_size=5, func="nanmean", reduce=True, fraction_accepted=1)
                                    , np.mean(rolling_window(a, 5).mean(axis=(2, 3)))))

        a[3, 3] = np.nan
        # return nan when only one value is missing
        self.assertTrue(np.allclose(focal_statistics(a, window_size=5, func="nanmean", reduce=True, fraction_accepted=1)
                                    , np.array([[np.nan]]), equal_nan=True))

    def test_focalstd_df_0(self):
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).std(axis=(2, 3)),
                                    focal_statistics(a, window_size=3, func="nanstd", std_df=0)[1:-1, 1:-1]))

    def test_focalstd_df_1(self):
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).std(axis=(2, 3), ddof=1),
                                    focal_statistics(a, window_size=3, func="nanstd", std_df=1)[1:-1, 1:-1]))

    def test_focalmin(self):
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).min(axis=(2, 3)),
                                    focal_statistics(a, window_size=3, func="nanmin")[1:-1, 1:-1]))

    def test_focalmax(self):
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).max(axis=(2, 3)),
                                    focal_statistics(a, window_size=3, func="nanmax")[1:-1, 1:-1]))

    # todo; test majority with its different modes


if __name__ == '__main__':
    unittest.main()

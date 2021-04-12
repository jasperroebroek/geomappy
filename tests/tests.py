import unittest
import numpy as np
from numpy import s_
from scipy.stats import pearsonr

from geomappy.utils import overlapping_arrays
from geomappy.rolling import rolling_window, rolling_sum
from geomappy.raster import Raster
from geomappy.focal_statistics import focal_mean, focal_statistics
from geomappy.focal_statistics import correlate_maps_base
from geomappy.focal_statistics import correlate_maps as correlate_maps_cython


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
            ind = s_[i - fringe:i + fringe + 1, j - fringe:j + fringe + 1]

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


class TestRollingSum(unittest.TestCase):
    def test_1D(self):
        np.random.seed(0)
        a = np.random.rand(100)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=1)))
        self.assertTrue(np.allclose(rolling_sum(a, 9), rolling_window(a, 9).sum(axis=1)))

    def test_2D(self):
        np.random.seed(0)
        a = np.random.rand(20, 20)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=(2, 3))))
        self.assertTrue(np.allclose(rolling_sum(a, 9), rolling_window(a, 9).sum(axis=(2, 3))))

    def test_3D(self):
        np.random.seed(0)
        a = np.random.rand(15, 15, 15)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=(3, 4, 5))))
        self.assertTrue(np.allclose(rolling_sum(a, 9), rolling_window(a, 9).sum(axis=(3, 4, 5))))

    def test_4D(self):
        np.random.seed(0)
        a = np.random.rand(12, 12, 12, 12)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=(4, 5, 6, 7))))
        self.assertTrue(np.allclose(rolling_sum(a, 9), rolling_window(a, 9).sum(axis=(4, 5, 6, 7))))

    def test_5D(self):
        np.random.seed(0)
        a = np.random.rand(10, 10, 10, 10, 10)
        self.assertTrue(np.allclose(rolling_sum(a, 3), rolling_window(a, 3).sum(axis=(5, 6, 7, 8, 9))))

    def test_reduce(self):
        np.random.seed(0)
        a = np.random.rand(5, 5)
        self.assertTrue(rolling_sum(a, window_size=5, reduce=True), a.sum())

    def test_assumptions(self):
        with self.assertRaises(ValueError):
            # window_size bigger than dimensions of array should raise ValueError
            rolling_sum(np.array([1, 2, 3]), 5)


class TestCorrelateRastersNumba(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCorrelateRastersNumba, self).__init__(*args, **kwargs)
        np.random.seed(0)
        self.map1 = np.random.rand(20, 20)
        self.map2 = np.random.rand(20, 20)

    def test_assumptions(self):
        with self.assertRaises((ValueError, IndexError)):
            # Only 2D is supported
            correlate_maps_cython(np.random.rand(10, 10, 10), np.random.rand(10, 10, 10))
        with self.assertRaises((ValueError, IndexError)):
            # Only 2D is supported
            correlate_maps_cython(np.random.rand(10), np.random.rand(10))
        with self.assertRaises(ValueError):
            # fraction accepted needs to be in range 0-1
            correlate_maps_cython(self.map1, self.map2, fraction_accepted=-0.1)
        with self.assertRaises(ValueError):
            # fraction accepted needs to be in range 0-1
            correlate_maps_cython(self.map1, self.map2, fraction_accepted=1.1)
        with self.assertRaises(ValueError):
            # window_size should be bigger than 1
            correlate_maps_cython(self.map1, self.map2, window_size=1)
        with self.assertRaises(ValueError):
            # window_size can't be even
            correlate_maps_cython(self.map1, self.map2, window_size=4)
        # window_size can't be even except when reduce=True
        correlate_maps_cython(self.map1, self.map2, window_size=4, reduce=True)

    def test_correlation(self):
        # fraction_accepted 0.7 and window_size 5
        c1 = correlate_maps_cython(self.map1, self.map2)
        c2 = correlate_maps_simple(self.map1, self.map2)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_fraction_accepted_0(self):
        c1 = correlate_maps_cython(self.map1, self.map2, fraction_accepted=0)
        c2 = correlate_maps_simple(self.map1, self.map2, fraction_accepted=0)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_fraction_accepted_025(self):
        c1 = correlate_maps_cython(self.map1, self.map2, fraction_accepted=0.25)
        c2 = correlate_maps_simple(self.map1, self.map2, fraction_accepted=0.25)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_fraction_accepted_1(self):
        c1 = correlate_maps_cython(self.map1, self.map2, fraction_accepted=1)
        c2 = correlate_maps_simple(self.map1, self.map2, fraction_accepted=1)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_window_size_15(self):
        c1 = correlate_maps_cython(self.map1, self.map2, window_size=15)
        c2 = correlate_maps_simple(self.map1, self.map2, window_size=15)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_correlation_value(self):
        # test the value of the correlation against scipys implementation
        np.random.seed(0)
        c1 = np.random.rand(5, 5)
        c2 = np.random.rand(5, 5)
        self.assertTrue(np.allclose(pearsonr(c1.flatten(), c2.flatten())[0],
                                    correlate_maps_cython(c1, c2, window_size=5)[2, 2]))

    def test_reduce(self):
        # Test if the right shape comes out
        self.assertTrue(correlate_maps_cython(self.map1, self.map2, window_size=4, reduce=True).shape == (5, 5))
        # Test if the right value comes out
        c1 = self.map1[:4, :4].flatten()
        c2 = self.map2[:4, :4].flatten()
        # print(correlate_maps_cython(self.map1, self.map2, window_size=4, reduce=True))
        # print(pearsonr(c1, c2))
        self.assertTrue(np.allclose(correlate_maps_cython(self.map1, self.map2, window_size=4, reduce=True)[0, 0],
                                    pearsonr(c1, c2)[0]))


class TestCorrelateRastersNumpy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCorrelateRastersNumpy, self).__init__(*args, **kwargs)
        np.random.seed(0)
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

    def test_correlation_fraction_accepted_025(self):
        c1 = correlate_maps_base(self.map1, self.map2, fraction_accepted=0.25)
        c2 = correlate_maps_simple(self.map1, self.map2, fraction_accepted=0.25)
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
        np.random.seed(0)
        c1 = np.random.rand(5, 5)
        c2 = np.random.rand(5, 5)
        self.assertTrue(np.allclose(pearsonr(c1.flatten(), c2.flatten())[0],
                                    correlate_maps_base(c1, c2, window_size=5)[2, 2]))

    def test_reduce(self):
        # As this is not yet implemented, it should raise an error
        with self.assertRaises(NotImplementedError):
            correlate_maps_base(self.map1, self.map2, window_size=5, reduce=True)


class TestRollingWindow(unittest.TestCase):
    def test_rolling_window(self):
        a = (np.random.rand(5, 5) > 0.5).astype(int)
        self.assertTrue(np.array_equal(rolling_window(a, window_size=4)[0, 0], a[:-1, :-1]))

    def test_reduce(self):
        a = (np.random.rand(6, 6) > 0.5).astype(int)
        self.assertTrue(np.array_equal(rolling_window(a, window_size=3)[0, 0], a[:3, :3]))

    def test_flatten(self):
        a = (np.random.rand(5, 5) > 0.5).astype(int)
        self.assertTrue(np.array_equal(rolling_window(a, window_size=4, flatten=True)[0, 0], a[:-1, :-1].flatten()))

    def test_5D(self):
        a = (np.random.rand(5, 5, 5, 5, 5) > 0.5).astype(int)
        self.assertTrue(rolling_window(a, window_size=4).ndim == 10)

    def test_errors_window_size_too_big(self):
        a = np.random.rand(5, 5)
        with self.assertRaises(ValueError):
            rolling_window(a, window_size=6)

    def test_errors_reduce_cant_divide(self):
        a = np.random.rand(5, 5)
        with self.assertRaises(ValueError):
            rolling_window(a, window_size=3, reduce=True)


class TestFocalStatistics(unittest.TestCase):
    def test_focalmean(self):
        np.random.seed(0)
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).mean(axis=(2, 3)),
                                    focal_statistics(a, window_size=3, func="mean")[1:-1, 1:-1]))

    def test_focalmean_reduce(self):
        np.random.seed(0)
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 5, reduce=True).mean(axis=(2, 3)),
                                    focal_statistics(a, window_size=5, func="mean", reduce=True)))

    def test_focalmean_reduce_fraction_accepted_0(self):
        np.random.seed(0)
        a = np.full((5, 5), np.nan)
        # return nan when nothing is present
        self.assertTrue(np.allclose(focal_statistics(a, window_size=5, func="mean", reduce=True, fraction_accepted=0)
                                    , np.array([[np.nan]]), equal_nan=True))

        # return value if only one value is present
        a[3, 3] = 10
        self.assertTrue(np.allclose(focal_statistics(a, window_size=5, func="mean", reduce=True, fraction_accepted=0)
                                    , np.nanmean(rolling_window(a, 5, reduce=True))))

    def test_focalmean_reduce_fraction_accepted_1(self):
        np.random.seed(0)
        a = np.random.rand(5, 5)
        # return value when all values are present
        self.assertTrue(np.allclose(focal_statistics(a, window_size=5, func="mean", reduce=True, fraction_accepted=1)
                                    , np.mean(rolling_window(a, 5).mean(axis=(2, 3)))))

        a[3, 3] = np.nan
        # return nan when only one value is missing
        self.assertTrue(np.allclose(focal_statistics(a, window_size=5, func="mean", reduce=True, fraction_accepted=1)
                                    , np.array([[np.nan]]), equal_nan=True))

    def test_focalstd_df_0(self):
        np.random.seed(0)
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).std(axis=(2, 3)),
                                    focal_statistics(a, window_size=3, func="std", std_df=0)[1:-1, 1:-1]))

    def test_focalstd_df_1(self):
        np.random.seed(0)
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).std(axis=(2, 3), ddof=1),
                                    focal_statistics(a, window_size=3, func="std", std_df=1)[1:-1, 1:-1]))

    def test_focalmin(self):
        np.random.seed(0)
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).min(axis=(2, 3)),
                                    focal_statistics(a, window_size=3, func="min")[1:-1, 1:-1]))

    def test_focalnanmin(self):
        np.random.seed(0)
        a = np.random.rand(5, 5)
        min = a.min()
        a[1, 1] = np.nan
        self.assertTrue(focal_statistics(a, window_size=5, func="min")[2, 2] == min)

    def test_focalnanmin_reduce(self):
        np.random.seed(0)
        a = np.random.rand(5, 5)
        min = a.min()
        a[1, 1] = np.nan
        self.assertTrue(focal_statistics(a, window_size=5, func="min", reduce=True) == min)

    def test_focalmax(self):
        np.random.seed(0)
        a = np.random.rand(100, 100)
        self.assertTrue(np.allclose(rolling_window(a, 3).max(axis=(2, 3)),
                                    focal_statistics(a, window_size=3, func="max")[1:-1, 1:-1]))

    def test_focalnanmax(self):
        np.random.seed(0)
        a = np.random.rand(5, 5)
        max = a.max()
        a[1, 1] = np.nan
        self.assertTrue(focal_statistics(a, window_size=5, func="max")[2, 2] == max)

    def test_focalnanmax_reduce(self):
        np.random.seed(0)
        a = np.random.rand(5, 5)
        max = a.max()
        a[1, 1] = np.nan
        self.assertTrue(focal_statistics(a, window_size=5, func="max", reduce=True) == max)

    def test_errors(self):
        np.random.seed(0)
        a = np.random.rand(100, 100)
        with self.assertRaises(IndexError):
            focal_statistics(a, window_size=101, func="max")
        with self.assertRaises(ValueError):
            focal_statistics(a, window_size=3, fraction_accepted=1.1, func="max")
        with self.assertRaises(TypeError):
            focal_statistics(a, window_size="a", func="max")
        with self.assertRaises(ValueError):
            focal_statistics(np.random.rand(100, 100, 100), window_size=3, func="max")
        with self.assertRaises(ValueError):
            focal_statistics(a, window_size=9, reduce=True, func="max")

    # todo; test majority with its different modes


class TestRaster(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestRaster, self).__init__(*args, **kwargs)
        self.map1 = Raster("../data/wtd.tif", tiles=(8, 8))
        self.map2 = Raster("../data/tree_height.asc", tiles=(8, 8))

    def test_tile_correlation(self):
        loc = "test.tif"
        self.map1.window_size = 5
        self.map2.window_size = 5
        self.map1.correlate(self.map2, output_file=loc, window_size=5, ind=32, overwrite=True)
        t = Raster(loc)
        c1 = t[0]
        t.close(verbose=False)
        c2 = correlate_maps_cython(self.map1[32], self.map2[32], window_size=5)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    # This should be tested once in a while to make sure everything works okay, but it takes a couple of minutes
    def test_correlation(self):
        loc = "test.tif"
        self.map1.correlate(self.map2, output_file=loc, window_size=5, overwrite=True)
        t = Raster(loc)
        c1 = t[0]
        t.close(verbose=False)
        c2 = correlate_maps_cython(self.map1.get_file_data(), self.map2.get_file_data(), window_size=5)
        self.assertTrue(np.allclose(c1[self.map1.ind_inner], c2[self.map1.ind_inner], equal_nan=True))

    def test_tile_focal_stats(self):
        loc = "test.tif"
        self.map1.focal_mean(ind=32, window_size=5, output_file=loc, overwrite=True, reduce=True)
        t = Raster(loc)
        c1 = t[0]
        t.close(verbose=False)
        c2 = focal_mean(self.map1[32], window_size=5, reduce=True)
        self.assertTrue(np.allclose(c1, c2, equal_nan=True))

    def test_focal_stats(self):
        loc = "test.tif"
        self.map1.focal_mean(window_size=5, output_file=loc, overwrite=True, reduce=True)
        t = Raster(loc)
        c1 = t[0]
        t.close(verbose=False)
        c2 = focal_mean(self.map1.get_file_data(), window_size=5, reduce=True)
        self.assertTrue(np.allclose(c1[self.map1.ind_inner], c2[self.map1.ind_inner], equal_nan=True))


if __name__ == '__main__':
    unittest.main()

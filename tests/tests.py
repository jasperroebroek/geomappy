import unittest
import numpy as np
from thesis import *
from scipy.stats import pearsonr


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


class TestCorrelateMaps(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCorrelateMaps, self).__init__(*args, **kwargs)
        # If available you can test it with actual maps, if not randomly generated grids work fine
        # self.map1 = Map("tree_height.asc")[0][::10, ::10]
        # self.map2 = Map("wtd.tif")[0][::10, ::10]
        self.map1 = np.random.rand(20, 20)
        self.map2 = np.random.rand(20, 20)

    def test_assumptions(self):
        with self.assertRaises(ValueError):
            # Only 2D is supported
            correlate_maps(np.random.rand(10, 10, 10), np.random.rand(10, 10, 10))
        with self.assertRaises(ValueError):
            # Only 2D is supported
            correlate_maps(np.random.rand(10), np.random.rand(10))

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


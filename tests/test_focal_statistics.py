import pytest
import numpy as np
import scipy.stats

import geomappy as mp


def test_focal_stats_values():
    a = np.random.rand(5, 5)

    # Values when not reducing
    assert np.allclose(mp.focal_statistics.focal_mean(a)[2, 2], a.mean())
    assert np.allclose(mp.focal_statistics.focal_min(a)[2, 2], a.min())
    assert np.allclose(mp.focal_statistics.focal_max(a)[2, 2], a.max())
    assert np.allclose(mp.focal_statistics.focal_std(a)[2, 2], a.std())
    assert np.allclose(mp.focal_statistics.focal_std(a, std_df=1)[2, 2], a.std(ddof=1))

    # Values when reducing
    assert np.allclose(mp.focal_statistics.focal_mean(a, reduce=True)[0, 0], a.mean())
    assert np.allclose(mp.focal_statistics.focal_min(a, reduce=True)[0, 0], a.min())
    assert np.allclose(mp.focal_statistics.focal_max(a, reduce=True)[0, 0], a.max())
    assert np.allclose(mp.focal_statistics.focal_std(a, reduce=True)[0, 0], a.std())
    assert np.allclose(mp.focal_statistics.focal_std(a, std_df=1, reduce=True)[0, 0], a.std(ddof=1))

    rs = np.random.RandomState(0)
    a = rs.randint(0, 10, 25).reshape(5, 5)

    # Value when reducing
    assert scipy.stats.mode(a.flatten()).mode[0] == \
           mp.focal_statistics.focal_majority(a, window_size=5, majority_mode='ascending')[2, 2]
    # Values when not reducing
    assert scipy.stats.mode(a.flatten()).mode[0] == \
           mp.focal_statistics.focal_majority(a, window_size=5, reduce=True, majority_mode='ascending')[0, 0]

    # Same number of observations in several classes lead to NaN in majority_mode='nan'
    a = np.arange(100).reshape(10, 10)
    assert np.isnan(mp.focal_statistics.focal_majority(a, window_size=10, reduce=True, majority_mode='nan'))

    # Same number of observations in several classes lead to lowest number in majority_mode='ascending'
    assert mp.focal_statistics.focal_majority(a, window_size=10, reduce=True, majority_mode='ascending') == 0

    # Same number of observations in several classes lead to highest number in majority_mode='descending'
    assert mp.focal_statistics.focal_majority(a, window_size=10, reduce=True, majority_mode='descending') == 99


def test_focal_stats_shape():
    for fs in [mp.focal_statistics.focal_mean,
               mp.focal_statistics.focal_min,
               mp.focal_statistics.focal_max,
               mp.focal_statistics.focal_std,
               mp.focal_statistics.focal_majority]:
        a = np.random.rand(10, 10)
        assert a.shape == fs(a, window_size=3).shape
        assert fs(a, window_size=10, reduce=True).shape == (1, 1)


def test_focal_stats_errors():
    for fs in [mp.focal_statistics.focal_mean,
               mp.focal_statistics.focal_min,
               mp.focal_statistics.focal_max,
               mp.focal_statistics.focal_std,
               mp.focal_statistics.focal_majority]:

        with pytest.raises(TypeError):
            fs(np.random.rand(10, 10), verbose=1)

        with pytest.raises(TypeError):
            fs(np.random.rand(10, 10), reduce=1)

        # not 2D
        with pytest.raises(IndexError):
            a = np.random.rand(10, 10, 10)
            fs(a)

        with pytest.raises(TypeError):
            a = np.random.rand(10, 10)
            fs(a, window_size="5")

        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            fs(a, window_size=1)

        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            fs(a, window_size=11)

        # uneven window_size is not supported
        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            fs(a, window_size=4)

        # Not exactly divided in reduce mode
        with pytest.raises((NotImplementedError, ValueError)):
            a = np.random.rand(10, 10)
            fs(a, window_size=4, reduce=True)

        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            fs(a, fraction_accepted=-0.1)

        with pytest.raises(ValueError):
            a = np.random.rand(10, 10)
            fs(a, fraction_accepted=1.1)


def test_focal_stats_nan_behaviour():
    for fs in [mp.focal_statistics.focal_mean,
               mp.focal_statistics.focal_min,
               mp.focal_statistics.focal_max,
               mp.focal_statistics.focal_std,
               mp.focal_statistics.focal_majority]:

        a = np.random.rand(5, 5)
        a[2, 2] = np.nan
        assert np.isnan(fs(a)[2, 2])

    for fs, np_fs in [(mp.focal_statistics.focal_mean, np.nanmean),
                      (mp.focal_statistics.focal_min, np.nanmin),
                      (mp.focal_statistics.focal_max, np.nanmax),
                      (mp.focal_statistics.focal_std, np.nanstd)]:

        a = np.random.rand(5, 5)
        a[1, 1] = np.nan

        assert np.allclose(fs(a)[2, 2], np_fs(a))
        assert not np.isnan(fs(a, fraction_accepted=0)[2, 2])
        assert np.isnan(fs(a, fraction_accepted=1)[2, 2])

    a = np.ones((5, 5)).astype(float)
    a[1, 1] = np.nan
    assert mp.focal_statistics.focal_majority(a)[2, 2] == 1
    assert not np.isnan(mp.focal_statistics.focal_majority(a, fraction_accepted=0)[2, 2])
    assert np.isnan(mp.focal_statistics.focal_majority(a, fraction_accepted=1)[2, 2])


def test_focal_stats_dtype():
    for fs in [mp.focal_statistics.focal_mean,
               mp.focal_statistics.focal_min,
               mp.focal_statistics.focal_max,
               mp.focal_statistics.focal_std,
               mp.focal_statistics.focal_majority]:

        a = np.random.rand(5, 5).astype(np.int)
        assert fs(a).dtype == np.float64

        a = np.random.rand(5, 5).astype(np.float)
        assert fs(a).dtype == np.float64

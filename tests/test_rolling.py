import geomappy as mp
import numpy as np
import pytest


def test_rolling_dimensions():
    a = np.random.rand(10)
    b = np.random.rand(10, 10)
    c = np.random.rand(10, 10, 10)
    d = np.random.rand(5, 5, 5, 5)

    assert mp.rolling.rolling_window(a, window_size=5).shape == (6, 5)
    assert mp.rolling.rolling_window(a, window_size=5, reduce=True).shape == (2, 5)
    assert mp.rolling.rolling_window(a, window_size=5, flatten=True).shape == (6, 5)
    assert mp.rolling.rolling_window(a, window_size=5, reduce=True, flatten=True).shape == (2, 5)

    assert mp.rolling.rolling_window(b, window_size=5).shape == (6, 6, 5, 5)
    assert mp.rolling.rolling_window(b, window_size=5, reduce=True).shape == (2, 2, 5, 5)
    assert mp.rolling.rolling_window(b, window_size=5, flatten=True).shape == (6, 6, 25)
    assert mp.rolling.rolling_window(b, window_size=5, reduce=True, flatten=True).shape == (2, 2, 25)

    assert mp.rolling.rolling_window(c, window_size=5).shape == (6, 6, 6, 5, 5, 5)
    assert mp.rolling.rolling_window(c, window_size=5, reduce=True).shape == (2, 2, 2, 5, 5, 5)
    assert mp.rolling.rolling_window(c, window_size=5, flatten=True).shape == (6, 6, 6, 125)
    assert mp.rolling.rolling_window(c, window_size=5, reduce=True, flatten=True).shape == (2, 2, 2, 125)

    assert mp.rolling.rolling_window(d, window_size=5, reduce=True, flatten=True).shape == (1, 1, 1, 1, 625)


def test_rolling_values():
    a = np.random.rand(10)
    b = np.random.rand(10, 10)

    assert mp.rolling.rolling_window(a, window_size=5)[0, 4] == a[4]
    assert mp.rolling.rolling_window(a, window_size=5, reduce=True)[1, 4] == a[9]
    assert mp.rolling.rolling_window(a, window_size=5, flatten=True)[0, 4] == a[4]
    assert mp.rolling.rolling_window(a, window_size=5, reduce=True, flatten=True)[1, 4] == a[9]

    assert mp.rolling.rolling_window(b, window_size=5)[2, 2, 0, 0] == b[2, 2]
    assert mp.rolling.rolling_window(b, window_size=5)[2, 2, 2, 2] == b[4, 4]


def test_rolling_errors():
    a = np.random.rand(10)

    # negative window size
    with pytest.raises(ValueError):
        mp.rolling.rolling_window(a, window_size=-1)

    # window size bigger than array
    with pytest.raises(ValueError):
        mp.rolling.rolling_window(a, window_size=11)

    # in reduction mode the window size needs to divide the input array exactly
    with pytest.raises(ValueError):
        mp.rolling.rolling_window(a, window_size=4, reduce=True)


def test_rolling_sum():
    # 1D
    a = np.random.rand(100)
    assert mp.rolling.rolling_sum(a, window_size=5)[0] == a[:5].sum()

    # 2D
    a = np.random.rand(10, 10)
    assert np.allclose(mp.rolling.rolling_sum(a, window_size=5)[0, 0], a[:5, :5].sum())
    assert np.allclose(mp.rolling.rolling_sum(a, window_size=5)[-1, -1], a[-5:, -5:].sum())

    # 3D
    a = np.random.rand(10, 10, 10)
    assert np.allclose(mp.rolling.rolling_sum(a, window_size=5)[0, 0, 0], a[:5, :5, :5].sum())
    assert np.allclose(mp.rolling.rolling_sum(a, window_size=5)[-1, -1, -1], a[-5:, -5:, -5:].sum())

    # 4D
    a = np.random.rand(10, 10, 10, 10)
    assert np.allclose(mp.rolling.rolling_sum(a, window_size=5)[0, 0, 0, 0], a[:5, :5, :5, :5].sum())
    assert np.allclose(mp.rolling.rolling_sum(a, window_size=5)[-1, -1, -1, -1], a[-5:, -5:, -5:, -5:].sum())

    # values
    a = np.random.rand(5, 5)
    assert np.allclose(a.sum(), mp.rolling.rolling_sum(a, window_size=5, reduce=True))


def test_rolling_sum_errors():
    a = np.random.rand(10, 10)

    # window 1 or lower
    with pytest.raises(ValueError):
        mp.rolling.rolling_sum(a, window_size=1)

    # window bigger than input data
    with pytest.raises(ValueError):
        mp.rolling.rolling_sum(a, window_size=11)


def test_rolling_mean():
    a = np.random.rand(10, 10, 10)
    assert np.allclose(mp.rolling.rolling_mean(a, window_size=5)[0, 0, 0], a[:5, :5, :5].mean())
    assert np.allclose(mp.rolling.rolling_mean(a, window_size=5)[-1, -1, -1], a[-5:, -5:, -5:].mean())

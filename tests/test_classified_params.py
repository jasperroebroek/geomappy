import numpy as np
from matplotlib.colors import BoundaryNorm

from geomappy.classified import parse_classified_plot_params


def test_scalar_params():
    a = np.random.RandomState(0).randint(0, 5, (10, 10))
    cmap, norm = parse_classified_plot_params(np.ma.fix_invalid(a))
    assert cmap.name == 'from_list'
    assert cmap.N == 5
    assert isinstance(norm, BoundaryNorm)
    assert norm.vmin == -1
    assert norm.vmax == 5
    assert norm._n_regions == 5
    assert norm.Ncmap == 5
    assert norm.extend == 'neither'
    assert not cmap.colorbar_extend


def test_scalar_params_levels():
    a = np.random.RandomState(0).randint(0, 5, (10, 10))
    cmap, norm = parse_classified_plot_params(np.ma.fix_invalid(a), levels=np.arange(0, 9))
    assert cmap.N == 9
    assert norm.vmin == -1
    assert norm.vmax == 9
    assert norm._n_regions == 9
    assert norm.Ncmap == 9

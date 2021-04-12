import matplotlib.pyplot as plt
import numpy as np
from pyproj import Proj
from ._read import RasterReader


class RasterReaderSet(object):
    def __init__(self, subdatasets, **kwargs):
        super(RasterReaderSet, self).__setattr__('_set', {})
        super(RasterReaderSet, self).__setattr__('_plot_columns', 3)
        for subdataset in subdatasets:
            name = subdataset[subdataset.rfind(":")+1:]
            self._set[name] = RasterReader(subdataset, **kwargs)
            super(RasterReaderSet, self).__setattr__(name, self._set[name])

    @property
    def plot_columns(self):
        return self._plot_columns

    @plot_columns.setter
    def plot_columns(self, columns):
        if not isinstance(columns, int):
            raise TypeError("columns needs to be an integer")

        self._plot_columns = columns

    def plot_params(self):
        columns = self.plot_columns
        if columns > len(self._set):
            return {"nrows": 1, "ncols": len(self._set)}
        else:
            nrows = len(self._set) // columns
            if len(self._set) % columns > 0:
                nrows += 1
            return {"nrows": nrows, "ncols": columns}

    def _plot_area(self, func, *args, **kwargs):
        if kwargs is None:
            kwargs = {}
        for i, key in enumerate(self._set.keys()):
            if i == 0:
                ax = getattr(self._set[key], func)(*args, **kwargs)
                kwargs.update({"ax": ax})
            else:
                getattr(self._set[key], func)(*args, **kwargs)
        return ax

    def plot_world(self, *args, **kwargs):
        return self._plot_area("plot_world", *args, **kwargs)

    def plot_tile(self, *args, **kwargs):
        return self._plot_area("plot_tile", *args, **kwargs)

    def _plot_data(self, func, *args, **kwargs):
        if kwargs is None:
            kwargs = {}
        fontsize = kwargs.get('fontsize', 10)
        figsize = kwargs.pop('figsize', (10, 10))
        f, ax = plt.subplots(**self.plot_params(), figsize=figsize)
        for i, key in enumerate(self._set.keys()):
            kwargs.update({"ax": ax[i]})
            getattr(self._set[key], func)(*args, **kwargs)
            ax[i].set_title(key, fontsize=fontsize)
        return ax

    def plot_map(self, *args, **kwargs):
        return self._plot_data("plot_map", *args, **kwargs)

    def plot(self, *args, **kwargs):
        return self.plot_map(*args, **kwargs)

    def plot_classified_map(self, *args, **kwargs):
        return self._plot_data("plot_classified_map", *args, **kwargs)

    def __setattr__(self, attr, value):
        for key in self._set:
            setattr(self._set[key], attr, value)

    def __getattr__(self, attr):
        if attr in dir(self):
            return attr

        if "plot" in attr:
            raise NotImplementedError("plotting interface not accessible through RasterReaderSet")

        test_key = self._set[list(self._set.keys())[0]]

        if hasattr(test_key, attr):
            if callable(getattr(test_key, attr)) and \
               not isinstance(getattr(test_key, attr), Proj):
                def wrapper(*args, **kwargs):
                    r = {key: getattr(self._set[key], attr)(*args, **kwargs) for key in self._set}
                    if not np.all([val is None for val in r.values()]):
                        return r
                    else:
                        return None
                return wrapper
            else:
                return {key: getattr(self._set[key], attr) for key in self._set}
        else:
            raise AttributeError(attr)

    def __repr__(self):
        return f"Set of Rasters: {list(self._set.keys())}"

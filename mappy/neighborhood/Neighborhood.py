"""
These objects, with the base being Neighborhood, can be used to in the rolling_functions to create non patterns as is
used in for example ArcGIS.
"""
import numpy as np
from mappy.plotting import plot_map
import matplotlib.pyplot as plt


class Neighborhood:
    def __init__(self):
        self._shape = None
        self._mask_flag = False
        self._mask = None

    @property
    def shape(self):
        return self._shape

    @property
    def mask_flag(self):
        return self._mask_flag

    @property
    def mask(self):
        return self._mask.copy()

    @property
    def ndim(self):
        return len(self._sape)

    @property
    def value_count(self):
        if self._mask_flag:
            return self._mask.sum()
        else:
            return np.prod(self._shape)

    def plot(self):
        if len(self.shape) == 2:
            plot_map(self._mask, figsize=(5, 5))
            plt.show()
        else:
            print(self._mask)

    def __repr__(self):
        if not self._mask_flag:
            s = (f"unmasked neighborhood of {len(self.shape)} dimensions; {self.shape}")
        else:
            s = (f"masked neighborhood of {len(self.shape)} dimensions; {self.shape}")
        return s


class NbrRectangle(Neighborhood):
    def __init__(self, window_size):
        self._mask_flag = False
        self._mask = None

        window_size = list(window_size)
        for number in window_size:
            if isinstance(number, int):
                if number < 2:
                    raise ValueError("window_size should be bigger than 1")
            else:
                raise TypeError("each element of window_size should be an integer")

        self._shape = window_size


class NbrCircular(Neighborhood):
    def __init__(self, radius, ndim=2):
        if isinstance(radius, int):
            if radius < 1:
                raise ValueError("Radius should be 1 or higher")
        else:
            raise TypeError("Radius should be an integer")

        if isinstance(ndim, int):
            if radius < 1:
                raise ValueError("Radius should be 1 or higher")
        else:
            raise TypeError("Dimensions should be an integer")

        diameter = radius * 2 + 1
        coords = []
        for i in range(ndim):
            ind = [np.newaxis] * ndim
            ind[i] = slice(None, None)
            ind = tuple(ind)
            coords.append(np.linspace(-1, 1, diameter)[ind])
        mask = coords[0] ** 2
        for i in range(1, len(coords)):
            mask = mask + coords[i] ** 2

        self._mask = np.sqrt(mask) <= 1
        self._mask_flag = True
        self._shape = [diameter] * ndim


class NbrIrregular(Neighborhood):
    def __init__(self, mask):
        if isinstance(mask, np.ndarray):
            self._mask = mask
        else:
            raise TypeError("Mask should be an ndarray")

        self._mask_flag = True
        self._shape = mask.shape


if __name__ == "__main__":
    print(NbrRectangle((5,10)))
    print(NbrCircular(250))
    print(NbrCircular(3))
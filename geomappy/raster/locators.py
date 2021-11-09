import numpy as np
import rasterio as rio


class BaseLocator:
    def __init__(self, loc=None, raster=None):
        if not loc is None:
            self.window, self.indexes = self._parse_loc(loc)
        self.r = raster

    def _parse_loc(self):
        raise NotImplementedError("BaseLocator cannot parse any input")

    def __getitem__(self, loc):
        if self.r.mode == 'w':
            raise NotImplementedError("Reading a file in 'w' mode is not allowed")
        window, indexes = self._parse_loc(loc)
        return self.r._get_data(window, indexes)

    def __setitem__(self, loc, data):
        if self.r.mode == 'r':
            raise NotImplementedError("Writing to a file in 'r' mode is not allowed")
        window, indexes = self._parse_loc(loc)
        if indexes is None:
            indexes = 1
        elif not isinstance(loc, int) and len(indexes) == 1:
            indexes = indexes[0]
        self.r._set_data(window, indexes, data)


class GeoLocator(BaseLocator):
    def _parse_loc(self, loc):
        error_msg = "Input to GeoLocator not understood. Needs to contain an iterator of coordinates and optionally " \
                    "an layer indices (int or iterable)."
        if not hasattr(loc, "__iter__"):
            raise ValueError(error_msg)
        elif isinstance(loc, (tuple, list, np.ndarray, rio.coords.BoundingBox)) and len(loc) == 4:
            ind = loc
            indexes = None
        elif isinstance(loc, tuple) and len(loc) == 2 and \
                isinstance(loc[0], (tuple, list, np.ndarray, rio.coords.BoundingBox)) and len(loc[0]) == 4:
            ind = loc[0]
            indexes = loc[1]
        else:
            raise ValueError(error_msg)

        ind = self.r.get_pointer(ind)
        window = self.r._tiles[ind]
        return window, indexes


class TileLocator(BaseLocator):
    def _parse_loc(self, loc):
        if isinstance(loc, int):
            ind = loc
            indexes = None
        elif isinstance(loc, tuple) and isinstance(loc[0], int):
            ind = loc[0]
            indexes = loc[1]
        else:
            raise ValueError("Input iTileLocator not understood. Needs to be a integer indicating a tile of the "
                             "raster, optionally followed by rasterio indexes.")

        ind = self.r.get_pointer(ind)
        window = self.r._tiles[ind]
        return window, indexes


class IdxLocator(BaseLocator):
    def _parse_loc(self, loc):
        msg = ("Input IdxLocator not understood. Needs to be one or two slices, optionally followed by rasterio "
               "indexes")

        if isinstance(loc, slice):
            ind = loc, slice(None)
            indexes = None
        elif isinstance(loc, int):
            ind = slice(loc, loc + 1), slice(None)
            indexes = None
        elif isinstance(loc, tuple) and len(loc) in (2, 3):
            if isinstance(loc[0], slice):
                ind_0 = loc[0]
            elif isinstance(loc[0], int):
                ind_0 = slice(loc[0], loc[0] + 1)
            else:
                raise ValueError(msg)

            if isinstance(loc[1], slice):
                ind_1 = loc[1]
            elif isinstance(loc[1], int):
                ind_1 = slice(loc[1], loc[1] + 1)
            else:
                raise ValueError(msg)

            ind = ind_0, ind_1

            if len(loc) == 3:
                indexes = loc[2]
            else:
                indexes = None
        else:
            raise ValueError(msg)

        ind = self.r.get_pointer(ind)
        window = self.r._tiles[ind]
        return window, indexes

import os

import numpy as np
import rasterio as rio

from ._base import RasterBase


class RasterWriter(RasterBase):
    """
    Subclass of RasterBase, file opened in 'w' mode.

    Parameters
    ----------
    location : str
        Location of the map
    tiles : int or tuple of ints, optional
        Tiles object. See property
    force_equal_tiles : bool, optional
        Force tiles of equal size. This defaults to False, which last the last tiles be slightly larger.
    window_size : int, optional
        Window size to be set on the map. See property
    ref_map : str, optional
        Location of file that a rasterio profile is pulled from and used for the creation of self. The reference map
        will not do anything if a profile is passed to the 'profile' parameter.
    overwrite : bool, optional
        Allowed to overwrite a file if it already exists. Default is False
    compress : bool, optional
        compress the output file, default is False. If profile is provided the compression parameter is taken from there
    dtype : numpy dtype, optional
        Type of data that is going to be written to the file. This parameter doesn't work in combination with passing a
        rasterio profile to the 'profile' parameter. The default is 'np.float64'
    nodata : numeric, optional
        nodata value that is used for the rasterio profile. The default is np.nan. This parameter doesn't work in
        combination with passing a rasterio profile to the 'profile' parameter.
    count : int, optional
        Count of indexes of the file. If -1 it will be taken from the reference file. The default is -1.
    profile : dict, optional
        custom rasterio profile can be passed which will be used directly to create the file with. All the other
        parameters are neglected.

    Raises
    ------
    IOError
        - `location` doesn't exist to pull a profile from and no reference map is given
        - reference maps is given but can't be found
        - can't overwrite an already existing file if not specified with the 'overwrite' parameter
    """

    def __init__(self, location, *, tiles=1, force_equal_tiles=False, window_size=1, ref_map=None, overwrite=False,
                 compress=False, dtype=np.float64, nodata=np.nan, count=-1, profile=None):
        self._mode = "w"

        # todo; check logic
        # todo; remove ref_map
        if profile is None:
            # load the rasterio profile either from a reference map or the location itself if it already exists
            if ref_map is None:
                if not os.path.isfile(location):
                    raise IOError("Location doesn't exist and no reference map or profile given")
                with rio.open(location) as f:
                    self._profile = f.profile
            else:
                if not os.path.isfile(ref_map):
                    raise IOError("Reference map can't be found")
                with rio.open(ref_map) as f:
                    self._profile = f.profile

            # todo; create folder if non - existent!!
            self._profile['dtype'] = dtype
            self._profile['nodata'] = np.array((nodata,)).astype(dtype)[0]
            self._profile['driver'] = "GTiff"
            if compress:
                self._profile['compress'] = 'lzw'
            if count != -1:
                if not isinstance(count, int):
                    raise TypeError("count should be an integer")
                if count < 1:
                    raise ValueError("count should be a positive integer")
                self._profile['count'] = count
        else:
            self._profile = profile

        # check if file exists and if the object is allowed to overwrite data
        if os.path.isfile(location):
            if not overwrite:
                raise IOError(f"Can't overwrite if not explicitly stated with overwrite parameter\n{location}")

        # todo; work on APPROXIMATE_STATISTICS
        #   https://gdal.org/doxygen/classGDALRasterBand.html#a48883c1dae195b21b37b51b10e910f9b
        #   https://github.com/mapbox/rasterio/issues/244
        #   https://rasterio.readthedocs.io/en/latest/topics/tags.html

        # todo; create **kwargs that feed into the rio.open function

        self._fp = rio.open(location, "w+", **self._profile, BIGTIFF="YES")
        self._location = self._fp.name

        # setting parameters by calling property functions
        self.window_size = window_size
        self.set_tiles(tiles, force_equal_tiles)

        self._current_ind = 0

        self.collector.append(self)
        self.mmap_collector = {}

    def __setitem__(self, i, data):
        """
        writing data to file with same indexing as RasterReader
        """
        ind, indexes = self.ind_user_input(i)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        ind = self.get_pointer(ind)
        if data.shape[-2:] != (self.get_height(ind), self.get_width(ind)):
            data = data[:, self.ind_inner[0], self.ind_inner[1]]

        self.write(data, indexes=indexes, window=self._tiles[ind])

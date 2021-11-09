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
        Location of the raster. If it already exists the overwrite parameter needs to be set to True or an error
        is raised. If the profile that is constructed here is the same as the profile of the already existing
        raster, the raster is opened in 'r+' mode, otherwise the original file is overwritten.
    tiles : int or tuple of ints, optional
        Tiles object. See property
    force_equal_tiles : bool, optional
        Force tiles of equal size. This defaults to False, which last the last tiles be slightly larger.
    window_size : int, optional
        Window size to be set on the map. See property
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

    def __init__(self, fp, *, tiles=1, force_equal_tiles=False, window_size=1, overwrite=False,
                 compress=False, dtype=np.float64, nodata=np.nan, count=-1, profile=None):

        if profile is None:
            # Attempt reading rasterio profile from existing file
            if not os.path.isfile(fp):
                raise IOError("Location doesn't exist and no reference map or profile given")
            with rio.open(fp) as f:
                profile = f.profile

            # todo; create folder if non - existent!!
            profile['dtype'] = dtype
            profile['nodata'] = np.asarray((nodata,), dtype=dtype)[0]
            profile['driver'] = "GTiff"
            if compress:
                profile['compress'] = 'lzw'
            if count != -1:
                profile['count'] = count
        else:
            profile = profile

        # check if file exists and if the object is allowed to overwrite data
        if os.path.isfile(fp):
            if not overwrite:
                raise IOError(f"Can't overwrite if not explicitly stated with overwrite parameter\n{fp}")
            with rio.open(fp) as f:
                test_profile = f.profile

            if profile == test_profile:
                mode = 'r+'
            else:
                mode = 'w+'
        else:
            mode = 'w+'

        # todo; work on APPROXIMATE_STATISTICS
        #   https://gdal.org/doxygen/classGDALRasterBand.html#a48883c1dae195b21b37b51b10e910f9b
        #   https://github.com/mapbox/rasterio/issues/244
        #   https://rasterio.readthedocs.io/en/latest/topics/tags.html

        # todo; create **kwargs that feed into the rio.open function

        self._fp = rio.open(fp, mode, **profile, BIGTIFF="YES")
        self._mode = "w"
        self._init(window_size, tiles, force_equal_tiles)

    def _set_data(self, window, indexes, data):
        self.write(data, indexes=indexes, window=window)

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

        self._set_data(self._tiles[ind], indexes, data)

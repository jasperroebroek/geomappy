#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import rasterio as rio
from map.MapBase import MapBase


class MapWrite(MapBase):
    """
    Subclass of MapBase, file opened in 'w' mode.
    
    Instance attributes
    -------------------
    _writing_buffer : ndarray
        this array is the temporary place where an array is store before it is written to disk
    
    Parameters
    ----------
    location : str
        Location of the map
    tiles : int or tuple of ints
        Tiles object. See property        
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
        Count of layers of the file. If -1 it will be taken from the reference file. The default is 1.
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

    def __init__(self, location, *, tiles=1, window_size=1, ref_map=None, overwrite=False, compress=False,
                 dtype=np.float64, nodata=np.nan, profile=None, count=-1):
        self._location = location
        self._mode = "w"

        if isinstance(profile, type(None)):
            # load the rasterio profile either from a reference map or the location itself if it already exists
            if isinstance(ref_map, type(None)):
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
            # todo; adapt default nodata to data type
            self._profile['dtype'] = dtype
            nodata = np.array((nodata, )).astype(dtype)[0]
            self._profile['nodata'] = nodata
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
        self._file = rio.open(location, "w", **self._profile)

        # setting parameters by calling property functions
        self.window_size = window_size
        self.tiles = tiles

        self._current_ind = 0
        self._writing_buffer = None

        self.collector.append(self)

        if isinstance(self._file.crs, type(None)):
            self._epsg = None
        else:
            self._epsg = self._file.crs.to_epsg()

    def get_writing_buffer(self):
        """
        Returns the current writing buffer
        
        Returns
        -------
        np.ndarray of the current writing buffer or None when it is not set
        """
        return self._writing_buffer.copy()

    def set_writing_buffer(self, writing_buffer):
        """
        set buffer that will be used when writing data. Layer information should be presented on the third axis.
        
        Parameters
        ----------
        writing_buffer : np.ndarray
            data that will be used when writing to the output file
        
        Raises
        ------
        TypeError
            writing_buffer is not an ndarray
        ValueError
            Shape of writing_buffer is not of the dimensions and shape that is expected with the current tiles
        """
        if not isinstance(writing_buffer, np.ndarray):
            raise TypeError("Buffer needs to be a numpy array")
        if writing_buffer.ndim not in (2, 3):
            raise ValueError(f"ndarray not of right dimensions: {writing_buffer.ndim}")
        if self.profile['count'] > 1 and writing_buffer.shape[-1] != self.profile['count']:
            raise ValueError(
                f"Layers don't match\ncount: {self.profile['count']}\nwriting_buffer: {writing_buffer.shape[-1]}")
        if writing_buffer.shape != self.get_shape():
            writing_buffer = writing_buffer[self.ind_inner]
        if writing_buffer.shape != self.get_shape():
            raise ValueError(
               f"Shapes don't match:\n- shape writing_buffer: {writing_buffer.shape}\n- tile shape: {self.get_shape()}")

        self._writing_buffer = writing_buffer

    writing_buffer = property(get_writing_buffer, set_writing_buffer)

    def write(self, ind):
        """
        write data to file at the current tile
        
        Parameters
        ----------
        ind : .
            see self.get_pointer()
        
        Raises
        ------
        NameError
            when writing buffer is not set
        """
        ind = self.get_pointer(ind)
        if isinstance(self.writing_buffer, type(None)):
            raise NameError("No writing buffer found")
        if self.profile['count'] == 1:
            self._file.write(self.writing_buffer, 1, window=self._tiles[ind])
        else:
            for i in range(1, self.profile['count'] + 1):
                self._file.write(self.writing_buffer[:, :, i - 1], i, window=self._tiles[ind])

    def __setitem__(self, ind, writing_buffer):
        """
        redirects to self.set_writing_buffer and self.write
        """
        self.writing_buffer = writing_buffer
        self.write(ind)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper around rasterio functionality. Object creation happens by calling Raster() which
distributes the requests to `RasterReader()`` and ``RasterWriter()`` for reading and writing respectively
(rasterio mode='r' and mode='w'). It is possible for both reading and writing to do this chunk by
chunk in tiles for files that are bigger than the installed RAM. A window_size parameter can be used
for the calculation of focal statistics. In reading this will add a buffer of data around the actual
tile to be able to calculate values on the border. By setting the same window_size in a ``RasterWriter``
object trims this border without values before writing it to the file as if everything happened in
one calculation.
"""
import os

import numpy as np
import rasterio as rio

from ._base import RasterBase
from ._read import RasterReader
from ._write import RasterWriter
from ._set import RasterReaderSet


class Raster:
    """
    Creating a an interface to raster data. Depending on the `mode` a ``RasterReader`` or
    ``RasterWriter`` object will be returned. If reading the data and several subdatasets
    are present a RasterReaderSet will be created, which mimics a dictionary of ``RasterReader``
    objects, but implements the same properties as the ``RasterReader`` (e.g. set.shape returns
    a dictionary of shapes of the subdatasets).

    Parameters
    ----------
    fp : str, file object or pathlib.Path object
        A filename, URL, a file object opened in binary ('rb') mode,
        or a Path object (see rasterio documentation).
    mode : {'r', 'w'}
        Mode of opening the file. Corresponds  with standard rasterio functionality
    **kwargs
        Arguments to be passed to either RasterReader or RasterWriter class creation functions

    Returns
    -------
    If mode == "r":
        ``RasterReader`` object
    If mode == "w":
        ``RasterWriter`` object
    """
    def __new__(self, fp, mode="r", **kwargs):
        # distribute creation based on mode
        if mode not in ("r", "w"):
            raise ValueError("Mode not recognized. Needs to be 'r' or 'w'")

        if mode == "r":
            file = rio.open(fp)
            subdatasets = file.subdatasets
            file.close()

            if len(subdatasets) == 0:
                return RasterReader(fp, **kwargs)
            elif len(subdatasets) == 1:
                return RasterReader(subdatasets[0], **kwargs)
            else:
                return RasterReaderSet(subdatasets, **kwargs)

        if mode == "w":
            return RasterWriter(fp, **kwargs)

    @staticmethod
    def close(verbose=True):
        """
        function closing all connections created with the RasterBase class and its subclasses.
        """
        print()
        for m in RasterBase.collector:
            if verbose:
                print(f"close file: '{m.location}'")
            m.close(clean=False, verbose=False)
        RasterBase.collector = []

    @staticmethod
    def get_tiles():
        """
        returning tile settings for all open maps
        """
        return {m.location: m.tiles for m in RasterBase.collector}

    @staticmethod
    def set_tiles(tiles):
        """
        setting tile settings for all open maps

        Parameters
        ----------
        tiles : int or tuple
            Parameter for setting the RasterBase.tiles property
        """
        for m in RasterBase.collector:
            m.tiles = tiles

    @staticmethod
    def get_window_size():
        """
        retrieve window_size settings for all open maps
        """
        return {m.location: m.window_size for m in RasterBase.collector}

    @staticmethod
    def set_window_size(window_size):
        """
        set window_size settings for all open maps
        todo; force_equal_size

        Parameters
        ----------
        window_size : int, optional
            'window_size' parameter for RasterBase.window_size property
        """
        for m in RasterBase.collector:
            m.window_size = window_size

    @staticmethod
    def get_rasters(self):
        """
        Returns a list of all open maps
        """
        return RasterBase.collector

    @staticmethod
    def equal(m1, m2):
        """
        function to check equality of two maps, on all parameters as well as values

        Parameters
        ----------
        m1, m2 : RasterReader
            Maps to be compared on settings and values

        Returns
        -------
        equality : [bool]
        """
        if not isinstance(m1, RasterReader) or not isinstance(m1, RasterReader):
            raise TypeError("Only RasterReader objects can be compared in this manner")

        if not m1.params == m2.params:
            return False
        if not m1.profile == m2.profile:
            return False

        params = [m1.params, m2.params]
        m1.set_tiles_as_chunks()
        m2.set_tiles_as_chunks()

        try:
            for i in m1:
                if not np.allclose(m1.iloc[i], m2.iloc[i], equal_nan=True):
                    return False
        finally:
            m1.params, m2.params = params

        return True

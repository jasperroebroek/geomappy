#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .MapBase import MapBase
from .MapRead import MapRead
from .MapWrite import MapWrite
import numpy as np
import warnings


class Map:
    """
    Creating a map object. Depending on the `mode` a MapRead() or MapWrite() object will be returned.
    
    Parameters
    ----------
    location : str
        Pathname to the raster file, either to create or the read.
    mode : {'r','w'}
        Mode of opening the file. Corresponds  with standard rasterio functionality
    **kwargs
        Arguments to be passed to either MapRead or MapWrite class creation functions
        
    Returns
    -------
    If mode == "r":
        `MapRead` object
    If mode == "w":
        `MapWrite` object
    
    Raises
    ------
    ValueError
        Error is raised when mode is not 'r' and not 'w'
    """

    def __new__(self, location, mode="r", **kwargs):
        # distribute creation based on mode
        if mode not in ("r", "w"):
            raise ValueError("Mode not recognized. Needs to be 'r' or 'w'")

        if mode == "r":
            return MapRead(location, **kwargs)
        if mode == "w":
            return MapWrite(location, **kwargs)

    @staticmethod
    def close():
        """
        function closing all connections created with the MapBase class and its subclasses.
        """
        print()
        for m in MapBase.collector:
            print(f"close file: '{m.location}'")
            m.close(clean=False, verbose=False)
        MapBase.collector = []

    @staticmethod
    def get_tiles():
        """
        returning tile settings for all open maps
        """
        l = {}
        for m in MapBase.collector:
            l.update({m.location: m.tiles})
        return l

    @staticmethod
    def set_tiles(tiles):
        """
        setting tile settings for all open maps

        Parameters
        ----------
        tiles : int or tuple
            Parameter for setting the MapBase.tiles property
        """
        l = []
        for m in MapBase.collector:
            m.tiles = tiles
            l.append(m.tiles)
        l = np.array(l)
        if ~np.all(l == l[0, :]):
            warnings.warn("Not all tiles are set equal")

    @staticmethod
    def get_window_size():
        """
        retrieve window_size settings for all open maps
        """
        l = {}
        for m in MapBase.collector:
            l.update({m.location: m.window_size})
        return l

    @staticmethod
    def set_window_size(window_size):
        """
        set window_size settings for all open maps

        Parameters
        ----------
        window_size : int, optional
            'window_size' parameter for MapBase.window_size property
        """
        for m in MapBase.collector:
            m.window_size = window_size

    @staticmethod
    def maps():
        """
        Returns a list of all open maps
        """
        return MapBase.collector

    @staticmethod
    def equal(m1, m2):
        """
        function to check equality of two maps, on all parameters as well as values
        
        Parameters
        ----------
        m1, m2 : MapRead
            Maps to be compared on settings and values
            
        Returns
        -------
        equality : [bool]
        """
        # todo; test this function
        if isinstance(m1, MapWrite) or isinstance(m2, MapWrite):
            return False
        if m1.tiles != m2.tiles:
            print("Tiles don't match")
            return False
        if m1.window_size != m2.window_size:
            print("Window size doesn't match")
            return False
        if m1.get_file_shape() != m2.get_file_shape():
            print("File shape doesn't match")
            return False
        if not np.allclose(m1.get_file_bounds(), m2.get_file_bounds()):
            print("Bounds don't match")
            return False

        for i in m1:
            try:
                d = np.allclose(m1[i], m2[i], equal_nan=True)
                if not d:
                    return False
            except (ValueError, IndexError):
                return False

        return True

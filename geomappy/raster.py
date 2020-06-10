#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper around rasterio functionality. Object creation happens by calling Raster() which distributes the requests to
_RasterReader() and _RasterWriter() for reading and writing respectively (rasterio mode='r' and mode='w'). It is
possible for both reading and writing to do this chunk by chunk in tiles for files that are bigger than the installed
RAM. A window_size parameter can be used for the calculation of focal statistics. In reading this will add a buffer of
data around the actual tile to be able to calculate values on the border. By setting the same window_size in a
_RasterWriter object trims this border without values before writing it to the file as if everything happened in one calculation.
"""

import copy
import os
import warnings

import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import rasterio as rio
from numpy.lib.index_tricks import s_
from pyproj import Proj
from shapely.geometry import Point, Polygon

from .basemap import basemap as basemap_function
from geomappy.plotting import plot_map, plot_classified_map
from geomappy.utils import progress_bar
from .focal_statistics import correlate_maps, focal_statistics
from .profile import resample_profile
from .bounds import bounds_to_platecarree, bounds_to_polygons, bounds_to_data_projection


class Raster:
    """
    Creating a map object. Depending on the `mode` a _RasterReader() or _RasterWriter() object will be returned.
    
    Parameters
    ----------
    location : str
        Pathname to the raster file, either to create or the read.
    mode : {'r','w'}
        Mode of opening the file. Corresponds  with standard rasterio functionality
    **kwargs
        Arguments to be passed to either _RasterReader or _RasterWriter class creation functions
        
    Returns
    -------
    If mode == "r":
        `_RasterReader` object
    If mode == "w":
        `_RasterWriter` object
    
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
            return _RasterReader(location, **kwargs)
        if mode == "w":
            return _RasterWriter(location, **kwargs)

    @staticmethod
    def close(verbose=True):
        """
        function closing all connections created with the _RasterBase class and its subclasses.
        """
        print()
        for m in _RasterBase.collector:
            if verbose:
                print(f"close file: '{m.location}'")
            m.close(clean=False, verbose=False)
        _RasterBase.collector = []

    @staticmethod
    def get_tiles():
        """
        returning tile settings for all open maps
        """
        return {m.location: m.tiles for m in _RasterBase.collector}

    @staticmethod
    def set_tiles(tiles):
        """
        setting tile settings for all open maps

        Parameters
        ----------
        tiles : int or tuple
            Parameter for setting the _RasterBase.tiles property
        """
        for m in _RasterBase.collector:
            m.tiles = tiles

    @staticmethod
    def get_window_size():
        """
        retrieve window_size settings for all open maps
        """
        return {m.location: m.window_size for m in _RasterBase.collector}

    @staticmethod
    def set_window_size(window_size):
        """
        set window_size settings for all open maps

        Parameters
        ----------
        window_size : int, optional
            'window_size' parameter for _RasterBase.window_size property
        """
        for m in _RasterBase.collector:
            m.window_size = window_size

    @staticmethod
    def maps():
        """
        Returns a list of all open maps
        """
        return _RasterBase.collector

    @staticmethod
    def equal(m1, m2):
        """
        function to check equality of two maps, on all parameters as well as values
        
        Parameters
        ----------
        m1, m2 : _RasterReader
            Maps to be compared on settings and values
            
        Returns
        -------
        equality : [bool]
        """
        # todo; test this function
        if isinstance(m1, _RasterWriter) or isinstance(m2, _RasterWriter):
            print("One of the objects is not readable")
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


class _RasterBase:
    """
    Main map object, based on rasterio functionality. The rasterio pointer is exposed through _RasterBase._file

    Attributes
    ----------
    collector : list
        Every new instance of _RasterBase, or its children, will be added to this list. This is convenient for closing of
        all file connections with the following loop:

            >>> for i in _RasterBase.collector:
            ..:    i.close()

        this functionality is callable by:

            >>> Raster.close()

    _c_tiles : int
        Number of tiles that are set on the raster. Set and get via tiles property
    _current_ind : int
        last index passed to the object (last call to self.get_pointer())
    _data_proj : `Proj`
        Proj object for transformations
    _file : rasterio pointer
        rasterio file pointer
    _fringe : int
        amount of cells in each direction for windowed calculation.
        fringe = window_size // 2
    _h_tiles : int
        number of tiles in horizontal direction set on the raster
    _horizontal_bins : list of int
        bins of the horizontal shape of the raster. for example: a file with shape (1000,800) will have the following
        horizontal_bins variable if h_tiles is set to 4: [0,200,400,600,800]
    _ind_inner : tuple of slice
        numpy index slice that removes the padded cells that are added when reading data with a window_size set to
        something bigger than 1. for example:
            window_size = 5
            fringe = 2
            ind_inner = [fringe:-fringe,fringe:-fringe]
    _iter : list of tuples of ints
        list of tuples containing the indices of the tiles on the final raster. (1,2) corresponds with the first row and
        second tile. The index of this tuple in the list is the index that should be passed to the object to obtain the
        data corresponding to that specific tile.
    _location : str
        pathname to file
    _profile : dict
        rasterio profile corresponding to the filepointer
    _tiles : list of rio.windows
        list of windows to be iterated over to access all the data in the file
    _v_tiles : int
        same as h_tiles but in vertical direction
    _vertical_bins : list of int
        same as horizontal_bins but in vertical direction
    _window_size : int
        window size to be considered when performing windowed operations on the raster

    window_size
        get and set _window_size
    tiles
        get and set _tiles
    epsg
        get and set epsg code if not present
    ind_inner
        get slice to remove fringes introduced by window_size
    c_tiles
        get number of tiles
    signature
        get _RasterBase signature
    profile
        get rasterio file profile
    location
        file location
    bounds
        get file bounds
    shape
        get file shape
    """

    collector = []

    def __init__(self, **kwargs):
        raise NotImplementedError("_RasterBase is used as the base class for _RasterReader and _RasterWriter instances. "
                                  "This class can't be used directly.")
        self._location = ""
        self._profile = {}
        self._mode = "undefined"
        self._file = None
        self._current_ind = 0
        self._c_tiles = 1
        self._h_tiles = 1
        self._v_tiles = 1
        self._window_size = 1
        self._fringe = 1
        self._ind_inner = None
        self._horizontal_bins = []
        self._vertical_bins = []
        self._iter = []
        self._tiles = []
        self._epsg = None
        self._data_proj = None
        self._transform = None

    @property
    def location(self):
        """
        Location of opened file in the object
        """
        return copy.deepcopy(self._location)

    def get_window_size(self):
        """
        gives the current window size
        """
        return copy.deepcopy(self._window_size)

    def set_window_size(self, window_size=1):
        """
        Sets the window_size and fringe parameters.

        Parameters
        ----------
        window_size : int
            size of the window used in windowed calculations

        Raises
        ------
        ValueError
            1: window_size is smaller than 1
            2: window_size is even
        """
        window_size = int(window_size)
        if window_size < 1:
            raise ValueError("window_size needs to be bigger than, or equal to 1")
        if window_size % 2 == 0:
            raise ValueError("window_size needs to be uneven")

        self._window_size = window_size
        self._fringe = window_size // 2
        self._ind_inner = s_[self._fringe:-self._fringe, self._fringe:-self._fringe]

        try:
            # this will fail the first time that it is called, but that problem is solved in the __init__ functions
            # by calling self.set_tiles with user inputted or default tile settings
            self.tiles = self.tiles
        except AttributeError:
            pass

    window_size = property(get_window_size, set_window_size)

    def get_tiles(self):
        """
        function to retrieve the current tile settings
        """
        return copy.deepcopy(self._v_tiles), copy.deepcopy(self._h_tiles)

    def set_tiles(self, tiles=1, force_equal_tiles=False):
        """
        Function to set the tiles.

        If tiles is given as a list, v_tiles and h_tiles are calculated from this parameter. Otherwise c_tiles is
        calculated by the v_tiles * h_tiles. Tiles as an integer is an approximate parameter. No guarantees that
        c_tiles will exactly be the amount of tiles that were requested. Tiles as a list is also an approximate
        parameter. If the shape of the file is not divisible by the values they are respectively raised with one until
        they reach a value that lets the raster be split up in equal chunks.

        Parameters
        ----------
        tiles : int or tuple of ints
            Passing an integer will create 'roughly' that amount of tiles. A tuple is read as vertical and horizontal
            tile count. These are still not guaranteed to work exactly. Internally this function looks for the closest
            parameters that can achieve to split the data up in equal parts. This behaviour might change in the future.
        force_equal_tiles : bool, optional
            If set to True the tiles will be adapted untill a perfect fit is created, which is the default. If set to
            False the last tiles on both axes can be sligthly bigger than the rest.

        Raises
        ------
        ValueError
            1: tuple size not equal to 2
            2: tiles not passed as positive integers
            3: c_tiles smaller than 0
        TypeError
            Tiles are not given as integers
        """
        if type(tiles) in (tuple, list):
            if len(tiles) != 2:
                raise ValueError("If a tuple is passed to set_window_size it needs "
                                 "to be of length 2; horizontal and vertical bin count.")
            if type(tiles[0]) != int or type(tiles[1]) != int:
                raise TypeError("Tiles need to be passed as integers")
            if tiles[0] < 1 or tiles[1] < 1:
                raise ValueError("Tiles have to be positive integers")

            self._v_tiles = tiles[0]
            self._h_tiles = tiles[1]

        else:
            if type(tiles) != int:
                raise TypeError("Tiles need to be passed as an integer, or as a list of integers")
            if tiles < 0:
                raise ValueError("Tiles have to be positive integers")

            # routine to infer h_tiles and v_tiles from c_tiles if set
            self._v_tiles = int(np.sqrt(tiles))
            self._h_tiles = int(tiles / self._v_tiles)

        shape = self._file.shape

        # routine to make sure the data is split up in equal parts
        if force_equal_tiles:
            while True:
                if shape[0] // self._v_tiles != shape[0] / self._v_tiles:
                    self._v_tiles += 1
                elif shape[1] // self._h_tiles != shape[1] / self._h_tiles:
                    self._h_tiles += 1
                else:
                    if type(tiles) in (tuple, list):
                        if (tiles[0] != self._v_tiles and tiles[0] > 1) or \
                                (tiles[1] != self._h_tiles and tiles[1] > 1):
                            print(f"v_tiles set to {self._v_tiles}")
                            print(f"h_tiles set to {self._h_tiles}")
                    break

            # creating the actual tiles that will be used in the reading and writing of the rasters
            self._vertical_bins = [ix for ix in range(0, shape[0] + 1, shape[0] // self._v_tiles)]
            self._horizontal_bins = [ix for ix in range(0, shape[1] + 1, shape[1] // self._h_tiles)]

        else:
            vertical_step = shape[0] // self._v_tiles
            horizontal_step = shape[1] // self._h_tiles

            # creating the actual tiles that will be used in the reading and writing of the rasters
            self._vertical_bins = [i * vertical_step for i in range(self._v_tiles)] + [shape[0]]
            self._horizontal_bins = [i * horizontal_step for i in range(self._h_tiles)] + [shape[1]]

        # recalculation of c_tiles
        self._c_tiles = self._v_tiles * self._h_tiles

        self._iter = [(i, j) for i in range(1, len(self._vertical_bins))
                      for j in range(1, len(self._horizontal_bins))]

        # store of the rasterio window objects
        self._tiles = []

        for i, j in self._iter:
            if self._mode == "r":
                # in reading mode a buffer around the raster is used
                s = ((self._vertical_bins[i - 1] - self._fringe,
                      self._vertical_bins[i] + self._fringe),
                     (self._horizontal_bins[j - 1] - self._fringe,
                      self._horizontal_bins[j] + self._fringe))
            if self._mode == "w":
                # in writing the buffer is not needed
                s = ((self._vertical_bins[i - 1],
                      self._vertical_bins[i]),
                     (self._horizontal_bins[j - 1],
                      self._horizontal_bins[j]))

            # rasterio window object is stored in self._tiles
            self._tiles.append(rio.windows.Window.from_slices(*s, boundless=True))

        self._tiles.append(None)

        self._current_ind = 0

    tiles = property(get_tiles, set_tiles)

    @property
    def ind_inner(self):
        """
        2D slice to remove fringes from ndarrays in the following way:
            >>> m[0][m.ind_inner]
        """
        return self._ind_inner

    @property
    def c_tiles(self):
        """
        Gives the amount of tiles set on the map
        """
        return copy.deepcopy(self._c_tiles)

    @property
    def signature(self):
        """
        Returns a dictionary that can be used to open files in an identical manner
        """
        return {'tiles': self.tiles,
                'window_size': self.window_size}

    def get_epsg(self):
        if isinstance(self._epsg, type(None)):
            raise TypeError("This file does not contain any epsg code. It can be set through set_epsg()")
        return self._epsg

    def set_epsg(self, epsg):
        if isinstance(self._file.crs, type(None)):
            if isinstance(epsg, type(None)):
                raise TypeError("EPSG can't be found in the file and is not provided in the initialisation")
            self._epsg = epsg
        else:
            self._epsg = self._file.crs.to_epsg()

        self._data_proj = Proj(init=f"epsg:{self._epsg}", preserve_units=False)
        if self._epsg == 4326 or self._epsg == "4326":
            self._transform = ccrs.PlateCarree()
        else:
            self._transform = ccrs.epsg(self._epsg)

    epsg = property(get_epsg, set_epsg)

    def get_pointer(self, ind):
        """
        Converts different types of pointer to the right index of the self._tiles list

        Parameters
        ----------
        ind :
            [bounds] -> [left,bottom,right,top]
                Rasterio bounds object or list of four numbers, the bounds of the
                new window. A tile is created based on these bounds like by passing
                a slice and gets added as the last entry of self._tiles at position
                self._c_tiles. If it is a list or tuple it is assumed to be coordinates
                in the latlon system (EPSG:4326) while a Rasterio BoundingBox is assumed
                to be in the coordinate system of the data.
            [slice]
                Works like passing an int twice. It finds the outer coordinates
                of both tiles and creates a new window based on those new
                bounds. This window gets added to the list self._tiles at index
                self._c_tiles, the last entry. This tile will not be accessed
                when looping over the object, but all other operations can be
                performed on this  new tile.
            [int]
                Index of self._tiles range(0,self._c_tiles)
            [None]
                Can be used to access the file as a whole

        Returns
        -------
        ind : int
            Index in range(0, self._c_tiles+1)

        Raises
        ------
        TypeError
            1: if ind is a tuple of length two:
                one of both numbers is not an integer
            3: if ind is a slice:
                start/stop parameter is not an integer

        IndexError
            1: if ind is a tuple of length two:
                a: first number of tuple not in range of v_tiles
                b: second number of tuple not in range of h_tiles
            3: if ind is a slice:
                numbers are not valid as an index for the self._tiles list
            4: if ind is an int:
                Index out of range

        ValueError
            2: if ind is a tuple or boundingbox and the coordinates are not possible
               to convert to lat and lon values

        KeyError
            5: None of the previous cases are applicable when passing an unknown
               type or when a list/tuple is passed that doesn't contain exactly
               2 or 4 parameters

        Examples
        --------
        self[4]
            get data at self._tiles[4]
        self[3:10]
            create new window. Read everything within the bounds of tile 3
            and tile 10.
        self[(0,30,30,60)]
            create new window with the specified bounds

        to access the slice capability in other fuctions than __getitem__:
        1: pass a slice directly -> slice(1,2)
        2: use numpy -> s_[1:2]
        """
        if isinstance(ind, type(None)):
            ind = self._file.bounds

        if isinstance(ind, (tuple, list, rio.coords.BoundingBox)):
            if not isinstance(ind, rio.coords.BoundingBox):
                x0, y0, x1, y1 = ind

                if x0 < -180 or x0 > 180:
                    raise ValueError(f"Left coordinate of the globe: {x0}")
                if y0 < -90 or y0 > 90:
                    raise ValueError(f"Bottom coordinate of the globe: {y0}")
                if x1 < -180 or x1 > 180:
                    raise ValueError(f"Right coordinate of the globe: {x1}")
                if y1 < -90 or y1 > 90:
                    raise ValueError(f"Top coordinate of the globe: {y1}")

                bounds1 = bounds_to_data_projection(self._data_proj, (x0, y0, x1, y1))
                bounds2 = bounds_to_data_projection(self._data_proj, (x0, y1, x1, y0))
                x = (bounds1[0], bounds1[2], bounds2[0], bounds2[2])
                y = (bounds1[1], bounds1[3], bounds2[1], bounds2[3])

                ind = rio.coords.BoundingBox(np.min(x), np.min(y), np.max(x), np.max(y))

            # add window object to self._tiles list
            temporary_window = rio.windows.from_bounds(*ind, self._file.transform)

            # round the entries and create the window
            col_off = np.round(temporary_window.col_off).astype(int)
            row_off = np.round(temporary_window.row_off).astype(int)
            width = np.round(temporary_window.width).astype(int)
            height = np.round(temporary_window.height).astype(int)

            self._tiles[self._c_tiles] = rio.windows.Window(col_off=col_off, row_off=row_off,
                                                            width=width, height=height)
            ind = self.c_tiles

        elif isinstance(ind, slice):
            if not isinstance(ind.start, int):
                raise TypeError("Start of slice not given or not an integer")
            if ind.start < 0 or ind.start > (self._c_tiles - 1):
                raise IndexError("Start of slice out of range")

            if not isinstance(ind.stop, int):
                raise TypeError("Stop of slice not given or not an integer")
            if ind.stop < 0 or ind.stop > (self._c_tiles - 1):
                raise IndexError("Stop of slice out of range")

            i, j = ind.start, ind.stop
            col_off = min(self._tiles[i].col_off,
                          self._tiles[j].col_off)
            row_off = min(self._tiles[i].row_off,
                          self._tiles[j].row_off)
            height = max(self._tiles[i].height + self._tiles[i].row_off,
                         self._tiles[j].height + self._tiles[j].row_off)
            width = max(self._tiles[i].width + self._tiles[i].col_off,
                        self._tiles[j].width + self._tiles[j].col_off)

            # reset the parameters relative to column and rof offsets
            height = height - row_off
            width = width - col_off

            self._tiles[self._c_tiles] = rio.windows.Window(col_off, row_off, width, height)
            ind = self.c_tiles

        elif isinstance(ind, int):
            if ind not in list(range(-1, self._c_tiles + 1)):
                raise IndexError("Index out of range")
            if ind == -1:
                ind = self._current_ind
        else:
            raise KeyError("ind parameter not understood")

        # store current index in self._current_ind
        self._current_ind = ind

        return ind

    def get_profile(self, ind=-1):
        """
        Rasterio profile of a tile in the opened file

        Parameters
        ----------
        ind : . , optional
            See get_pointer(). If set it calculates width, height and transform for the given Index, while None, the
            default will yield the rasterio profile from the file directly

        Returns
        -------
        dict
        """
        profile = copy.deepcopy(self._profile)
        height, width = self.get_shape(ind)
        left, bottom, right, top = self.get_bounds(ind)
        transform = rio.transform.from_bounds(west=left, south=bottom, east=right,
                                              north=top, width=width, height=height)
        profile.update({'height': height, 'width': width, 'transform': transform})
        return profile

    def get_file_profile(self):
        """
        Rasterio profile of the rasterio file
        """
        return self.get_profile(ind=None)

    profile = property(get_file_profile)

    def get_bounds(self, ind=-1):
        """
        return rasterio bounds object of the current window

        Parameters
        ----------
        ind : .
            see self.get_pointer()

        Returns
        -------
        rasterio bounds object from tile
        """
        ind = self.get_pointer(ind)
        return self._file.window_bounds(self._tiles[ind])

    def get_file_bounds(self):
        """
        return rasterio bounds object of the whole map

        Returns
        -------
        rasterio bounds object of file
        """
        return self.get_bounds(ind=None)

    bounds = property(get_file_bounds)

    def get_shape(self, ind=-1):
        """
        Function to retrieve the shape at a given Index

        Parameters
        ----------
        ind : .
            see self.get_pointer()

        Returns
        -------
        numpy shape object
        """
        ind = self.get_pointer(ind)
        s = self._tiles[ind]
        if self._profile['count'] == 1:
            return tuple(np.around((s.height, s.width)).astype(int))
        else:
            return tuple(np.around((s.height, s.width, self._profile['count'])).astype(int))

    def get_file_shape(self):
        """
        Function to retrieve the shape of the file

        Returns
        -------
        numpy shape object
        """
        return self.get_shape(ind=None)

    shape = property(get_file_shape)

    def plot_world(self, ind=-1, numbers=False, tiles=True, ax=None, constrain_bounds=None, **kwargs):
        """
        Plots world with the outline of the file. If tiles is true, the different tiles are plotted on the map. If
        numbers is True not only the tiles but also the number of the tiles are plotted on the map. The current tile is
        plotted in red (ind=-1) or any other tile that might be needed. If the file doesn't contain the whole world
        a line is plotted in green.

        Blue: tiles
        Green: file boundaries
        Red: current data (which can be overwritten by providing `ind` directly

        Parameters
        ----------
        ind : ., optional
            see `get_pointer`. The default is -1, which will highlight the last accessed tile
        numbers : bool, optional
            plot the numbers of the tiles on the map
        tiles : bool, optional
            plot the borders of the tiles on the map
        ax : :obj:`~matplotlib.axes.Axes`, optional
            Axes on which to plot the figure
        constrain_bounds : list, optional
            Takes a four number list, or rasterio bounds object. It constrains the world view to a specific view.
        **kwargs
            arguments for the Basemap function

        Returns
        -------
        GeoAxis
        """
        if type(numbers) != bool:
            raise TypeError("numbers needs to a boolean variable")
        if type(tiles) != bool:
            raise TypeError("Tiles needs to be a boolean variable")

        if isinstance(constrain_bounds, type(None)):
            extent = [-180, -90, 180, 90]
        else:
            extent = constrain_bounds

        ax = basemap_function(*extent, ax=ax, **kwargs)

        # plot line around the file
        bounds = self._file.bounds
        gdf = bounds_to_polygons([bounds])
        gdf.crs = f"EPSG:{self.epsg}"
        gdf.plot(ax=ax, edgecolor="green", facecolor="none", transform=self._transform, zorder=2)

        # type checking and conversion
        ind = self.get_pointer(ind)
        ind_reset = ind
        # plot borders of current tile
        bounds_current_tile = self.get_bounds(ind)
        gdf = bounds_to_polygons([bounds_current_tile])
        gdf.crs = f"EPSG:{self.epsg}"
        gdf.plot(ax=ax, edgecolors="red", facecolor='none', transform=self._transform, zorder=3)

        if tiles or numbers:
            # plot borders around all tiles
            bounds_list = [self.get_bounds(i) for i in self]
            gdf = bounds_to_polygons(bounds_list)
            gdf.crs = f"EPSG:{self.epsg}"
            gdf.plot(ax=ax, edgecolor="blue", facecolor="none", transform=self._transform, zorder=1)

        if numbers:
            if numbers:
                gdf["x"] = gdf.apply(lambda x: (x.bounds[0] + x.bounds[2]) / 2, axis=1)
                gdf["y"] = gdf.apply(lambda x: (x.bounds[1] + x.bounds[3]) / 2, axis=1)

                for index, row in gdf.iterrows():
                    font = {'family': 'serif',
                            'color': 'darkred',
                            'weight': 'bold'
                            }
                    x = ax.text(row.x, row.y, index, fontdict=font, ha='center', va='center', transform=self._transform)
                    x.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='grey'))

        self.get_pointer(ind_reset)
        return ax

    def plot_file(self, ind=-1, epsg=None, **kwargs):
        """
        Plots the area of the file with tiles and numbers

        Plots the area of the file with the outline of the file. If tiles is true, the different tiles are plotted on
        the map. If numbers is True not only the tiles but also the number of the tiles are plotted on the map. The
        current tile is plotted in red (ind=-1) or any other tile that might be needed. If the file doesn't contain the
        whole world a line is plotted in green. For parameters, see self.plot_world()

        Parameters
        ----------
        ind : .
            see self.get_pointer()
        epsg : int, optional
            epsg code of the plot. `None` is the default, which will plot the file in it's native projection
        """
        if isinstance(epsg, type(None)):
            epsg = self.epsg
        bounds = bounds_to_platecarree(self._data_proj, self._file.bounds)
        ax = self.plot_world(ind=ind, constrain_bounds=bounds, epsg=epsg, **kwargs)

        bounds = self.get_bounds(None)
        ax.set_extent((bounds[0], bounds[2], bounds[1], bounds[3]), crs=self._transform)

        return ax

    def close(self, clean=True, verbose=True):
        """
        Function to close the pointer to the file and delete the handle

        Parameters
        ----------
        clean : bool, optional
            Default is True, in which case the file will be closed and the entry in the BaseMap.collector will be
            removed. If False, the file will be close but the entry will not be removed. This is only needed if looping
            over the BaseMap.collector to close all files, because you can't remove items from a list while looping over
            that list.
        verbose: bool, optional
            Default is True, in which case the location of the file will be printed once it is closed. For external use
            where printing is not required this flag can be set to False
        """
        try:
            self._file.close()
            if clean:
                _RasterBase.collector.remove(self)
            if verbose:
                print(f"close file: '{self._location}'")
            del self
        except (AttributeError, ValueError) as error:
            # if this happens, the file has been removed before the object,
            # which already made sure that file connections were closed properly
            # so no need to worry about it
            print(f"test: {error}")
            raise AttributeError("test")

    def __str__(self):
        return f"Raster at '{self._location}'"

    def __repr__(self):
        return (
            f"Raster object\n"
            f"location: {self._location}'\n"
            f"mode: '{self._mode}'\n"
            f"Access: {not self._file.closed}\n"
            f"Tiles: {self._c_tiles} => {{{self._v_tiles, self._h_tiles}}}\n"
            f"Window size: {self.window_size}\n"
            f"Shape: {self.shape}\n"
            f"Tile shape: {self.get_shape(0)}"
        )

    def __iter__(self):
        """
        Functionality to be used in a loop

        Examples
        --------
        1: The following example loops over a created map. In a for loop the object returns
           the indices of the tiles that were created. These indices can be used to both
           read (M_loc[i]) and write (M_loc[i] = np.ndarray) data.

            >>> loc = "/Users/Downloads/...."
            >>> M_loc = map(loc, tiles = 3)
            >>> for i in M_loc:
                    print(i)

            0
            1
            2

        Yields
        ------
        current index [int]
        """
        for i in range(self._c_tiles):
            yield i

    def __enter__(self):
        """
        function is used in "with" syntax

        Returns
        -------
        self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        function is used at the end of a "with" statement to close the open file
        """
        self.close()


class _RasterWriter(_RasterBase):
    """
    Subclass of _RasterBase, file opened in 'w' mode.

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
        Count of layers of the file. If -1 it will be taken from the reference file. The default is -1.
    profile : dict, optional
        custom rasterio profile can be passed which will be used directly to create the file with. All the other
        parameters are neglected.
    epsg : int, optional
        EPGS code of the data. This parameter is only used if the code can't be found in the file

    Raises
    ------
    IOError
        - `location` doesn't exist to pull a profile from and no reference map is given
        - reference maps is given but can't be found
        - can't overwrite an already existing file if not specified with the 'overwrite' parameter
    """

    def __init__(self, location, *, tiles=1, window_size=1, ref_map=None, overwrite=False, compress=False,
                 dtype=np.float64, nodata=np.nan, count=-1, profile=None, epsg=None):
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
            self._profile['dtype'] = dtype
            nodata = np.array((nodata,)).astype(dtype)[0]
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

        # todo; create **kwargs that feed into the rio.open function
        self._file = rio.open(location, "w", **self._profile, BIGTIFF="YES")

        # setting parameters by calling property functions
        self.window_size = window_size
        self.tiles = tiles
        self.epsg = epsg

        self._current_ind = 0
        self._writing_buffer = None

        self.collector.append(self)

    def get_writing_buffer(self):
        """
        Returns the current writing buffer

        Returns
        -------
        np.ndarray of the current writing buffer or None when it is not set
        """
        return self._writing_buffer

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
        shape = self.get_shape(ind)
        writing_buffer = self.writing_buffer

        if writing_buffer.shape != shape:
            writing_buffer = writing_buffer[self.ind_inner]
        if writing_buffer.shape != shape:
            raise ValueError(
                f"Shapes don't match:\n"
                f"- shape writing_buffer: {writing_buffer.shape}\n"
                f"- tile shape: {shape}")

        if isinstance(writing_buffer, type(None)):
            raise NameError("No writing buffer found")
        if self.profile['count'] == 1:
            self._file.write(writing_buffer, 1, window=self._tiles[ind])
        else:
            for i in range(1, self.profile['count'] + 1):
                self._file.write(writing_buffer[:, :, i - 1], i, window=self._tiles[ind])

    def __setitem__(self, ind, writing_buffer):
        """
        redirects to self.set_writing_buffer and self.write
        """
        self.writing_buffer = writing_buffer
        self.write(ind)


class _RasterReader(_RasterBase):
    """
    Instance of _RasterBase, file opened in 'r' mode.

    Parameters
    ----------
    location : str
        Location of the map
    variable : str, optional
        the specific variable to use when looking at a NetCDF file.
    tiles : int or tuple of ints, optional
        See tiles property in _RasterBase
    window_size : int, optional
        Window size to be set on the map. See property in _RasterBase
    fill_value : numeric, optional
        Fill value used in rasterio.read call.
    epsg : int, optional
        EPGS code of the data. This parameter is only used if the code can't be determined from the file's crs. The
        default is 4326.

    Attributes
    ----------
    See Also _RasterBase

    values : ndarray
        all data of the file

    Raises
    ------
    TypeError
        Location not a string
    IOError
        Location of file not found
    """

    def __init__(self, location, *, variable=None, tiles=1, window_size=1, fill_value=None, epsg=None):
        if type(location) != str:
            raise TypeError("Location not recognised")
        if not os.path.isfile(location):
            raise IOError(f"Location can't be found:\n'{location}'")

        self._location = location
        self._mode = "r"

        # open file in reading mode
        if location.endswith(".nc"):
            if isinstance(variable, type(None)):
                file = rio.open(location)
                report = file.subdatasets
                file.close()
                raise ValueError(f"When opening a netcdf file, a variable needs to be specified:\n"
                                 f"Either set a `variable` or specify the variable in the rasterio way:"
                                 f"netcdf:location.nc:variable.\n"
                                 f"The options are:\n"
                                 f"{report}")
            else:
                location = f"netcdf:{location}:{variable}"

        self._file = rio.open(location)
        self._profile = self._file.profile

        if isinstance(fill_value, type(None)):
            if 'float' in self._profile['dtype']:
                self._fill_value = np.nan
            else:
                self._fill_value = self._profile['nodata']
        else:
            if np.isnan(fill_value) and 'float' not in self._profile['dtype']:
                raise TypeError("Trying to set NaN on a not float array")
            self._fill_value = fill_value

        # setting parameters by calling internal property settings functions
        self.window_size = window_size
        self.tiles = tiles
        self.epsg = epsg

        self._current_ind = 0

        # collector of the base class. This list contains all files opened with the Raster classes
        self.collector.append(self)

    def get_data(self, ind=-1, layers=None):
        """
        Returns the data associated with the index of the tile. Layers are set on the third axis.

        Parameters
        ----------
        ind : .
            see _RasterBase.get_pointer(). If set to None it will read the whole file.
        layers : int or list, optional
            The number of the layer required or a list of layers (which might contain duplicates if needed).
            The default is None, which will load all layers.

        Returns
        -------
        numpy.ndarray of shape self.get_shape()
        """
        ind = self.get_pointer(ind)

        data = self._file.read(indexes=layers, window=self._tiles[ind], boundless=True, fill_value=self._fill_value)

        # if only a single layer is present compress the data to a 2D array
        data = np.squeeze(data)

        # move layer information to the last axis
        if data.ndim == 3:
            data = np.moveaxis(data, 0, -1)

        return data

    def get_file_data(self, layers=None):
        """
        Read the whole file

        Parameters
        ----------
        layers : int or list, optional
            The number of the layer required or a list of layers (which might contain duplicates if needed).
            The default is None, which will load all layers.

        Returns
        -------
        np.array with all data of the file
        """
        return self.get_data(ind=None, layers=layers)

    values = property(get_file_data)

    def sample_raster(self, points, layers=None):
        """
        Sample the raster at points

        Parameters
        ----------
        points : list or GeoDataFrame
            Either list of tuples with Lat/Lon values or a GeoDataFrame with points. If polygons are present in the
            GeoDataFrame the centroid will be used to obtain the values.
        layers : int or list, optional
            The number of the layer required or a list of layers (which might contain duplicates if needed).
            The default is None, which will use all layers.

        Returns
        -------
        (GeoSeries, array)
            points in the georefence system of the data and array with values
        """
        # Create a Polygon of the area that can be sampled
        bounds = self.get_file_bounds()
        box_x = [bounds[0], bounds[2], bounds[2], bounds[0]]
        box_y = [bounds[3], bounds[3], bounds[1], bounds[1]]
        bounding_box = Polygon(zip(box_x, box_y))
        outline = gpd.GeoDataFrame(geometry=[bounding_box])
        outline.crs = self._data_proj.definition_string()
        outline = outline.to_crs({'init': 'epsg:4326'})

        if not isinstance(points, (gpd.GeoSeries, gpd.GeoDataFrame)):
            if not isinstance(points, (tuple, list)):
                raise TypeError("points should either be a GeoDataFrame or a list of tuples")
            points = gpd.GeoDataFrame(geometry=[Point(point[0], point[1]) for point in points],
                                      crs={'init': 'epsg:4326'})

        points = points.loc[points.geometry.apply(lambda x: outline.contains(x)).values, :] \
            .to_crs(self._data_proj.definition_string())

        geom_types = points.geometry.type
        point_idx = np.asarray((geom_types == "Point"))
        points.loc[~point_idx, 'geometry'] = points.loc[~point_idx, 'geometry'].centroid

        if points.empty:
            raise IndexError("Geometries are all outside the bounds")

        sampling_points = points.geometry.apply(lambda x: (x.x, x.y)).values.tolist()
        values = self._file.sample(sampling_points, indexes=layers)

        return points.geometry, np.array([x for x in values])

    def cross_profile(self, c1, c2, n=100, layers=1):
        """
        Routine to create a cross profile, based on self.sample_raster

        Parameters
        ----------
        c1, c2 : tuple
            Location of the start and end of the cross profile (Lat, Lon)
        n : int, optional
            Number of sampling points
        layers : int or list, optional
            The number of the layer required or a list of layers (which might contain duplicates if needed).
            The default is None, which will use all layers.
        """
        points = np.linspace(c1, c2, num=n).tolist()
        return self.sample_raster(points=points, layers=layers)

    def _focal_stat_iter(self, output_file=None, ind=None, layers=1, *, func=None, overwrite=False, compress=False,
                         p_bar=True, verbose=False, reduce=False, window_size=None, dtype=None, majority_mode="nan",
                         **kwargs):
        """
        Function that calculates focal statistics, in tiled fashion if self.c_tiles is bigger than 1. The result is
        outputted to `output_file`

        Parameters
        ----------
        func : {"mean", "min", "max", "std", "majority"}
            function to be applied to the map in a windowed fashion.
        output_file : str
            location of output file.
            todo; should have a default behaviour that is slightly more functional
        ind : . , optional
            see self.get_pointer(). If set, no tiled behaviour occurs.
        layers : int, optional
            the layer that will be used to do the calculations
        overwrite : bool, optional
            If allowed to overwrite the output_file. The default is False.
        compress : bool, optional
            Compress output data, the default is False
        p_bar : bool, optional
            Show the progress bar. If verbose is True p_bar will be False. The default behaviour is True.
        verbose : bool, optional
            Verbosity; the default is False
        reduce : bool, optional
            If True, the dimensions of the output map are divided by `window_size`. If False the resulting file has the
            same shape as the input, which is the default.
        window_size : int, optional
            `window_size` parameter of the focal_statistics function. The default is None, in which case it will be
            taken from the object: self.window_size.
        majority_mode : {"nan", "ascending", "descending"}, optional
            nan: when more than one class has the same score NaN will be assigned ascending: the first occurrence of the
            maximum count will be assigned descending: the last occurence of the maximum count will be assigned.
            Parameter only used when the `func` is majority.
        **kwargs
            passed to focal_statistics()

        Notes
        -----
        the following parameter will be passed on directly to the focal_statistics calculation through **kwargs

        fraction_accepted : float, optional
            Fraction of the window that has to contain not-nans for the function to calculate the correlation.
            The default is 0.7.
        """
        if isinstance(func, type(None)):
            raise TypeError("No function given")
        if not isinstance(output_file, str):
            raise TypeError("Filename not understood")

        if not overwrite and os.path.isfile(output_file):
            print(f"\n{output_file}:\nOutput file already exists. Can only overwrite this file if explicitly stated "
                  f"with the overwrite' parameter. Continuing without performing operation ...\n")
            return None

        if not isinstance(reduce, bool):
            raise TypeError("reduce parameter needs to be a boolean")

        self_old_window_size = self.window_size

        if not reduce:
            if isinstance(window_size, type(None)):
                if self.window_size < 3:
                    raise ValueError("Can't run focal statistics with a window size of 1, unless 'reduce' is True")
                window_size = self.window_size
            else:
                if window_size < 3:
                    raise ValueError("Can't run focal statistics with a window size of 1")
                self.window_size = window_size
            profile = self.profile
        else:
            self.window_size = 1
            if isinstance(window_size, type(None)):
                raise TypeError("Window_size needs to be provided in reduction mode")
            if window_size < 2:
                raise ValueError("Window_size needs to be bigger than 1 in reduction mode")
            profile = resample_profile(self.profile, 1 / window_size)

        if not isinstance(dtype, type(None)):
            profile['dtype'] = dtype
        if func == "majority" and majority_mode == "nan":
            profile['dtype'] = np.float64
        if not isinstance(compress, type(None)):
            profile.update({'compress': compress})
        profile.update({'count': 1, 'driver': "GTiff"})

        if isinstance(ind, type(None)):
            with _RasterWriter(output_file, tiles=(self._v_tiles, self._h_tiles), window_size=self.window_size,
                               overwrite=overwrite, profile=profile) as f:
                for i in self:
                    if verbose:
                        print(f"\nTILE: {i + 1}/{self._c_tiles}")
                    elif p_bar:
                        progress_bar((i + 1) / self._c_tiles)

                    data = self[i, layers]
                    # if data is empty, write directly
                    if ~np.isnan(self[i]).sum() == 0:
                        f[i] = np.full(f.get_shape(), np.nan)
                    else:
                        f[i] = focal_statistics(data, func=func, window_size=window_size, verbose=verbose,
                                                reduce=reduce, majority_mode=majority_mode, **kwargs)

                if p_bar:
                    print()
        else:
            index_profile = self.get_profile(ind)
            height = index_profile['height']
            width = index_profile['width']
            transform = index_profile['transform']

            profile.update({'height': height, 'width': width, 'transform': transform})

            if reduce:
                profile = resample_profile(profile, 1 / window_size)

            with _RasterWriter(output_file, overwrite=overwrite, profile=profile) as f:
                f[0] = focal_statistics(self[ind, layers], func=func, window_size=window_size, verbose=verbose,
                                        reduce=reduce, majority_mode=majority_mode, **kwargs)

        self.window_size = self_old_window_size

    def focal_mean(self, **kwargs):
        """
        Function passes call to _RasterReader._focal_stat_iter with func = "nanmean". Function forces float64 dtype.
        """
        kwargs['dtype'] = np.float64
        return self._focal_stat_iter(func="mean", **kwargs)

    def focal_min(self, **kwargs):
        """
        Function passes call to _RasterReader._focal_stat_iter with func = "nanmin"
        """
        return self._focal_stat_iter(func="min", **kwargs)

    def focal_max(self, **kwargs):
        """
        Function passes call to _RasterReader._focal_stat_iter with func = "nanmax"
        """
        return self._focal_stat_iter(func="max", **kwargs)

    def focal_std(self, **kwargs):
        """
        Function passes call to _RasterReader._focal_stat_iter with func = "nanstd".

        Function forces float dtype.
        """
        kwargs['dtype'] = np.float64
        return self._focal_stat_iter(func="std", **kwargs)

    def focal_majority(self, **kwargs):
        """
        Function passes call to _RasterReader._focal_stat_iter with func = "majority". If `majority_mode` is not given as a
        parameter or set to "nan" the dtype will be forced to float64.
        """
        return self._focal_stat_iter(func="majority", **kwargs)

    def correlate(self, other=None, ind=None, self_layers=1, other_layers=1, *, output_file=None, window_size=None,
                  fraction_accepted=0.7, verbose=False, overwrite=False, compress=False, p_bar=True, parallel=False):
        """
        Correlate self and other and output the result to output_file.

        Parameters
        ----------
        other : _RasterReader
            map to correlate with
        ind : . , optional
            see self.get_pointer(). If set, no tiled behaviour occurs.
        self_layers, other_layers : int, optional
            the layer that will be used to calculate the correlations
        output_file : str
            Location of output file
        window_size : int, optional
            Size of the window used for the correlation calculations. It should be bigger than 1, the default is the
            window_size set on self (`_RasterReader`).
        fraction_accepted : float, optional
            Fraction of the window that has to contain not-nans for the function to calculate the correlation.
            The default is 0.7.
        verbose : bool, optional
            Verbosity, default is False
        overwrite : bool, optional
            If allowed to overwrite the output_file, default is False
        compress : bool, optional
            Compress calculated data, default is False.
        p_bar : bool, optional
            Show the progress bar. If verbose is True p_bar will be False. Default value is True.

        Raises
        ------
        TypeError
            Other is not of type _RasterReader
        """
        # todo; implement reduce
        # todo; implement parallel

        if not isinstance(other, _RasterReader):
            raise TypeError("Other not correctly passed")

        if self._v_tiles != other._v_tiles:
            raise ValueError("v_tiles don't match")
        if self._h_tiles != other._h_tiles:
            raise ValueError("h_tiles don't match")

        if not isinstance(self_layers, int) or not isinstance(other_layers, int):
            raise TypeError("Layers can only be an integer for correlation calculations")

        if self.get_file_shape() != other.get_file_shape():
            raise ValueError("Shapes  of the files don't match")
        if not np.allclose(self.get_file_bounds(), other.get_file_bounds()):
            raise ValueError(f"Bounds don't match:\n{self.get_file_bounds()}\n{other.get_file_bounds()}")

        self_old_window_size = self.window_size
        other_old_window_size = other.window_size
        if not isinstance(window_size, type(None)):
            self.window_size = window_size
            other.window_size = window_size

        if self.window_size != other.window_size:
            raise ValueError("window sizes don't match")
        if self.window_size < 3:
            raise ValueError("Can't run correlation with a window size of 1")

        if not overwrite:
            if os.path.isfile(output_file):
                print(f"Output file already exists. Can only overwrite this file if explicitly stated with the "
                      f"'overwrite' parameter. \n{output_file}\nContinuing without performing operation ...\n")
                return None

        if isinstance(ind, type(None)):
            with _RasterWriter(output_file, tiles=(self._v_tiles, self._h_tiles), window_size=self.window_size,
                               ref_map=self._location, overwrite=overwrite, compress=compress, dtype=np.float64,
                               count=1) as f:
                for i in self:
                    if verbose:
                        print(f"\nTILE: {i + 1}/{self._c_tiles}")
                    elif p_bar:
                        progress_bar((i + 1) / self._c_tiles)

                    f[i] = correlate_maps(self[i, self_layers], other[i, other_layers], window_size=self.window_size,
                                             fraction_accepted=fraction_accepted, verbose=verbose)

                if p_bar:
                    print()
        else:
            profile = self.get_profile(ind=ind)
            profile.update({'driver': "GTiff", 'count': 1, 'dtype': 'float64'})
            if not isinstance(compress, type(None)):
                profile.update({'compress': compress})

            with _RasterWriter(output_file, overwrite=overwrite, profile=profile, window_size=self.window_size) as f:
                f[0] = correlate_maps(self[ind, self_layers], other[ind, other_layers], window_size=self.window_size,
                                         fraction_accepted=fraction_accepted, verbose=verbose)

        self.window_size = self_old_window_size
        other.window_size = other_old_window_size

    def export_tile(self, ind, output_file, layers=None, compress=None):
        """
        exports a tile of the currently opened map.

        Function takes 'ind', which means that a tile or a custom window can be exported. The data is written to
        'output_file'.

        Parameters
        ----------
        ind : .
            see self.get_pointer()
        output_file : str
            path to the location where the data will be written to
        layers : int or tuple, optional
            the layer that is exported. If a tuple is provided several layers are exported, None means all layers by
            default.
        compress : str, optional
            rasterio compression parameter
        """
        data = self.get_data(ind, layers=layers)
        profile = self.get_profile(ind)
        profile.update({'driver': "GTiff", 'count': data.shape[-1]})
        if not isinstance(compress, type(None)):
            profile.update({'compress': compress})
        with rio.open(output_file, mode="w", **profile) as dst:
            dst.write(np.moveaxis(data, -1, 0))

    def _plot(self, classified, ind=None, layers=1, *, basemap=False, figsize=(10, 10), ax=None, log=False, epsg=None,
              xticks=30, yticks=30, resolution="110m", fontsize=10, basemap_kwargs=None, bounds=None, **kwargs):
        """
        Plot data at given index (ind). Classified or not, depends on the first parameter.

        Parameters
        ----------
        classified : bool
            Switch between `plot_classified_map` and `plot_map`
        ind : ., optional
            check self.get_pointer(). The default is None which will get all data.
        layers : int or tuple, optional
            The layer that is plotted. If `classified` is True only a single layer is accepted (as integer) while False
            will accept a tuple representing RGB or RGBA.
        basemap : bool, optional
            plot a basemap behind the data
        figsize : tuple, optional
            matplotlib figsize parameter
        ax : Axes, optional
            matplotlib axes where plot is drawn on
        log : bool, optional
            Plot the colors on a log scale if `classified` is False and only a single layer is selected.
        epsg : int, optional
            EPSG code that will be used to render the plot, the default is the projection of the data itself.
        xticks : float or list, optional
            parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
            means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
            coordinates in the list are used.
        yticks : float or list, optional
            parameter that describes the distance between two gridlines in PlateCarree coordinate terms. The default 30
            means that every 30 degrees a gridline gets drawn. If a list is passed, the procedure is skipped and the
            coordinates in the list are used.
        resolution : {"110m", "50m", "10m"} , optional
            coastline resolution
        fontsize : float/tuple, optional
            fontsize for both the lon/lat ticks and the ticks on the colorbar if one number, if a list of is passed it
            represents the basemap fontsize and the colorbar fontsize.
        basemap_kwargs : dict, optional
            kwargs going to the basemap command
        bounds : list, optional
            extend of the plot, if not provided it will plot the extent belonging to the Index. If the string "global"
            is set the global extent will be used.
        **kwargs
            kwargs going to the plot_map() command

        Returns
        -------
        (:obj:`~matplotlib.axes.Axes` or GeoAxis, legend)
        """
        if classified and not isinstance(layers, int):
            raise TypeError("layers can only be an integer for classified plotting")
        if not classified and isinstance(layers, (tuple, list)) and len(layers) > 4:
            raise IndexError("layers can only be a maximum of four integers, as the would index for RGB(A)")

        data = self.get_data(ind, layers=layers)
        extent = self.get_bounds(ind)

        if isinstance(bounds, type(None)):
            bounds = bounds_to_platecarree(self._data_proj, extent)
            set_bounds = False
        elif bounds == "global":
            bounds = [-180, -90, 180, 90]
            set_bounds = True
        else:
            set_bounds = True

        if not classified and isinstance(layers, int) and log:
            data = np.log(data)

        if not isinstance(fontsize, (tuple, list)):
            fontsize = (fontsize, fontsize)
        kwargs.update({'fontsize': fontsize[1]})

        if basemap:
            if isinstance(basemap_kwargs, type(None)):
                basemap_kwargs = {}

            if 'xticks' not in basemap_kwargs:
                basemap_kwargs.update({'xticks': xticks})
            if 'yticks' not in basemap_kwargs:
                basemap_kwargs.update({'yticks': yticks})
            if 'fontsize' not in basemap_kwargs:
                basemap_kwargs.update({'fontsize': fontsize[0]})
            if 'resolution' not in basemap_kwargs:
                basemap_kwargs.update({'resolution': resolution})

            if isinstance(self.epsg, type(None)):
                raise RuntimeError("This object does not contain a EPSG code")
            elif self.epsg == 4326:
                transform = ccrs.PlateCarree()
            else:
                transform = ccrs.epsg(self.epsg)

            plot_epsg = self.epsg if isinstance(epsg, type(None)) else epsg

            ax = basemap_function(*bounds, ax=ax, epsg=plot_epsg, figsize=figsize, **basemap_kwargs)
            kwargs.update({'transform': transform, 'extent': (extent[0], extent[2], extent[1], extent[3])})

            if not set_bounds:
                if plot_epsg == self.epsg:
                    ax.set_extent((extent[0], extent[2], extent[1], extent[3]), crs=self._transform)
                elif plot_epsg == 4326 and isinstance(ind, (list, tuple)) and not isinstance(ind,
                                                                                             rio.coords.BoundingBox):
                    ax.set_extent((ind[0], ind[2], ind[1], ind[3]))

        if classified:
            return plot_classified_map(data, ax=ax, figsize=figsize, **kwargs)
        else:
            return plot_map(data, ax=ax, figsize=figsize, **kwargs)

    def plot(self, *args, **kwargs):
        """
        alias for self.plot_map
        """
        return self.plot_map(*args, **kwargs)

    def plot_map(self, ind=None, layers=1, **kwargs):
        """
        Plot map wrapper around `plot_map`. It redirects to `self._plot` with parameter `classified` = False
        """
        return self._plot(classified=False, ind=ind, layers=layers, **kwargs)

    def plot_classified_map(self, ind=None, layers=1, **kwargs):
        """
        Plot map wrapper around `plot_classified_map`. It redirects to `self._plot` with parameter `classified` = True
        """
        return self._plot(classified=True, ind=ind, layers=layers, **kwargs)

    def __getitem__(self, ind):
        """
        pointer to internal function get_data().
        """
        if isinstance(ind, tuple) and len(ind) == 2:
            ind, layers = ind
        else:
            layers = None
        return self.get_data(ind, layers=layers)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.lib.index_tricks import s_
import rasterio as rio
import copy
from pyproj import Proj
import pyproj
from shapely.ops import transform
import cartopy.crs as ccrs
from functools import partial

from .misc import bounds_to_platecarree, bounds_to_data_projection
from ..plotting import basemap as basemap_function
from ..raster_functions.bounds_to_polygons import bounds_to_polygons


class MapBase:
    """
    Main map object, based on rasterio functionality. The rasterio pointer is exposed through MapBase._file
    
    Attributes
    ----------
    collector : list
        Every new instance of MapBase, or its children, will be added to this list. This is convenient for closing of
        all file connections with the following loop:
            
            >>> for i in MapBase.collector:
            ..:    i.close()
            
        this functionality is callable by:
            
            >>> Map.close()

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
        set epsg code if not present
    ind_inner
        get slice to remove fringes introduced by window_size
    c_tiles
        get number of tiles
    signature
        get MapBase signature
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
        raise NotImplementedError("MapBase is used as the base class for MapRead and MapWrite instances. "
                                  "This class is not directly callable.")
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
        self._data_proj = None

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

        shape = self.get_file_shape()

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

        self._current_ind = 0

    tiles = property(get_tiles, set_tiles)

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
        if not isinstance(ind, type(None)):
            height, width = self.get_shape(ind)
            left, bottom, right, top = self.get_bounds(ind)
            transform = rio.transform.from_bounds(west=left,
                                                  south=bottom,
                                                  east=right,
                                                  north=top,
                                                  width=width,
                                                  height=height)
            profile.update({'height': height, 'width': width, 'transform': transform})
        return profile


    def get_file_profile(self):
        """
        Rasterio profile of the rasterio file
        """
        return self.get_profile(ind=None)

    profile = property(get_file_profile)

    @property
    def location(self):
        """
        Location of opened file in the object
        """
        return copy.deepcopy(self._location)

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
                self._c_tiles.
            [slice]
                Works like passing an int twice. It finds the outer coordinates
                of both tiles and creates a new window based on those new 
                bounds. This window gets added to the list self._tiles at index
                self._c_tiles, the last entry. This tile will not be accessed
                when looping over the object, but all other operations can be
                performed on this  new tile.
            [int]
                Index of self._tiles range(0,self._c_tiles)

        Returns
        -------
        ind : int
            Index in range(0,self._c_tiles+1)
        
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
        if isinstance(ind, (tuple, list, rio.coords.BoundingBox)):
            if isinstance(ind, (tuple, list)):
                x0, y0, x1, y1 = bounds_to_platecarree(self._data_proj, ind)

                if x0 < -180 or x0 > 180:
                    raise ValueError("BoundingBox left coordinate of the globe")
                if y0 < -90 or y0 > 90:
                    raise ValueError("BoundingBox bottom coordinate of the globe")
                if x1 < -180 or x1 > 180:
                    raise ValueError("BoundingBox right coordinate of the globe")
                if y1 < -90 or y1 > 90:
                    raise ValueError("BoundingBox top coordinate of the globe")

                ind = rio.coords.BoundingBox(*bounds_to_data_projection(self._data_proj, ind))

            # create the space in the self._tiles list if it doesn't exist yet
            try:
                self._tiles[self._c_tiles] = 0
            except IndexError:
                self._tiles.append(0)
            finally:
                # add window object to self._tiles list
                temporary_window = rio.windows.from_bounds(*ind, self._file.transform)

                # round the entries and create the window
                col_off = np.round(temporary_window.col_off).astype(int)
                row_off = np.round(temporary_window.row_off).astype(int)
                width = np.round(temporary_window.width).astype(int)
                height = np.round(temporary_window.height).astype(int)

                self._tiles[self._c_tiles] = rio.windows.Window(col_off=col_off,
                                                                row_off=row_off,
                                                                width=width,
                                                                height=height)
                ind = self._c_tiles

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

            # if self._tiles[self.index] doesn't exist, create it
            try:
                self._tiles[self._c_tiles] = 0
            except IndexError:
                self._tiles.append(0)
            finally:
                self._tiles[self._c_tiles] = rio.windows.Window(col_off, row_off, width, height)
                ind = self._c_tiles

        elif isinstance(ind, int):
            if ind not in list(range(-1, self._c_tiles + 1)):
                raise IndexError("Index out of range")
            if ind == -1:
                return self._current_ind
        else:
            raise KeyError("ind parameter not understood")

        # store current index in self._current_ind
        self._current_ind = ind

        return ind

    def get_bounds(self, ind=-1):
        """
        return rasterio bounds object of the current window
        
        Parameters
        ----------
        ind : .
            see self.get_pointer(), the default is None which gets the bounds of the file.
        
        Returns
        -------
        rasterio bounds object from tile
        """
        if isinstance(ind, type(None)):
            return copy.deepcopy(self._file.bounds)
        else:
            # type and bound checking happens in get_pointer()
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
        Function to retrieve the shape of the current data in memory
        
        Parameters
        ----------
        ind : .
            see self.get_pointer(), the default is None which gets the shape of the file.
            
        Returns
        -------
        numpy shape object
        """
        if isinstance(ind, type(None)):
            s = copy.deepcopy(self._file.shape)
            if self.profile['count'] == 1:
                return s
            else:
                return s + (self.profile['count'], )
        else:
            ind = self.get_pointer(ind)
            s = self._tiles[ind]
            if self.profile['count'] == 1:
                return tuple(np.around((s.height, s.width)).astype(int))
            else:
                return tuple(np.around((s.height, s.width, self.profile['count'])).astype(int))

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

        # type checking and conversion
        ind = self.get_pointer(ind)

        if isinstance(constrain_bounds, type(None)):
            extent = [-180, -90, 180, 90]
        else:
            extent = constrain_bounds

        ax = basemap_function(*extent, ax=ax, **kwargs)

        # plot line around the file
        bounds = self.get_file_bounds()
        gdf = bounds_to_polygons([bounds])
        gdf.crs = f"EPSG:{self.epsg}"
        gdf.plot(ax=ax, edgecolor="green", facecolor="none", transform=self._transform, zorder=2)

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

        return ax

    def plot_file(self, ind=-1, **kwargs):
        """
        Plots the area of the file with tiles and numbers

        Plots the area of the file with the outline of the file. If tiles is true, the different tiles are plotted on
        the map. If numbers is True not only the tiles but also the number of the tiles are plotted on the map. The
        current tile is plotted in red (ind=-1) or any other tile that might be needed. If the file doesn't contain the
        whole world a line is plotted in green. For parameters, see self.plot_world()
        """
        bounds = self.get_file_bounds()
        gdf = bounds_to_polygons([bounds])
        gdf.crs = {'init': f"epsg:{self.epsg}"}
        project = partial(
            pyproj.transform,
            Proj(init=f'epsg:{self.epsg}'),  # source coordinate system
            Proj(init='epsg:4326'))  # destiation

        bounds = transform(project, gdf.geometry[0]).bounds
        return self.plot_world(ind=ind, constrain_bounds=bounds, **kwargs)

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
                MapBase.collector.remove(self)
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
        return f"Map at '{self._location}'"

    def __repr__(self):
        return (
            f"Map object\n"
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

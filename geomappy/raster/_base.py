import copy
import os
from abc import abstractmethod

import numpy as np
import pyproj
import rasterio as rio
from cartopy import crs as ccrs
from pyproj import Proj
from rasterio.coords import BoundingBox

from .locators import GeoLocator, TileLocator, IdxLocator
from ..basemap import basemap as basemap_function
from ..bounds import bounds_to_data_projection, bounds_to_polygons
from ..utils import progress_bar as progress_bar_func


class RasterBase:
    """
    Main Raster object, based on rasterio functionality. The rasterio pointer is exposed
    through ``RasterBase.rio``

    Attributes
    ----------
    collector : list
        Every new instance of ``RasterBase``, or its children, will be added to this list.
        This is convenient for closing of all file connections with the following loop:

            >>> for i in RasterBase.collector:
            ..:    i.close()

        this functionality is callable by:

            >>> Raster.close()

    _c_tiles : int
        Number of tiles that are set on the raster. Set and get via tiles property
    _current_ind : int
        last index passed to the object (last call to self.get_pointer())
    _data_proj : `Proj`
        Proj object for transformations
    _data_projection : cartopy CRS
        Projection of the data in Cartopy coordinate system
    _force_equal_tiles : bool
        Forces tiles to be created with exactly the same shape. By default this is False
    _fp : rasterio pointer
        rasterio file pointer
    _fringe : int
        amount of cells in each direction for windowed calculation.
        fringe = window_size // 2
    _h_tiles : int
        number of tiles in horizontal direction set on the raster
    _horizontal_bins : list of int
        bins of the horizontal shape of the raster. for example: a file with shape (1000,800)
        will have the following horizontal_bins variable if h_tiles is set to 4: [0,200,400,600,800]
    _ind_inner : tuple of slice
        numpy index slice that removes the padded cells that are added when reading data with
        a window_size set to something bigger than 1. for example:

            >>> window_size = 5
            >>> fringe = 2
            >>> ind_inner = [fringe:-fringe, fringe:-fringe]

    _iter : list of tuples of ints
        list of tuples containing the indices of the tiles on the final raster. (1,2) corresponds
        with the first row and second tile. The index of this tuple in the list is the index that
        should be passed to the object to obtain the data corresponding to that specific tile.
    _mode : str
        Mode of open file {"r", "w"}
    _tiles : list of rio.windows
        list of windows to be iterated over to access all the data in the file
    _v_tiles : int
        same as h_tiles but in vertical direction
    _vertical_bins : list of int
        same as horizontal_bins but in vertical direction
    _window_size : int
        window size to be considered when performing windowed operations on the raster

    Properties
    ----------
    rio
        rasterio file object
    location
        location of the data
    window_size
        get and set _window_size
    tiles
        get and set _tiles
    ind_inner
        get slice to remove fringes introduced by window_size
    c_tiles
        get number of tiles
    proj
        ``pyproj.Proj`` object corresponding to the data
    projection
        Cartopy projection corresponding to the data
    epsg
        get epsg code of data if existent
    crs
        get the pyproj CRS of the file. The raterio CRS is accessible through raster._fp.crs
    dtype
        get dtype if all dtypes are the same
    """
    collector = []

    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "RasterBase is used as the base class for RasterReader and RasterWriter instances. "
            "This class can't be used directly.")
        self._mode = None
        self._fp = None
        self._profile = {}
        self._location = ""
        self._current_ind = 0
        self._c_tiles = 1
        self._h_tiles = 1
        self._v_tiles = 1
        self._force_equal_tiles = False
        self._window_size = 1
        self._fringe = 1
        self._ind_inner = None
        self._horizontal_bins = []
        self._vertical_bins = []
        self._iter = []
        self._tiles = []
        self._data_proj = None
        self._data_projection = None
        self.mmap_collector = {}

    def _init(self, window_size, tiles, force_equal_tiles):
        self.window_size = window_size
        self.set_tiles(tiles, force_equal_tiles)
        self._current_ind = 0
        self._location = self._fp.name
        self._profile = copy.deepcopy(self._fp.profile)

        # collector of the base class. This list contains all files opened with the Raster classes
        self.collector.append(self)
        self.mmap_collector = {}

        # initialise locators
        self.geo = GeoLocator(raster=self)
        self.iloc = TileLocator(raster=self)
        self.idx = IdxLocator(raster=self)

    @property
    def rio(self):
        return self._fp

    @property
    def location(self):
        """
        Location of opened file in the object
        """
        return self.rio.name

    def get_window_size(self):
        """
        gives the current window size
        """
        return self._window_size

    def set_window_size(self, window_size=1):
        """
        Sets the window_size and fringe parameters.

        Parameters
        ----------
        window_size : int
            Size of the window used in windowed calculations. Needs to be uneven.

        """
        if not isinstance(window_size, int):
            raise TypeError("window_size needs to be bigger an integer")
        if window_size < 1:
            raise ValueError("window_size needs to be bigger than, or equal to 1")
        if window_size % 2 == 0:
            raise ValueError("window_size needs to be uneven")

        self._window_size = window_size
        self._fringe = window_size // 2

        if window_size == 1:
            self._ind_inner = np.s_[:, :]
        else:
            self._ind_inner = np.s_[self._fringe:-self._fringe, self._fringe:-self._fringe]

        if hasattr(self, "_force_equal_tiles") and self._force_equal_tiles is not None:
            self.set_tiles(self.tiles, self._force_equal_tiles)

    window_size = property(get_window_size, set_window_size)

    def get_tiles(self):
        """
        function to retrieve the current tile settings
        """
        return self._v_tiles, self._h_tiles

    def set_tiles(self, tiles=1, force_equal_tiles=False):
        """
        Function to set the tiles.

        If tiles is given as a list, v_tiles and h_tiles are calculated from this parameter.
        Otherwise c_tiles is calculated by the v_tiles * h_tiles. Tiles as an integer is an
        approximate parameter. No guarantees that c_tiles will exactly be the amount of tiles
        that were requested.

        Parameters
        ----------
        tiles : int or tuple of ints
            Passing an integer will create 'roughly' that amount of tiles. A tuple is read as
            vertical and horizontal tile count.
        force_equal_tiles : bool, optional
            If set to False the last tiles on both axes can be sligthly bigger than the rest.
            If set to True the tiles will be adapted untill a perfect fit is created, which
            causes the tiles to be approximate.
        """
        if isinstance(tiles, (tuple, list)):
            if len(tiles) != 2:
                raise ValueError("If a tuple is passed to window_size it needs "
                                 "to be of length 2; horizontal and vertical bin count.")
            if not isinstance(tiles[0], int) or not isinstance(tiles[1], int):
                raise TypeError("Tiles need to be passed as integers")
            if tiles[0] < 1 or tiles[1] < 1:
                raise ValueError("Tiles have to be positive integers")

            self._v_tiles = tiles[0]
            self._h_tiles = tiles[1]

        else:
            if not isinstance(tiles, int):
                raise TypeError("Tiles need to be passed as an integer, or as a list of integers")
            if tiles < 0:
                raise ValueError("Tiles have to be positive integers")

            # routine to infer h_tiles and v_tiles from c_tiles if set
            self._v_tiles = int(np.sqrt(tiles))
            self._h_tiles = int(tiles / self._v_tiles)

        shape = self.height, self.width

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

        # make an additional space for an index that is not predefined
        self._tiles.append(None)

        self._force_equal_tiles = force_equal_tiles
        self._current_ind = 0

    tiles = property(get_tiles, set_tiles)

    def set_tiles_as_chunks(self):
        self.window_size = 1
        self._tiles = [w[1] for w in self._fp.block_windows()]
        self._tiles.append(None)
        self._c_tiles = len(self._tiles) - 1
        self._force_equal_tiles = False
        self._current_ind = 0

        self._h_tiles = self.width // self._tiles[0].width
        self._v_tiles = self.height // self._tiles[0].height
        self._horizontal_bins = list(np.linspace(0, self.width, self._h_tiles + 1, dtype=np.int, endpoint=True))
        self._vertical_bins = list(np.linspace(0, self.height, self._v_tiles + 1, dtype=np.int, endpoint=True))

        self._iter = [(x, y) for x in range(1, self._v_tiles + 1) for y in range(1, self._h_tiles + 1)]

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
        return self._c_tiles

    @property
    def proj(self):
        if hasattr(self, "_data_proj"):
            return self._data_proj
        else:
            try:
                self._data_proj = Proj(self._fp.crs)
            except pyproj.exceptions.CRSError:
                raise ValueError(f"Internal rasterio CRS to Proj conversion has been unsuccesful: '{self._fp.crs}' "
                                 f"please specify it manually by `raster.proj = proj`")
            return self._data_proj

    @proj.setter
    def proj(self, proj):
        if not isinstance(proj, Proj):
            raise TypeError("proj should be an instance of pyproj.Proj")
        self._data_proj = proj

    @property
    def epsg(self):
        epsg = None
        if self._fp.crs is not None:
            epsg = self._fp.crs.to_epsg()
        if epsg is None:
            epsg = self.proj.crs.to_epsg()

        return epsg

    @property
    def projection(self):
        if hasattr(self, "_data_projection"):
            return self._data_projection
        else:
            epsg = self.epsg

        if epsg == 4326 or epsg == "4326":
            self._data_projection = ccrs.PlateCarree()
        else:
            try:
                self._data_projection = ccrs.epsg(epsg)
            except ValueError:
                raise ValueError(f"Cartopy CRS can't be created automatically from EPSG code: {self.epsg}."
                                 f"This can be resolved by setting `raster.projection = cartopy.CRS()`")

        return self._data_projection

    @projection.setter
    def projection(self, projection):
        if not isinstance(projection, ccrs.CRS):
            raise TypeError("projection needs to be a cartopy CRS object")
        else:
            self._data_projection = projection

    @property
    def crs(self):
        return self.proj.crs

    def get_pointer(self, ind):
        """
        Converts different types of pointer to the right index of the self._tiles list

        Parameters
        ----------
        ind :
            bounds -> [left,bottom,right,top]
                Rasterio bounds object or list of four numbers, the bounds of the
                new window. A tile is created based on these bounds like by passing
                a slice and gets added as the last entry of self._tiles at position
                self._c_tiles. If it is a list or tuple it is assumed to be coordinates
                in the latlon system (EPSG:4326) while a Rasterio BoundingBox is assumed
                to be in the coordinate system of the data.
            slices -> [slice, slice]
                A pair of slices in row, column order
            int
                Index of self._tiles range(0,self._c_tiles)
            tile selection -> [int, int]
                Two ints, in the bounds of vertical and horizontal tiles. Are converted
                internally to int.
            None
                Can be used to access the file as a whole

        Returns
        -------
        ind : int
            Index in range(0, self._c_tiles+1)

        Examples
        --------
        self[4]
            get data at self._tiles[4]
        self[1000:1500, :]
            get data from a slice of the raster
        self[(0,30,30,60)]
            create new window with the specified bounds

        to access the slice capability in other fuctions than __getitem__:
        1: pass a slice directly -> slice(1,2)
        2: use numpy -> np.s_[1:2]
        """
        if ind is None:
            ind = self.bounds

        if isinstance(ind, (tuple, list, np.ndarray, rio.coords.BoundingBox)) and len(ind) == 4:
            if not isinstance(ind, rio.coords.BoundingBox):
                x0, y0, x1, y1 = ind

                if x0 < -180 or x0 > 180:
                    raise ValueError(f"Left coordinate off the globe: {x0}")
                if y0 < -90 or y0 > 90:
                    raise ValueError(f"Bottom coordinate off the globe: {y0}")
                if x1 < -180 or x1 > 180:
                    raise ValueError(f"Right coordinate off the globe: {x1}")
                if y1 < -90 or y1 > 90:
                    raise ValueError(f"Top coordinate off the globe: {y1}")

                bounds1 = bounds_to_data_projection(self.proj, (x0, y0, x1, y1))
                bounds2 = bounds_to_data_projection(self.proj, (x0, y1, x1, y0))
                x = (bounds1[0], bounds1[2], bounds2[0], bounds2[2])
                y = (bounds1[1], bounds1[3], bounds2[1], bounds2[3])

                ind = rio.coords.BoundingBox(np.min(x), np.min(y), np.max(x), np.max(y))

            # add window object to self._tiles list
            self._tiles[self._c_tiles] = rio.windows.from_bounds(*ind, self.transform).round_shape()
            ind = self.c_tiles

        if isinstance(ind, tuple) and len(ind) == 2 and isinstance(ind[0], slice) and isinstance(ind[1], slice):
            ind = list(ind)

            if ind[0].start is None:
                ind[0] = slice(0, ind[0].stop)
            if ind[0].stop is None:
                ind[0] = slice(ind[0].start, self.height)
            if ind[1].start is None:
                ind[1] = slice(0, ind[1].stop)
            if ind[1].stop is None:
                ind[1] = slice(ind[1].start, self.width)

            self._tiles[self._c_tiles] = rio.windows.Window(row_off=ind[0].start,
                                                            col_off=ind[1].start,
                                                            height=ind[0].stop - ind[0].start,
                                                            width=ind[1].stop - ind[1].start)
            ind = self.c_tiles

        if isinstance(ind, (list, tuple)) and len(ind) == 2 and isinstance(ind[0], int) and isinstance(ind[1], int):
            if ind[0] > self._v_tiles-1 or ind[0] < 0:
                raise IndexError("Vertical index out of bounds")
            if ind[1] > self._h_tiles-1 or ind[1] < 0:
                raise IndexError("Horizontal tiles out of bounds")

            ind = self._h_tiles * ind[0] + ind[1]

        elif isinstance(ind, int):
            if ind not in list(range(-1, self._c_tiles + 1)):
                raise IndexError("Index out of range")
            if ind == -1:
                ind = self._current_ind

        else:
            raise KeyError(f"ind parameter not understood {ind}")

        # store current index in self._current_ind
        self._current_ind = ind

        return ind

    def ind_user_input(self, i):
        """
        Indexing occurs with the following rules:
        1) layers are on the last place

        2) int -> indexing in tiles
        3) array-like of length 4 -> using bounds
        4) two slices -> numpy indexing

        to form the combinations
        None                                    retrieve all data                   len .       x
        int                                     indexing in c_tiles                 len .       x
        int, [layers]                           indexing in c_tiles with layers     len 2       x
        [left, bottom, right, top]              indexing on bounds                  len 4       x
        [left, bottom, right, top], [layers]    indexing on bounds with layers      len 2       x
        slice, slice                            numpy indexing                      len 2
        slice, slice, [layers]                  numpy indexing with layers          len 3
        """
        indexes = -1
        if i is None:
            ind = None
            indexes = None
        elif isinstance(i, int):
            ind = i
            indexes = None
        elif len(i) == 4:
            ind = i
            indexes = None
        elif isinstance(i[0], (tuple, list, np.ndarray, rio.coords.BoundingBox)) and len(i[0]) == 4:
            ind = i[0]
            indexes = i[1]
        elif i[0] is None and len(i) == 2:
            ind = i[0]
            indexes = i[1]
        elif isinstance(i[0], int):
            ind = i[0]
            indexes = i[1]
        elif isinstance(i[0], slice):
            ind = i[0], i[1]
            if len(i) == 2:
                indexes = None
            elif len(i) == 3:
                indexes = i[2]

        if indexes == -1:
            raise IndexError("Indexing not understood. Check 'help(RasterReader.ind_user_input)'")

        return ind, indexes

    def get_transform(self, ind=-1):
        """
        Affine transform of a tile

        Parameters
        ----------
        ind : . , optional
            See get_pointer(). If set it calculates width, height and transform for the given Index, while None, the
            default will yield the rasterio profile from the file directly

        Returns
        -------
        Affine transformation
        """
        if ind is None:
            return self.transform

        ind = self.get_pointer(ind)
        return rio.windows.transform(self._tiles[ind], self.transform)

    def get_profile(self, ind=-1):
        """
        Rasterio profile of a tile

        Parameters
        ----------
        ind : . , optional
            See get_pointer(). If set it calculates width, height and transform for the given Index, while None, the
            default will yield the rasterio profile from the file directly

        Returns
        -------
        dict
        """
        profile = copy.deepcopy(self._fp.profile)
        if ind is None:
            return profile

        height = self.get_height(ind)
        width = self.get_width(ind)
        transform = self.get_transform(ind)
        profile.update({'height': height, 'width': width, 'transform': transform})

        return profile

    @property
    def profile(self):
        """
        Rasterio profile of the rasterio file
        """
        return self.get_profile(ind=None)

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
        if ind is None:
            return self.bounds

        ind = self.get_pointer(ind)
        return BoundingBox(*self.window_bounds(self._tiles[ind]))

    def get_shape(self, ind=-1, indexes=None):
        """
        Function to retrieve the shape at a given Index

        Parameters
        ----------
        ind : .
            see self.get_pointer()
        indexes : int or list-like
            Indexes corresponding to layers.

        Returns
        -------
        numpy shape object
        """
        if indexes is None:
            count = self.count
        elif isinstance(indexes, int):
            count = 1
        else:
            count = len(indexes)

        height = self.get_height(ind)
        width = self.get_width(ind)

        if count == 1:
            return height, width
        else:
            return count, height, width

    @property
    def shape(self):
        """
        Shape of the file
        """
        return self.get_shape(ind=None)

    def get_width(self, ind=-1):
        if ind is None:
            return self.width

        ind = self.get_pointer(ind)
        return round(self._tiles[ind].width)

    def get_height(self, ind=-1):
        if ind is None:
            return self.height

        ind = self.get_pointer(ind)
        return round(self._tiles[ind].height)

    @property
    def dtype(self):
        dtypes = np.asarray(self.dtypes)
        if np.all(dtypes == dtypes[0]):
            return dtypes[0]
        else:
            raise ValueError(f"Not all dtypes are equal, thus it can't be condensed:'\n{dtypes}")

    def plot_world(self, ind=-1, numbers=False, tiles=True, ax=None, extent="global", projection=None,
                   **kwargs):
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
        extent : list, optional
            Takes a four number list, or rasterio bounds object. It constrains the world view
            to a specific view. If not lat-lon, the extent_projection needs to be specified
        projection : cartopy.CRS
            Projection of the plot. The default is None, which yields ccrs.PlateCarree(). It accepts
            'native', which plots the map on the projection of the data.
        **kwargs
            arguments for the Basemap function

        Returns
        -------
        GeoAxis
        """
        if not isinstance(numbers, bool):
            raise TypeError("numbers needs to a boolean variable")
        if not isinstance(tiles, bool):
            raise TypeError("Tiles needs to be a boolean variable")

        if projection == "native":
            projection = self.projection

        ax = basemap_function(extent, ax=ax, projection=projection, **kwargs)

        # plot line around the file
        bounds = self.bounds
        gdf = bounds_to_polygons([bounds])
        gdf.crs = self.proj.definition_string()
        gdf.plot(ax=ax, edgecolor="green", facecolor="none", transform=self.projection, zorder=2)

        # type checking and conversion
        ind = self.get_pointer(ind)
        ind_reset = ind
        # plot borders of current tile
        bounds_current_tile = self.get_bounds(ind)
        gdf = bounds_to_polygons([bounds_current_tile])
        gdf.crs = self.proj.definition_string()
        gdf.plot(ax=ax, edgecolors="red", facecolor='none', transform=self.projection, zorder=3)

        if tiles or numbers:
            # plot borders around all tiles
            bounds_list = [self.get_bounds(i) for i in self]
            gdf = bounds_to_polygons(bounds_list)
            gdf.crs = self.proj.definition_string()
            gdf.plot(ax=ax, edgecolor="blue", facecolor="none", transform=self.projection, zorder=1)

        if numbers:
            if numbers:
                gdf["x"] = gdf.apply(lambda x: (x.bounds[0] + x.bounds[2]) / 2, axis=1)
                gdf["y"] = gdf.apply(lambda x: (x.bounds[1] + x.bounds[3]) / 2, axis=1)

                for index, row in gdf.iterrows():
                    font = {'family': 'serif',
                            'color': 'darkred',
                            'weight': 'bold'
                            }
                    x = ax.text(row.x, row.y, index, fontdict=font, ha='center', va='center', transform=self.projection)
                    x.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='grey'))

        self.get_pointer(ind_reset)
        return ax

    def plot_file(self, ind=-1, **kwargs):
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
        """
        ax = self.plot_world(ind=ind, extent=self.bounds, projection=self.projection,
                             extent_projection=self.projection, **kwargs)

        return ax

    def generate_memmap(self, ind=None, indexes=None):
        if ind is None:
            all_flag = True
        else:
            all_flag = False

        ind = self.get_pointer(ind)
        if indexes is None:
            indexes = self.indexes

        c = 100000000
        if not os.path.exists("_tmp_memmap"):
            os.mkdir("_tmp_memmap")

        while True:
            if os.path.isfile(f"_tmp_memmap/{c}.npy"):
                c += 1
            else:
                break

        path = f"_tmp_memmap/{c}.npy"
        memmap = np.memmap(path, dtype=self.dtype, mode='w+', offset=0, shape=self.get_shape(ind, indexes))

        if self.mode == "r":
            if all_flag:
                params = self.get_params()
                self.set_tiles_as_chunks()
                for i in self:
                    slices = self._tiles[i].toslices()
                    if memmap.ndim == 3:
                        memmap[:, slices[0], slices[1]] = self.get_data(i, indexes)
                    else:
                        memmap[slices[0], slices[1]] = self.get_data(i, indexes)
                self.set_params(params)
            else:
                memmap[:] = self.get_data(ind, indexes)

        self.mmap_collector[path] = memmap
        return memmap

    def close_memmap(self):
        for mmap_loc in list(self.mmap_collector):
            del self.mmap_collector[mmap_loc]
            os.remove(mmap_loc)
        try:
            if len(os.listdir("_tmp_memmap")) == 0:
                os.rmdir("_tmp_memmap")
        except OSError:
            pass

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
            self._fp.close()
            if clean:
                RasterBase.collector.remove(self)
            if verbose:
                print(f"close file: '{self._location}'")
            if len(self.mmap_collector) > 0:
                self.close_memmap()
            del self

        except (AttributeError, ValueError) as error:
            # if this happens, the file has been removed before the object,
            # which already made sure that file connections were closed properly
            # so no need to worry about it
            print(error)

    def get_params(self):
        return (self.window_size, self.tiles, self._force_equal_tiles)

    def set_params(self, params):
        window_size, tiles, force_equal_tiles = params
        self._init(window_size=window_size, tiles=tiles, force_equal_tiles=force_equal_tiles)

    params = property(get_params, set_params)

    def __getstate__(self):
        return (self.location, *self.get_params(), self._profile)

    def __setstate__(self, state):
        location, window_size, tiles, force_equal_tiles, profile = state
        if self._mode == 'w':
            kwargs = {'overwrite': True, 'profile': profile}
        else:
            kwargs = {}
        self.__init__(fp=location, window_size=window_size, tiles=tiles, force_equal_tiles=force_equal_tiles,
                      **kwargs)

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        elif not self.__getstate__() == other.__getstate__():
            return False
        else:
            return True

    def __repr__(self):
        return f"Raster at '{self.location}'"

    def iter(self, progress_bar=True):
        """Same function as standard iterator functionality, but has the possibility to plot a progress bar"""
        for i in range(self.c_tiles):
            if progress_bar:
                progress_bar_func((i+1) / self.c_tiles)
            yield i

    def __iter__(self):
        """
        Functionality to be used in a loop

        Examples
        --------
        The following example loops over a created map. In a for loop the object returns
        the indices of the tiles that were created. These indices can be used to both
        read (M_loc[i]) and write (M_loc[i] = np.ndarray) data.

        >>> loc = "/Users/Downloads/...."
        >>> M_loc = mp.Raster(loc, tiles = 4)
        >>> for i in M_loc:
                print(i)

        0
        1
        2
        3

        Yields
        ------
        current index [int]
        """
        for i in range(self._c_tiles):
            yield i

    def __enter__(self):
        """
        function is used in "with" syntax
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        function is used at the end of a "with" statement to close the open file
        """
        self.close(verbose=False)

    def __getattr__(self, attr):
        if '_fp' in dir(self):
            if attr in dir(self._fp):
                return getattr(self._fp, attr)
            else:
                raise AttributeError(f"Raster does not have {attr}")

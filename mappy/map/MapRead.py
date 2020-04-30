#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import rasterio as rio
from shapely.geometry import Point, Polygon
from mappy.plotting import basemap as basemap_function
from mappy.plotting import plot_world, plot_map, plot_classified_map
from .MapBase import MapBase
from .MapWrite import MapWrite
from ..progress_bar.progress_bar import progress_bar
from ..raster_functions import resample_profile, focal_statistics, correlate_maps


class MapRead(MapBase):
    """
    Instance of MapBase, file opened in 'r' mode.
    
    Parameters
    ----------
    location : str
        Location of the map
    tiles : int or tuple of ints
        See tiles property in MapBase
    window_size : int
        Window size to be set on the map. See property in MapBase
    fill_value : numeric
        Fill value used in rasterio.read call.
    
    Raises
    ------
    TypeError
        Location not a string
    IOError
        Location of file not found

    todo; Create option for multiple layers. Some functions already support it,
     like plot and get_data but others don't yet.
    """
    def __init__(self, location, *, tiles=1, window_size=1, fill_value=None):
        if type(location) != str:
            raise TypeError("Location not recognised")
        if not os.path.isfile(location):
            raise IOError(f"Location can't be found:\n'{location}'")

        self._location = location
        self._mode = "r"

        # open file in reading mode and set the global nan_value
        self._file = rio.open(location)
        self._profile = self._file.profile

        # todo; change fill value into a property
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

        self._current_ind = 0

        # collector of the base class. This list contains all files opened with the Map classes
        self.collector.append(self)

        if isinstance(self._file.crs, type(None)):
            self._epsg = None
        else:
            self._epsg = self._file.crs.to_epsg()

    def get_data(self, ind=-1, layers=None):
        """
        Returns the data associated with the index of the tile. Layers are set on the third axis.
        
        Parameters
        ----------
        ind : .
            see MapBase.get_pointer()
        layers : int or list
            The number of the layer required or a list of layers (which might contain duplicates if needed).
            The default is None, which will load all layers.
        
        Returns
        -------
        numpy array of shape self.get_shape()
        
        """
        # type and bound checking happens in self.get_pointer()
        ind = self.get_pointer(ind)

        data = self._file.read(indexes=layers, window=self._tiles[ind], boundless=True, fill_value=self._fill_value)

        # if only a single layer is present compress the data to a 2D array
        if data.shape[0] == 1:
            data = np.squeeze(data)

        # move layer information to ts axis
        if data.ndim == 3:
            data = np.moveaxis(data, 0, -1)

        return data

    def get_file_data(self, layers=None):
        """
        Reads all data from the file without a fringe
        
        Returns
        -------
        np.array with all data of the file
        """
        data = self._file.read(indexes=layers, fill_value=self._fill_value)

        # if only a single layer is present compress the data to a 2D array
        if data.shape[0] == 1:
            data = np.squeeze(data)

        if data.ndim == 3:
            data = np.moveaxis(data, 0, -1)

        return data

    def sample_raster(self, points):
        """
        Function to sample the raster at a list of given points

        Parameters
        ----------
        points 
            1: [list]
                list of tuples with longitude and latitude
            2: [DataFrame]
                must contain a Lon and Lat column
            
        Returns
        -------
        list of values in the same order as the points were given
        
        Raises
        ------
        TypeError
            1: wrong type of parameter passed to plotting
            2: points not a list of lists or pandas dataframe
        ValueError
            1: single points don't have exactly two coordinates
            2: longitude out of bounds
            3: latitude out of bounds
        """
        # todo; evaluate this function
        # todo; integrate with geopandas
        # todo; 3D
        # todo; outside bounds

        # Create a Polygon of the area that can be sampled
        bounds = self.get_file_bounds()
        box_x = [bounds[0], bounds[2], bounds[2], bounds[0]]
        box_y = [bounds[3], bounds[3], bounds[1], bounds[1]]
        bounding_box = Polygon(zip(box_x, box_y))

        # store the sampled values
        sampled_values = []
        # store the shapely Points
        points_geometry = []

        if type(points) == pd.DataFrame:
            points = points.apply(lambda x: (x.Lon, x.Lat), axis=1).to_list()

        if type(points) not in (list, tuple):
            raise TypeError("Points can only be passed as a list or dataframe")
        else:
            if type(points[0]) not in (list, tuple):
                raise TypeError("Points is not perceived as a list of lists")

        for point in points:
            if len(point) != 2:
                raise ValueError(f"point doesn't have two coordinates: {point}")
            if point[0] < -180 or point[0] > 180:
                raise ValueError(f"Longitude out of range: {point[0]}")
            if point[1] < -90 or point[1] > 90:
                raise ValueError(f"Latitude out of range: {point[1]}")

            # create shapely Point
            points_geometry.append(Point(point[0], point[1]))
            # check if point is inside the bounds of the file
            if bounding_box.contains(points_geometry[-1]):
                # add value to store
                sampled_values.append(list(self._file.sample([point]))[0][0])
            else:
                # add NaN to store
                sampled_values.append(np.nan)

        return sampled_values

    def cross_profile(self, c1, c2, n=100):
        """
        Routine to create a cross profile, based on self.sample_raster

        Parameters
        ----------
        c1, c2 : tuple
            Location of the start and end of the cross profile (Lat, Lon)
        n : int, optional
            Number of sampling points
        """
        points = np.linspace(c1, c2, num=n).tolist()
        return np.array(self.sample_raster(points=points))

    def _focal_stat_iter(self, output_file=None, *, func=None, overwrite=False, compress=False, p_bar=True,
                         verbose=False, reduce=False, window_size=None, dtype=None, majority_mode="nan", **kwargs):
        """
        Function that calculates focal statistics, in tiled fashion if c_tiles is bigger than 1. The result is outputted
        to 'output_file'
        
        Parameters
        ----------
        func : {"nanmean", "nanmin", "nanmax", "nanstd", "majority"}
            todo; check for the presence of nans and switch behaviour if possible
            function to be applied to the map in a windowed fashion.
        output_file : str
            location of output file.
            todo; should have a default behaviour that is slightly more functional
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
            If `reduce` is False window_size of the whole object is set temporarily. If `reduce` is True, the
            window_size of the whole object is set to 1. The default is None, in which case it will be taken from the
            object: self.window_size.
        majority_mode : {"nan", "ascending", "descending"}, optional
            nan: when more than one class has the same score NaN will be assigned ascending: the first occurrence of the
            maximum count will be assigned descending: the last occurence of the maximum count will be assigned.
            Parameter only used when the `func` is majority.
        **kwargs
            passed to focal_statistics()
        """
        # todo; make it accept ind
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

        window_size_old = self.window_size
        if not reduce:
            if isinstance(window_size, type(None)):
                if self.window_size < 3:
                    raise ValueError("Can't run focal statistics with a window size of 1, unless 'reduce' is True")
                window_size = self.window_size
            else:
                if window_size < 3:
                    raise ValueError("Can't run focal statistics with a window size of 1, unless 'reduce' is True")
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

        old_settings = np.seterr(all='ignore')  # silence all numpy warnings

        with MapWrite(output_file, tiles=(self._v_tiles, self._h_tiles), window_size=self.window_size,
                      overwrite=overwrite, compress=compress, profile=profile) as f:
            for i in self:
                if verbose:
                    print(f"\nTILE: {i + 1}/{self._c_tiles}")
                elif p_bar:
                    progress_bar((i + 1) / self._c_tiles)

                data = self[i]
                # if data is empty, write directly
                if ~np.isnan(self[i]).sum() == 0:
                    # todo; check if this is already done in the focal statistics wrapper
                    f[i] = np.full(f.get_shape(), np.nan)
                else:
                    f[i] = focal_statistics(data, func=func, window_size=window_size, verbose=verbose, reduce=reduce,
                                            majority_mode=majority_mode, **kwargs)

            if p_bar:
                print()

        self.window_size = window_size_old
        np.seterr(**old_settings)  # reset to default

    def focal_mean(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "nanmean". Function forces float64 dtype.
        """
        kwargs['dtype'] = np.float64
        return self._focal_stat_iter(func="nanmean", **kwargs)

    def focal_min(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "nanmin"
        """
        return self._focal_stat_iter(func="nanmin", **kwargs)

    def focal_max(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "nanmax"
        """
        return self._focal_stat_iter(func="nanmax", **kwargs)

    def focal_std(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "nanstd".
        
        Function forces float dtype.
        """
        kwargs['dtype'] = np.float64
        return self._focal_stat_iter(func="nanstd", **kwargs)

    def focal_majority(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "majority". If `majority_mode` is not given as a
        parameter or set to "nan" the dtype will be forced to float64.
        """
        return self._focal_stat_iter(func="majority", **kwargs)

    def correlate(self, other=None, *, output_file=None, window_size=None, fraction_accepted=0.7, verbose=False,
                  overwrite=False, compress=False, p_bar=True, parallel=False):
        """
        Correlate self and other and output the result to output_file.
        
        Parameters
        ----------
        other : MapRead
            map to correlate with
        output_file : str
            Location of output file
        window_size
            todo; create window_size parameter
        fraction_accepted

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
            1: Other is not of type MapRead
        ValueError
            1: v_tiles don't match
            2: h_tiles don't match
            3: shapes don't match
            4: bounds don't match
            5: window_size too small
        """
        # todo; make it accept ind
        if type(other) != MapRead:
            raise TypeError("Other not correctly passed")
        if self._v_tiles != other._v_tiles:
            raise ValueError("v_tiles don't match")
        if self._h_tiles != other._h_tiles:
            raise ValueError("h_tiles don't match")
        if self.window_size != other.window_size:
            raise ValueError("window sizes don't match")
        if self.get_file_shape() != other.get_file_shape():
            raise ValueError("Shapes  of the files don't match")
        if not np.allclose(self.get_file_bounds(), other.get_file_bounds()):
            raise ValueError(f"Bounds don't match:\n{self.get_file_bounds()}\n{other.get_file_bounds()}")
        if self.window_size < 3:
            raise ValueError("Can't run correlation with a window size of 1")

        if not overwrite:
            if os.path.isfile(output_file):
                print(f"Output file already exists. Can only overwrite this file if explicitly stated with the "
                      f"'overwrite' parameter. \n{output_file}\nContinuing without performing operation ...\n")
                return None

        # todo; if updating to new version of correlate_maps this is not necessary anymore
        old_settings = np.seterr(all='ignore')  # silence all numpy warnings

        with MapWrite(output_file, tiles=(self._v_tiles, self._h_tiles), window_size=self.window_size,
                      ref_map=self._location, overwrite=overwrite, compress=compress, dtype=np.float64) as f:
            for i in self:
                if verbose:
                    print(f"\nTILE: {i + 1}/{self._c_tiles}")
                elif p_bar:
                    progress_bar((i + 1) / self._c_tiles)

                f[i] = correlate_maps(self[i], other[i], window_size=self.window_size,
                                      fraction_accepted=fraction_accepted, verbose=verbose)

            if p_bar:
                print()

        np.seterr(**old_settings)  # reset to default

    def export_tile(self, ind=-1, output_file=None):
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

        """
        # todo; this function only works with one layer
        height, width = self.get_shape(ind)
        left, bottom, right, top = self.get_bounds(ind)
        transform = rio.transform.from_bounds(west=left,
                                              south=bottom,
                                              east=right,
                                              north=top,
                                              width=width,
                                              height=height)

        data = self[ind]
        profile = self.profile
        profile.update({'height': height, 'width': width, 'transform': transform, 'driver': "GTiff"})

        with rio.open(output_file, mode="w", **profile) as dst:
            dst.write_band(1, data)

    def plot(self, ind=None, *, mode="ind", basemap=False, figsize=(10, 10), ax=None, log=False, epsg=4326,
             basemap_kwargs=None, **kwargs):
        """
        plot data at given index (ind)
        
        Parameters
        ----------
        ind : .
            check self.get_pointer(). The default is None which will set `mode` to "all"
        mode : {"ind", "all"}
            if ind is passed, only the data at the given index is plotted. If all is given, it will plot the whole file.
        basemap : bool, optional
            plot a basemap behind the data
        figsize : tuple, optional
            matplotlib figsize parameter
        ax : Axes, optional
            matplotlib axes where plot is drawn on
        log : bool, optional
            plot the colors on a log scale
        epsg : int, optional
            todo; add a possibility to take the native one
            EPSG code that will be used to render the plot, the default is 4326
        basemap_kwargs
            kwargs going to the basemap command
        **kwargs
            kwargs going to the plot_map() command
        # todo; at least add resolution, xticks and yticks as parameters here

        Returns
        -------
        :obj:`~matplotlib.axes.Axes` or GeoAxis if basemap=True
        """
        if isinstance(ind, type(None)):
            mode = "all"

        if mode == "ind":
            ind = self.get_pointer(ind)
            data = self.get_data(ind)
            bounds = self._file.window_bounds(self._tiles[ind])
        elif mode == "all":
            data = self.get_file_data()
            bounds = self._file.bounds
        else:
            raise ValueError("Mode not recognised")

        if data.ndim not in (2, 3):
            print(f"Can't plot this data. Dimensions : {data.ndim}")
            return None
        elif data.ndim == 3 and data.shape[-1] not in (3, 4):
            # todo; create the posibility to select a layer
            print(f"Can't plot this data; only RGB(A) accepted on the third axis.")
            return None

        if log:
            data = np.log(data)

        if basemap:
            if isinstance(basemap_kwargs, type(None)):
                basemap_kwargs = {}

            ax = basemap_function(*bounds, ax=ax, epsg=epsg, figsize=figsize, **basemap_kwargs)

            if isinstance(self.epsg, type(None)):
                raise RuntimeError("This object does not contain a EPSG code. It can be set through set_epsg()")
            elif self.epsg == 4326:
                transform = ccrs.PlateCarree()
            else:
                transform = ccrs.epsg(self.epsg)

            ax = plot_map(data, transform=transform, extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
                          ax=ax, **kwargs)

        else:
            ax = plot_map(data, ax=ax, figsize=figsize, **kwargs)

        return ax

    def plot_classified(self, ind=None, *, mode="ind", basemap=False, figsize=(10, 10), ax=None, epsg=4326,
                        basemap_kwargs=None, **kwargs):
        """
        Plots data in a classified way. Look at plot_classified_map for the implementation.

        Parameters
        ----------
        ind : .
            check self.get_pointer(). The default is None which will set `mode` to "all"
        mode : {"ind", "all"}
            if ind is passed, only the data at the given index is plotted. If all is given, it will plot the whole file.
        basemap : bool, optional
            plot a basemap behind the data
        figsize : tuple, optional
            matplotlib figsize parameter
        ax : Axes, optional
            matplotlib axes where plot is drawn on
        epsg : int, optional
            EPSG code that will be used to render the plot, the default is 4326
        basemap_kwargs : dict, optional
            kwargs for basemap
        **kwargs
            kwargs going to the plot_classified_map() command itself

        Returns
        -------
        :obj:`~matplotlib.axes.Axes` or GeoAxis if basemap=True
        """
        if isinstance(ind, type(None)):
            mode = "all"

        if mode == "ind":
            ind = self.get_pointer(ind)
            data = self.get_data(ind)
            bounds = self._file.window_bounds(self._tiles[ind])
        elif mode == "all":
            data = self.get_file_data()
            bounds = self._file.bounds
        else:
            raise ValueError("Mode not recognised")

        # todo, rebuild it to accept taking a layer
        data = np.squeeze(data)
        if data.ndim > 2:
            raise ValueError(f"This function is only applicable for 2D data. This file contains stacked layers. "
                             f"{data.shape}")

        title = kwargs.pop('title', '')
        legend_kwargs = kwargs.pop('legend_kwargs', {})
        legend_kwargs.update({'title': title})

        if basemap:
            if isinstance(basemap_kwargs, type(None)):
                basemap_kwargs = {}

            ax = basemap_function(*bounds, ax=ax, epsg=epsg, figsize=figsize, **basemap_kwargs)

            if isinstance(self.epsg, type(None)):
                raise RuntimeError("This object does not contain a EPSG code. It can be set through set_epsg()")
            elif self.epsg == 4326:
                transform = ccrs.PlateCarree()
            else:
                transform = ccrs.epsg(self.epsg)

            ax = plot_classified_map(data, ax=ax, legend_kwargs=legend_kwargs, transform=transform,
                                     extent=(bounds[0], bounds[2], bounds[1], bounds[3]), **kwargs)
        else:
            ax = plot_classified_map(data, ax=ax, figsize=figsize, legend_kwargs=legend_kwargs, **kwargs)

        return ax

    def __getitem__(self, ind):
        """
        pointer to internal function get_data().
        """
        if isinstance(ind, tuple) and len(ind) == 2:
            ind, layers = ind
        else:
            layers = None
        return self.get_data(ind, layers=layers)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import rasterio as rio
from shapely.geometry import Point, Polygon

from mappy.plotting import basemap as basemap_function
from mappy.plotting import plot_map, plot_classified_map
from .MapBase import MapBase
from .MapWrite import MapWrite
from .misc import bounds_to_platecarree
from ..progress_bar.progress_bar import progress_bar
from ..raster_functions import resample_profile, focal_statistics, correlate_maps


class MapRead(MapBase):
    """
    Instance of MapBase, file opened in 'r' mode.
    
    Parameters
    ----------
    location : str
        Location of the map
    tiles : int or tuple of ints, optional
        See tiles property in MapBase
    window_size : int, optional
        Window size to be set on the map. See property in MapBase
    fill_value : numeric, optional
        Fill value used in rasterio.read call.
    epsg : int, optional
        EPGS code of the data. This parameter is only used if the code can't be found in the file

    Attributes
    ----------
    See Also MapBase

    values : ndarray
        all data of the file

    Raises
    ------
    TypeError
        Location not a string
    IOError
        Location of file not found
    """

    def __init__(self, location, *, tiles=1, window_size=1, fill_value=None, epsg=None):
        if type(location) != str:
            raise TypeError("Location not recognised")
        if not os.path.isfile(location):
            raise IOError(f"Location can't be found:\n'{location}'")

        self._location = location
        self._mode = "r"

        # open file in reading mode and set the global nan_value
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

        # collector of the base class. This list contains all files opened with the Map classes
        self.collector.append(self)

    def get_data(self, ind=-1, layers=None):
        """
        Returns the data associated with the index of the tile. Layers are set on the third axis.
        
        Parameters
        ----------
        ind : .
            see MapBase.get_pointer(). If set to None it will read the whole file.
        layers : int or list, optional
            The number of the layer required or a list of layers (which might contain duplicates if needed).
            The default is None, which will load all layers.
        
        Returns
        -------
        numpy array of shape self.get_shape()
        
        """
        if isinstance(ind, type(None)):
            ind = bounds_to_platecarree(self._data_proj, self.get_file_bounds())

        # type and bound checking happens in self.get_pointer()
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
        if not isinstance(compress, type(None)):
            profile.update({'compress': compress})
        profile.update({'count': 1, 'driver': "GTiff"})

        if isinstance(ind, type(None)):
            with MapWrite(output_file, tiles=(self._v_tiles, self._h_tiles), window_size=self.window_size,
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

            with MapWrite(output_file, overwrite=overwrite, profile=profile) as f:
                f[0] = focal_statistics(self[ind, layers], func=func, window_size=window_size, verbose=verbose,
                                        reduce=reduce, majority_mode=majority_mode, **kwargs)

        self.window_size = self_old_window_size

    def focal_mean(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "nanmean". Function forces float64 dtype.
        """
        kwargs['dtype'] = np.float64
        return self._focal_stat_iter(func="mean", **kwargs)

    def focal_min(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "nanmin"
        """
        return self._focal_stat_iter(func="min", **kwargs)

    def focal_max(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "nanmax"
        """
        return self._focal_stat_iter(func="max", **kwargs)

    def focal_std(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "nanstd".
        
        Function forces float dtype.
        """
        kwargs['dtype'] = np.float64
        return self._focal_stat_iter(func="std", **kwargs)

    def focal_majority(self, **kwargs):
        """
        Function passes call to MapRead._focal_stat_iter with func = "majority". If `majority_mode` is not given as a
        parameter or set to "nan" the dtype will be forced to float64.
        """
        return self._focal_stat_iter(func="majority", **kwargs)

    def correlate(self, other=None, ind=None, self_layers=1, other_layers=1, *, output_file=None, window_size=None,
                  fraction_accepted=0.7, verbose=False, overwrite=False, compress=False, p_bar=True, parallel=False):
        """
        Correlate self and other and output the result to output_file.
        
        Parameters
        ----------
        other : MapRead
            map to correlate with
        ind : . , optional
            see self.get_pointer(). If set, no tiled behaviour occurs.
        self_layers, other_layers : int, optional
            the layer that will be used to calculate the correlations
        output_file : str
            Location of output file
        window_size : int, optional
            Size of the window used for the correlation calculations. It should be bigger than 1, the default is the
            window_size set on self (`MapRead`).
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
            Other is not of type MapRead
        """
        # todo; implement reduce
        # todo; implement parallel

        if not isinstance(other, MapRead):
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
            with MapWrite(output_file, tiles=(self._v_tiles, self._h_tiles), window_size=self.window_size,
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

            with MapWrite(output_file, overwrite=overwrite, profile=profile, window_size=self.window_size) as f:
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
        if not classified and isinstance(layers, (tuple, list)) and len(layers > 4):
            raise IndexError("layers can only be a maximum of four integers, as the would index for RGB(A)")

        data = self.get_data(ind, layers=layers)
        extent = self.get_bounds(ind)

        if isinstance(bounds, type(None)):
            bounds = bounds_to_platecarree(self._data_proj, extent)
        elif bounds == "global":
            bounds = [-180, -90, 180, 90]

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

        if classified:
            return plot_classified_map(data, ax=ax, figsize=figsize, **kwargs)
        else:
            return plot_map(data, ax=ax, figsize=figsize, **kwargs)


    def plot(self, *args, **kwargs):
        """
        alias for self.plot_map
        """
        return self(*args, **kwargs)

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

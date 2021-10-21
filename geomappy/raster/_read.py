import copy
import os
import warnings

import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import Polygon, Point

from ..basemap import basemap as basemap_function
from ..plotting import plot_classified_map, plot_map
from ..focal_statistics import focal_statistics, correlate_maps
from ..profile import resample_profile
from ..utils import progress_bar
from ._base import RasterBase
from ._write import RasterWriter


class RasterReader(RasterBase):
    """
    Instance of RasterBase, file opened in 'r' mode.

    Parameters
    ----------
    fp : str
        Location of the map
    tiles : int or tuple of ints, optional
        See tiles property in RasterBase
    force_equal_tiles : bool, optional
        Force tiles of equal size. This defaults to False, which last the last tiles be slightly larger.
    window_size : int, optional
        Window size to be set on the map. See property in RasterBase
    fill_value : numeric, optional
        Fill value used in rasterio.read call.

    Attributes
    ----------
    See also ``RasterBase``

    values : ndarray
        all data of the file
    """
    def __init__(self, fp, *, tiles=1, force_equal_tiles=False, window_size=1, fill_value=None):

        self._fp = rio.open(fp, mode='r')
        self._profile = copy.deepcopy(self._fp.profile)
        self._location = self._fp.name
        self._mode = "r"

        dtype = np.dtype(self.profile['dtype'])
        if fill_value is None:
            if np.issubdtype(dtype, np.floating):
                self._fill_value = np.nan
            else:
                warnings.warn("The raster contains integer data and no fill_value. This results in the data "
                      "being returned without removing `nodata`")
                self._fill_value = self.profile['nodata']
                if self._fill_value is None:
                    raise ValueError("no fill_value was given and the file does not have a nodata value to "
                                     "use as a backup")
        else:
            self._fill_value = dtype.type(fill_value)

        self.window_size = window_size
        self.set_tiles(tiles, force_equal_tiles)

        self._current_ind = 0

        # collector of the base class. This list contains all files opened with the Raster classes
        self.collector.append(self)
        self.mmap_collector = {}

    def get_data(self, ind=-1, indexes=None):
        """
        Returns the data associated with the index of the tile. Layers are set on the first axis.

        Parameters
        ----------
        ind : .
            see RasterBase.get_pointer(). If set to None it will read the whole file.
        indexes : int or list, optional
            The number of the layer required or a list of indexes (which might contain duplicates if needed).
            The default is None, which will load all indexes.

        Returns
        -------
        numpy.ndarray of shape self.get_shape()
        """
        ind = self.get_pointer(ind)
        if ind < self._c_tiles:
            if isinstance(indexes, int):
                c = 1
            elif indexes is None:
                c = len(self.indexes)
            else:
                c = len(indexes)

            data = np.full((c, self.get_height(ind), self.get_width(ind)),
                           fill_value=self._fill_value, dtype=self.dtype)

            window = self._tiles[ind]
            pad_height = -window.row_off if window.row_off < 0 else 0
            pad_width = -window.col_off if window.col_off < 0 else 0
            window = rio.windows.Window(row_off=window.row_off + pad_height,
                                        col_off=window.col_off + pad_width,
                                        height=window.height - pad_height,
                                        width=window.width - pad_width)

            data[:, pad_height:, pad_width:] = \
                self.read(indexes=indexes, window=window, boundless=True, fill_value=self._fill_value)

        else:
            data = self.read(indexes=indexes, window=self._tiles[ind], boundless=True, fill_value=self._fill_value)

        if data.shape[0] == 1:
            return data[0]
        else:
            return data

    def get_file_data(self, indexes=None):
        """
        Read the whole file

        Parameters
        ----------
        indexes : int or list, optional
            The number of the layer required or a list of indexes (which might contain duplicates if needed).
            The default is None, which will load all indexes.

        Returns
        -------
        np.array with all data of the file
        """
        return self.get_data(ind=None, indexes=indexes)

    values = property(get_file_data)

    def sample_raster(self, points, indexes=None):
        """
        Sample the raster at points

        Parameters
        ----------
        points : list or GeoDataFrame
            Either list of tuples with Lat/Lon values or a GeoDataFrame with points. If polygons are present in the
            GeoDataFrame the centroid will be used to obtain the values.
        indexes : int or list, optional
            The number of the layer required or a list of indexes (which might contain duplicates if needed).
            The default is None, which will use all indexes.

        Returns
        -------
        (GeoSeries, array)
            points in the georefence system of the data and array with values
        """
        # Create a Polygon of the area that can be sampled
        bounds = self.bounds
        box_x = [bounds[0], bounds[2], bounds[2], bounds[0]]
        box_y = [bounds[3], bounds[3], bounds[1], bounds[1]]
        bounding_box = Polygon(zip(box_x, box_y))
        outline = gpd.GeoDataFrame(geometry=[bounding_box])
        outline.crs = self.proj.definition_string()
        outline = outline.to_crs({'init': 'epsg:4326'})

        if not isinstance(points, (gpd.GeoSeries, gpd.GeoDataFrame)):
            if not isinstance(points, (tuple, list)):
                raise TypeError("points should either be a GeoDataFrame or a list of tuples")
            points = gpd.GeoDataFrame(geometry=[Point(point[0], point[1]) for point in points],
                                      crs={'init': 'epsg:4326'})

        points = points.loc[points.geometry.apply(lambda x: outline.contains(x)).values, :] \
            .to_crs(self.proj.definition_string())

        geom_types = points.geometry.type
        point_idx = np.asarray((geom_types == "Point"))
        points.loc[~point_idx, 'geometry'] = points.loc[~point_idx, 'geometry'].centroid

        if points.empty:
            raise IndexError("Geometries are all outside the bounds")

        sampling_points = points.geometry.apply(lambda x: (x.x, x.y)).values.tolist()
        values = self.sample(sampling_points, indexes=indexes)

        return points.geometry, np.array([x for x in values])

    def cross_profile(self, c1, c2, n=100, indexes=1):
        """
        Routine to create a cross profile, based on self.sample_raster

        Parameters
        ----------
        c1, c2 : tuple
            Location of the start and end of the cross profile (Lat, Lon)
        n : int, optional
            Number of sampling points
        indexes : int or list, optional
            The number of the layer required or a list of indexes (which might contain duplicates if needed).
            The default is None, which will use all indexes.
        """
        points = np.linspace(c1, c2, num=n).tolist()
        return self.sample_raster(points=points, indexes=indexes)

    def _focal_stat_iter(self, func=None, *, output_file=None, ind=None, indexes=1, overwrite=False, compress=True,
                         p_bar=True, reduce=False, window_size=None, dtype=None, majority_mode="nan", parallel=False,
                         **kwargs):
        """
        Function that calculates focal statistics, in tiled fashion if self.c_tiles is bigger than 1. The result is
        outputted to `output_file`

        Parameters
        ----------
        func : {"mean", "min", "max", "std", "majority"}
            function to be applied to the map in a windowed fashion.
        output_file : str, optional
            location of output file. By default it is 'focal_func_x' with x a number that has not been used
            in the current directory.
        ind : . , optional
            see self.get_pointer(). If set, no tiled behaviour occurs.
        indexes : int, optional
            the layer that will be used to do the calculations
        overwrite : bool, optional
            If allowed to overwrite the output_file. The default is False.
        compress : bool, optional
            Compress output data, the default is True.
        p_bar : bool, optional
            Show the progress bar. Default is True.
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
        if func is None:
            raise TypeError("No function given")
        if not isinstance(reduce, bool):
            raise TypeError("reduce parameter needs to be a boolean")
        if output_file is None:
            i = 0
            while True:
                output_file = f"focal_{func}_{i}.tif"
                if not os.path.isfile(output_file):
                    break
                i += 1

        reset_window_size = self.window_size

        if not reduce:
            if window_size is None:
                window_size = self.window_size
            if window_size < 3:
                raise ValueError("Can't run focal statistics with a window lower than 3, unless 'reduce' is True")
            self.window_size = window_size
            profile = self.profile
        else:
            if window_size is None:
                raise TypeError("window_size needs to be provided when `reduce=True`")
            if window_size < 2:
                raise ValueError("Window_size needs to be bigger than 1 in reduction mode")
            self.window_size = 1
            profile = resample_profile(self.profile, 1 / window_size)

        if dtype is not None:
            profile['dtype'] = dtype
        if func == "majority" and majority_mode == "nan":
            profile['dtype'] = np.float64
        if compress:
            profile.update({'compress': "LZW"})
        profile.update({'count': 1, 'driver': "GTiff"})

        if ind is None:
            with RasterWriter(output_file, tiles=(self._v_tiles, self._h_tiles), window_size=self.window_size,
                              force_equal_tiles=self._force_equal_tiles, overwrite=overwrite, profile=profile) as f:

                for i in self:
                    if p_bar:
                        progress_bar((i + 1) / self._c_tiles)

                    data = self[i, indexes]

                    # if data is empty, write directly
                    if (~np.isnan(data)).sum() == 0:
                        f[i] = np.full(f.get_shape(i), np.nan)
                    else:
                        f[i] = focal_statistics(data, func=func, window_size=window_size, reduce=reduce,
                                                majority_mode=majority_mode, **kwargs)

                if p_bar:
                    print()


        else:
            profile.update({'height': self.get_height(ind),
                            'width': self.get_width(ind),
                            'transform': self.get_transform(ind)})
            if reduce:
                profile = resample_profile(profile, 1 / window_size)

            with RasterWriter(output_file, overwrite=overwrite, profile=profile) as f:
                f[0] = focal_statistics(self[ind, indexes], func=func, window_size=window_size,
                                        reduce=reduce, majority_mode=majority_mode, **kwargs)

        self.window_size = reset_window_size

    def focal_mean(self, **kwargs):
        """
        Function passes call to RasterReader._focal_stat_iter with func = "nanmean". Function forces float64 dtype.
        """
        kwargs['dtype'] = np.float64
        return self._focal_stat_iter(func="mean", **kwargs)

    def focal_min(self, **kwargs):
        """
        Function passes call to RasterReader._focal_stat_iter with func = "nanmin"
        """
        return self._focal_stat_iter(func="min", **kwargs)

    def focal_max(self, **kwargs):
        """
        Function passes call to RasterReader._focal_stat_iter with func = "nanmax"
        """
        return self._focal_stat_iter(func="max", **kwargs)

    def focal_std(self, **kwargs):
        """
        Function passes call to RasterReader._focal_stat_iter with func = "nanstd".

        Function forces float dtype.
        """
        kwargs['dtype'] = np.float64
        return self._focal_stat_iter(func="std", **kwargs)

    def focal_majority(self, **kwargs):
        """
        Function passes call to RasterReader._focal_stat_iter with func = "majority". If `majority_mode` is not given as a
        parameter or set to "nan" the dtype will be forced to float64.
        """
        return self._focal_stat_iter(func="majority", **kwargs)

    def correlate(self, other=None, *, ind=None, self_indexes=1, other_indexes=1, output_file=None, window_size=None,
                  overwrite=False, compress=True, p_bar=True, reduce=False, parallel=False, **kwargs):
        """
        Correlate self and other and output the result to output_file.

        Parameters
        ----------
        other : RasterReader
            map to correlate with
        ind : . , optional
            see self.get_pointer(). If set, no tiled behaviour occurs.
        self_indexes, other_indexes : int, optional
            the layer that will be used to calculate the correlations
        output_file : str
            Location of output file
        window_size : int, optional
            Size of the window used for the correlation calculations. It should be bigger than 1, the default is the
            window_size set on self (`_RasterReader`).
        overwrite : bool, optional
            If allowed to overwrite the output_file, default is False
        compress : bool, optional
            Compress calculated data, default is True.
        p_bar : bool, optional
            Show the progress bar. If verbose is True p_bar will be False. Default value is True.
        reduce : bool, optional
            If True, the dimensions of the output map are divided by `window_size`. If False the resulting file has the
            same shape as the input, which is the default.

        Notes
        -----
        the following parameter will be passed on directly to the focal_statistics calculation through **kwargs

        fraction_accepted : float, optional
            Fraction of the window that has to contain not-nans for the function to calculate the correlation.
            The default is 0.7.
        """
        if not isinstance(other, RasterReader):
            raise TypeError("Other not correctly passed")
        if not isinstance(self_indexes, int) or not isinstance(other_indexes, int):
            raise TypeError("Indexes can only be an integer for correlation calculations")

        # todo; move to Raster/RasterBase
        if self._v_tiles != other._v_tiles:
            raise ValueError("v_tiles don't match")
        if self._h_tiles != other._h_tiles:
            raise ValueError("h_tiles don't match")
        if self._force_equal_tiles != other._force_equal_tiles:
            raise ValueError("force_equal_tiles needs to be set equally on both Rasters")
        if self.shape != other.shape:
            raise ValueError("Shapes  of the files don't match")
        if not np.allclose(self.bounds, other.bounds):
            raise ValueError(f"Bounds don't match:\n{self.bounds}\n{other.bounds}")

        if output_file is None:
            i = 0
            while True:
                output_file = f"focal_correlate_{i}.tif"
                if not os.path.isfile(output_file):
                    break
                i += 1

        reset_window_size = self.window_size, other.window_size

        if not reduce:
            if window_size is None:
                window_size = self.window_size
            if window_size < 3:
                raise ValueError("Can't run focal statistics with a window lower than 3, unless 'reduce' is True")
            self.window_size = window_size
            other.window_size = window_size
            profile = self.profile
        else:
            if window_size is None:
                raise TypeError("window_size needs to be provided when `reduce=True`")
            if window_size < 2:
                raise ValueError("window_size needs to be bigger than 1 in reduction mode")
            self.window_size = 1
            other.window_size = 1
            profile = resample_profile(self.profile, 1 / window_size)

        profile['dtype'] = np.float64
        if compress:
            profile.update({'compress': "LZW"})
        profile.update({'count': 1, 'driver': "GTiff"})

        if ind is None:
            with RasterWriter(output_file, tiles=(self._v_tiles, self._h_tiles), window_size=self.window_size,
                              force_equal_tiles=self._force_equal_tiles, overwrite=overwrite, profile=profile) as f:

                for i in self:
                    if p_bar:
                        progress_bar((i + 1) / self._c_tiles)

                    self_data = self[i, self_indexes]
                    other_data = other[i, other_indexes]

                    # if data is empty, write directly
                    if (~np.isnan(self_data)).sum() == 0 or (~np.isnan(other_data)).sum() == 0:
                        f[i] = np.full(f.get_shape(i), np.nan)
                    else:
                        f[i] = correlate_maps(self_data, other_data, window_size=window_size, reduce=reduce,
                                              **kwargs)

                if p_bar:
                    print()

        else:
            profile.update({'height': self.get_height(ind),
                            'width': self.get_width(ind),
                            'transform': self.get_transform(ind)})
            if reduce:
                profile = resample_profile(profile, 1 / window_size)

            with RasterWriter(output_file, overwrite=overwrite, profile=profile) as f:
                f[0] = correlate_maps(self[ind, self_indexes], other[ind, other_indexes], window_size=window_size,
                                      reduce=reduce, **kwargs)

        self.window_size = reset_window_size[0]
        other.window_size = reset_window_size[1]

    focal_correlate = correlate

    def export_tile(self, ind, output_file, indexes=None, compress=True):
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
        indexes : int or tuple, optional
            the layer that is exported. If a tuple is provided several indexes are exported, None means all indexes by
            default.
        compress : bool, optional
            rasterio compression parameter
        """
        data = self.get_data(ind, indexes=indexes)
        profile = self.get_profile(ind)

        profile.update({'driver': "GTiff", 'count': data.shape[0]})
        if compress:
            profile.update({'compress': "LZW"})
        with rio.open(output_file, mode="w", **profile) as dst:
            dst.write(data)

    def _plot(self, classified, ind=None, indexes=1, *, basemap=False, projection='native', figsize=(10, 10), ax=None,
              xticks=30, yticks=30, resolution="110m", fontsize=10, basemap_kwargs=None, extent=None,
              **kwargs):
        """
        Plot data at given index (ind). Classified or not, depends on the first parameter.

        Parameters
        ----------
        classified : bool
            Switch between `plot_classified_map` and `plot_map`
        ind : ., optional
            check self.get_pointer(). The default is None which will get all data.
        indexes : int or tuple, optional
            The layer that is plotted. If `classified` is True only a single layer is accepted (as integer) while False
            will accept a tuple representing RGB or RGBA.
        basemap : bool, optional
            plot a basemap behind the data
        projection : cartopy CRS
            Plot projection. By default `native` which plots a map in the coordinate system of the data.
            (which works best). None yields crs.PlateCarre(), if no epsg code is set in the basemap_kwargs.
        figsize : tuple, optional
            matplotlib figsize parameter
        ax : Axes, optional
            matplotlib axes where plot is drawn on
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
        extent : list, optional
            extent of the plot, if not provided it will plot the extent belonging to the Index. If the string "global"
            is set the global extent will be used.
        **kwargs
            kwargs going to the plot_map() command

        Returns
        -------
        (:obj:`~matplotlib.axes.Axes` or GeoAxis, legend)
        """
        if classified and not isinstance(indexes, int):
            raise TypeError("indexes can only be an integer for classified plotting")
        if not classified and isinstance(indexes, (tuple, list)) and len(indexes) > 4:
            raise IndexError("indexes can only be a maximum of four integers, as the would index for RGB(A)")

        data = self.get_data(ind, indexes=indexes)
        if data.ndim > 2:
            data = np.moveaxis(data, 0, -1)
        bounds = self.get_bounds(ind)

        if isinstance(basemap_kwargs, type(None)):
            basemap_kwargs = {}

        if projection == 'native':
            projection = self.projection

        if extent is None:
            extent = bounds
            if basemap:
                basemap_kwargs.update({'extent_projection': self.projection})

        if not isinstance(fontsize, (tuple, list)):
            fontsize = (fontsize, fontsize)
        kwargs.update({'fontsize': fontsize[1]})

        if basemap:
            if 'xticks' not in basemap_kwargs:
                basemap_kwargs.update({'xticks': xticks})
            if 'yticks' not in basemap_kwargs:
                basemap_kwargs.update({'yticks': yticks})
            if 'fontsize' not in basemap_kwargs:
                basemap_kwargs.update({'fontsize': fontsize[0]})
            if 'resolution' not in basemap_kwargs:
                basemap_kwargs.update({'resolution': resolution})

            ax = basemap_function(extent, ax=ax, projection=projection, figsize=figsize, **basemap_kwargs)
            kwargs.update({'transform': self.projection, 'extent': (bounds[0], bounds[2], bounds[1], bounds[3])})

        if classified:
            return plot_classified_map(data, ax=ax, figsize=figsize, **kwargs)
        else:
            return plot_map(data, ax=ax, figsize=figsize, **kwargs)

    def plot(self, *args, **kwargs):
        """
        alias for self.plot_map
        """
        return self.plot_map(*args, **kwargs)

    def plot_map(self, ind=None, indexes=1, **kwargs):
        """
        Plot map wrapper around `plot_map`. It redirects to `self._plot` with parameter `classified` = False
        """
        return self._plot(classified=False, ind=ind, indexes=indexes, **kwargs)

    def plot_classified_map(self, ind=None, indexes=1, **kwargs):
        """
        Plot map wrapper around `plot_classified_map`. It redirects to `self._plot` with parameter `classified` = True
        """
        return self._plot(classified=True, ind=ind, indexes=indexes, **kwargs)

    def __getitem__(self, i):
        """
        pointer to internal function get_data().
        """
        ind, indexes = self.ind_user_input(i)
        return self.get_data(ind, indexes=indexes)

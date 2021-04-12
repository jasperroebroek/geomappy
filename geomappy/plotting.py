import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from geopandas.plotting import plot_polygon_collection, plot_linestring_collection, plot_point_collection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, BoundaryNorm, Normalize, ListedColormap, NoNorm
from shapely.geometry import Point

from .utils import nandigitize
from .colors import cmap_discrete, create_colorbar_axes, add_colorbar, legend_patches


# PLOTTING
class Plot:
    """
    Plot object. Handles the creation of the Axes if none is provided, stores the parameters for plotting and deals
    with the Axes makeup when the plot is drawn.
    """
    def __init__(self, ax, figsize, legend, params, **kwargs):
        """Sets the parameters and creates the Axes if not provided"""
        if isinstance(ax, type(None)):
            _, ax = plt.subplots(figsize=figsize)

        self.ax = ax
        self.kwargs = kwargs
        self.params = params
        self.legend = legend

    def draw(self, **draw_kwargs):
        """Uses the _draw() method of sub-objects for the actual representation"""
        fontsize = self.kwargs.pop('fontsize', None)
        self._draw(**draw_kwargs)
        self.ax.tick_params(axis="x", labelsize=fontsize)
        self.ax.tick_params(axis="y", labelsize=fontsize)

        if not isinstance(self.legend, type(None)):
            legend = self.legend.draw(self.ax, self.params)
        else:
            legend = None
        return self.ax, legend


class PlotRaster(Plot):
    """
    Plot object for raster data. If a GeoAxes is provided the `extent` and `transform` will be inferred if not provided.
    If this leads to the wrong result they can be passed in directly as keyword arguments.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.ax, cartopy.mpl.geoaxes.GeoAxes):
            if 'extent' not in kwargs:
                self.kwargs['extent'] = self.ax.get_extent()
            if 'transform' not in kwargs:
                self.kwargs["transform"] = self.ax.projection

    def _draw(self):
        self.ax.imshow(self.params.values, norm=self.params.norm, cmap=self.params.cmap, origin='upper', **self.kwargs)


class PlotShapes(Plot):
    """
    Plot object for polygon/point data. If a GeoAxes is provided the `transform` will be inferred if not provided.
    If this leads to the wrong result it can be passed in directly as keyword argument.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.ax, cartopy.mpl.geoaxes.GeoAxes):
            if 'transform' not in kwargs:
                self.kwargs["transform"] = ccrs.PlateCarree()

    @staticmethod
    def _create_geometry_values_and_sizes(lat=None, lon=None, values=None, s=None, df=None):
        """
        Function that deals with the input data for `plot_shapes` and `plot_classified_shapes`. Lat and Lon will be used
        if `df` is not given. Values and s will be cast to the length of the geometries, or set to 1 and None
        respectively if not provided.

        Parameters
        ----------
        lat, lon : array-like
            Latitude and Longitude
        values : array-like or numeric or str
            Values at each geometry. A single numeric value will be cast to all geometries. If `df` is set a string can
            be passed to values which will be interpreted as the name of the column holding the values.
        s : array-like, optional
            Size for each geometry. A single numeric value will be cast to all geometries. If `df` is set a string will
            be interpreted as the name of the column holding the sizes. If None is set no sizes will be set.
        df : GeoDataFrame, optional
            GeoDataFrame containing the geometries that will be used for plotting in a 'geometry' column.

        Returns
        -------
        geometry, values, markersize
        markersize can be None if None is passed in as `s`
        """
        if isinstance(df, type(None)):
            lon = np.array(lon).flatten()
            lat = np.array(lat).flatten()
            values = np.array(values).flatten()
            if lon.size != lat.size:
                raise IndexError("Mismatch in length of `lat` and `lon`")

            if isinstance(s, type(None)):
                markersize = None
            else:
                markersize = np.array(s).flatten()
            geometry = gpd.GeoSeries([Point(lon[i], lat[i]) for i in range(len(lon))])

        else:
            geometry = df['geometry']
            if isinstance(values, str):
                if values in df.columns:
                    values = df.loc[:, values].values
                else:
                    values = np.array([None])
            else:
                values = np.array(values).flatten()

            if isinstance(s, str):
                if s in df.columns:
                    markersize = df.loc[:, s].values
                else:
                    markersize = None
            else:
                markersize = np.array(s).flatten()

        if values.size == 1:
            if isinstance(values[0], type(None)):
                values = np.array(1)
            values = values.repeat(geometry.size)
        elif values.size != geometry.size:
            raise IndexError("Mismatch length sizes and geometries")

        if not isinstance(markersize, type(None)):
            if markersize.size == 1:
                if isinstance(markersize[0], type(None)):
                    markersize = None
                else:
                    markersize = markersize.repeat(geometry.size)
            elif markersize.size != geometry.size:
                raise IndexError("Mismatch length of `s` and coordindates")

        return geometry, values, markersize

    @staticmethod
    def _plot_geometries(ax, df, colors, linewidth, markersize, **kwargs):
        """
        internal plotting function for geometries, called by plot_shapes and plot_classified_shapes

        Parameters
        ----------
        ax : matplotlib.Axes
            axes to plot on
        df : GeoDataFrame
            geodataframe containing the geometries
        colors : pd.Series
            Series object containing the colors
        linewidth : numeric
            linewidth of the geometries
        markersize : pd.Series
            size of points in `df`
        **kwargs
            Keyword arguments for the geopandas plotting functions: plot_point_collection, plot_polygon_collection and
            plot_linestring_collection

        Notes
        -----
        Setting either ``facecolor`` or ``edgecolor`` does the same as in geopandas. It overwrite the behaviour of this
        function.
        """
        geom_types = df.geometry.type
        poly_idx = np.asarray((geom_types == "Polygon") | (geom_types == "MultiPolygon"))
        line_idx = np.asarray((geom_types == "LineString") | (geom_types == "MultiLineString"))
        point_idx = np.asarray((geom_types == "Point") | (geom_types == "MultiPoint"))

        facecolor = kwargs.pop('facecolor', np.array(colors))
        edgecolor = kwargs.pop('edgecolor', np.array(colors))

        if not isinstance(facecolor, np.ndarray):
            facecolor = pd.Series([facecolor] * df.shape[0])
        if not isinstance(edgecolor, np.ndarray):
            edgecolor = pd.Series([edgecolor] * df.shape[0])

        # plot all Polygons and all MultiPolygon components in the same collection
        polys = df.geometry[poly_idx]
        if not polys.empty:
            plot_polygon_collection(ax, polys, facecolor=facecolor[poly_idx], edgecolor=edgecolor[poly_idx],
                                    linewidth=linewidth, **kwargs)

        # plot all LineStrings and MultiLineString components in same collection
        lines = df.geometry[line_idx]
        if not lines.empty:
            plot_linestring_collection(ax, lines, facecolor=facecolor[line_idx], edgecolor=edgecolor[line_idx],
                                       linewidth=linewidth, **kwargs)

        # plot all Points in the same collection
        points = df.geometry[point_idx]
        if not points.empty:
            if isinstance(markersize, np.ndarray):
                markersize = markersize[point_idx]
            plot_point_collection(ax, points, facecolor=facecolor[point_idx], edgecolor=edgecolor[point_idx],
                                  markersize=markersize, linewidth=linewidth, **kwargs)

    def _draw(self, geometry, markersize, linewidth):
        """Sets the colors that will be passed on to `PlotShapes._plot_geometries`"""
        values = self.params.values
        colors = pd.Series(self.params.cmap(self.params.norm(values)).tolist())
        colors[np.isnan(values)] = self.params._nan_color
        self._plot_geometries(self.ax, geometry, colors, linewidth, markersize, **self.kwargs)


# PLOTTING PARAMETER OBJECTS
class PlotParams:
    """
    Parameter object used in the `Plot` objects.

    Attributes
    ----------
    norm : matplotlib norm
        Norm object to normalize the values. It is the only required parameter that does not have a default, together
        with the values.
    values : array-like
        Values that will be plotted. It has no default, and asking for it while it has not been set, raises an error.
    cmap : matplotlib colormap, optional
        Colormap to color the data. The default is 'viridis'
    bins : array-like, optional
        Iterable containing bins that can be used for either classification or binning values
    labels : array-like, optional
        Labels for the legend
    labels_overwritten : bool, optional
        Switch to overwrite the default labels with the labels provided here. If a `Colorbar` is used this switch needs
        to be activated to actually overwrite the defaults
    colors : array-like, optional
        Colors for `LegendPatches`. The default is None, in which case the legend will silently not be drawn.
    nan_color : matplotlib color, optionl
        Color for NaN values. It is both set in the property, as well as in the cmap instance
    extend : {"both", "neither", "max", "min"}, optional
        Extend for the colorbar. The default is 'neither'
    """
    def __init__(self, cmap="viridis", norm=None, colors=None, nan_color="White", bins=None, labels=None, extend=None,
                 values=None, labels_overwritten=False):
        if isinstance(cmap, type(None)):
            self._cmap = plt.cm.get_cmap("viridis")
        elif isinstance(cmap, str):
            self._cmap = plt.cm.get_cmap(cmap)
        elif isinstance(cmap, Colormap):
            self._cmap = cmap
        else:
            raise TypeError("cmap not recognized")

        self._cmap.set_bad(nan_color)
        self._nan_color = nan_color
        self._colors = colors
        self._bins = bins
        self._labels = labels
        self._extend = extend
        self._labels_overwritten = labels_overwritten

        # If values are not given to this function, it will raise an error when it tries to obtain them
        if not isinstance(values, type(None)):
            self._values = values

        # If norm not given to this function, it will raise an error when it tries to obtain them
        if not isinstance(norm, type(None)):
            self._norm = norm

    @property
    def cmap(self):
        if hasattr(self, "_cmap"):
            return self._cmap
        else:
            raise AttributeError("cmap not found in the PlotParams object")

    @property
    def norm(self):
        if hasattr(self, "_norm"):
            return self._norm
        else:
            raise AttributeError("norm not found in the PlotParams object")

    @property
    def colors(self):
        if hasattr(self, "_colors"):
            return self._colors
        else:
            return None

    @property
    def nan_color(self):
        if hasattr(self, "_nan_color"):
            return self._nan_color
        else:
            return "White"

    @property
    def bins(self):
        if hasattr(self, "_bins"):
            return self._bins
        else:
            return None

    @property
    def labels(self):
        if hasattr(self, "_labels"):
            return self._labels
        else:
            return None

    @property
    def labels_overwritten(self):
        if hasattr(self, "_labels_overwritten"):
            return self._labels_overwritten
        else:
            return False

    @property
    def extend(self):
        if hasattr(self, "_extend"):
            if isinstance(self._extend, type(None)):
                return 'neither'
            else:
                return self._extend
        else:
            return "neither"

    @property
    def values(self):
        if hasattr(self, "_values"):
            return self._values
        else:
            raise AttributeError("Values have not been set")


class ScalarPlotParams(PlotParams):
    """
    Plot parameters for a scalar data

    Parameters
    ----------
    values : array-like
        The data
    bins : array-like, optional
        Iterable containing bins that are use to aggregate the data. If not provided the data will be drawn linearly
    labels : array-like, optional
        If set it will overwrite the labels on the Legend
    vmin, vmax : numeric, optional
        Cutoff values if no `bins` are provided.
    cmap : matplotlib colormap, optional
        Matplotlib colormap to color the data, or a string that represents one.
    nan_color : matplotlib color, optional
        Color for NaN values, the default is white
    clip_legend : bool, optional
        If bins are provided that are outside the range of data they will be clipped. The default is False. An Error
        wil be raised if all bins are clipped.
    """
    def __init__(self, values, bins=None, labels=None, vmin=None, vmax=None, cmap=None, nan_color='White',
                 clip_legend=False):
        super().__init__(cmap=cmap)
        cmap = self._cmap

        values = np.array(values)
        if np.issubdtype(values.dtype, np.floating):
            data = values[~np.isnan(values)]
        else:
            data = values.flatten()

        minimum = data.min()
        maximum = data.max()

        if isinstance(bins, type(None)):
            if isinstance(vmin, type(None)):
                vmin = minimum
            if isinstance(vmax, type(None)):
                vmax = maximum

            if minimum < vmin and maximum > vmax:
                extend = 'both'
            elif minimum < vmin and not maximum > vmax:
                extend = 'min'
            elif not minimum < vmin and maximum > vmax:
                extend = 'max'
            elif not minimum < vmin and not maximum > vmax:
                extend = 'neither'

            self._norm = Normalize(vmin, vmax)
            self._extend = extend

        else:
            bins = np.unique(np.sort(np.array(bins)))

            boundaries = bins.copy()

            if clip_legend:
                bins = bins[np.logical_and(bins >= minimum, bins <= maximum)]
                if bins.size < 2:
                    raise IndexError("Clip_legend has removed all bins. Please set it to False or provide bins in the "
                                     "data range")

            if minimum < bins[0]:
                boundaries = np.hstack([minimum, boundaries])
                extend_min = True
                self._labels = [f"< {bins[0]}"]
            else:
                extend_min = False
                self._labels = [f"{bins[0]} - {bins[1]}"]

            self._labels = self._labels + [f"{bins[i - 1]} - {bins[i]}" for i in range(1 + (not extend_min), len(bins))]

            if maximum > bins[-1]:
                boundaries = np.hstack([boundaries, maximum])
                extend_max = True
                self._labels = self._labels + [f"> {bins[-1]}"]
            else:
                extend_max = False

            if extend_min and extend_max:
                extend = "both"
            elif not extend_min and not extend_max:
                extend = "neither"
            elif not extend_min and extend_max:
                extend = "max"
            elif extend_min and not extend_max:
                extend = "min"

            colors = cmap(np.linspace(0, 1, boundaries.size - 1))
            cmap = ListedColormap(colors)

            end = -1 if extend_max else None
            self._cmap = ListedColormap(colors[int(extend_min):end, :])
            self._cmap.set_under(cmap(0))
            self._cmap.set_over(cmap(cmap.N))

            self._colors = colors
            self._extend = extend
            self._bins = bins
            self._norm = BoundaryNorm(bins, len(bins) - 1)

        if not isinstance(labels, type(None)):
            self._labels_overwritten = True
            self._labels = labels
        else:
            self._labels_overwritten = False

        self._cmap.set_bad(nan_color)
        self._nan_color = nan_color
        self._values = values


class ClassifiedPlotParams(PlotParams):
    """
    Plot parameters for a data with classes

    Parameters
    ----------
    values : array-like
        The data
    bins : array-like, optional
        Iterable containing the classes. If not provided all unique values will be used.
    labels : array-like, optional
        If set it will overwrite the labels on the Legend
    cmap : matplotlib colormap, optional
        Matplotlib colormap to color the data, or a string that represents one.
    nan_color : matplotlib color, optional
        Color for NaN values, the default is white
    clip_legend : bool, optional
        If bins are provided that are outside the range of data they will be clipped. The default is False. An Error
        wil be raised if all bins are clipped.
    suppress_warnings : bool, optional
        Only 9 bins will be drawn by default. Set this parameter to True to avoid this limitation
    """
    def __init__(self, values, bins=None, labels=None, colors=None, cmap=None, nan_color="White", clip_legend=False,
                 suppress_warnings=False):
        data = values[~np.isnan(values)]

        if isinstance(bins, type(None)):
            bins = np.unique(data)
        else:
            bins = np.unique(np.sort(np.array(bins)))

        if len(bins) > 9 and not suppress_warnings:
            raise ValueError("Number of bins above 9, this creates issues with visibility. This error can be suppressed"
                             "by setting suppress_warnings to True. Be aware that the default colormap will cause "
                             "issues")

        if not isinstance(colors, type(None)):
            if len(bins) != len(colors):
                raise IndexError(f"length of bins and colors don't match\nbins: {len(bins)}\ncolors: {len(colors)}")
            colors = np.array(colors)
        else:
            if isinstance(cmap, type(None)):
                if len(bins) > 9:
                    raise IndexError("More than 9 classes are present, in which case cmap or colors needs to be set "
                                     "explicitly")
                colors = cmap_discrete(cmap="Set1", n=9, return_type="list")[:len(bins)]
            else:
                colors = cmap_discrete(cmap=cmap, n=len(bins), return_type='list')

        if not isinstance(labels, type(None)):
            if len(bins) != len(labels):
                raise IndexError("length of bins and labels don't match")
            labels = np.array(labels)
        else:
            labels = bins

        classes = np.unique(data)
        if not np.all(np.isin(classes, bins)):
            raise ValueError("Not all classes in the data are represented in the bins")

        if clip_legend:
            present = np.isin(bins, classes)
            bins = bins[present]
            labels = labels[present]
            colors = colors[present]

        boundaries = np.hstack((bins[0]-1, [(bins[i]+bins[i-1])/2 for i in range(1, len(bins))], bins[-1]+1))
        self._norm = BoundaryNorm(boundaries, len(bins))
        self._cmap = ListedColormap(colors)
        self._nan_color = nan_color
        self._cmap.set_bad(nan_color)
        self._colors = colors
        self._labels = labels
        self._labels_overwritten = True
        self._bins = bins
        self._values = values


# LEGEND OBJECTS (both colorbar and legend_patches)
class Legend:
    """Parent object for the legends. It contains a method `create` to create a Colorbar or LegendPatches object.
    A custom Legend object can be created, it only needs to expose a `draw` method with parameters `ax` and `params`"""
    @staticmethod
    def create(legend, aspect=30, pad_fraction=0.6, legend_ax=None, fontsize=None,  legend_kwargs=None):
        """
        Creates a Legend object of type Colorbar or LegendPatches

        Parameters
        ----------
        legend : {'colorbar', 'legend'} or Legend or None
            Switch for the creation of a legend. `colorbar` will create a ``Colorbar`` object while `legend` will
            create a ``LegendPatches`` object. A Legend object can be passed in directly, in which case nothing happens
        aspect : numeric, optional
            Aspect of ``Colorbar``
        pad_fraction : numeric, optional
            Pad fraction of ``Colorbar`
        legend_ax : matplotlib.Axes, otional
            Axes the legend will be drawn on. If not provided it will be drawn on the current Axes
        fontsize : numeric, optional
            Fontsize for the legend
        legend_kwargs : dict
            Keyword arguments for ``Colobar`` or ``LegendPatches`

        Returns
        -------
`       Legend (Colorbar/LegendPatches)
        """
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {}
        fontsize = legend_kwargs.pop('fontsize', fontsize)
        if legend == 'colorbar':
            return Colorbar(aspect=aspect, pad_fraction=pad_fraction, legend_ax=legend_ax, fontsize=fontsize,
                            **legend_kwargs)
        elif legend == 'legend':
            return LegendPatches(legend_ax=legend_ax, fontsize=fontsize, **legend_kwargs)
        else:
            return legend

    def draw(self, ax=None, params=None):
        raise NotImplementedError("Legend object does not have a `draw` method")


class Colorbar(Legend):
    """
    Legend object that represents a Colorbar

    Parameters
    ----------
    aspect : float, optional
        The aspect ratio of the colorbar
    pad_fraction : float, optional
        The fraction of the height of the colorbar that the colorbar is removed from the image
    legend_ax : matplotlib.Axes, otional
        Axes the legend will be drawn on. If not provided it will be drawn on the current Axes
    fontsize : numeric, optional
        Fontsize for the legend
    kwargs : dict
        Keyword arguments for ``cbar_decorater function``
    """
    def __init__(self, aspect=30, pad_fraction=0.6, legend_ax=None, fontsize=None, **kwargs):
        self._aspect = aspect
        self._pad_fraction = pad_fraction
        self._legend_ax = legend_ax
        self._fontsize = kwargs.pop('fontsize', fontsize)
        self._kwargs = kwargs

    @staticmethod
    def cbar_decorator(cbar, ticks=None, ticklabels=None, title="", label="", tick_params=None, title_font=None,
                       label_font=None, fontsize=None):
        if not isinstance(ticks, type(None)):
            cbar.set_ticks(ticks)
            if not isinstance(ticklabels, type(None)):
                if len(ticklabels) != len(ticks):
                    raise IndexError("Length of ticks and ticklabels do not match")
                cbar.set_ticklabels(ticklabels)

        if isinstance(tick_params, type(None)):
            tick_params = {}
        if isinstance(title_font, type(None)):
            title_font = {}
        if isinstance(label_font, type(None)):
            label_font = {}

        if 'labelsize' not in tick_params:
            tick_params['labelsize'] = fontsize
        if 'fontsize' not in title_font:
            title_font['fontsize'] = fontsize
        if 'fontsize' not in label_font:
            label_font['fontsize'] = fontsize

        cbar.ax.set_title(title, **title_font)
        cbar.ax.tick_params(**tick_params)
        cbar.set_label(label, **label_font)

    def draw(self, ax, params):
        sm = ScalarMappable(norm=params.norm, cmap=params.cmap)
        cbar = add_colorbar(im=sm, ax=ax, cax=self._legend_ax, aspect=self._aspect, pad_fraction=self._pad_fraction,
                            extend=self._kwargs.pop('extend', params.extend), shrink=self._kwargs.pop('shrink', 1),
                            position=self._kwargs.pop("position", "right"))

        if not isinstance(params.bins, type(None)):
            if len(params.bins) == params.cmap.N and isinstance(params, ClassifiedPlotParams):
                boundaries = cbar._boundaries
                tick_locations = [(boundaries[i] + boundaries[i - 1]) / 2
                                  for i in range(1, len(boundaries))]
            elif isinstance(params.norm, BoundaryNorm):
                tick_locations = params.bins
        else:
            tick_locations = None

        if params._labels_overwritten:
            labels = params.labels
        else:
            labels = None

        self.cbar_decorator(cbar, ticks=tick_locations, ticklabels=labels, fontsize=self._fontsize, **self._kwargs)

        return cbar

    def example(self, ax=None, params=None, cmap="Greys", norm=None, bins=None, bin_labels=None, extend='neither',
                figsize=(5, 5)):
        """Create an example plot with Colorbar"""
        if isinstance(params, type(None)):
            if isinstance(norm, type(None)):
                norm = NoNorm()
            params = PlotParams(cmap=cmap, norm=norm, bins=bins, labels=bin_labels, extend=extend)

        if isinstance(ax, type(None)):
            f, ax = plt.subplots(figsize=figsize)

        self.draw(ax=ax, params=params)
        if not isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()


class LegendPatches(Legend):
    """
    Legend object that represents a legend with patches

    Parameters
    ----------
    legend_ax : matplotlib.Axes, otional
        Axes the legend will be drawn on. If not provided it will be drawn on the current Axes
    fontsize : numeric, optional
        Fontsize for the legend
    facecolor : matplotlib color, optional
        Color of the legend box, default is white
    edgecolor : matplotlib color, optional
        Color of the edge of the legend box, default is lightgrey
    patch_type : str, optional
        Parameter defining the legend patches. "Patch" will create legend patches, while any other value will be
        interpreted as linestyle.
    patch_edgecolor : matplotlib color, optional
        Color of the edge of the legend patches, default is lightgrey
    title : str, optional
        Title of the legend box
    align_left : bool, optional
        Align the title of the legend box to the left. The default is True
    handles_kwargs : dict, optional
        # todo; move this function here
        Keyword arguments for the `legend_patches` function
    kwargs
        Keyword arguments for the `ax.legend` function call.
    """
    def __init__(self, legend_ax=None, facecolor='white', edgecolor='lightgrey', fontsize=None, patch_type="patch",
                 patch_edgecolor="lightgrey", title=None, align_left=True, handles_kwargs=None, **kwargs):
        self._facecolor = facecolor
        self._edgecolor = edgecolor
        self._patch_edgecolor = patch_edgecolor
        self._patch_type = patch_type
        self._title = title
        self._align_left = align_left
        self._legend_ax = legend_ax

        self._fontsize = kwargs.pop('fontsize', fontsize)
        self._kwargs = kwargs

        if isinstance(handles_kwargs, type(None)):
            self._handles_kwargs = {}
        else:
            self._handles_kwargs = handles_kwargs

    def draw(self, ax, params):
        if not isinstance(self._legend_ax, type(None)):
            ax = self._legend_ax

        if not isinstance(params.colors, type(None)):
            handles = legend_patches(colors=params.colors, labels=params.labels, type=self._patch_type,
                                     edgecolor=self._patch_edgecolor, **self._handles_kwargs)
            leg = ax.legend(handles=handles, title=self._title, title_fontsize=self._fontsize, fontsize=self._fontsize,
                            **self._kwargs)
            if self._align_left:
                leg._legend_box.align = "left"
        else:
            leg = None

        return leg

    def example(self, ax=None, params=None, figsize=(5, 5), colors=None, labels=None, **kwargs):
        """Create an example plot with LegendPatches"""
        if isinstance(ax, type(None)):
            f, ax = plt.subplots(figsize=figsize)

        if isinstance(params, type(None)):
            params = PlotParams()
            if isinstance(colors, type(None)):
                params._colors = cmap_discrete("Greys", n=5, return_type="list")
            if isinstance(labels, type(None)):
                params._labels = [1, 2, 3, 4, 5]

        self.draw(ax=ax, params=params, **kwargs)
        if not isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()


# Plotting functions
def plot_map(m, bins=None, bin_labels=None, cmap=None, vmin=None, vmax=None, legend="colorbar", clip_legend=False,
             ax=None, figsize=None, legend_ax=None, legend_kwargs=None, fontsize=None, aspect=30,
             pad_fraction=0.6, force_equal_figsize=False, nan_color="White", **kwargs):
    """
    Plot a scalar raster

    Parameters
    ----------
    m : array-like
        Input array. Needs to be either 2D or 3D if the third axis contains RGB(A) information
    bins : array-like, optional
        List of bins that will be used to create a BoundaryNorm instance to discretise the plotting. This does not work
        in conjunction with vmin and vmax. Bins in that case will take the upper hand.
    bin_labels : array-like, optional
        This parameter can be used to override the labels on the colorbar. Should have the same length as bins.
    cmap : matplotlib.cmap or str, optional
        Matplotlib cmap instance or string the will be recognized by matplotlib
    vmin, vmax : float, optional
        vmin and vmax parameters for plt.imshow(). This does have no effect in conjunction with bins being provided.
    legend : {'colorbar', 'legend', Legend, None}, optional
        Legend type that will be plotted. The 'legend' type will only work if bins are specified.
    clip_legend : bool, optional
        Clip bins that are do not fit in the data. If False the colormap will remain intact over the whole provided
        bins, which potentially lowers contrast a lot, but works great for having a common colorbar for several plots.
    ax : `matplotlib.Axes`, optional
        Axes object. If not provided it will be created on the fly.
    figsize : tuple, optional
        Matplotlib figsize parameter.
    legend_ax : `matplotlib.Axes`, optional
        Axes object that the legend will be drawn on
    legend_kwargs : dict, optional
        Extra parameters to create and decorate the colorbar or the call to `plt.legend` if `legend` == "legend"
        For the colorbar creation: shrink, position and extend (which would override the internal behaviour)
        For the colorbar decorator see `cbar_decorate`.
    fontsize : float, optional
        Fontsize of the legend
    aspect : float, optional
        aspect ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    force_equal_figsize : bool, optional
        when plotting with a colorbar the figure is going be slightly smaller than when you are using `legend` or non
        at all. This parameter can be used to force equal sizes, meaning that the version with a `legend` is going to
        be slightly reduced. This depdends on equal values for `aspect` and `pad_fraction` being provided.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'White'
    **kwargs
        Keyword arguments for plt.imshow()

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the `extent` of the data should be provided
    if there is not a perfect overlap. If provided to this parameter it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different than the data projection.

    Returns
    -------
    (Axes or GeoAxes, legend)
    legend depends on the `legend` parameter
    """
    if m.ndim not in (2, 3):
        raise ValueError("Input data needs to be 2D or present RGB(A) values on the third axis.")
    if m.ndim == 3 and (m.shape[-1] not in (3, 4) or np.issubdtype(m.dtype, np.bool_)):
        raise ValueError(f"3D arrays are only acceptable if presenting RGB(A) information. It does not work with "
                         f"boolean variables.\nShape: {m.shape} \ndtype: {m.dtype}")

    if not isinstance(bins, type(None)) and len(bins) == 1:
        # If only one bin is present the data will be converted to a boolean array
        nan_mask = np.isnan(m)
        m = (m > bins[0]).astype(float)
        m[nan_mask] = np.nan
        plot_params = ClassifiedPlotParams(values=m, bins=[0, 1], labels=["False", "True"],
                                           colors=['Lightgrey', 'Red'], nan_color=nan_color)

    elif not np.issubdtype(m.dtype, np.bool_) and m.ndim == 2:
        # 2D Array with values that can be adapted using bins or vmin/vmax
        plot_params = ScalarPlotParams(values=m, bins=bins, labels=bin_labels, vmin=vmin, vmax=vmax, cmap=cmap,
                                       nan_color=nan_color, clip_legend=clip_legend)

    elif m.ndim == 3:
        # Case of RGB(A) values on the third axis
        plot_params = PlotParams(cmap=cmap, values=m, norm=NoNorm())
        legend = None

    else:
        # Case of boolean raster
        plot_params = ClassifiedPlotParams(values=m, bins=[0, 1], labels=["False", "True"], colors=['Lightgrey', 'Red'])

    legend = Legend.create(legend, aspect=aspect, pad_fraction=pad_fraction, legend_ax=legend_ax, fontsize=fontsize,
                           legend_kwargs=legend_kwargs)

    ax, legend = PlotRaster(ax=ax, figsize=figsize, fontsize=fontsize, params=plot_params, legend=legend, **kwargs).draw()

    if force_equal_figsize and legend != 'colorbar':
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {}
        create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction,
                             position=legend_kwargs.get("position", "right")).axis("off")

    return ax, legend


def plot_classified_map(m, bins=None, colors=None, cmap=None, labels=None, legend="legend", clip_legend=False,
                        ax=None, figsize=None, suppress_warnings=False, legend_ax=None, legend_kwargs=None,
                        fontsize=None, aspect=30, pad_fraction=0.6, force_equal_figsize=False, nan_color="White",
                        **kwargs):
    """
    Plot a classified raster

    Parameters
    ----------
    m : array
        Raster
    bins : list, optional
        List of bins as used in np.digitize . By default this parameter is not necessary, the unique values are
        taken from the input data
    colors : list, optional
        List of colors in a format understandable by matplotlib. By default colors will be obtained from `cmap`
    cmap : matplotlib cmap or str, optional
        Can be used to set a colormap when no colors are provided. `Set1` is the default.
    labels : list, optional
        list of labels for the different classes. By default the unique values are taken as labels
    legend : {'legend', 'colorbar', Legend, None}, optional
        Presence and type of legend. 'Legend' wil insert patches, 'colorbar' wil insert a colorbar and None will
        prevent any legend to be printed.
    clip_legend : bool, optional
        remove the items from the legend that don't occur on the map but are passed in
    ax : axes, optional
        matplotlib axes to plot the map on. If not given it is created on the fly. A cartopty GeoAxis can be provided.
    figsize : tuple, optional
        Matplotlib figsize parameter.
    suppress_warnings : bool, optional
        By default 9 classes is the maximum that can be plotted. If set to True this maximum is removed.
    legend_ax : `matplotlib.Axes`, optional
        Axes object that the legend will be drawn on
    legend_kwargs : dict, optional
        Extra parameters to create and decorate the colorbar or the call to `plt.legend` if `legend` == "legend"
        For the colorbar creation: shrink, position and extend (which would override the internal behaviour)
        For the colorbar decorator see `cbar_decorate`.
    fontsize : float, optional
        Fontsize of the legend
    aspect : float, optional
        aspact ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    force_equal_figsize : bool, optional
        when plotting with a colorbar the figure is going be slightly smaller than when you are using `legend` or non
        at all. This parameter can be used to force equal sizes, meaning that the version with a `legend` is going to
        be slightly reduced. This depdends on equal values for `aspect` and `pad_fraction` being provided.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'white'
    **kwargs : dict, optional
        kwargs for the plt.imshow command

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Returns
    -------
    (Axes or GeoAxes, legend)
    legend depends on the `legend` parameter
    """
    plot_params = ClassifiedPlotParams(values=m, bins=bins, labels=labels, colors=colors, cmap=cmap,
                                       nan_color=nan_color, clip_legend=clip_legend, suppress_warnings=suppress_warnings)

    legend = Legend.create(legend, aspect=aspect, pad_fraction=pad_fraction, legend_ax=legend_ax, fontsize=fontsize,
                           legend_kwargs=legend_kwargs)

    ax, legend = PlotRaster(ax=ax, figsize=figsize, fontsize=fontsize, params=plot_params, legend=legend, **kwargs)\
        .draw()

    if force_equal_figsize and legend != 'colorbar':
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {}
        create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction,
                             position=legend_kwargs.get("position", "right")).axis("off")

    return ax, legend


def plot_shapes(lat=None, lon=None, values=None, s=None, df=None, bins=None, bin_labels=None, cmap=None, vmin=None,
                vmax=None, legend="colorbar", clip_legend=False, ax=None, figsize=None, legend_ax=None,
                legend_kwargs=None, fontsize=None, aspect=30, pad_fraction=0.6, linewidth=1, force_equal_figsize=None,
                nan_color="White", **kwargs):
    """
    Plot shapes in a continuous fashion

    Parameters
    ----------
    lat, lon : array-like
        Latitude and Longitude
    values : array-like or numeric or str
        Values at each pair of latitude and longitude entries if list like. A single numeric value will be cast to all
        geometries. If `df` is set, a string can be passed to values which will be interpreted as the name of the column
        holding the values.
    s : array-like, optional
        Size values for each pair of latitude and longitude entries if list like. A single numeric value will be cast
        to all geometries. If `df` is set a string will be interpreted as the name of the column holding the sizes. If
        None is set no sizes will be set.
    df : GeoDataFrame, optional
        Optional GeoDataframe which can be used to plot different shapes than points. A geometry column is expected
    bins : array-like, optional
        List of bins that will be used to create a BoundaryNorm instance to discretise the plotting. This does not work
        in conjunction with vmin and vmax. Bins in that case will take the upper hand.
    bin_labels : array-like, optional
        This parameter can be used to override the labels on the colorbar. Should have the same length as bins.
    cmap : matplotlib.cmap or str, optional
        Matplotlib cmap instance or string that will be recognized by matplotlib
    vmin, vmax : float, optional
        vmin and vmax parameters for plt.imshow(). This does have no effect in conjunction with bins being provided.
    legend : {'colorbar', 'legend', False}, optional
        Legend type that will be plotted. The 'legend' type will only work if bins are specified.
    clip_legend : bool, optional
        Clip the legend to the minimum and maximum of bins are provided. If False the colormap will remain intact over
        the whole provided bins, which potentially lowers contrast a lot.
    ax : matplotlib.Axes, optional
        Axes object. If not provided it will be created on the fly.
    figsize : tuple, optional
        Matplotlib figsize parameter. Default is (10,10)
    legend_ax : `matplotlib.Axes`, optional
        Axes object that the legend will be drawn on
    legend_kwargs : dict, optional
        Extra parameters to create and decorate the colorbar or the call to `plt.legend` if `legend` == "legend"
        For the colorbar creation: shrink, position and extend (which would override the internal behaviour)
        For the colorbar decorator see `cbar_decorate`.
    fontsize : float, optional
        Fontsize of the legend
    aspect : float, optional
        aspact ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    linewidth : numeric, optional
        width of the line around the shapes
    force_equal_figsize : bool, optional
        when plotting with a colorbar the figure is going be slightly smaller than when you are using `legend` or non
        at all. This parameter can be used to force equal sizes, meaning that the version with a `legend` is going to
        be slightly reduced. This depdends on equal values for `aspect` and `pad_fraction` being provided.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'white'
    **kwargs
        kwargs for the _plot_geometries function

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Setting either ``facecolor`` or ``edgecolor`` does the same as in geopandas. It overwrite the behaviour of this
    function.

    Returns
    -------
    (Axes or GeoAxes, legend)
    legend depends on the `legend` parameter
    """
    if isinstance(values, type(None)):
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = "lightgrey"
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = "black"
        legend = None

    geometry, values, markersize = PlotShapes._create_geometry_values_and_sizes(lat, lon, values, s, df)

    if not isinstance(bins, type(None)) and len(bins) == 1:
        # If only one bin is present the data will be converted to a boolean array
        nan_mask = np.isnan(values)
        values = (values > bins[0]).astype(float)
        values[nan_mask] = np.nan
        plot_params = ClassifiedPlotParams(values=values, bins=[0, 1], labels=["False", "True"],
                                           colors=['Lightgrey', 'Red'], nan_color=nan_color)

    elif not np.issubdtype(values.dtype, np.bool_):
        # 1D Array with values that can be adapted using bins or vmin/vmax
        plot_params = ScalarPlotParams(values=values, bins=bins, labels=bin_labels, vmin=vmin, vmax=vmax, cmap=cmap,
                                       nan_color=nan_color, clip_legend=clip_legend)

    else:
        # Case of boolean raster
        plot_params = ClassifiedPlotParams(values=values, bins=[0, 1], labels=['False', 'True'],
                                           colors=['Lightgrey', 'Red'])

    legend = Legend.create(legend, aspect=aspect, pad_fraction=pad_fraction, legend_ax=legend_ax, fontsize=fontsize,
                           legend_kwargs=legend_kwargs)

    ax, legend = PlotShapes(ax=ax, figsize=figsize, fontsize=fontsize, params=plot_params, legend=legend, **kwargs)\
        .draw(geometry=geometry, markersize=markersize, linewidth=linewidth)

    if force_equal_figsize and legend != 'colorbar':
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {}
        create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction,
                             position=legend_kwargs.get("position", "right")).axis("off")

    return ax, legend


def plot_classified_shapes(lat=None, lon=None, values=None, s=None, df=None, bins=None, colors=None, cmap="Set1",
                           labels=None, legend="legend", clip_legend=False, ax=None, figsize=None,
                           suppress_warnings=False, legend_ax=None, legend_kwargs=None, fontsize=None, aspect=30,
                           pad_fraction=0.6, linewidth=1, force_equal_figsize=False, nan_color="White", **kwargs):
    """
    Plot shapes with discrete classes or index

    Parameters
    ----------
    lat, lon : array-like
        Latitude and Longitude
    values : array-like or numeric or str
        Values at each pair of latitude and longitude entries if list like. A single numeric value will be cast to all
        geometries. If `df` is set a string can be passed to values which will be interpreted as the name of the column
        holding the values.
    s : array-like, optional
        Size values for each pair of latitude and longitude entries if list like. A single numeric value will be cast
        to all geometries. If `df` is set a string will be interpreted as the name of the column holding the sizes. If
        None is set no sizes will be set.
    df : GeoDataFrame, optional
        Optional GeoDataframe which can be used to plot different shapes than points.
    bins : list, optional
        list of either bins as used in np.digitize or unique values corresponding to `colors` and `labels`. By default
        this parameter is not necessary, the unique values are taken from the input map
    colors : list, optional
        List of colors in a format understandable by matplotlib. By default color will be taken from cmap
    cmap : matplotlib cmap or str
        Can be used to set a colormap when no colors are provided. The default is 'Set1'
    labels : list, optional
        list of labels for the different classes. By default the unique values are taken as labels
    legend : {'legend', 'colorbar', Legend, None}, optional
        Presence and type of legend. 'Legend' wil insert patches, 'colorbar' wil insert a colorbar and None will
        prevent any legend to be printed.
    clip_legend : bool, optional
        remove the items from the legend that don't occur on the map but are passed in
    ax : axes, optional
        matplotlib axes to plot the map on. If not given it is created on the fly. A cartopty GeoAxis can be provided.
    figsize : tuple, optional
        Matplotlib figsize parameter.
    suppress_warnings : bool, optional
        By default 9 classes is the maximum that can be plotted. If set to True this maximum is removed
    legend_ax : `matplotlib.Axes`, optional
        Axes object that the legend will be drawn on
    legend_kwargs : dict, optional
        Extra parameters to create and decorate the colorbar or the call to `plt.legend` if `legend` == "legend"
        For the colorbar creation: shrink, position and extend (which would override the internal behaviour)
        For the colorbar decorator see `cbar_decorate`.
    fontsize : float, optional
        Fontsize of the legend
    aspect : float, optional
        aspact ratio of the colorbar
    pad_fraction : float, optional
        pad_fraction between the Axes and the colorbar if generated
    linewidth : numeric, optional
        width of the line around the shapes
    force_equal_figsize : bool, optional
        when plotting with a colorbar the figure is going be slightly smaller than when you are using `legend` or non
        at all. This parameter can be used to force equal sizes, meaning that the version with a `legend` is going to
        be slightly reduced. This depdends on equal values for `aspect` and `pad_fraction` being provided.
    nan_color : matplotlib color, optional
        Color used for shapes with NaN value. The default is 'white'
    **kwargs : dict, optional
        kwargs for the `PlotShapes._plot_geometries` function

    Notes
    -----
    When providing a GeoAxes in the 'ax' parameter it needs to be noted that the 'extent' of the data should be provided
    if there is not a perfect overlap. If provided to this function it will be handled by **kwargs. The same goes for
    'transform' if the plotting projection is different from the data projection.

    Setting either ``facecolor`` or ``edgecolor`` does the same as in geopandas. It overwrite the behaviour of this
    function.

    Returns
    -------
    (Axes or GeoAxes, legend)
    legend depends on the `legend` parameter
    """
    if isinstance(values, type(None)):
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = "lightgrey"
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = "black"
        legend = None

    geometry, values, markersize = PlotShapes._create_geometry_values_and_sizes(lat, lon, values, s, df)

    plot_params = ClassifiedPlotParams(values=values, bins=bins, labels=labels, colors=colors, cmap=cmap,
                                       nan_color=nan_color, clip_legend=clip_legend,
                                       suppress_warnings=suppress_warnings)

    legend = Legend.create(legend, aspect=aspect, pad_fraction=pad_fraction, legend_ax=legend_ax, fontsize=fontsize,
                           legend_kwargs=legend_kwargs)

    ax, legend = PlotShapes(ax=ax, figsize=figsize, fontsize=fontsize, params=plot_params, legend=legend, **kwargs) \
        .draw(geometry=geometry, markersize=markersize, linewidth=linewidth)

    if force_equal_figsize and legend != 'colorbar':
        if isinstance(legend_kwargs, type(None)):
            legend_kwargs = {}
        create_colorbar_axes(ax=ax, aspect=aspect, pad_fraction=pad_fraction,
                             position=legend_kwargs.get("position", "right")).axis("off")

    return ax, legend

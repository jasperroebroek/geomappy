import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.patches import Patch
from .colors import legend_patches as lp
from shapely.geometry import Point


def cbar_decorator(cbar, ticks=None, ticklabels=None, title="", label="", tick_params=None, title_font=None,
                   label_font=None, fontsize=None):
    # todo; move this to .Plot
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper around rasterio functionality. Object creation happens by calling Map() which distributes the requests to
MapRead() and MapWrite() for reading and writing respectively (rasterio mode='r' and mode='w'). It is possible for both
reading and writing to do this chunk by chunk in tiles for files that are bigger than the installed RAM. A window_size
parameter can be used for the calculation of focal statistics. In reading this will add a buffer of data around the
actual tile to be able to calculate values on the border. By setting the same window_size in a MapWrite object trims
this border without values before writing it to the file as if everything happened in one calculation.
"""

from map.MapBase import MapBase
from map.MapWrite import MapWrite
from map.MapRead import MapRead
from map.Map import Map


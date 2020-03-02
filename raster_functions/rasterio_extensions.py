#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import rasterio as rio
from rasterio.warp import reproject, Resampling
import copy


def resample_profile(m, scale):
    """
    Creates a rasterio profile with the dimensions resampled by a factor

    Parameters
    ----------
    m : str, rasterio profile
        location of the map or rasterio profile
    scale : float
        factor with which height and width will be multiplied

    Returns
    -------
    rasterio profile
    """
    if type(m) == str:
        with rio.open(m) as ref_map:
            profile = ref_map.profile
            transform = ref_map.transform
    elif type(m) == rio.profiles.Profile:
        m = copy.deepcopy(m)
        profile = m
        transform = m['transform']
    else:
        raise TypeError("M not understood")

    new_transform = rio.Affine(transform[0] / scale, transform[1], transform[2],
                               transform[3], transform[4] / scale, transform[5])

    profile.update({'transform': new_transform,
                    'width': int(profile['width'] * scale),
                    'height': int(profile['height'] * scale)})
    return profile


def reproject_map_like(input_map=None, ref_map=None, output_map=None, dtype=np.float64):
    """
    Reprojecting a map like a reference map.

    Parameters
    ----------
    input_map : str
        Path to the input map
    ref_map : str
        Path to the reference map where the profile is pulled from
    output_map : str
        Path to the location where the transformed map is written to.
    dtype : numpy.dtype, optional
        export dtype

    Raises
    ------
    IOError
        reference map or input map not found or output_map already exists
    """
    # todo; make it possible to write several bands instead of just one
    # todo; implement different resampling options
    # todo; make it accept Map
    if not os.path.isfile(ref_map):
        raise IOError("no reference map provided")
    if not os.path.isfile(input_map):
        raise IOError("no input map provided")
    if os.path.isfile(output_map):
        raise IOError("output file name already exists")

    with rio.open(ref_map) as ref_file:
        ref_transform = ref_file.transform
        ref_crs = ref_file.crs
        shape = ref_file.shape

    with rio.open(input_map) as src:
        current_transform = src.transform
        current_crs = src.crs

        if isinstance(current_crs, type(None)):
            print("input map does not have a CRS. Therefore the crs of the reference map is assumed")
            current_crs = ref_crs

        new_map = np.ndarray(shape, dtype=dtype)
        print("start reprojecting")
        reproject(src.read(1), new_map,
                  src_transform=current_transform,
                  src_crs=current_crs,
                  dst_transform=ref_transform,
                  dst_crs=ref_crs,
                  resampling=Resampling.bilinear)

    print("writing file")
    with rio.open(output_map, "w", driver="GTiff", dtype=str(new_map.dtype), height=shape[0], width=shape[1],
                  crs=ref_crs, count=1, transform=ref_transform) as dst:
        dst.write(new_map, 1)

    print("reprojection completed")


def empty_map_like(ref_map, output_map, dtype=np.float64):
    """
    This function creates an empty tif file based on a reference map where
    the function gets the shape, transform and crs parameters from.

    Parameters
    ----------
    ref_map : str
        Path to the map where the profile is pulled from
    output_map : str
        Path to the location where an empty array is written to
    dtype : numpy.dtype, optional
        type of data written to the file

    Raises
    ------
    IOError
        if reference map doesn't exist or the output_map does already exist
    """

    if not os.path.isfile(ref_map):
        raise IOError("no reference map provided")
    if os.path.isfile(output_map):
        raise IOError("output file name already exists")

    with rio.open(ref_map) as ref_file:
        ref_transform = ref_file.transform
        ref_crs = ref_file.crs
        shape = ref_file.shape

    new_map = np.full(shape, 0, dtype=dtype)

    # todo; implement different `count` options
    with rio.open(output_map, "w", driver="GTiff", dtype=str(new_map.dtype),
                  height=shape[0], width=shape[1], crs=ref_crs, count=1,
                  transform=ref_transform) as dst:
        dst.write(new_map, 1)


def export_map_like(m, ref_map=None, output_map=None):
    """
    Exporting map, taking the needed parameters from a reference_map

    Parameters
    ----------
    m : :obj:`~numpy.ndarray`
        2D input map
    ref_map : str
        Path of the reference map
        todo; make the direct insertion of a rasterio profile possible
    output_map : str
        Path where the data will be written to. If extension is given
        it will be added (.tif)

    Raises
    ------
    TypeError
        if no input data is given
    ValueError
        if data is not 2D
    IOError
        if output_map location already exists
    """
    if type(m) != np.ndarray:
        raise TypeError("Input data should be a ndarray")
    if m.ndim != 2:
        raise ValueError("map should be a 2D ndarray")

    # reference map
    with rio.open(ref_map) as src:
        ref_transform = src.transform
        ref_crs = src.crs
        shape = (src.height, src.width)

    # todo; create functionality to overwrite the file if needed
    if os.path.isfile(output_map):
        raise IOError("Output location already exists")

    # if file extension is not given or is not tif, it will be added
    if output_map[-4:] != ".tif":
        output_map = output_map.strip() + ".tif"

    # create the new file
    with rio.open(output_map,
                  "w", driver="GTiff", dtype=str(m.dtype), height=shape[0], width=shape[1],
                  crs=ref_crs, count=1, transform=ref_transform) as src:
        # todo; more than one layer
        src.write(m, 1)


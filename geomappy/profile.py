import copy
import rasterio as rio


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
    if isinstance(m, str):
        with rio.open(m) as ref_map:
            profile = ref_map.profile
    elif isinstance(m, rio.profiles.Profile):
        profile = copy.deepcopy(m)
    else:
        raise TypeError("M should be a filepath or rasterio profile")

    transform = profile['transform']
    new_transform = rio.Affine(transform[0] / scale, transform[1], transform[2],
                               transform[3], transform[4] / scale, transform[5])

    profile.update({'transform': new_transform,
                    'width': int(profile['width'] * scale),
                    'height': int(profile['height'] * scale)})
    return profile

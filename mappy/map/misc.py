from pyproj import Proj, transform

def bounds_to_platecarree(proj, bounds):
    p_lat_lon = Proj(init="epsg:4326", preserve_units=False)
    return (*transform(proj, p_lat_lon, bounds[0], bounds[1]), *transform(proj, p_lat_lon, bounds[2], bounds[3]))

def bounds_to_data_projection(proj, bounds):
    p_lat_lon = Proj(init="epsg:4326", preserve_units=False)
    return (*transform(p_lat_lon, proj, bounds[0], bounds[1]), *transform(p_lat_lon, proj, bounds[2], bounds[3]))

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='geomappy',
    version='0.0.1',
    packages=['geomappy'],
    url='',
    license='GNU GPLv3',
    author='jasperroebroek',
    author_email='roebroek.jasper@gmail.com',
    description='Plot maps on a basemap, with data in raster and polygon formats',
    install_requires=["cartopy>=0.18", "rasterio", "geopandas", "descartes", "packaging"],
    extra_require={"develop": ["numba", "cython"]},
    ext_modules=cythonize("geomappy/focal_statistics/c_focal_correlation.pyx"),
    include_dirs=[numpy.get_include()]
)

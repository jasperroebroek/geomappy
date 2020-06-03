from setuptools import setup

setup(
    name='geomappy',
    version='0.0.1',
    packages=['geomappy'],
    url='',
    license='GNU GPLv3',
    author='jasperroebroek',
    author_email='roebroek.jasper@gmail.com',
    description='Plot maps on a basemap, with data in raster and polygon formats',
    install_requires=["cartopy>=0.17", "rasterio", "geopandas"]
)

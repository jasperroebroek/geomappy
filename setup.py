from setuptools import setup, find_packages

setup(
    name='geomappy',
    version='0.0.2',
    packages=find_packages(),
    url='',
    license='GNU GPLv3',
    author='jasperroebroek',
    author_email='roebroek.jasper@gmail.com',
    description='Plot maps on a basemap',
    install_requires=["cartopy>=0.20", "geopandas", "rioxarray"],
    extras_require={
        'develop': ['sphinx', 'sphinx_rtd_theme', 'numpydoc', 'jupyter', 'pytest']
    }
)

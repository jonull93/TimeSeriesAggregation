# setup.py
from setuptools import setup, find_packages

setup(
    name='TimeSeriesAggregator',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    # Add other parameters as necessary
    author='Jonathan Ullmark',
    url=r'https://github.com/jonull93/TimeSeriesAggregator',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
    ],

)
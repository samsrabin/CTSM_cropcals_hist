# %% Setup

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Directory where input file(s) can be found
# indir = "/Volumes/Reacher/CESM_runs/numa_20211014_rx/normal/"
# generate_gdds = False
# indir = "/Volumes/Reacher/CESM_runs/numa_20211014_rx/"
indir = "/Volumes/Reacher/CESM_runs/f10_f10_mg37/"
# indir = "/Volumes/Reacher/CESM_runs/f10_f10_mg37/2021-11-10/"

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import glob
import cftime
import cartopy.crs as ccrs
from matplotlib import cm

import os

import sys
sys.path.append(my_ctsm_python_gallery)
import utils

import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")


# %% Import realized sowing and harvest dates

# Either the name of a file within $indir, or a pattern that will return a list of files.
pattern = "*h2.*-01-01-00000.nc"

# Get list of all files in $indir matching $pattern
filelist = glob.glob(indir + pattern)

# Import
this_ds = utils.import_ds(filelist, \
    myVars=["SDATES", "HDATES"], 
    myVegtypes=utils.define_mgdcrop_list())


# %% Compare sdates and hdates

sdates1 = utils.grid_one_variable(\
    this_ds, 
    "SDATES", 
    time=1)
hdates2 = utils.grid_one_variable(\
    this_ds, 
    "HDATES", 
    time=2)

print(sdates1.values)
print(this_ds.HDATES.values)
np.bitwise_or(this_ds.HDATES.values == sdates1-1, 
              np.bitwise_and(this_ds.HDATES.values==-1, 
                             sdates1==-1))






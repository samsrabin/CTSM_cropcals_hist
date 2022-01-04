# %% Setup

# Years of interest
y1 = 1980
yN = 2010

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Directory where input file(s) can be found
indir = "/Users/Shared/CESM_runs/f10_f10_mg37_1850/"

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


# %% Import sowing and harvest dates
dates_ds = utils.import_ds(glob.glob(indir + "*h2.*"), \
    myVars=["SDATES", "HDATES"], 
    myVegtypes=utils.define_mgdcrop_list())
y1_import_str = f"{y1+1}-01-01"
yN_import_str = f"{yN+1}-01-01"
print(f"Using netCDF time steps {y1_import_str} through {yN_import_str}")
dates_ds = utils.xr_flexsel(dates_ds, \
    time__values=slice(y1_import_str,
                       yN_import_str))


# %% Check that simulated sowing and harvest dates do not vary between years

for v in ["SDATES", "HDATES"]:
    for t in np.arange(dates_ds.dims["time"]):
        if t==0:
            continue
        if not np.all((dates_ds[v].isel(time=0) == dates_ds[v].isel(time=t)).values):
            raise RuntimeError(f"{v} timestep {t} does not match timestep 0")
print("✅ CLM output sowing and harvest dates do not vary between years.")


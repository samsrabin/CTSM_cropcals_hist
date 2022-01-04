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


# %% Import expected sowing dates

sdate_inFile = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"

# Get run info:
# Max number of growing seasons per year
if "mxgrowseas" in dates_ds:
    mxgrowseas = dates_ds.dims["mxgrowseas"]
else:
    mxgrowseas = 1
    
# Which vegetation types were simulated?
itype_veg_toImport = np.unique(dates_ds.patches1d_itype_veg)

sdate_varList = []
for i in itype_veg_toImport:
    for g in np.arange(mxgrowseas):
        thisVar = f"sdate{g+1}_{i}"
        sdate_varList = sdate_varList + [thisVar]

sdates_rx = utils.import_ds(sdate_inFile, myVars=sdate_varList)


# %% Check that input and output sdates match

sdates_grid = utils.grid_one_variable(\
    dates_ds, 
    "SDATES")

all_ok = True
for i, vt_str in enumerate(dates_ds.vegtype_str.values):
    
    # Input
    vt = dates_ds.ivt.values[i]
    thisVar = f"sdate1_{vt}"
    if thisVar not in sdates_rx:
        continue
    in_map = sdates_rx[thisVar].squeeze(drop=True)
    
    # Output
    out_map = sdates_grid.sel(ivt_str=vt_str).squeeze(drop=True)
    
    # Check for differences
    diff_map = out_map - in_map
    if np.any(diff_map.values[np.invert(np.isnan(diff_map.values))]):
        print(f"Difference(s) found in {vt_str}")
        all_ok = False
        
if all_ok:
    print("✅ Input and output sdates match!")


# %% Setup

# which_cases = "main2.2022"
which_cases = "main2"
# which_cases = "ctsm_lu_5.0_vs_5.2"
# which_cases = "originalCLM"
# which_cases = "originalBaseline" # As originalCLM, but without cmip6
# which_cases = "diagnose"

# Include irrigation?
incl_irrig = True

# Yields will be set to zero unless HUI at harvest is ≥ min_viable_hui
min_viable_hui = 1

# Include diagnostic info in figures?
diagnostics = True

# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import cropcal_module as cc
from cropcal_figs_module import *

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

import numpy as np
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
from matplotlib import cm
import datetime as dt
import re
import importlib
import pandas as pd
import cftime
from fig_global_timeseries import global_timeseries_yieldetc, global_timeseries_irrig_inclcrops, global_timeseries_irrig_allcrops
from fig_maps_allCrops import *
from fig_maps_eachCrop import maps_eachCrop

# Ignore these annoying warnings
import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")


# %% Define functions etc.

cropList_combined_clm_nototal = ["Corn", "Rice", "Cotton", "Soybean", "Sugarcane", "Wheat"]
cropList_combined_clm_nototal.sort()
cropList_combined_clm = cropList_combined_clm_nototal.copy()
cropList_combined_clm.append("Total")
# cropList_combined_faostat = ["Maize", "Rice, paddy", "Seed cotton", "Soybeans", "Sugar cane", "Wheat"]

fao_to_clm_dict = {"Maize": "Corn",
                   "Rice, paddy": "Rice",
                   "Seed cotton": "Cotton",
                   "Soybeans": "Soybean",
                   "Sugar cane": "Sugarcane",
                   "Wheat": "Wheat",
                   "Total": "Total"}

plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

# Add GRAINC variants with too-long seasons set to 0
# CLM max growing season length, mxmat, is stored in the following files:
#	 * clm5_1: lnd/clm2/paramdata/ctsm51_params.c211112.nc
#	 * clm5_0: lnd/clm2/paramdata/clm50_params.c211112.nc
#	 * clm4_5: lnd/clm2/paramdata/clm45_params.c211112.nc
paramfile_dir = "/Users/Shared/CESM_inputdata/lnd/clm2/paramdata/"
my_clm_ver = 50
my_clm_subver = "c211112"
pattern = os.path.join(paramfile_dir, f"*{my_clm_ver}_params.{my_clm_subver}.nc")
paramfile = glob.glob(pattern)
if len(paramfile) != 1:
    raise RuntimeError(f"Expected to find 1 match of {pattern}; found {len(paramfile)}")
paramfile_ds = xr.open_dataset(paramfile[0])
# Import max growing season length (stored in netCDF as nanoseconds!)
paramfile_mxmats = paramfile_ds["mxmat"].values / np.timedelta64(1, 'D')
# Import sowing windows start/end (stored in netCDF as MMDD)
paramfile_sow_start = paramfile_ds["min_NH_planting_date"].values
paramfile_sow_end = paramfile_ds["max_NH_planting_date"].values
def sowing_date_int2date(date_int):
    if np.isnan(date_int):
        return ""
    date_float = date_int / 100
    day = round(100*(date_float % 1))
    month = round(date_float - day/100)
    
    # print(date_int)
    # print(month)
    # print(day)
    # print(date_float % 1)
    # sinfirnei
    
    return f"YYYY-{str(month).rjust(2 ,'0')}-{str(day).rjust(2 ,'0')}"
    
# Import PFT name list
paramfile_pftnames = [x.decode("UTF-8").replace(" ", "") for x in paramfile_ds["pftname"].values]
# Save as dicts
mxmats = {}
sowing_windows = {}
for i, mxmat in enumerate(paramfile_mxmats):
    mxmats[paramfile_pftnames[i]] = mxmat
    if np.isnan(paramfile_sow_start[i]):
        sowing_windows[paramfile_pftnames[i]] = ""
    else:
        sowing_windows[paramfile_pftnames[i]] = f"{sowing_date_int2date(paramfile_sow_start[i])} to {sowing_date_int2date(paramfile_sow_end[i])}"
        
print(sowing_windows)
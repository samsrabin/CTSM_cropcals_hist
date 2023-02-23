# %% Setup

my_clm_subver = "c211112"
outDir = "/Users/Shared/CESM_inputdata/lnd/clm2/paramdata"

# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

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

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils



# %% Define functions etc.

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


# %% Import

inDir = "/Users/Shared/CESM_inputdata/lnd/clm2/paramdata/"
paramfile_versions = ['45', '50']

params = {}
all_pft_params = []
for v in paramfile_versions:
    pattern = os.path.join(inDir, f"*{v}_params.{my_clm_subver}.nc")
    paramfile = glob.glob(pattern)
    if len(paramfile) != 1:
        raise RuntimeError(f"Expected to find 1 match of {pattern}; found {len(paramfile)}")
    params[v] = xr.open_dataset(paramfile[0])
    params[v]["pftname2"] = [x.decode("UTF-8").replace(" ", "") for x in params[v]["pftname"].values]
    # # Import max growing season length (stored in netCDF as nanoseconds!)
    # paramfile_mxmats = paramfile_ds["mxmat"].values / np.timedelta64(1, 'D')
    # # Import sowing windows start/end (stored in netCDF as MMDD)
    # paramfile_sow_start = paramfile_ds["min_NH_planting_date"].values
    # paramfile_sow_end = paramfile_ds["max_NH_planting_date"].values
    for param_name in params[v]:
        if "pft" in params[v][param_name].dims:
            all_pft_params.append(param_name)

all_pft_params = list(np.sort(np.unique(all_pft_params)))

    
# %% Print info about a PFT

# pftname = 'rice'
# pftname = 'irrigated_rice'
pftname = 'tropical_corn'

param_values_lists = [[] for v in paramfile_versions]
param_names = []
long_names = []
for param_name in all_pft_params:
    param_values = []
    long_name = ""
    for v in paramfile_versions:
        if param_name in params[v]:
            pftnum = list(params[v]["pftname2"].values).index(pftname)
            param_value = params[v][param_name].isel(pft=pftnum).values
            if isinstance(param_value, np.ndarray):
                if param_value.ndim==0:
                    param_value = float(param_value)
                elif param_value.ndim==1 and len(param_value)==1:
                    param_value = param_value[0]
            param_values.append(param_value)
            if long_name=="" and 'long_name' in params[v][param_name].attrs:
                long_name = params[v][param_name].attrs['long_name']
        else:
            param_values.append("n/a")
    if len(np.unique(param_values)) > 1:
        param_names.append(param_name)
        long_names.append(long_name)
        print(f"{param_name} ({long_name}): {param_values}")
        for i in np.arange(len(paramfile_versions)):
            param_values_lists[i].append(param_values[i])

data = {'name': param_names,
        'long_name': long_names}
for i, v in enumerate(paramfile_versions):
    data[f"v{v}"] = param_values_lists[i]
df = pd.DataFrame(data)

outFile_suffix = ""
for v in paramfile_versions:
    outFile_suffix += "_" + v
outFile = os.path.join(outDir, f"compare_params_{my_clm_subver}" + outFile_suffix + f"_{pftname}.csv")
df.to_csv(outFile)


# %% Print info about a variable

param_name = "grnfill"

cropList = ['cotton', 'rice', 'sugarcane', 'spring_wheat', 'temperate_corn', 'temperate_soybean', 'tropical_corn', 'tropical_soybean']

for pftname in cropList:
    irr_pftname = "irrigated_" + pftname
    param_values = []
    for v in paramfile_versions:
        if param_name in params[v]:
            
            # Get value
            pftnum = list(params[v]["pftname2"].values).index(pftname)
            param_value = params[v][param_name].isel(pft=pftnum).values
            
            # Make sure irrigated value is the same
            irr_pftnum = list(params[v]["pftname2"].values).index(irr_pftname)
            irr_param_value = params[v][param_name].isel(pft=irr_pftnum).values
            if not np.array_equal(param_value, irr_param_value):
                raise RuntimeError("Irrigated and rainfed parameter values differ")
            
            # Save value
            if isinstance(param_value, np.ndarray):
                if param_value.ndim==0:
                    param_value = float(param_value)
                elif param_value.ndim==1 and len(param_value)==1:
                    param_value = param_value[0]
            param_values.append(param_value)
        else:
            param_values.append("n/a")
    if param_values[0] != param_values[1]:
        print(f"∆ {pftname}: {param_values[0]} → {param_values[1]}")
    else:
        print(f"  {pftname}: {param_values[0]}")
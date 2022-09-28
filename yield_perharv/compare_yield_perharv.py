# %% Options

# Years to include (set to None to leave that edge unbounded)
y1 = 2000
yN = 2009

# Directory where outputs are downloaded
indir = "/Users/Shared/CESM_runs/yield_perharv_f10_f10_mg37/lnd.291a1d5/hist/"

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"


# %% Imports

from re import I
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import shutil
import os
import time
import datetime as dt
import cftime
import warnings
import glob
import cartopy.crs as ccrs
from matplotlib import cm

import sys
sys.path.append(my_ctsm_python_gallery)
import utils


# %% Import/process daily outputs

h1_pattern = indir + "*h1.*"
h1_filelist = glob.glob(h1_pattern)
if not h1_filelist:
    raise RuntimeError(f"No files found matching pattern: {h1_pattern}")

daily_ds = utils.import_ds(h1_filelist, \
    myVars=["GRAINC_TO_FOOD"], 
    myVegtypes=utils.define_mgdcrop_list())

# 2001-01-01, e.g., is unfortunately the output for 2000-12-31
daily_ds = daily_ds.shift(time=-1)

daily_ds = daily_ds.sel(time=slice("2000-01-01","2009-12-31"))

# Collapse to annual. Even though you want the sum of grainc_to_food,
# that will mess up all the other variables. So first take the mean...
# daily_ann_ds = daily_ds.resample(time="1Y").mean()
daily_ann_ds = utils.resample(daily_ds, "mean", time="1Y")
# ... and then fix GRAINC_TO_FOOD
daily_ann_ds["GRAINC_TO_FOOD"] = daily_ann_ds["GRAINC_TO_FOOD"] * 3600*24*365

grainc_to_food = daily_ann_ds["GRAINC_TO_FOOD"]


# %% Import/process per-harvest outputs

h2_file = indir + "yield_perharv_f10_f10_mg37.clm2.h2.2000-01-01-00000.nc"

perharv_ds = utils.import_ds(h2_file, \
    myVars=["GRAINC_TO_FOOD_PERHARV"], 
    myVegtypes=utils.define_mgdcrop_list())

# grainc_to_food_perharv = perharv_ds.GRAINC_TO_FOOD_PERHARV.sum(dim="mxharvests")
perharv_ds = perharv_ds.sum(dim="mxharvests")

# Align dates correctly
perharv_ds["time"] = perharv_ds["time"] - dt.timedelta(days=1)

perharv_ds = perharv_ds.sel(time=slice("2000-01-01","2009-12-31"))


grainc_to_food_perharv = perharv_ds.GRAINC_TO_FOOD_PERHARV


# %% Import/process annual outputs

h2_file = indir + "yield_perharv_f10_f10_mg37.clm2.h2.2000-01-01-00000.nc"

ann_ds = utils.import_ds(h2_file, \
    myVars=["GRAINC_TO_FOOD_ANN"], 
    myVegtypes=utils.define_mgdcrop_list())

# Align dates correctly
ann_ds["time"] = ann_ds["time"] - dt.timedelta(days=1)

ann_ds = ann_ds.sel(time=slice("2000-01-01","2009-12-31"))

grainc_to_food_ann = ann_ds.GRAINC_TO_FOOD_ANN


# %% Import/grid annual outputs from NCL script

ncl_file = indir + "yield_perharv_f10_f10_mg37_Crop_GRAINC_TO_FOOD_200001-200912_Sum.nc"

ann_ncl_ds = xr.open_dataset(ncl_file)

grainc_to_food_ann_ncl_map = ann_ncl_ds.GRAINC_TO_FOOD

# # Ignore first timestep
# grainc_to_food_ann = grainc_to_food_ann.drop_isel(time=0)


# %% Find max differences in non-gridded variables

def get_maxdiff_ind(da1, da2):
    absdiff = np.abs(da2.values - da1.values)
    maxdiff = np.nanmax(absdiff)
    inds = np.where(absdiff == maxdiff)
    inds_zipped = list(zip(*inds))
    val1 = da1.values[inds]
    val2 = da2.values[inds]
    return maxdiff, val1, val2, inds_zipped

def get_lonlat_year_pft(inds, ds):
    lon = ds.patches1d_lon.values[inds[1]]
    lat = ds.patches1d_lat.values[inds[1]]
    yearlist = np.arange(2000,2010)
    year = yearlist[inds[0]]
    pft = ds.patches1d_itype_veg_str.values[inds[1]]
    pft_int = ds.patches1d_itype_veg.values[inds[1]]
    return lon, lat, year, pft, pft_int

maxdiff, val1, val2, inds_zipped = get_maxdiff_ind(grainc_to_food, grainc_to_food_perharv)
for i, inds in enumerate(inds_zipped):
    lon, lat, year, pft, pft_int = get_lonlat_year_pft(inds, ann_ds)
    print(f"Max diff, original vs. perharv: ({lon},{lat}; p={inds[1]}) {pft} ({pft_int}) {year}: {maxdiff} ({val1[i]} vs. {val2[i]})")

maxdiff, val1, val2, inds_zipped = get_maxdiff_ind(grainc_to_food_perharv, grainc_to_food_ann)
for i, inds in enumerate(inds_zipped):
    lon, lat, year, pft, pft_int = get_lonlat_year_pft(inds, ann_ds)
    print(f"Max diff, perharv vs. annual: ({lon},{lat}; p={inds[1]}) {pft} ({pft_int}) {year}: {maxdiff} ({val1[i]} vs. {val2[i]})")


# %% Find max differences in gridded variables

def get_lonlat_year_pft(inds, da):
    lon = da.lon.values[inds[3]]
    lat = da.lat.values[inds[2]]
    yearlist = np.arange(2000,2010)
    year = yearlist[inds[0]]
    pft = da.ivt_str.values[inds[1]]
    pft_int = utils.ivt_str2int(pft)
    return lon, lat, year, pft, pft_int

grainc_to_food_map = utils.grid_one_variable(daily_ann_ds, "GRAINC_TO_FOOD")
grainc_to_food_perharv_map = utils.grid_one_variable(perharv_ds, "GRAINC_TO_FOOD_PERHARV")
grainc_to_food_ann_map = utils.grid_one_variable(ann_ds, "GRAINC_TO_FOOD_ANN")

np.nanmax(np.abs(grainc_to_food_map - grainc_to_food_perharv_map).values)
np.nanmax(np.abs(grainc_to_food_perharv_map - grainc_to_food_ann_map).values)

maxdiff, val1, val2, inds_zipped = get_maxdiff_ind(grainc_to_food_map, grainc_to_food_perharv_map)
for i, inds in enumerate(inds_zipped):
    lon, lat, year, pft, pft_int = get_lonlat_year_pft(inds, grainc_to_food_map)
    print(f"Max diff, MAPPED original vs. perharv: ({lon},{lat}) {pft} ({pft_int}) {year}: {maxdiff} ({val1[i]} vs. {val2[i]})")

maxdiff, val1, val2, inds_zipped = get_maxdiff_ind(grainc_to_food_perharv_map, grainc_to_food_ann_map)
for i, inds in enumerate(inds_zipped):
    lon, lat, year, pft, pft_int = get_lonlat_year_pft(inds, grainc_to_food_perharv_map)
    print(f"Max diff, MAPPED perharv vs. annual: ({lon},{lat}) {pft} ({pft_int}) {year}: {maxdiff} ({val1[i]} vs. {val2[i]})")



# %% Describe all patch-years with discrepancy above a certain threshold

def describe_bad_patchyears(da1, da2, ds):
    absdiff = np.abs(da2.values - da1.values)
    thresh = 0.001
    # inds = np.where(absdiff > thresh)
    # inds = np.where(np.bitwise_and(absdiff > thresh, da1.values==0.0))
    # inds = np.where(np.bitwise_and(absdiff <= thresh, da1.values>100.0))
    inds = np.where(np.isinf(da2.values))
    print(f"{len(inds[0])}/{np.prod(np.shape(da1.values))} bad patch-years")
    inds_zipped = zip(*inds)
    for inds in inds_zipped:
        lon, lat, year, pft, pft_int = get_lonlat_year_pft(inds, ann_ds)
        val1 = da1.values[inds]
        val2 = da2.values[inds]
        print(f"Big diff, perharv vs. annual: ({lon},{lat}) {pft} ({pft_int}) {year}: {np.abs(val1-val2)} ({val1} vs. {val2})")

describe_bad_patchyears(grainc_to_food, grainc_to_food_perharv, ann_ds)





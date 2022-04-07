# %% Setup

# Years of interest (do not include extra year needed for finishing last growing season)
y1 = 1980
yN = 2009

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Directory where model output file(s) can be found (figure files will be saved in subdir here)
# indir1 = "/Users/Shared/CESM_runs/f10_f10_mg37_GDDtest/"
# indir1 = "/Users/Shared/CESM_runs/1537-crop-date-outputs3/"
# indir0 = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_pre1628/"
# indir0 = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_justmerged/"
# indir0 = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_justmerged_neg1/"
# indir0 = "/Users/Shared/CESM_runs/1537-crop-date-outputs3/"
# indir1 = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_restart/"
# indir1 = "/Users/Shared/CESM_runs/1537-crop-date-outputs3/"
# indir1 = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_justmerged_neg1_restart/"

# indir0 = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-30/"
# indir1 = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-31/"
# indir1 = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-04-ts01/"

indir0 = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-05-orig/"
indir1 = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-05-gddforced/"

# Directory to save output figures
outdir = indir1 + "figs/"

import enum
from multiprocessing.sharedctypes import Value
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import glob
import cftime
import cartopy.crs as ccrs
from matplotlib import cm
import datetime as dt
import nc_time_axis

import os

import sys
sys.path.append(my_ctsm_python_gallery)
import utils

import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")

def import_rx_dates(s_or_h, date_inFile, dates_ds1_orig):
    # Get run info:
    # Max number of growing seasons per year
    if "mxsowings" in dates_ds1_orig:
        mxsowings = dates_ds1_orig.dims["mxsowings"]
    else:
        mxsowings = 1
        
    # Which vegetation types were simulated?
    itype_veg_toImport = np.unique(dates_ds1_orig.patches1d_itype_veg)

    date_varList = []
    for i in itype_veg_toImport:
        for g in np.arange(mxsowings):
            thisVar = f"{s_or_h}date{g+1}_{i}"
            date_varList = date_varList + [thisVar]

    ds = utils.import_ds(date_inFile, myVars=date_varList)
    
    for v in ds:
        ds = ds.rename({v: v.replace(f"{s_or_h}date","gs")})
    
    return ds

def thisCrop_map_to_patches(lon_points, lat_points, map_ds, vegtype_int):
    # xarray pointwise indexing; see https://xarray.pydata.org/en/stable/user-guide/indexing.html#more-advanced-indexing
    return map_ds[f"gs1_{vegtype_int}"].sel( \
        lon=xr.DataArray(lon_points, dims="patch"),
        lat=xr.DataArray(lat_points, dims="patch")).squeeze(drop=True)

def get_gs_length_1patch(sdates, hdates, align=True, verbose=False):
    
    if align:
        sdates, hdates = align_shdates_1patch(sdates, hdates)
    
    # Get growing season length
    gs_length = hdates - sdates
    gs_length[gs_length <= 0] = gs_length[gs_length <= 0] + 365
    
    if verbose:
        print(np.concatenate((np.expand_dims(sdates, axis=1), 
                            np.expand_dims(hdates, axis=1), 
                            np.expand_dims(gs_length, axis=1)), axis=1))
    
    return gs_length    

def get_year_from_cftime(cftime_date):
    # Subtract 1 because the date for annual files is when it was SAVED
    return cftime_date.year - 1

def check_and_trim_years(y1, yN, get_year_from_cftime, ds_in):
    ### In annual outputs, file with name Y is actually results from year Y-1.
    ### Note that time values refer to when it was SAVED. So 1981-01-01 is for year 1980.

    # Check that all desired years are included
    if get_year_from_cftime(ds_in.time.values[0]) > y1:
        raise RuntimeError(f"Requested y1 is {y1} but first year in outputs is {get_year_from_cftime(ds_in.time.values[0])}")
    elif get_year_from_cftime(ds_in.time.values[-1]) < y1:
        raise RuntimeError(f"Requested yN is {yN} but last year in outputs is {get_year_from_cftime(ds_in.time.values[-1])}")
    
    # Remove years outside range of interest
    ### Include an extra year at the end to finish out final seasons.
    ds_in = ds_in.sel(time=slice(f"{y1+1}-01-01", f"{yN+2}-01-01"))
    
    # Make sure you have the expected number of timesteps (including extra year)
    Nyears_expected = yN - y1 + 2
    if ds_in.dims["time"] != Nyears_expected:
        raise RuntimeError(f"Expected {Nyears_expected} timesteps in output but got {ds_in.dims['time']}")
    
    return ds_in

def get_Nharv(array_in, these_dims):
    # Sum over time and mxevents to get number of events in time series for each patch
    sum_indices = tuple(these_dims.index(x) for x in ["time", "mxharvests"])
    Nevents_eachPatch = np.sum(array_in > 0, axis=sum_indices)
    return Nevents_eachPatch

def set_firstharv_nan(this_ds, this_var, firstharv_nan_inds):
    this_da = this_ds[this_var]
    this_array = this_da.values
    this_array[0,0,firstharv_nan_inds] = np.nan
    this_da.values = this_array
    this_ds[this_var] = this_da
    return this_ds

def extract_gs_timeseries(this_ds, this_var, in_da, include_these, Npatches, Ngs):
    this_array = in_da.values

    # Rearrange to (Ngs*mxevents, Npatches)
    this_array = this_array.reshape(-1, Npatches)
    include_these = include_these.reshape(-1, Npatches)

    # Extract valid events
    this_array = this_array.transpose()
    include_these = include_these.transpose()
    this_array = this_array[include_these].reshape(Npatches,Ngs)
    this_array = this_array.transpose()
    
    # Always ignore last sowing date, because for some cells this growing season will be incomplete.
    if this_var == "SDATES":
        this_array = this_array[:-1,:]

    # Set up and fill output DataArray
    out_da = xr.DataArray(this_array, \
        coords=this_ds.coords,
        dims=this_ds.dims)

    # Save output DataArray to Dataset
    this_ds[this_var] = out_da
    return this_ds

def time_to_gs(Ngs, this_ds, extra_var_list):
    ### Checks ###
    # Relies on max harvests per year >= 2
    mxharvests = len(this_ds.mxharvests)
    if mxharvests < 2:
        raise RuntimeError(f"get_gs_length_1patch() assumes max harvests per year == 2, not {mxharvests}")
    # Untested with max harvests per year > 2
    if mxharvests > 2:
        print(f"Warning: Untested with max harvests per year ({mxharvests}) > 2")
    # All sowing dates should be positive
    if np.any(this_ds["SDATES"] <= 0).values:
        raise ValueError("All sowing dates should be positive... right?")
    # Throw an error if harvests happened on the day of planting
    if (np.any(np.equal(this_ds["SDATES"], this_ds["HDATES"]))):
        raise RuntimeError("Harvest on day of planting")
    
    ### Setup ###
    Npatches = this_ds.dims["patch"]
    # Set up empty Dataset with time axis as "gs" (growing season) instead of what CLM puts out
    gs_years = [t.year-1 for t in this_ds.time.values[:-1]]
    new_ds_gs = xr.Dataset(coords={
        "gs": gs_years,
        "patch": this_ds.patch,
        })
    
    ### Ignore harvests from the old, non-prescribed sowing date from the year before this run began. ###
    cond1 = this_ds["HDATES"].isel(time=0, mxharvests=0) \
      < this_ds["SDATES"].isel(time=0, mxsowings=0)
    firstharv_nan_inds = np.where(cond1)[0]
    # (Only necessary before "don't allow harvest on day of planting" was working)
    cond2 = np.bitwise_and( \
        this_ds["HDATES"].isel(time=0, mxharvests=0) \
        == this_ds["SDATES"].isel(time=0, mxsowings=0), \
        this_ds["HDATES"].isel(time=0, mxharvests=1) > 0)
    firstharv_nan_inds = np.where(np.bitwise_or(cond1, cond2))[0]
    for v in ["HDATES"] + extra_var_list:
        this_ds = set_firstharv_nan(this_ds, v, firstharv_nan_inds)
    
    ### In some cells, the last growing season did complete, but we have to ignore it because it's extra. This block determines which members of HDATES (and other mxharvests-dimensioned variables) should be ignored for this reason. ###
    hdates = this_ds["HDATES"].values
    Nharvests_eachPatch = get_Nharv(hdates, this_ds["HDATES"].dims)
    if np.max(Nharvests_eachPatch) > Ngs + 1:
        raise ValueError(f"Expected at most {Ngs+1} harvests in any patch; found at least one patch with {np.max(Nharvests_eachPatch)}")
    h = mxharvests
    while np.any(Nharvests_eachPatch > Ngs):
        h = h - 1
        if h < 0:
            raise RuntimeError("Unable to ignore enough harvests")
        hdates[-1,h,Nharvests_eachPatch > Ngs] = np.nan
        Nharvests_eachPatch = get_Nharv(hdates, this_ds["HDATES"].dims)
    
    # So: Which events are we manually excluding?
    sdate_included = this_ds["SDATES"].values > 0
    hdate_manually_excluded = np.isnan(hdates)
    hdate_included = np.bitwise_not(np.bitwise_or(hdate_manually_excluded, hdates<=0))

    ### Extract the series of sowings and harvests ###
    new_ds_gs = extract_gs_timeseries(new_ds_gs, "SDATES", this_ds["SDATES"], sdate_included, Npatches, Ngs+1)
    for v in ["HDATES"] + extra_var_list:
        new_ds_gs = extract_gs_timeseries(new_ds_gs, v, this_ds[v], hdate_included, Npatches, Ngs)
    return new_ds_gs


# %% Import output sowing and harvest dates, etc.

print("Importing CLM output sowing and harvest dates...")

extra_annual_vars = ["GDDACCUM_PERHARV", "GDDHARV_PERHARV", "HARVEST_REASON_PERHARV", "HUI_PERHARV"]

dates_ds1_orig = utils.import_ds(glob.glob(indir1 + "*h2.*"), \
    myVars=["SDATES", "HDATES"] + extra_annual_vars, 
    myVegtypes=utils.define_mgdcrop_list())
dates_ds1_orig = check_and_trim_years(y1, yN, get_year_from_cftime, dates_ds1_orig)

dates_ds0_orig = utils.import_ds(glob.glob(indir0 + "*h2.*"), \
    myVars=["SDATES", "HDATES"] + extra_annual_vars, 
    myVegtypes=utils.define_mgdcrop_list())
dates_ds0_orig = check_and_trim_years(y1, yN, get_year_from_cftime, dates_ds0_orig)

# How many growing seasons can we use? Ignore last season because it can be incomplete for some gridcells.
Ngs = dates_ds1_orig.dims['time'] - 1

print("Done.")


# %% Align sowing and harvest dates/etc.

dates_ds0 = time_to_gs(Ngs, dates_ds0_orig, extra_annual_vars)
dates_ds1 = time_to_gs(Ngs, dates_ds1_orig, extra_annual_vars)

# %% Check that simulated sowing dates do not vary between years

verbose = True

t1 = 0 # 0-indexed
ok = True
v = "SDATES"

t1_yr = dates_ds1.gs.values[t1]
t1_vals = np.squeeze(dates_ds1[v].isel(gs=t1).values)

for t in np.arange(t1+1, dates_ds1.dims["gs"]):
    t_yr = dates_ds1.gs.values[t]
    t_vals = np.squeeze(dates_ds1[v].isel(gs=t).values)
    ok_p = np.squeeze(t1_vals == t_vals)
    if not np.all(ok_p):
        ok = False
        print(f"{v} timestep {t} does not match timestep {t1}")
        if verbose:
            for thisPatch in np.where(np.bitwise_not(ok_p))[0]:
                thisLon = dates_ds1.patches1d_lon.values[thisPatch]
                thisLat = dates_ds1.patches1d_lat.values[thisPatch]
                thisCrop = dates_ds1.patches1d_itype_veg_str.values[thisPatch]
                thisStr = f"Patch {thisPatch} (lon {thisLon} lat {thisLat}) {thisCrop}"
                print(f"{thisStr}: Sowing {t1_yr} jday {int(t1_vals[thisPatch])}, {t_yr} jday {int(t_vals[thisPatch])}")
            break

if ok:
    print(f"✅ dates_ds1: CLM output sowing dates do not vary through {dates_ds1.dims['gs'] - t1} growing seasons of output.")





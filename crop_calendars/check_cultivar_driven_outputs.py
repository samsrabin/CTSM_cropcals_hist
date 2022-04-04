# %% Setup

# Years of interest (do not include extra year needed for finishing last growing season)
y1 = 1980
yN = 2009

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Directory where model output file(s) can be found (figure files will be saved in subdir here)
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_GDDtest/"
# indir = "/Users/Shared/CESM_runs/1537-crop-date-outputs3/"
# indir_orig = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_pre1628/"
# indir_orig = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_justmerged/"
# indir_orig = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_justmerged_neg1/"
# indir_orig = "/Users/Shared/CESM_runs/1537-crop-date-outputs3/"
# indir = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_restart/"
# indir = "/Users/Shared/CESM_runs/1537-crop-date-outputs3/"
# indir = "/Users/Shared/CESM_runs/1537-crop-date-outputs3_justmerged_neg1_restart/"

indir_orig = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-30/"
indir = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-31/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-04-ts01/"

# Directory to save output figures
outdir = indir + "figs/"

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

def import_rx_dates(s_or_h, date_inFile, dates_ds):
    # Get run info:
    # Max number of growing seasons per year
    if "mxsowings" in dates_ds:
        mxsowings = dates_ds.dims["mxsowings"]
    else:
        mxsowings = 1
        
    # Which vegetation types were simulated?
    itype_veg_toImport = np.unique(dates_ds.patches1d_itype_veg)

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

def align_shdates_1patch(sdates, hdates, verbose=False):
    # Relies on max harvests per year >= 2
    if hdates.shape[1] < 2:
        raise RuntimeError(f"get_gs_length_1patch() assumes max harvests per year == 2, not {hdates.shape[1]}")
    # Untested with max harvests per year > 2
    elif hdates.shape[1] > 2:
        print(f"Warning: Untested with max harvests per year ({hdates.shape[1]}) > 2")
        
    if verbose:
        print("sdates, original:")
        print(sdates)
        print("hdates, original:")
        print(hdates)
    
    # All sowing dates should be positive
    if np.any(np.bitwise_not(sdates > 0)):
        raise ValueError("All sowing dates should be positive... right?")
        
    # Ignore harvests from the old, non-prescribed sowing date from the year before this run began.
    # # (Second condition can be removed once you get "don't allow harvest on day of planting" working.)
    # if hdates[0,0] < sdates[0] or (hdates[0,0] == sdates[0] and hdates[0,1] == sdates[0]) :
    if hdates[0,0] < sdates[0] or (hdates[0,0] == sdates[0] and hdates[0,1] > 0) :
        if verbose: print("Removing first harvest")
        hdates[0,0] = np.nan
    
    # Throw an error if any other harvests happened on the day of planting
    if (np.any(np.equal(sdates, hdates))):
        raise RuntimeError("Harvest on day of planting")
    
    # Extract the series of sowings and harvests.
    sdates = sdates[sdates>0] # Since we already checked that all sowing dates are positive, this just has the function of collapsing the array to 1d, if it wasn't already
    hdates = hdates[hdates>0]
    
    # Always ignore last sowing date, because for some cells this growing season will be incomplete.
    sdates = sdates[:-1]
    
    # In other cells, the last growing season did complete, but we have to ignore it.
    if hdates.size == sdates.size + 1:
        hdates = hdates[:-1]
    
    # At the end of this, sdates and hdates should be the same shape
    if sdates.shape != hdates.shape:
        raise RuntimeError(f"sdates {sdates.shape} and hdates {hdates.shape} failed to align")
    
    return sdates, hdates

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


# %% Import output sowing and harvest dates

print("Importing CLM output sowing and harvest dates...")

dates_ds = utils.import_ds(glob.glob(indir + "*h2.*"), \
    myVars=["SDATES", "HDATES"], 
    myVegtypes=utils.define_mgdcrop_list())
dates_ds = check_and_trim_years(y1, yN, get_year_from_cftime, dates_ds)

dates_ds_orig = utils.import_ds(glob.glob(indir_orig + "*h2.*"), \
    myVars=["SDATES", "HDATES"], 
    myVegtypes=utils.define_mgdcrop_list())
dates_ds_orig = check_and_trim_years(y1, yN, get_year_from_cftime, dates_ds_orig)

# How many growing seasons can we use? Ignore last season because it can be incomplete for some gridcells.
Ngs = dates_ds.dims['time'] - 1

print("Done.")


# %% Check that simulated sowing dates do not vary between years

verbose = True

t1 = 0 # 0-indexed
ok = True
v = "SDATES"

t1_yr = get_year_from_cftime(dates_ds.time.values[t1])
t1_vals = np.squeeze(dates_ds[v].isel(time=t1).values)

for t in np.arange(t1+1, dates_ds.dims["time"]):
    t_yr = get_year_from_cftime(dates_ds.time.values[t])
    t_vals = np.squeeze(dates_ds[v].isel(time=t).values)
    ok_p = np.squeeze(t1_vals == t_vals)
    if not np.all(ok_p):
        ok = False
        print(f"{v} timestep {t} does not match timestep {t1}")
        if verbose:
            for thisPatch in np.where(np.bitwise_not(ok_p))[0]:
                thisLon = dates_ds.patches1d_lon.values[thisPatch]
                thisLat = dates_ds.patches1d_lat.values[thisPatch]
                thisCrop = dates_ds.patches1d_itype_veg_str.values[thisPatch]
                thisStr = f"Patch {thisPatch} (lon {thisLon} lat {thisLat}) {thisCrop}"
                print(f"{thisStr}: Sowing {t1_yr} jday {int(t1_vals[thisPatch])}, {t_yr} jday {int(t_vals[thisPatch])}")
            break

if ok:
    print(f"✅ CLM output sowing dates do not vary through {dates_ds.dims['time'] - t1} years of output.")


# %% Align sowing and harvest dates
### There's probably a more efficient way to do this than looping through patches!

# Set up empty arrays
Npatch = dates_ds.dims["patch"]
mxsowings = dates_ds.dims["mxsowings"]

sdates_aligned = np.empty((Ngs, mxsowings, Npatch))
hdates_aligned = np.empty((Ngs, mxsowings, Npatch))
gs_lengths = np.empty((Ngs, mxsowings, Npatch))
for p in np.arange(Npatch):
# for p in [4]:
    sdates, hdates = align_shdates_1patch( \
        dates_ds.SDATES.values[:,:,p],
        dates_ds.HDATES.values[:,:,p],
        verbose = False)
    sdates_aligned[:,:,p] = np.expand_dims(sdates, axis=1)
    hdates_aligned[:,:,p] = np.expand_dims(hdates, axis=1)
    tmp = get_gs_length_1patch(sdates, hdates, align=False)
    gs_lengths[:,:,p] = np.expand_dims(tmp, axis=1)
    
sdates_aligned_orig = np.empty((Ngs, mxsowings, Npatch))
hdates_aligned_orig = np.empty((Ngs, mxsowings, Npatch))
gs_lengths_orig = np.empty((Ngs, mxsowings, Npatch))
for p in np.arange(Npatch):
    sdates, hdates = align_shdates_1patch( \
        dates_ds_orig.SDATES.values[:,:,p],
        dates_ds_orig.HDATES.values[:,:,p],
        verbose = False)
    sdates_aligned_orig[:,:,p] = np.expand_dims(sdates, axis=1)
    hdates_aligned_orig[:,:,p] = np.expand_dims(hdates, axis=1)
    tmp = get_gs_length_1patch(sdates, hdates, align=False)
    gs_lengths_orig[:,:,p] = np.expand_dims(tmp, axis=1)

print("Done.")


# %% Look at some random patches

print("Some random examples:")
for p in np.sort(np.random.choice(np.arange(Npatch), 10, replace=False)):
    thisLon = np.round(dates_ds.patches1d_lon.values[p],5)
    thisLat = np.round(dates_ds.patches1d_lat.values[p],5)
    thisCrop = dates_ds.patches1d_itype_veg_str.values[p]
    thisPatch = dates_ds.patch.values[p]
    print(f"Patch {thisPatch} (lon {thisLon} lat {thisLat}) {thisCrop}")
    these_sdates = sdates_aligned[:,:,p]
    these_hdates = hdates_aligned[:,:,p]
    these_gslengths = gs_lengths[:,:,p]
    for g in np.arange(Ngs):
        syear = get_year_from_cftime(dates_ds.time.values[g])
        sdate = int(these_sdates[g,0])
        hdate = int(these_hdates[g,0])
        hyear = syear + (hdate<=sdate)
        gslen = int(these_gslengths[g,0])
        print(f"    Planted {syear} day {sdate}, harvested {hyear} day {hdate} ({gslen} days)")


# %% Check that original and new versions are identical

print(f"sdates: {np.array_equal(sdates_aligned_orig, sdates_aligned)}")
print(f"hdates: {np.array_equal(hdates_aligned_orig, hdates_aligned)}")
print(f"gs_lengths: {np.array_equal(gs_lengths_orig, gs_lengths)}")

bad_patches = np.where(np.any( \
    np.bitwise_or(sdates_aligned_orig != sdates_aligned, 
                  hdates_aligned_orig != hdates_aligned),axis=1)[0])[0]
for p in bad_patches:
    print(f"Patch {p}:")
    # print("Original:")
    # print(np.concatenate(
    #     (np.expand_dims(sdates_aligned_orig[:,0,p], axis=1),
    #      hdates_aligned_orig[:,:,p],
    #      np.expand_dims(gs_lengths_orig[:,0,p], axis=1)),
    #     axis=1))
    # print("New:")
    # print(np.concatenate(
    #     (np.expand_dims(sdates_aligned[:,0,p], axis=1),
    #      hdates_aligned[:,:,p],
    #      np.expand_dims(gs_lengths[:,0,p], axis=1)),
    #     axis=1))
    for s in np.arange(Ngs):
        if sdates_aligned[s,0,p] != sdates_aligned_orig[s,0,p]:
            print(f"Year {s+1}: Sowing date changed from {sdates_aligned_orig[s,0,p]} to {sdates_aligned[s,0,p]}")
        if hdates_aligned[s,0,p] != hdates_aligned_orig[s,0,p]:
            print(f"Year {s+1}: Harvest date changed from {hdates_aligned_orig[s,0,p]} to {hdates_aligned[s,0,p]}")
        if gs_lengths[s,0,p] != gs_lengths_orig[s,0,p]:
            print(f"Year {s+1}: Season length changed from {gs_lengths_orig[s,0,p]} to {gs_lengths[s,0,p]}")
        # print(f"First gs changed by {gs_lengths[0,0,p]-gs_lengths_orig[0,0,p]} day(s)")
    print("")


# %% Look at raw date outputs for a patch

p = 0

print("sdates_orig:")
print(dates_ds_orig.SDATES.values[:,:,p])
print("hdates:")
print(dates_ds_orig.HDATES.values[:,:,p])

print("sdates:")
print(dates_ds.SDATES.values[:,:,p])
print("hdates:")
print(dates_ds.HDATES.values[:,:,p])
    

# %%  Check GDDHARV
### "Growing degree days (gdd) needed to harvest"

gddharv_ds = utils.import_ds(glob.glob(indir + "*h1.*"), \
    myVars="GDDHARV", 
    myVegtypes=utils.define_mgdcrop_list())

# %% Set up empty array
gddharv = np.empty((Ngs, mxsowings, Npatch))

# Get year list
yearList_base = []
for y in np.arange(Ngs):
    yearList_base = yearList_base + [dates_ds.time.values[y].year]
yearList_base

has_year_zero = gddharv_ds.time.values[0].has_year_zero

# for p in np.arange(Npatch):
for p in np.arange(10):
    sdates = np.squeeze(sdates_aligned[:,:,p])
    hdates = np.squeeze(hdates_aligned[:,:,p])
    yearList = yearList_base + (hdates<=sdates).astype(int)
    dateList = []
    for y, thisYear in enumerate(yearList):
        cfdtnl = cftime.datetime.fromordinal(hdates[y], calendar="noleap")
        thisDate = cftime.DatetimeNoLeap(thisYear, cfdtnl.month, cfdtnl.day, 0, 0, 0, 0, has_year_zero=has_year_zero)
        dateList = dateList + [thisDate]
    # print(dateList)
    print(gddharv_ds.GDDHARV.isel(patch=p).sel(time=dateList).values)
    
    # break
    

# %% Test: See sdate, hdate, and growing season length

# for p in np.arange(2):
for p in [965]:
    print(f"Patch {p}:")
    thisPatch_sdates = dates_ds.SDATES.values[:,:,p]
    thisPatch_hdates = dates_ds.HDATES.values[:,:,p]
    get_gs_length_1patch(thisPatch_sdates, thisPatch_hdates, verbose=True)
    print(" ")


# %% Test: Look at time series for one patch

thisVar = "CPHASE"
thisPatch = 342

thisVar_ds = utils.import_ds(glob.glob(indir + "*h1.*"), \
    myVars=thisVar, 
    myVegtypes=utils.define_mgdcrop_list())

plt.plot(thisVar_ds[thisVar][:,thisPatch])









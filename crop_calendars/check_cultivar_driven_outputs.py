# %% Setup

# Years of interest (do not include extra year needed for finishing last growing season)
y1 = 1980
yN = 2009

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# CLM max growing season length, mxmat, is stored in the following files:
#   * clm5_1: lnd/clm2/paramdata/ctsm51_params.c211112.nc
#   * clm5_0: lnd/clm2/paramdata/clm50_params.c211112.nc
#   * clm4_5: lnd/clm2/paramdata/clm45_params.c211112.nc
paramfile_dir = "/Users/Shared/CESM_inputdata/lnd/clm2/paramdata/"
my_clm_ver = 51
my_clm_subver = "c211112"

# Prescribed sowing and harvest dates
sdates_rx_file = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"
hdates_rx_file = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"

# Directory where model output file(s) can be found (figure files will be saved in subdir here)
indirs = list()
# indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-05-orig/",
#                    used_clm_mxmat = True))
indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-05-gddforced/",
                   used_clm_mxmat = True))
# indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-08-gddforced/",
#                    used_clm_mxmat = True))
indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-11-gddforced/",
                   used_clm_mxmat = False))

if len(indirs) != 2:
    raise RuntimeError(f"For now, indirs must have 2 members (found {len(indirs)}")

import enum
from multiprocessing.sharedctypes import Value
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# Directory to save output figures
indir0 = indirs[0]["path"]
outdir_figs = os.path.join(indirs[1]["path"], f"figs_comp_{os.path.basename(os.path.dirname(indir0))}")
if not os.path.exists(outdir_figs):
    os.makedirs(outdir_figs)

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
    data_vars = dict()
    for v in this_ds.data_vars:
        if "time" not in this_ds[v].dims:
            data_vars[v] = this_ds[v]
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
    
    ### Save additional variables ###
    new_ds_gs = new_ds_gs.assign(variables=data_vars)
    new_ds_gs.coords["lon"] = this_ds.coords["lon"]
    new_ds_gs.coords["lat"] = this_ds.coords["lat"]
    
    return new_ds_gs

fontsize_titles = 8
fontsize_axislabels = 8
fontsize_ticklabels = 7
bin_width = 30
lat_bin_edges = np.arange(0, 91, bin_width)
def make_map(ax, this_map, this_title, ylabel, vmin, vmax, bin_width, fontsize_ticklabels, fontsize_titles, cmap): 
    im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
            this_map, shading="auto",
            vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="white")
    ax.coastlines(linewidth=0.3)
    if this_title:
        ax.set_title(this_title, fontsize=fontsize_titles)
    if ylabel:
        ax.set_ylabel(ylabel)
    # cbar = plt.colorbar(im1, orientation="horizontal", fraction=0.1, pad=0.02)
    # cbar.ax.tick_params(labelsize=fontsize_ticklabels)
    
    # ax.yaxis.set_tick_params(width=0.2)
    # ticks = np.arange(-90, 91, bin_width)
    # ticklabels = [str(x) for x in ticks]
    # for i,x in enumerate(ticks):
    #     if x%2:
    #         ticklabels[i] = ''
    # # ticklabels = []
    # plt.yticks(np.arange(-90,91,bin_width), labels=ticklabels,
    #            fontsize=fontsize_ticklabels,
    #            fontweight=0.1)
    return im1

def make_axis(fig, ny, nx, n):
    ax = fig.add_subplot(ny,nx,n,projection=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False) # Turn off box outline
    return ax

# Get vegtype str for figure titles
def get_vegtype_str_for_title(vegtype_str):
    vegtype_str3 = vegtype_str
    if "irrigated" in vegtype_str:
        vegtype_str3 = vegtype_str3.replace("irrigated_", "") + " (ir)"
    else:
        vegtype_str3 = vegtype_str3 + " (rf)"
    if "temperate" in vegtype_str:
        vegtype_str3 = vegtype_str3.replace("temperate_", "temp. ")
    if "tropical" in vegtype_str:
        vegtype_str3 = vegtype_str3.replace("tropical_", "trop. ")
    elif "soybean" in vegtype_str:
        vegtype_str3 = "temp. " + vegtype_str3
    if "soybean" in vegtype_str:
        vegtype_str3 = vegtype_str3.replace("soybean", "soy")
    if "spring" in vegtype_str:
        vegtype_str3 = vegtype_str3.replace("spring_", "spr. ")
    if "winter" in vegtype_str:
        vegtype_str3 = vegtype_str3.replace("winter_", "win.")
    return vegtype_str3

def mask_immature(this_ds, this_vegtype, gridded_da):
    reason_gridded = utils.grid_one_variable(this_ds, "HARVEST_REASON_PERHARV", \
                vegtype=this_vegtype).squeeze(drop=True)
    gridded_da = gridded_da.where(reason_gridded == 1)
    return gridded_da


def remove_outliers(gridded_da):
    gs_axis = gridded_da.dims.index("gs")
    pctle25 = np.nanpercentile(gridded_da, q=25, axis=gs_axis)
    pctle75 = np.nanpercentile(gridded_da, q=75, axis=gs_axis)
    iqr = pctle75 - pctle25
    outlier_thresh_lo = pctle25 - iqr
    outlier_thresh_up = pctle75 - iqr
    not_outlier = np.bitwise_and(gridded_da > outlier_thresh_lo, gridded_da < outlier_thresh_up)
    gridded_da = gridded_da.where(not_outlier)
    return gridded_da


# %% Import output sowing and harvest dates, etc.

print("Importing CLM output sowing and harvest dates...")

extra_annual_vars = ["GDDACCUM_PERHARV", "GDDHARV_PERHARV", "HARVEST_REASON_PERHARV", "HUI_PERHARV"]

dates_ds1_orig = utils.import_ds(glob.glob(indirs[1]["path"] + "*h2.*"), \
    myVars=["SDATES", "HDATES"] + extra_annual_vars, 
    myVegtypes=utils.define_mgdcrop_list())
dates_ds1_orig = check_and_trim_years(y1, yN, get_year_from_cftime, dates_ds1_orig)

dates_ds0_orig = utils.import_ds(glob.glob(indirs[0]["path"] + "*h2.*"), \
    myVars=["SDATES", "HDATES"] + extra_annual_vars, 
    myVegtypes=utils.define_mgdcrop_list())
dates_ds0_orig = check_and_trim_years(y1, yN, get_year_from_cftime, dates_ds0_orig)

# How many growing seasons can we use? Ignore last season because it can be incomplete for some gridcells.
Ngs = dates_ds1_orig.dims['time'] - 1

# CLM max growing season length, mxmat, is stored in the following files:
#   * clm5_1: lnd/clm2/paramdata/ctsm51_params.c211112.nc
#   * clm5_0: lnd/clm2/paramdata/clm50_params.c211112.nc
#   * clm4_5: lnd/clm2/paramdata/clm45_params.c211112.nc
pattern = os.path.join(paramfile_dir,f"*{my_clm_ver}_params.{my_clm_subver}.nc")
paramfile = glob.glob(pattern)
if len(paramfile) != 1:
    raise RuntimeError(f"Expected to find 1 match of {pattern}; found {len(paramfile)}")
paramfile_ds = xr.open_dataset(paramfile[0])
# Import max growing season length (stored in netCDF as nanoseconds!)
paramfile_mxmats = paramfile_ds["mxmat"].values / np.timedelta64(1, 'D')
# Import PFT name list
paramfile_pftnames = [x.decode("UTF-8").replace(" ", "") for x in paramfile_ds["pftname"].values]

print("Done.")


# %% Import GGCMI sowing and harvest dates

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

sdates_rx_ds = import_rx_dates("s", sdates_rx_file, dates_ds0_orig)
hdates_rx_ds = import_rx_dates("h", hdates_rx_file, dates_ds0_orig)

# Get GGCMI growing season lengths
def get_gs_len_da(this_da):
    tmp = this_da.values
    tmp[tmp < 0] = 365 + tmp[tmp < 0]
    this_da.values = tmp
    return this_da
gs_len_rx_ds = hdates_rx_ds.copy()
for v in gs_len_rx_ds:
    if v == "time_bounds":
        continue
    gs_len_rx_ds[v] = get_gs_len_da(hdates_rx_ds[v] - sdates_rx_ds[v])


# %% Align output sowing and harvest dates/etc.

dates_ds0 = time_to_gs(Ngs, dates_ds0_orig, extra_annual_vars)
dates_ds1 = time_to_gs(Ngs, dates_ds1_orig, extra_annual_vars)

# Get growing season length
dates_ds0["GSLEN"] = get_gs_len_da(dates_ds0["HDATES"] - dates_ds0["SDATES"])
dates_ds1["GSLEN"] = get_gs_len_da(dates_ds1["HDATES"] - dates_ds1["SDATES"])


# %% Check that some things are constant across years for ds1

constantVars = ["SDATES", "GDDHARV_PERHARV"]
verbose = True

t1 = 0 # 0-indexed
for v in constantVars:
    ok = True

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
                    if v == "SDATES":
                        print(f"{thisStr}: Sowing {t1_yr} jday {int(t1_vals[thisPatch])}, {t_yr} jday {int(t_vals[thisPatch])}")
                    else:
                        print(f"{thisStr}: {t1_yr} {v} {int(t1_vals[thisPatch])}, {t_yr} {v} {int(t_vals[thisPatch])}")
                    break

    if ok:
        print(f"✅ dates_ds1: CLM output {v} do not vary through {dates_ds1.dims['gs'] - t1} growing seasons of output.")


# %% For both datasets, check that GDDACCUM_PERHARV <= HUI_PERHARV

verbose = True

def check_gddaccum_le_hui(this_ds, which_ds):
    if np.all(this_ds["GDDACCUM_PERHARV"] <= this_ds["HUI_PERHARV"]):
        print(f"✅ dates_ds{which_ds}: GDDACCUM_PERHARV always <= HUI_PERHARV")
    else: print(f"❌ dates_ds{which_ds}: GDDACCUM_PERHARV *not* always <= HUI_PERHARV")

check_gddaccum_le_hui(dates_ds0, 0)
check_gddaccum_le_hui(dates_ds1, 1)


# %% Make map of harvest reasons

thisVar = "HARVEST_REASON_PERHARV"

reason_list_text_all = [ \
    "???",                 # 0; should never actually be saved
    "Crop mature",         # 1
    "Max gs length",       # 2
    "Bad Dec31 sowing",    # 3
    "Sowing today",        # 4
    "Sowing tomorrow",     # 5
    "Sown a yr ago tmrw.", # 6
    "Sowing tmrw. (Jan 1)" # 7
    ]

reason_list = np.unique(np.concatenate( \
    (np.unique(dates_ds0.HARVEST_REASON_PERHARV.values), \
    np.unique(dates_ds1.HARVEST_REASON_PERHARV.values))))
reason_list = [int(x) for x in reason_list]

reason_list_text = [reason_list_text_all[x] for x in reason_list]
 
def get_reason_freq_map(Ngs, thisCrop_gridded, reason):
    map_yx = np.sum(thisCrop_gridded==reason, axis=0, keepdims=False) / Ngs
    notnan_yx = np.bitwise_not(np.isnan(thisCrop_gridded.isel(gs=0, drop=True)))
    map_yx = map_yx.where(notnan_yx)
    return map_yx

ny = 2
nx = len(reason_list)

figsize = (8, 4)
cbar_adj_bottom = 0.15
cbar_ax_rect = [0.15, 0.05, 0.7, 0.05]
cmap = plt.cm.viridis
if nx != 2:
    print(f"Since nx = {nx}, you may need to rework some parameters")

for v, vegtype_str in enumerate(dates_ds0.vegtype_str.values):
    if vegtype_str not in dates_ds0.patches1d_itype_veg_str.values:
        continue
    vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
    
    # Get vegtype str for figure titles
    vegtype_str3 = get_vegtype_str_for_title(vegtype_str)
    
    # Grid
    thisCrop0_gridded = utils.grid_one_variable(dates_ds0, thisVar, \
        vegtype=vegtype_int).squeeze(drop=True)
    thisCrop1_gridded = utils.grid_one_variable(dates_ds1, thisVar, \
        vegtype=vegtype_int).squeeze(drop=True)
    
    # Set up figure
    fig = plt.figure(figsize=figsize)
    
    # Map each reason's frequency
    for f, reason in enumerate(reason_list):
        reason_text = reason_list_text[f]
        
        ylabel = "CLM5-style" if f==0 else None
        map0_yx = get_reason_freq_map(Ngs, thisCrop0_gridded, reason)
        ax = make_axis(fig, ny, nx, f+1)
        im0 = make_map(ax, map0_yx, f"v0: {reason_text}", ylabel, 0.0, 1.0, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
        
        ylabel = "GGCMI-style" if f==0 else None
        ax = make_axis(fig, ny, nx, f+nx+1)
        map1_yx = get_reason_freq_map(Ngs, thisCrop1_gridded, reason)
        im1 = make_map(ax, map1_yx, f"v1: {reason_text}", ylabel, 0.0, 1.0, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
        
    fig.suptitle(f"Harvest reason: {vegtype_str3}")
    fig.subplots_adjust(bottom=cbar_adj_bottom)
    cbar_ax = fig.add_axes(cbar_ax_rect)
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal")
    cbar_ax.tick_params(labelsize=fontsize_ticklabels)
    plt.xlabel("Frequency", fontsize=fontsize_titles)
    
    # plt.show()
    # break
    
    # Save
    outfile = os.path.join(outdir_figs, f"harvest_reason_0vs1_{vegtype_str}.png")
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
            bbox_inches='tight')
    plt.close()
    

# %% Make map of means 

varList = ["GDDHARV_PERHARV", "HUI_PERHARV", "GSLEN", "GSLEN.onlyMature"]
# varList = ["GDDHARV_PERHARV"]
# varList = ["HUI_PERHARV"]
# varList = ["GSLEN"]
# varList = ["GSLEN.onlyMature"]
# varList = ["GSLEN", "GSLEN.onlyMature"]

for thisVar in varList:
    
    # Processing options
    title_prefix = ""
    filename_prefix = ""
    onlyMature = "onlyMature" in thisVar
    if onlyMature:
        thisVar = thisVar.replace(".onlyMature", "")
        title_prefix = title_prefix + " (if mat.)"
        filename_prefix = filename_prefix + "_ifmature"
    noOutliers = "noOutliers" in thisVar
    if noOutliers:
        thisVar = thisVar.replace(".noOutliers", "")
        title_prefix = title_prefix + " (no outl.)"
        filename_prefix = filename_prefix + "_nooutliers"
    useMedian = "useMedian" in thisVar

    ny = 2
    nx = 1
    vmin = 0.0
    cmap = plt.cm.viridis
    if thisVar == "GDDHARV_PERHARV":
        title_prefix = "Harv. thresh." + title_prefix
        filename_prefix = "harvest_thresh" + filename_prefix
        units = "GDD"
    elif thisVar == "HUI_PERHARV":
        title_prefix = "HUI" + title_prefix
        filename_prefix = "hui" + filename_prefix
        units = "GDD"
    elif thisVar == "GSLEN":
        title_prefix = "Seas. length" + title_prefix
        filename_prefix = "seas_length" + filename_prefix
        units = "Days"
        ny = 3
        nx = 1
        vmin = None
    else:
        raise RuntimeError(f"thisVar {thisVar} not recognized")
    
    figsize = (4, 4)
    cbar_adj_bottom = 0.15
    cbar_ax_rect = [0.15, 0.05, 0.7, 0.05]
    if nx != 1:
        print(f"Since nx = {nx}, you may need to rework some parameters")
    if ny == 3:
        cbar_width = 0.46
        cbar_ax_rect = [(1-cbar_width)/2, 0.05, cbar_width, 0.05]
    elif ny != 2:
        print(f"Since ny = {ny}, you may need to rework some parameters")

    for v, vegtype_str in enumerate(dates_ds0.vegtype_str.values):
        if vegtype_str not in dates_ds0.patches1d_itype_veg_str.values:
            continue
        vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
        
        # Get vegtype str used in parameter file
        if vegtype_str == "soybean":
            vegtype_str2 = "temperate_soybean"
        elif vegtype_str == "irrigated_soybean":
            vegtype_str2 = "irrigated_temperate_soybean"
        else:
            vegtype_str2 = vegtype_str
            
        # Get vegtype str for figure titles
        vegtype_str3 = get_vegtype_str_for_title(vegtype_str)
        
        # Grid
        thisCrop0_gridded = utils.grid_one_variable(dates_ds0, thisVar, \
            vegtype=vegtype_int).squeeze(drop=True)
        thisCrop1_gridded = utils.grid_one_variable(dates_ds1, thisVar, \
            vegtype=vegtype_int).squeeze(drop=True)
        
        # If needed, only include seasons where crop reached maturity
        if onlyMature:
            thisCrop0_gridded = mask_immature(dates_ds0, vegtype_int, thisCrop0_gridded)
            thisCrop1_gridded = mask_immature(dates_ds1, vegtype_int, thisCrop1_gridded)
            
        # If needed, remove outliers
        if noOutliers:
            thisCrop0_gridded = remove_outliers(thisCrop0_gridded)
            thisCrop1_gridded = remove_outliers(thisCrop1_gridded)
        
        # Set up figure 
        fig = plt.figure(figsize=figsize)
        subplot_title_suffixes = ["", ""]
        
        # Get means
        map0_yx = np.mean(thisCrop0_gridded, axis=0)
        map1_yx = np.mean(thisCrop1_gridded, axis=0)
        max0 = int(np.ceil(np.nanmax(map0_yx)))
        max1 = int(np.ceil(np.nanmax(map1_yx)))
        vmax = max(max0, max1)
        if vmin == None:
            vmin = int(np.floor(min(np.nanmin(map0_yx), np.nanmin(map1_yx))))
        if thisVar == "GSLEN":
            mxmat = int(paramfile_mxmats[paramfile_pftnames.index(vegtype_str2)])
            units = f"Days (mxmat: {mxmat})"
            if not mxmat > 0:
                raise RuntimeError(f"Error getting mxmat: {mxmat}")
            
            longest_gs = max(max0, max1)
            subplot_title_suffixes = [f" (max={max0})",
                                      f" (max={max1})"]
            if indirs[0]["used_clm_mxmat"]:
                if max0 > mxmat:
                    raise RuntimeError(f"v0: mxmat {mxmat} but max simulated {max0}")
            if indirs[1]["used_clm_mxmat"]:
                if max1 > mxmat:
                    raise RuntimeError(f"v1: mxmat {mxmat} but max simulated {max1}")
            ggcmi_yx = gs_len_rx_ds[f"gs1_{vegtype_int}"].isel(time=0, drop=True)
            ggcmi_yx = ggcmi_yx.where(np.bitwise_not(np.isnan(map1_yx)))
            ggcmi_max = int(np.nanmax(ggcmi_yx.values))
            vmax = max(max0, max1, ggcmi_max)
            
            if vmax > mxmat:
                Nok = mxmat - vmin + 1
                Nbad = vmax - mxmat + 1
                cmap_to_mxmat = plt.cm.viridis(np.linspace(0, 1, num=Nok))
                cmap_after_mxmat = plt.cm.OrRd(np.linspace(0, 1, num=Nbad))
                cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', np.vstack((cmap_to_mxmat, cmap_after_mxmat)))
        
        ylabel = "CLM5-style"
        ax = make_axis(fig, ny, nx, 1)
        im0 = make_map(ax, map0_yx, f"v0{subplot_title_suffixes[0]}", ylabel, vmin, vmax, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
        
        ylabel = "GGCMI-style"
        ax = make_axis(fig, ny, nx, 2)
        im1 = make_map(ax, map1_yx, f"v1{subplot_title_suffixes[1]}", ylabel, vmin, vmax, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
        
        if thisVar == "GSLEN":
            ax = make_axis(fig, ny, nx, 3)
            im1 = make_map(ax, ggcmi_yx, f"GGCMI (max={ggcmi_max})", ylabel, vmin, vmax, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
            
        fig.suptitle(f"{title_prefix}:\n{vegtype_str3}", y=1.04)
        fig.subplots_adjust(bottom=cbar_adj_bottom)
        cbar_ax = fig.add_axes(cbar_ax_rect)
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal")
        cbar_ax.tick_params(labelsize=fontsize_ticklabels)
        plt.xlabel(units, fontsize=fontsize_titles)
        
        # plt.show()
        # print(os.path.join(outdir_figs, f"{filename_prefix}_0vs1_{vegtype_str}.png"))
        # break
        
        # Save
        outfile = os.path.join(outdir_figs, f"{filename_prefix}_0vs1_{vegtype_str}.png")
        plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
                bbox_inches='tight')
        plt.close()

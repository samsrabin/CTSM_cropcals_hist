# %% Setup

# Years of interest (do not include extra year needed for finishing last growing season)
# y1 = 1980
# yN = 2009 # 2009
y1 = 1951
yN = 1952 # 2009

# Save map figures to files?
save_figs = True

# Where is the script running?
import socket
hostname = socket.gethostname()

# Import the CTSM Python utilities
if hostname == "Sams-2021-MacBook-Pro.local":
    my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
    import sys
    sys.path.append(my_ctsm_python_gallery)
    import utils
else:
    # Only possible because I have export PYTHONPATH=$HOME in my .bash_profile
    from ctsm_python_gallery_myfork.ctsm_py import utils

# Directory where input file(s) can be found (figure files will be saved in subdir here)
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_1850/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-29/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/tmp/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-30/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601.02.72441c4e"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601.03.ba902039"
# indir = "/glade/scratch/samrabin/archive/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1950-2013.ggcmi2/lnd/hist"
indir = "/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1950-2013.ggcmi2"

# sdate_inFile = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"
# hdate_inFile = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.20220602_230029.nc"
# sdate_inFile = "/glade/u/home/samrabin/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"
# hdate_inFile = "/glade/u/home/samrabin/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"
sdate_inFile = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"
hdate_inFile = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"


# Directory to save output netCDF
import os
if hostname == "Sams-2021-MacBook-Pro.local":
    outdir = "/Users/Shared/CESM_work/crop_dates/"
else:
    outdir = "/glade/u/home/samrabin/crop_dates/"
if save_figs:
    outdir_figs = indir + "figs/"
    if not os.path.exists(outdir_figs):
        os.makedirs(outdir_figs)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import glob
import cftime
import cartopy.crs as ccrs
from matplotlib import cm
import datetime as dt

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
    
    
# Check that, during period of interest, simulated harvest always happens the day before sowing
# Could vectorize this, but it gets complicated because some cells are sown Jan. 1 and some aren't.
def check_harvest_daybefore_sowing(dates_ds):
    print("   Checking that harvest is always the day before sowing...")
    verbose = True

    patchList = dates_ds.patch.values
    ok_p = np.full((dates_ds.dims["patch"]), True)

    for p, thisPatch in enumerate(patchList):
        
        if (p+1) % 10000 == 0:
            print(f"{p+1} / {len(patchList)}")
            
        thisLon = dates_ds.patches1d_lon.values[p]
        thisLat = dates_ds.patches1d_lat.values[p]
        # thisLon = np.round(dates_ds.patches1d_lon.values[p], decimals=2)
        # thisLat = np.round(dates_ds.patches1d_lat.values[p], decimals=2)
        thisCrop = dates_ds.patches1d_itype_veg_str.values[p]
        thisIVT = dates_ds.patches1d_itype_veg.values[p]
        thisStr = f"Patch {thisPatch} (lon {thisLon} lat {thisLat}) {thisCrop} ({thisIVT})"
        sim_sp = dates_ds["SDATES"].sel(patch=thisPatch).values
        sim_hp = dates_ds["HDATES"].sel(patch=thisPatch).values
        
        if "time" in dates_ds.dims and dates_ds.dims["time"] == 1:
            thisYear = dates_ds["time"].values[0].year
        else:
            thisYear = None
        
        # There should be no missing sowings
        if any(sim_sp < 1):
            ok_p[p] = False
            if verbose:
                if np.all(sim_sp < 1):
                    print(f"{thisStr}: Sowing never happened")
                elif thisYear:
                    print(f"{thisStr}: Sowing didn't happen in {thisYear}")
                else:
                    print(f"{thisStr}: Sowing didn't happen some year(s); first {y1+int(np.argwhere(sim_sp < 1)[0])}")
            continue

        # Should only need to consider one sowing and one harvest
        if (sim_sp.ndim > 1):
            if (sim_sp.shape[1] > 1):
                ok_p[p] = False
                if verbose: print(f"{thisStr}: Expected mxsowings 1 but found {sim_sp.shape[1]}")
                continue
            sim_sp = sim_sp[:,0]
        if (sim_sp.ndim > 1):
            if np.any(sim_hp[:,1:] > 0):
                ok_p[p] = False
                if verbose: print(f"{thisStr}: More than 1 harvest found in some year(s)")
                continue
            sim_hp = sim_hp[:,0]

        # Align
        if sim_sp[0] > 1:
            sim_hp = sim_hp[1:]
        else:
            sim_hp = sim_hp[0:-1]
        sim_sp = sim_sp[0:-1]

        # We're going to be comparing each harvest to the sowing that FOLLOWS it.
        sim_sp = sim_sp[1:]
        sim_hp = sim_hp[0:-1]
            
        # There should no longer be any missing harvests
        if any(sim_hp < 1):
            ok_p[p] = False
            if verbose:
                if np.all(sim_hp < 1):
                    print(f"{thisStr}: Harvest never happened")
                elif thisYear:
                    print(f"{thisStr}: Harvest didn't happen in {thisYear}")
                else:
                    print(f"{thisStr}: Harvest didn't happen some growing season(s); first {y1+int(np.argwhere(sim_hp < 1)[0])}")
            continue

        # Harvest should always happen the day before the next sowing.
        exp_hp = ((sim_sp - 2)%365) + 1
        if not np.array_equal(sim_hp, exp_hp):
            ok_p[p] = False
            if verbose: print(f"{thisStr}: Not every harvest happens the day before next sowing")
            continue
        
    if np.all(ok_p):
        print("   ✅ CLM output sowing and harvest dates look good.")
    else:
        raise RuntimeError(f"   ❌ {sum(np.bitwise_not(ok_p))} patch(es) had problem(s) with CLM output sowing and/or harvest dates.")


def check_sdates(dates_ds, sdates_rx, verbose=False):
    print("   Checking that input and output sdates match...")

    sdates_grid = utils.grid_one_variable(\
        dates_ds, 
        "SDATES")

    all_ok = True
    any_found = False
    vegtypes_skipped = []
    vegtypes_included = []
    for i, vt_str in enumerate(dates_ds.vegtype_str.values):
        
        # Input
        vt = dates_ds.ivt.values[i]
        thisVar = f"gs1_{vt}"
        if thisVar not in sdates_rx:
            vegtypes_skipped = vegtypes_skipped + [vt_str]
            # print(f"    {vt_str} ({vt}) SKIPPED...")
            continue
        vegtypes_included = vegtypes_included + [vt_str]
        any_found = True
        if verbose: print(f"    {vt_str} ({vt})...")
        in_map = sdates_rx[thisVar].squeeze(drop=True)
        
        # Output
        out_map = sdates_grid.sel(ivt_str=vt_str).squeeze(drop=True)
        
        # Check for differences
        diff_map = out_map - in_map
        diff_map_notnan = diff_map.values[np.invert(np.isnan(diff_map.values))]
        if np.any(diff_map_notnan):
            print(f"Difference(s) found in {vt_str}")
            here = np.where(diff_map_notnan)
            print("in:")
            in_map_notnan = in_map.values[np.invert(np.isnan(diff_map.values))]
            print(in_map_notnan[here][0:4])
            out_map_notnan = out_map.values[np.invert(np.isnan(diff_map.values))]
            print("out:")
            print(out_map_notnan[here][0:4])
            print("diff:")
            print(diff_map_notnan[here][0:4])
            ieboeurbeo
            all_ok = False

    if not (any_found):
        raise RuntimeError("No matching variables found in sdates_rx!")

    # Sanity checks for included vegetation types
    vegtypes_skipped = np.unique([x.replace("irrigated_","") for x in vegtypes_skipped])
    vegtypes_skipped_weird = [x for x in vegtypes_skipped if x in vegtypes_included]
    if np.array_equal(vegtypes_included, [x.replace("irrigated_","") for x in vegtypes_included]):
        print("\nWARNING: No irrigated crops included!!!\n")
    elif vegtypes_skipped_weird:
        print(f"\nWarning: Some crop types had output rainfed patches but no irrigated patches: {vegtypes_skipped_weird}")
        
    if all_ok:
        print("   ✅ Input and output sdates match!")
    else:
        raise RuntimeError("   ❌ Input and output sdates differ.")


def get_values_at_harvest(thisCrop_hdates_rx, in_da, time_indsP1, new_dt_axis, newVar):
    # There's almost certainly a more efficient way to do this than looping through patches!
    for p in np.arange(thisCrop_hdates_rx.size):
        thisPatch_da = in_da.isel(patch=p)
        
        # Extract time range of interest plus extra year for cells where growing season crosses the new year
        thisCell_gdds_da = thisPatch_da.isel(time=np.where(doy==thisCrop_hdates_rx.sel(patch=p).values)[0])
        
        # Extract the actual time range of interest for this cell, depending on whether its growing season crosses the new year
        if thisCrop_gany[p]:
            thisCell_gdds_da = thisCell_gdds_da.isel(time=time_indsP1[1:])
        else:
            thisCell_gdds_da = thisCell_gdds_da.isel(time=time_indsP1[:-1])
        
        # Set to standard datetime axis for outputs
        thisCell_gdds_da = thisCell_gdds_da.assign_coords(time=new_dt_axis)
        
        # Add to new DataArray
        if p==0:
            out_da = thisCell_gdds_da
            out_da = out_da.rename(newVar)
        else:
            out_da = xr.concat([out_da, thisCell_gdds_da], dim="patch")
        
    return out_da


# Set up output Dataset(s)
def setup_output_ds(in_ds, thisVar, Nyears, new_dt_axis):
    out_ds = in_ds.isel(time=np.arange(Nyears))
    out_ds = out_ds.assign_coords(time=new_dt_axis)
    del out_ds[thisVar]
    return out_ds

# Get and grid mean GDDs in GGCMI growing season
def yp_list_to_ds(yp_list, daily_ds, daily_incl_ds, dates_rx, longname_prefix):
    
    # Get means
    warnings.filterwarnings("ignore", message="Mean of empty slice") # Happens when you do np.nanmean() of an all-NaN array (or slice, if doing selected axis/es)
    p_list = [np.nanmean(x, axis=0) if not isinstance(x, type(None)) else x for x in yp_list]
    warnings.filterwarnings("always", message="Mean of empty slice")
    
    # Grid
    ds_out = xr.Dataset()
    for c, ra in enumerate(p_list):
        if isinstance(ra, type(None)):
            continue
        thisCrop_str = daily_incl_ds.vegtype_str.values[c]
        newVar = f"gdd1_{utils.ivt_str2int(thisCrop_str)}"
        ds = daily_ds.isel(patch=np.where(daily_ds.patches1d_itype_veg_str.values==thisCrop_str)[0])
        template_da = ds.patches1d_itype_veg_str
        da = xr.DataArray(data = ra,
                          coords = template_da.coords,
                          attrs = {'units': 'GDD',
                                   'long_name': f'{longname_prefix}{vegtype_str}'})
        
        # Grid this crop
        ds['tmp'] = da
        da_gridded = utils.grid_one_variable(ds, 'tmp', vegtype=thisCrop_str).squeeze(drop=True)
        
        # Add singleton time dimension and save to output Dataset
        da_gridded = da_gridded.expand_dims(time = dates_rx.time)
        ds_out[newVar] = da_gridded
        
    return ds_out



# %% Import output sowing and harvest dates

# Keep 1 extra year to avoid incomplete final growing season for crops harvested after Dec. 31.
y1_import_str = f"{y1+1}-01-01"
yN_import_str = f"{yN+2}-01-01"

print(f"Importing netCDF time steps {y1_import_str} through {yN_import_str} (years are +1 because of CTSM output naming)")

# Get h2 file (list)
if indir[-1] != os.path.sep:
    indir = indir + os.path.sep
h2_pattern = indir + "*h2.*"
h2_filelist = glob.glob(h2_pattern)
if not h2_filelist:
    raise RuntimeError(f"No files found matching pattern: {h2_pattern}")

sdates_rx = None
hdates_rx = None
gddaccum_ds = None
gddharv_ds = None
incorrectly_daily = False
skip_patches_for_isel_nan_lastyear = np.ndarray([])
for y, thisYear in enumerate(np.arange(y1+1,yN+3)):
    print(f'netCDF year {thisYear}...')
    
    dates_ds = utils.import_ds(h2_filelist, \
        myVars=["SDATES", "HDATES"], 
        myVegtypes=utils.define_mgdcrop_list(),
        timeSlice = slice(f"{thisYear}-01-01", f"{thisYear}-12-31"))
        
    if dates_ds.dims['time'] > 1:
        if dates_ds.dims['time'] == 365:
            if not incorrectly_daily:
                print("   ℹ️ You saved SDATES and HDATES daily, but you only needed annual. Fixing.")
            incorrectly_daily = True
            dates_ds = dates_ds.isel(time=-1)
    
    # check_harvest_daybefore_sowing(dates_ds)
    # Make sure NaN masks match
    sdates_all_nan = np.sum(~np.isnan(dates_ds.SDATES.values), axis=dates_ds.SDATES.dims.index('mxsowings')) == 0
    hdates_all_nan = np.sum(~np.isnan(dates_ds.HDATES.values), axis=dates_ds.HDATES.dims.index('mxharvests')) == 0
    N_unmatched_nans = np.sum(sdates_all_nan != hdates_all_nan)
    if N_unmatched_nans > 0:
        raise RuntimeError("Output SDATE and HDATE NaN masks do not match.")
    
    # Just work with non-NaN patches for now
    skip_patches_for_isel_nan = np.where(sdates_all_nan)[0]
    incl_patches_for_isel_nan = np.where(~sdates_all_nan)[0]
    different_nan_mask = y > 0 and ~np.array_equal(skip_patches_for_isel_nan_lastyear, skip_patches_for_isel_nan)
    if different_nan_mask:
        print('   Different NaN mask than last year')
        incl_thisyr_but_nan_lastyr = [dates_ds.patch.values[p] for p in incl_patches_for_isel_nan if p in skip_patches_for_isel_nan_lastyear]
    else:
        incl_thisyr_but_nan_lastyr = []
    skipping_patches_for_isel_nan = len(skip_patches_for_isel_nan) > 0
    if skipping_patches_for_isel_nan:
        print(f'   Ignoring {len(skip_patches_for_isel_nan)} patches with all-NaN sowing and harvest dates.')
        dates_incl_ds = dates_ds.isel(patch=incl_patches_for_isel_nan)
    else:
        dates_incl_ds = dates_ds
    
    # Some patches can have -1 sowing date?? Hopefully just an artifact of me incorrectly saving SDATES/HDATES daily.
    mxsowings = dates_ds.dims['mxharvests']
    mxsowings_dim = dates_ds.HDATES.dims.index('mxharvests')
    skip_patches_for_isel_sdatelt1 = np.where(dates_incl_ds.SDATES.values < 1)[1]
    skipping_patches_for_isel_sdatelt1 = len(skip_patches_for_isel_sdatelt1) > 0
    if skipping_patches_for_isel_sdatelt1:
        unique_hdates = np.unique(dates_incl_ds.HDATES.isel(mxharvests=0, patch=skip_patches_for_isel_sdatelt1).values)
        if incorrectly_daily and list(unique_hdates)==[364]:
            print(f'   ❗ {len(skip_patches_for_isel_sdatelt1)} patches have SDATE < 1, but this might have just been because of incorrectly daily outputs. Setting them to 365.')
            new_sdates_ar = dates_incl_ds.SDATES.values
            if mxsowings_dim != 0:
                raise RuntimeError("Code this up")
            new_sdates_ar[0, skip_patches_for_isel_sdatelt1] = 365
            dates_incl_ds['SDATES'] = xr.DataArray(data = new_sdates_ar,
                                                    coords = dates_incl_ds['SDATES'].coords,
                                                    attrs = dates_incl_ds['SDATES'].attrs)
        else:
            raise RuntimeError(f"{len(skip_patches_for_isel_sdatelt1)} patches have SDATE < 1. Unique affected hdates: {unique_hdates}")
        
    # Some patches can have -1 harvest date?? Hopefully just an artifact of me incorrectly saving SDATES/HDATES daily. Can also happen if patch wasn't active last year
    mxharvests = dates_ds.dims['mxharvests']
    mxharvests_dim = dates_ds.HDATES.dims.index('mxharvests')
    # If a patch was inactive last year but was either (a) harvested the last time it was active or (b) was never active, it will have -1 as its harvest date this year. Such instances are okay.
    hdates_thisyr = dates_incl_ds.HDATES.isel(mxharvests=0)
    skip_patches_for_isel_hdatelt1 = np.where(hdates_thisyr.values < 1)[0]
    skipping_patches_for_isel_hdatelt1 = len(skip_patches_for_isel_hdatelt1) > 0
    if incl_thisyr_but_nan_lastyr and list(skip_patches_for_isel_hdatelt1):
        hdates_thisyr_where_nan_lastyr = hdates_thisyr.sel(patch=incl_thisyr_but_nan_lastyr)
        sdates_thisyr_where_nan_lastyr = dates_incl_ds.SDATES.isel(mxsowings=0).sel(patch=incl_thisyr_but_nan_lastyr)
        if np.any(hdates_thisyr_where_nan_lastyr < 1):
            # patches_to_fix = hdates_thisyr_where_nan_lastyr.isel(patch=np.where(hdates_thisyr_where_nan_lastyr < 1)[0]).patch.values
            new_hdates = dates_incl_ds.HDATES.values
            if mxharvests_dim != 0:
                raise RuntimeError("Code this up")
            patch_list = list(hdates_thisyr.patch.values)
            here = [patch_list.index(x) for x in incl_thisyr_but_nan_lastyr]
            print(f"   ❗ {len(here)} patches have harvest date -1 because they weren't active last year (and were either never active or were harvested when last active). Ignoring, but you should have done a run with patches always active if they are ever active in the real LU timeseries.")
            new_hdates[0, here] = sdates_thisyr_where_nan_lastyr.values - 1
            dates_incl_ds['HDATES'] = xr.DataArray(data = new_hdates,
                                                coords = dates_incl_ds.HDATES.coords,
                                                attrs = dates_incl_ds.HDATES.attrs)
            # Recalculate these
            skip_patches_for_isel_hdatelt1 = np.where(dates_incl_ds.HDATES.isel(mxharvests=0).values < 1)[0]
            skipping_patches_for_isel_hdatelt1 = len(skip_patches_for_isel_hdatelt1) > 0

    # Resolve other issues
    if skipping_patches_for_isel_hdatelt1:
        unique_sdates = np.unique(dates_incl_ds.SDATES.isel(patch=skip_patches_for_isel_hdatelt1).values)
        if incorrectly_daily and list(unique_sdates)==[1]:
            print(f'   ❗ {len(skip_patches_for_isel_hdatelt1)} patches have HDATE < 1??? Seems like this might have just been because of incorrectly daily outputs; setting them to 365.')
            new_hdates_ar = dates_incl_ds.HDATES.values
            if mxharvests_dim != 0:
                raise RuntimeError("Code this up")
            new_hdates_ar[0, skip_patches_for_isel_hdatelt1] = 365
            dates_incl_ds['HDATES'] = xr.DataArray(data = new_hdates_ar,
                                                    coords = dates_incl_ds['HDATES'].coords,
                                                    attrs = dates_incl_ds['HDATES'].attrs)
        else:
            raise RuntimeError(f"{len(skip_patches_for_isel_hdatelt1)} patches have HDATE < 1. Unique affected sdates: {unique_sdates}")
    
    # Make sure there was only one harvest per year
    N_extra_harv = np.sum(np.nanmax(dates_incl_ds.HDATES.isel(mxharvests=slice(1,mxharvests)).values, axis=mxharvests_dim) >= 1)
    if N_extra_harv > 0:
        raise RuntimeError(f"{N_extra_harv} patches have >1 harvest.")
    
    # Make sure harvest happened the day before sowing
    sdates_clm = dates_incl_ds.SDATES.values.squeeze()
    hdates_clm = dates_incl_ds.HDATES.isel(mxharvests=0).values
    diffdates_clm = sdates_clm - hdates_clm
    diffdates_clm[(sdates_clm==1) & (hdates_clm==365)] = 1
    if list(np.unique(diffdates_clm)) != [1]:
        raise RuntimeError(f"Not all sdates-hdates are 1: {np.unique(diffdates_clm)}")
        
    # Import expected sowing dates. This will also be used as our template output file.
    if not sdates_rx:
        print("   Importing expected sowing dates...")
        sdates_rx = import_rx_dates("s", sdate_inFile, dates_incl_ds)
        
    check_sdates(dates_incl_ds, sdates_rx)
    
    if not hdates_rx:
        print("   Importing prescribed harvest dates...")
        hdates_rx = import_rx_dates("h", hdate_inFile, dates_incl_ds)
        # Determine cells where growing season crosses new year
        grows_across_newyear = hdates_rx < sdates_rx
        
    print(f"   Importing accumulated GDDs...")
    clm_gdd_var = "GDDACCUM"
    myVars = [clm_gdd_var]
    if save_figs:
        myVars.append("GDDHARV")
    h1_ds = utils.import_ds(glob.glob(indir + f"*h1.{thisYear-1}-01-01*"), myVars=myVars, myVegtypes=utils.define_mgdcrop_list())
    
    # Restrict to patches we're including
    if skipping_patches_for_isel_nan:
        if not np.array_equal(dates_ds.patch.values, h1_ds.patch.values):
            raise RuntimeError("dates_ds and h1_ds don't have the same patch list!")
        h1_incl_ds = h1_ds.isel(patch=incl_patches_for_isel_nan)
    else:
        h1_incl_ds = h1_ds

    if not np.any(h1_incl_ds[clm_gdd_var].values != 0):
        raise RuntimeError(f"All {clm_gdd_var} values are zero!")
    
    # Get day of year for each day in time axis
    doy = np.array([t.timetuple().tm_yday for t in h1_incl_ds.time.values])
    
    # Get standard datetime axis for outputs
    t1 = h1_incl_ds.time.values[0]
    Nyears = yN - y1 + 1
    new_dt_axis = np.array([cftime.datetime(y, 1, 1, 
                                            calendar=t1.calendar,
                                            has_year_zero=t1.has_year_zero)
                            for y in np.arange(y1, yN+1)])
    time_indsP1 = np.arange(Nyears + 1)
    
    if y==0:
        gddaccum_yp_list = [None for vegtype_str in h1_incl_ds.vegtype_str.values]
        if save_figs: gddharv_yp_list = [None for vegtype_str in h1_incl_ds.vegtype_str.values]
    
    incl_vegtype_indices = []
    for v, vegtype_str in enumerate(h1_incl_ds.vegtype_str.values):
        
        vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
        thisCrop_full_patchlist = list(utils.xr_flexsel(h1_ds, vegtype=vegtype_str).patch.values)
        
        # Get time series for each patch of this type
        thisCrop_ds = utils.xr_flexsel(h1_incl_ds, vegtype=vegtype_str)
        thisCrop_gddaccum_da = thisCrop_ds[clm_gdd_var]
        if save_figs: thisCrop_gddharv_da = thisCrop_ds['GDDHARV']
        if not thisCrop_gddaccum_da.size:
            continue
        print(f"      {vegtype_str}...")
        incl_vegtype_indices = incl_vegtype_indices + [v]
        
        # Get prescribed harvest dates for these patches
        lon_points = thisCrop_ds.patches1d_lon.values
        lat_points = thisCrop_ds.patches1d_lat.values
        thisCrop_hdates_rx = thisCrop_map_to_patches(lon_points, lat_points, hdates_rx, vegtype_int)
        # Get "grows across new year?" for these patches
        thisCrop_gany = thisCrop_map_to_patches(lon_points, lat_points, grows_across_newyear, vegtype_int)
        
        if isinstance(gddaccum_yp_list[v], type(None)):
            gddaccum_yp_list[v] = np.full((Nyears, len(thisCrop_full_patchlist)), np.nan)
            if save_figs: gddharv_yp_list[v] = np.full((Nyears, len(thisCrop_full_patchlist)), np.nan)
        
        # Get the accumulated GDDs at each prescribed harvest date
        gddaccum_atharv_p = np.full(thisCrop_hdates_rx.shape, np.nan)
        if save_figs: gddharv_atharv_p = np.full(thisCrop_hdates_rx.shape, np.nan)
        unique_rx_hdates = np.unique(thisCrop_hdates_rx.values)
        # Build an indexing tuple
        patches = []
        i_patches = []
        i_times = []
        for i, hdate in enumerate(unique_rx_hdates):
            here = np.where(thisCrop_hdates_rx.values == hdate)[0]
            patches += list(thisCrop_gddaccum_da.patch.values[here])
            i_patches += list(here)
            i_times += list(np.full((len(here),), int(hdate-1)))
        # Select using the indexing tuple
        gddaccum_atharv_p = thisCrop_gddaccum_da.values[(i_times, i_patches)]
        if save_figs: gddharv_atharv_p = thisCrop_gddharv_da.values[(i_times, i_patches)]
        if np.any(np.isnan(gddaccum_atharv_p)):
            print(f"         ❗ {np.sum(np.isnan(gddaccum_atharv_p))}/{len(gddaccum_atharv_p)} NaN after extracting GDDs accumulated at harvest")
        if save_figs and np.any(np.isnan(gddharv_atharv_p)):
            print(f"         ❗ {np.sum(np.isnan(gddharv_atharv_p))}/{len(gddharv_atharv_p)} NaN after extracting GDDHARV")
        # Sort patches back to correct order
        if not np.all(thisCrop_gddaccum_da.patch.values[:-1] <= thisCrop_gddaccum_da.patch.values[1:]):
            raise RuntimeError("This code depends on DataArray patch list being sorted.")
        sortorder = np.argsort(patches)
        gddaccum_atharv_p = gddaccum_atharv_p[np.array(sortorder)]
        if save_figs: gddharv_atharv_p = gddharv_atharv_p[np.array(sortorder)]
                
        # Assign these to growing seasons based on whether gs crossed new year
        thisYear_active_patch_indices = [thisCrop_full_patchlist.index(x) for x in thisCrop_ds.patch.values]
        thisCrop_sdates_rx = thisCrop_map_to_patches(lon_points, lat_points, sdates_rx, vegtype_int)
        where_gs_thisyr = np.where(thisCrop_sdates_rx < thisCrop_hdates_rx)[0]
        tmp_gddaccum = np.full(thisCrop_sdates_rx.shape, np.nan)
        tmp_gddaccum[where_gs_thisyr] = gddaccum_atharv_p[where_gs_thisyr]
        if save_figs:
            tmp_gddharv = np.full(tmp_gddaccum.shape, np.nan)
            tmp_gddharv[where_gs_thisyr] = gddharv_atharv_p[where_gs_thisyr]
        if y > 0:
            where_gs_lastyr = np.where(thisCrop_sdates_rx > thisCrop_hdates_rx)[0]
            # Make sure we're not about to overwrite any existing values.
            if np.any(~np.isnan(tmp_gddaccum[where_gs_lastyr])):
                raise RuntimeError("Unexpected non-NaN for last season's GDD accumulation")
            if save_figs and np.any(~np.isnan(tmp_gddharv[where_gs_lastyr])):
                raise RuntimeError("Unexpected non-NaN for last season's GDDHARV")
            # Fill.
            tmp_gddaccum[where_gs_lastyr] = gddaccum_atharv_p[where_gs_lastyr]
            if save_figs: tmp_gddharv[where_gs_lastyr] = gddharv_atharv_p[where_gs_lastyr]
            # Last year's season should be filled out now; make sure.
            if np.any(np.isnan(tmp_gddaccum[where_gs_lastyr])):
                raise RuntimeError("Unexpected NaN for last season's GDD accumulation. Maybe because it was inactive last year?")
            if save_figs and np.any(np.isnan(tmp_gddharv[where_gs_lastyr])):
                raise RuntimeError("Unexpected NaN for last season's GDDHARV. Maybe because it was inactive last year?")
        gddaccum_yp_list[v][y, thisYear_active_patch_indices] = tmp_gddaccum
        if save_figs: gddharv_yp_list[v][y, thisYear_active_patch_indices] = tmp_gddharv
        
    skip_patches_for_isel_nan_lastyear = skip_patches_for_isel_nan
    if y==1:
        break

print("Done")


# %% Get and grid mean GDDs in GGCMI growing season

longname_prefix = "GDD harvest target for "

print('Getting and gridding mean GDDs...')
gdd_maps_ds = yp_list_to_ds(gddaccum_yp_list, h1_ds, h1_incl_ds, sdates_rx, longname_prefix)
if save_figs: gddharv_maps_ds = yp_list_to_ds(gddharv_yp_list, h1_ds, h1_incl_ds, sdates_rx, longname_prefix)

# Fill NAs with dummy values
dummy_fill = -1
gdd_fill0_maps_ds = gdd_maps_ds.fillna(0)
gdd_maps_ds = gdd_maps_ds.fillna(dummy_fill)
print('Done getting and gridding means.')

# Add dummy variables for crops not actually simulated
print("Adding dummy variables...")
# Unnecessary?
template_ds = xr.open_dataset(sdate_inFile, decode_times=True)
all_vars = [v.replace("sdate","gdd") for v in template_ds if "sdate" in v]
all_longnames = [template_ds[v].attrs["long_name"].replace("Planting day ", longname_prefix) + " (dummy)" for v in template_ds if "sdate" in v]
dummy_vars = [v for v in all_vars if v not in gdd_maps_ds]
def make_dummy(thisCrop_gridded, addend):
    dummy_gridded = thisCrop_gridded
    dummy_gridded.values = dummy_gridded.values*0 + addend
    return dummy_gridded
for v in gdd_maps_ds:
    thisCrop_gridded = gdd_maps_ds[v].copy()
    thisCrop_fill0_gridded = gdd_fill0_maps_ds[v].copy()
    break
dummy_gridded = make_dummy(thisCrop_gridded, -1)
dummy_gridded0 = make_dummy(thisCrop_fill0_gridded, 0)

for v, thisVar in enumerate(dummy_vars):
    if thisVar in gdd_maps_ds:
        raise RuntimeError(f'{thisVar} is already in gdd_maps_ds. Why overwrite it with dummy?')
    dummy_gridded.name = thisVar
    dummy_gridded.attrs["long_name"] = all_longnames[v]
    gdd_maps_ds[thisVar] = dummy_gridded
    dummy_gridded0.name = thisVar
    dummy_gridded0.attrs["long_name"] = all_longnames[v]
    gdd_fill0_maps_ds[thisVar] = dummy_gridded0

# Add lon/lat attributes
def add_lonlat_attrs(ds):
    ds.lon.attrs = {\
        "long_name": "coordinate_longitude",
        "units": "degrees_east"}
    ds.lat.attrs = {\
        "long_name": "coordinate_latitude",
        "units": "degrees_north"}
    return ds
gdd_maps_ds = add_lonlat_attrs(gdd_maps_ds)
gdd_fill0_maps_ds = add_lonlat_attrs(gdd_fill0_maps_ds)
if save_figs: gddharv_maps_ds = add_lonlat_attrs(gddharv_maps_ds)

print("Done.")


# %% Save before/after map and boxplot figures, if doing so

# layout = "3x1"
layout = "2x2"
bin_width = 15
lat_bin_edges = np.arange(0, 91, bin_width)

fontsize_titles = 18
fontsize_axislabels = 15
fontsize_ticklabels = 15

vegtypes_included = h1_ds.vegtype_str.values[[i for i,c in enumerate(gddaccum_yp_list) if not isinstance(c,type(None))]]

def make_map(ax, this_map, this_title, vmax, bin_width, fontsize_ticklabels, fontsize_titles): 
    im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
            this_map, shading="auto",
            vmin=0, vmax=vmax)
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(this_title, fontsize=fontsize_titles)
    cbar = plt.colorbar(im1, orientation="horizontal", fraction=0.1, pad=0.02)
    cbar.ax.tick_params(labelsize=fontsize_ticklabels)
    
    ticks = np.arange(-60, 91, bin_width)
    ticklabels = [str(x) for x in ticks]
    for i,x in enumerate(ticks):
        if x%2:
            ticklabels[i] = ''
    plt.yticks(np.arange(-60,91,15), labels=ticklabels,
               fontsize=fontsize_ticklabels)
    plt.axis('off')
    
def get_non_nans(in_da, fillValue):
    in_da = in_da.where(in_da != fillValue)
    return in_da.values[~np.isnan(in_da.values)]

def set_boxplot_props(bp, color):
    linewidth = 3
    plt.setp(bp['boxes'], color=color, linewidth=linewidth)
    plt.setp(bp['whiskers'], color=color, linewidth=linewidth)
    plt.setp(bp['caps'], color=color, linewidth=linewidth)
    plt.setp(bp['medians'], color=color, linewidth=linewidth)
    plt.setp(bp['fliers'], markeredgecolor=color, markersize=6, linewidth=linewidth, markeredgewidth=linewidth/2)

Nbins = len(lat_bin_edges)-1
bin_names = ["All"]
for b in np.arange(Nbins):
    lower = lat_bin_edges[b]
    upper = lat_bin_edges[b+1]
    bin_names.append(f"{lower}–{upper}")
    
color_old = '#beaed4'
color_new = '#7fc97f'
def make_plot(data, offset):
    linewidth = 1.5
    offset = 0.4*offset
    bpl = plt.boxplot(data, positions=np.array(range(len(data)))*2.0+offset, widths=0.6, 
                      boxprops=dict(linewidth=linewidth), whiskerprops=dict(linewidth=linewidth), 
                      capprops=dict(linewidth=linewidth), medianprops=dict(linewidth=linewidth),
                      flierprops=dict(markeredgewidth=0.5))
    return bpl

if save_figs:
    
    # Maps
    ny = 3
    nx = 1
    print("Making before/after maps...")
    for v, vegtype_str in enumerate(vegtypes_included):
        vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
        thisVar = f"gdd1_{vegtype_int}"
        print(f"   {vegtype_str} ({vegtype_int})...")
        
        
        # Maps #####################
        
        gdd_map = gdd_maps_ds[thisVar].isel(time=0, drop=True)
        gdd_map_yx = gdd_map.where(gdd_map != dummy_fill)
        gddharv_map = gddharv_maps_ds[thisVar]
        if "time" in gddharv_map.dims:
            gddharv_map = gddharv_map.isel(time=0, drop=True)
        gddharv_map_yx = gddharv_map.where(gddharv_map != dummy_fill)
                
        vmax = max(np.max(gdd_map_yx), np.max(gddharv_map_yx))
        
        # Set up figure and first subplot
        if layout == "3x1":
            fig = plt.figure(figsize=(7.5,14))
            ax = fig.add_subplot(ny,nx,1,projection=ccrs.PlateCarree())
        elif layout == "2x2":
            fig = plt.figure(figsize=(24,12))
            spec = fig.add_gridspec(nrows=2, ncols=2,
                                    width_ratios=[0.4,0.6])
            ax = fig.add_subplot(spec[0,0],projection=ccrs.PlateCarree())
        else:
            raise RuntimeError(f"layout {layout} not recognized")
        
        thisMin = int(np.round(np.nanmin(gddharv_map_yx)))
        thisMax = int(np.round(np.nanmax(gddharv_map_yx)))
        thisTitle = f"{vegtype_str}: Old (range {thisMin}–{thisMax})"
        make_map(ax, gddharv_map_yx, thisTitle, vmax, bin_width,
                 fontsize_ticklabels, fontsize_titles)
        
        if layout == "3x1":
            ax = fig.add_subplot(ny,nx,2,projection=ccrs.PlateCarree())
        elif layout == "2x2":
            ax = fig.add_subplot(spec[1,0],projection=ccrs.PlateCarree())
        else:
            raise RuntimeError(f"layout {layout} not recognized")
        thisMin = int(np.round(np.nanmin(gdd_map_yx)))
        thisMax = int(np.round(np.nanmax(gdd_map_yx)))
        thisTitle = f"{vegtype_str}: New (range {thisMin}–{thisMax})"
        make_map(ax, gdd_map_yx, thisTitle, vmax, bin_width,
                 fontsize_ticklabels, fontsize_titles)
        
        # Boxplots #####################
        
        gdd_vector = get_non_nans(gdd_map, dummy_fill)
        gddharv_vector = get_non_nans(gddharv_map, dummy_fill)
        
        lat_abs = np.abs(gdd_map.lat.values)
        gdd_bybin_old = [gddharv_vector]
        gdd_bybin_new = [gdd_vector]
        for b in np.arange(Nbins):
            lower = lat_bin_edges[b]
            upper = lat_bin_edges[b+1]
            lat_inds = np.where((lat_abs>=lower) & (lat_abs<upper))[0]
            # gdd_map_thisBin = gdd_map.where(gdd_map.lat>=lower )
            gdd_vector_thisBin = get_non_nans(gdd_map[lat_inds,:], dummy_fill)
            gddharv_vector_thisBin = get_non_nans(gddharv_map[lat_inds,:], dummy_fill)
            gdd_bybin_old.append(gddharv_vector_thisBin)
            gdd_bybin_new.append(gdd_vector_thisBin)
                
        if layout == "3x1":
            ax = fig.add_subplot(ny,nx,3)
        elif layout == "2x2":
            ax = fig.add_subplot(spec[:,1])
        else:
            raise RuntimeError(f"layout {layout} not recognized")

        bpl = make_plot(gdd_bybin_old, -1)
        bpr = make_plot(gdd_bybin_new, 1)
        set_boxplot_props(bpl, color_old)
        set_boxplot_props(bpr, color_new)
        
        # draw temporary lines to create a legend
        plt.plot([], c=color_old, label='Old')
        plt.plot([], c=color_new, label='New')
        plt.legend(fontsize=fontsize_titles)
        
        plt.xticks(range(0, len(bin_names) * 2, 2), bin_names,
                   fontsize=fontsize_ticklabels)
        plt.yticks(fontsize=fontsize_ticklabels)
        plt.xlabel("|latitude| zone", fontsize=fontsize_axislabels)
        plt.ylabel("Growing degree-days", fontsize=fontsize_axislabels)
        plt.title(f"Zonal changes: {vegtype_str}", fontsize=fontsize_titles)
        
        outfile = f"{outdir_figs}/{thisVar}_{vegtype_str}_gs{y1}-{yN}.png"
        plt.savefig(outfile, dpi=300, transparent=False, facecolor='white', \
            bbox_inches='tight')
        plt.close()

    print("Done.")
    

# %% Check that all cells that had sdates are represented
print("Checking that all cells that had sdates are represented...")

sdates_grid = utils.grid_one_variable(dates_incl_ds, 'SDATES')

all_ok = True
for vt_str in vegtypes_included:
    vt_int = utils.ivt_str2int(vt_str)
    # print(f"{vt_int}: {vt_str}")
    
    map_gdd = gdd_maps_ds[f"gdd1_{vt_int}"].isel(time=0, drop=True)
    map_sdate = sdates_grid.isel(mxsowings=0, drop=True).sel(ivt_str=vt_str, drop=True)
    if "time" in map_sdate.dims:
        map_sdate = map_sdate.isel(time=0, drop=True)
    
    ok_gdd = map_gdd.where(map_gdd >= 0).notnull().values
    ok_sdate = map_sdate.where(map_sdate > 0).notnull().values
    missing_both = np.bitwise_and(np.bitwise_not(ok_gdd), np.bitwise_not(ok_sdate))
    ok_both = np.bitwise_and(ok_gdd, ok_sdate)
    ok_both = np.bitwise_or(ok_both, missing_both)
    if np.any(np.bitwise_not(ok_both)):
        all_ok = False
        gdd_butnot_sdate = np.bitwise_and(ok_gdd, np.bitwise_not(ok_sdate))
        sdate_butnot_gdd = np.bitwise_and(ok_sdate, np.bitwise_not(ok_gdd))
        if np.any(gdd_butnot_sdate):
            print(f"   {vt_int} {vt_str}: {np.sum(gdd_butnot_sdate)} cells in GDD but not sdate")
        if np.any(sdate_butnot_gdd):
            print(f"   {vt_int} {vt_str}: {np.sum(sdate_butnot_gdd)} cells in sdate but not GDD")

if not all_ok:
    print("❌ Mismatch between sdate and GDD outputs")
else:
    print("✅ All sdates have GDD and vice versa")


# %% Save to netCDF
print("Saving...")

# Get output file path
if not os.path.exists(outdir):
    os.makedirs(outdir)
datestr = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = outdir + "gdds_" + datestr + ".nc"
outfile_fill0 = outdir + "gdds_fill0_" + datestr + ".nc"

def save_gdds(sdate_inFile, hdate_inFile, outfile, gdd_maps_ds, sdates_rx):
    # Set up output file from template (i.e., prescribed sowing dates).
    template_ds = xr.open_dataset(sdate_inFile, decode_times=True)
    for v in template_ds:
        if "sdate" in v:
            template_ds = template_ds.drop(v)
    template_ds.to_netcdf(path=outfile, format="NETCDF3_CLASSIC")
    template_ds.close()

    # Add global attributes
    comment = f"Derived from CLM run plus crop calendar input files {os.path.basename(sdate_inFile) and {os.path.basename(hdate_inFile)}}."
    gdd_maps_ds.attrs = {\
        "author": "Sam Rabin (sam.rabin@gmail.com)",
        "comment": comment,
        "created": dt.datetime.now().astimezone().isoformat()
        }

    # Add time_bounds
    gdd_maps_ds["time_bounds"] = sdates_rx.time_bounds

    # Save cultivar GDDs
    gdd_maps_ds.to_netcdf(outfile, mode="a", format="NETCDF3_CLASSIC")

save_gdds(sdate_inFile, hdate_inFile, outfile, gdd_maps_ds, sdates_rx)
save_gdds(sdate_inFile, hdate_inFile, outfile_fill0, gdd_fill0_maps_ds, sdates_rx)

print("All done!")


# %% Misc.

# # %% View GDD for a single crop and cell

# tmp = accumGDD_ds[clm_gdd_var].values
# incl = np.bitwise_and(accumGDD_ds.patches1d_lon.values==270, accumGDD_ds.patches1d_lat.values==40)
# tmp2 = tmp[:,incl]
# plt.plot(tmp2)



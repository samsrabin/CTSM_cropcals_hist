# Where is the script running?
import socket
hostname = socket.gethostname()

# Import the CTSM Python utilities
import sys
if hostname == "Sams-2021-MacBook-Pro.local":
    sys.path.append("/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/")
    import utils
else:
    # Only possible because I have export PYTHONPATH=$HOME in my .bash_profile
    from ctsm_python_gallery_myfork.ctsm_py import utils

import numpy as np
import xarray as xr
import warnings
import os
import glob
import cftime


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


def import_rx_dates(s_or_h, date_inFile, incl_patches1d_itype_veg, mxsowings):
    
    if isinstance(date_inFile, xr.Dataset):
        return date_inFile
    elif not isinstance(date_inFile, str):
        raise RuntimeError(f'Importing {s_or_h}dates_rx: Expected date_inFile to be str or DataArray, not {type(date_inFile)}')
    
    # Which vegetation types were simulated?
    itype_veg_toImport = np.unique(incl_patches1d_itype_veg)

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
    
# Get and grid mean GDDs in GGCMI growing season
def yp_list_to_ds(yp_list, daily_ds, incl_vegtypes_str, dates_rx, longname_prefix):
    
    # Get means
    warnings.filterwarnings("ignore", message="Mean of empty slice") # Happens when you do np.nanmean() of an all-NaN array (or slice, if doing selected axis/es)
    p_list = [np.nanmean(x, axis=0) if not isinstance(x, type(None)) else x for x in yp_list]
    warnings.filterwarnings("always", message="Mean of empty slice")
    
    if isinstance(incl_vegtypes_str, xr.DataArray):
        incl_vegtypes_str = incl_vegtypes_str.values
    
    # Grid
    ds_out = xr.Dataset()
    for c, ra in enumerate(p_list):
        if isinstance(ra, type(None)):
            continue
        thisCrop_str = incl_vegtypes_str[c]
        print(f'   {thisCrop_str}...')
        newVar = f"gdd1_{utils.ivt_str2int(thisCrop_str)}"
        ds = daily_ds.isel(patch=np.where(daily_ds.patches1d_itype_veg_str.values==thisCrop_str)[0])
        template_da = ds.patches1d_itype_veg_str
        da = xr.DataArray(data = ra,
                          coords = template_da.coords,
                          attrs = {'units': 'GDD',
                                   'long_name': f'{longname_prefix}{thisCrop_str}'})
        
        # Grid this crop
        ds['tmp'] = da
        da_gridded = utils.grid_one_variable(ds, 'tmp', vegtype=thisCrop_str).squeeze(drop=True)
        
        # Add singleton time dimension and save to output Dataset
        da_gridded = da_gridded.expand_dims(time = dates_rx.time)
        ds_out[newVar] = da_gridded
        
    return ds_out




def import_and_process_1yr(y1, yN, y, thisYear, sdates_rx, hdates_rx, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices_list, incorrectly_daily, gddharv_in_h3, save_figs, indir, incl_vegtypes_str_in, h1_ds_file):
    print(f'netCDF year {thisYear}...')
    
    # Get h2 file (list)
    if indir[-1] != os.path.sep:
        indir = indir + os.path.sep
    h2_pattern = indir + "*h2.*"
    h2_filelist = glob.glob(h2_pattern)
    if not h2_filelist:
        raise RuntimeError(f"No files found matching pattern: {h2_pattern}")
    
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
    else:
        dates_ds = dates_ds.isel(time=0)
    
    # Make sure NaN masks match
    sdates_all_nan = np.sum(~np.isnan(dates_ds.SDATES.values), axis=dates_ds.SDATES.dims.index('mxsowings')) == 0
    hdates_all_nan = np.sum(~np.isnan(dates_ds.HDATES.values), axis=dates_ds.HDATES.dims.index('mxharvests')) == 0
    N_unmatched_nans = np.sum(sdates_all_nan != hdates_all_nan)
    if N_unmatched_nans > 0:
        raise RuntimeError("Output SDATE and HDATE NaN masks do not match.")
    if np.sum(~np.isnan(dates_ds.SDATES.values)) == 0:
        raise RuntimeError("All SDATES are NaN!")
    
    # Just work with non-NaN patches for now
    skip_patches_for_isel_nan = np.where(sdates_all_nan)[0]
    incl_patches_for_isel_nan = np.where(~sdates_all_nan)[0]
    different_nan_mask = y > 0 and not np.array_equal(skip_patches_for_isel_nan_lastyear, skip_patches_for_isel_nan)
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
    incl_patches1d_itype_veg = dates_incl_ds.patches1d_itype_veg
    
    if y==0:
        incl_vegtypes_str = dates_incl_ds.vegtype_str.values
    else:
        incl_vegtypes_str = incl_vegtypes_str_in
        if isinstance(incl_vegtypes_str, xr.DataArray):
            incl_vegtypes_str = incl_vegtypes_str.values
        if isinstance(incl_vegtypes_str, np.ndarray):
            incl_vegtypes_str = list(incl_vegtypes_str)
        if incl_vegtypes_str != list(dates_incl_ds.vegtype_str.values):
            raise RuntimeError(f'Included veg types differ. Previously {incl_vegtypes_str}, now {dates_incl_ds.vegtype_str.values}')
        
    if np.sum(~np.isnan(dates_incl_ds.SDATES.values)) == 0:
        raise RuntimeError("All SDATES are NaN after ignoring those patches!")
    
    # Some patches can have -1 sowing date?? Hopefully just an artifact of me incorrectly saving SDATES/HDATES daily.
    mxsowings = dates_ds.dims['mxsowings']
    mxsowings_dim = dates_ds.SDATES.dims.index('mxsowings')
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
    sdates_rx = import_rx_dates("s", sdates_rx, incl_patches1d_itype_veg, mxsowings)
        
    check_sdates(dates_incl_ds, sdates_rx)
    
    # Import hdates, if needed
    hdates_rx = import_rx_dates("h", hdates_rx, incl_patches1d_itype_veg, mxsowings) # Yes, mxsowings even when importing harvests
    # Determine cells where growing season crosses new year
    grows_across_newyear = hdates_rx < sdates_rx
        
    print(f"   Importing accumulated GDDs...")
    clm_gdd_var = "GDDACCUM"
    myVars = [clm_gdd_var]
    if save_figs and not gddharv_in_h3:
        myVars.append("GDDHARV")
    h1_ds = utils.import_ds(glob.glob(indir + f"*h1.{thisYear-1}-01-01*"), myVars=myVars, myVegtypes=utils.define_mgdcrop_list(), myVars_missing_ok=['GDDHARV'])
    if save_figs and 'GDDHARV' not in h1_ds:
        if not gddharv_in_h3:
            print('Trying to get GDDHARV from h3 file(s) instead.')
        try:
            h3_ds = utils.import_ds(glob.glob(indir + f"*h3.{thisYear-1}-01-01*"), myVars=['GDDHARV'], myVegtypes=utils.define_mgdcrop_list())
            h1_ds['GDDHARV'] = h3_ds['GDDHARV']
            if not gddharv_in_h3:
                print('Success! Will look in h3 files from now on.')
                gddharv_in_h3 = True
        except:
            print('Unable to import GDDHARV from h1 or h3 files. Disabling save_figs.')
            save_figs = False

    # Restrict to patches we're including
    if skipping_patches_for_isel_nan:
        if not np.array_equal(dates_ds.patch.values, h1_ds.patch.values):
            raise RuntimeError("dates_ds and h1_ds don't have the same patch list!")
        h1_incl_ds = h1_ds.isel(patch=incl_patches_for_isel_nan)
    else:
        h1_incl_ds = h1_ds

    if not np.any(h1_incl_ds[clm_gdd_var].values != 0):
        raise RuntimeError(f"All {clm_gdd_var} values are zero!")
    
    # Get standard datetime axis for outputs
    Nyears = yN - y1 + 1
    
    if len(gddaccum_yp_list)==0:
        lastYear_active_patch_indices_list = [None for vegtype_str in h1_incl_ds.vegtype_str.values]
        gddaccum_yp_list = [None for vegtype_str in h1_incl_ds.vegtype_str.values]
        if save_figs: gddharv_yp_list = [None for vegtype_str in h1_incl_ds.vegtype_str.values]
    
    incl_vegtype_indices = []
    for v, vegtype_str in enumerate(h1_incl_ds.vegtype_str.values):

        # Skipping Miscanthus because it seems to never be harvested even though it is sown. This causes problems in NaN mask check.
        if 'miscanthus' in vegtype_str:
            print(f'      SKIPPING {vegtype_str}')
            continue
        
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
        
        if isinstance(gddaccum_yp_list[v], type(None)):
            gddaccum_yp_list[v] = np.full((Nyears+1, len(thisCrop_full_patchlist)), np.nan)
            if save_figs: gddharv_yp_list[v] = np.full((Nyears+1, len(thisCrop_full_patchlist)), np.nan)
        
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
        # Sort patches back to correct order
        if not np.all(thisCrop_gddaccum_da.patch.values[:-1] <= thisCrop_gddaccum_da.patch.values[1:]):
            raise RuntimeError("This code depends on DataArray patch list being sorted.")
        sortorder = np.argsort(patches)
        i_patches = list(np.array(i_patches)[np.array(sortorder)])
        # Select using the indexing tuple
        gddaccum_atharv_p = thisCrop_gddaccum_da.values[(i_times, i_patches)]
        if save_figs: gddharv_atharv_p = thisCrop_gddharv_da.values[(i_times, i_patches)]
        if np.any(np.isnan(gddaccum_atharv_p)):
            print(f"         ❗ {np.sum(np.isnan(gddaccum_atharv_p))}/{len(gddaccum_atharv_p)} NaN after extracting GDDs accumulated at harvest")
        if save_figs and np.any(np.isnan(gddharv_atharv_p)):
            print(f"         ❗ {np.sum(np.isnan(gddharv_atharv_p))}/{len(gddharv_atharv_p)} NaN after extracting GDDHARV")
                
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
            lastYear_active_patch_indices = lastYear_active_patch_indices_list[v]
            where_gs_lastyr = np.where(thisCrop_sdates_rx > thisCrop_hdates_rx)[0]
            active_thisYear_where_gs_lastyr_indices = [thisYear_active_patch_indices[x] for x in where_gs_lastyr]
            if not np.array_equal(lastYear_active_patch_indices, thisYear_active_patch_indices):
                if incorrectly_daily:
                    print("         ❗ This year's active patch indices differ from last year's. Allowing because this might just be an artifact of incorrectly daily outputs, BUT RESULTS MUST NOT BE TRUSTED.")
                else:
                    raise RuntimeError("This year's active patch indices differ from last year's.")
            # Make sure we're not about to overwrite any existing values.
            if np.any(~np.isnan(gddaccum_yp_list[v][y-1, active_thisYear_where_gs_lastyr_indices])):
                if incorrectly_daily:
                    print("         ❗ Unexpected non-NaN for last season's GDD accumulation. Allowing because this might just be an artifact of incorrectly daily outputs, BUT RESULTS MUST NOT BE TRUSTED.")
                else:
                    raise RuntimeError("Unexpected non-NaN for last season's GDD accumulation")
            if save_figs and np.any(~np.isnan(gddharv_yp_list[v][y-1, active_thisYear_where_gs_lastyr_indices])):
                if incorrectly_daily:
                    print("         ❗ Unexpected non-NaN for last season's GDDHARV. Allowing because this might just be an artifact of incorrectly daily outputs, BUT RESULTS MUST NOT BE TRUSTED.")
                else:
                    raise RuntimeError("Unexpected non-NaN for last season's GDDHARV")
            # Fill.
            gddaccum_yp_list[v][y-1, active_thisYear_where_gs_lastyr_indices] = gddaccum_atharv_p[where_gs_lastyr]
            if save_figs: gddharv_yp_list[v][y-1, active_thisYear_where_gs_lastyr_indices] = gddharv_atharv_p[where_gs_lastyr]
            # Last year's season should be filled out now; make sure.
            if np.any(np.isnan(gddaccum_yp_list[v][y-1, active_thisYear_where_gs_lastyr_indices])):
                if incorrectly_daily:
                    print("         ❗ Unexpected NaN for last season's GDD accumulation. Allowing because this might just be an artifact of incorrectly daily outputs, BUT RESULTS MUST NOT BE TRUSTED.")
                else:
                    raise RuntimeError("Unexpected NaN for last season's GDD accumulation.")
            if save_figs and np.any(np.isnan(gddharv_yp_list[v][y-1, active_thisYear_where_gs_lastyr_indices])):
                if incorrectly_daily:
                    print("         ❗ Unexpected NaN for last season's GDDHARV. Allowing because this might just be an artifact of incorrectly daily outputs, BUT RESULTS MUST NOT BE TRUSTED.")
                else:
                    raise RuntimeError("Unexpected NaN for last season's GDDHARV.")
        gddaccum_yp_list[v][y, thisYear_active_patch_indices] = tmp_gddaccum
        if save_figs: gddharv_yp_list[v][y, thisYear_active_patch_indices] = tmp_gddharv
        
        # Make sure that NaN masks are the same for this year's sdates and 'filled-out' GDDs from last year
        if y>0:
            nanmask_output_sdates = np.isnan(dates_ds.SDATES.isel(mxsowings=0, patch=np.where(dates_ds.patches1d_itype_veg_str==vegtype_str)[0]).values)
            nanmask_output_gdds_lastyr = np.isnan(gddaccum_yp_list[v][y-1,:])
            if not np.array_equal(nanmask_output_gdds_lastyr, nanmask_output_sdates):
                if incorrectly_daily:
                    print("         ❗ NaN masks differ between this year's sdates and 'filled-out' GDDs from last year. Allowing because this might just be an artifact of incorrectly daily outputs, BUT RESULTS MUST NOT BE TRUSTED.")
                else:
                    raise RuntimeError("NaN masks differ between this year's sdates and 'filled-out' GDDs from last year")
        lastYear_active_patch_indices_list[v] = thisYear_active_patch_indices
                
    skip_patches_for_isel_nan_lastyear = skip_patches_for_isel_nan
    
    # Could save space by only saving variables needed for gridding
    print('   Saving h1_ds...')
    h1_ds.to_netcdf(h1_ds_file)
        
    return h1_ds, sdates_rx, hdates_rx, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices_list, incorrectly_daily, gddharv_in_h3, incl_vegtypes_str, incl_patches1d_itype_veg, mxsowings
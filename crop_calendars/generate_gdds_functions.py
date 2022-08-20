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


import sys
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

import numpy as np
import xarray as xr
import cartopy.crs as ccrs


# After importing a file, restrict it to years of interest.
def check_and_trim_years(y1, yN, ds_in):
    ### In annual outputs, file with name Y is actually results from year Y-1.
    ### Note that time values refer to when it was SAVED. So 1981-01-01 is for year 1980.
    
    def get_year_from_cftime(cftime_date):
        # Subtract 1 because the date for annual files is when it was SAVED
        return cftime_date.year - 1

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


 # Make sure GDDACCUM_PERHARV is always <= HUI_PERHARV
def check_gddaccum_le_hui(this_ds, msg_txt=" ", both_nan_ok = False, throw_error=False):
    gdd_lt_hui = this_ds["GDDACCUM_PERHARV"] <= this_ds["HUI_PERHARV"]
    if both_nan_ok:
        gdd_lt_hui = gdd_lt_hui | (np.isnan(this_ds["GDDACCUM_PERHARV"]) & np.isnan(this_ds["HUI_PERHARV"]))
    if np.all(gdd_lt_hui):
        print(f"✅{msg_txt}GDDACCUM_PERHARV always <= HUI_PERHARV")
    else: 
        msg = f"❌{msg_txt}GDDACCUM_PERHARV *not* always <= HUI_PERHARV"
        if throw_error:
            print(msg)
        else:
            raise RuntimeError(msg)


def check_rx_obeyed(vegtype_list, rx_ds, dates_ds, which_ds, output_var, gdd_min=None):
    all_ok = 2
    diff_str_list = []
    gdd_tolerance = 0
    for vegtype_str in vegtype_list:
        ds_thisVeg = dates_ds.isel(patch=np.where(dates_ds.patches1d_itype_veg_str == vegtype_str)[0])
        patch_inds_lon_thisVeg = ds_thisVeg.patches1d_ixy.values.astype(int) - 1
        patch_inds_lat_thisVeg = ds_thisVeg.patches1d_jxy.values.astype(int) - 1
        patch_lons_thisVeg = ds_thisVeg.patches1d_lon
        patch_lats_thisVeg = ds_thisVeg.patches1d_lat
    
        vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
        rx_da = rx_ds[f"gs1_{vegtype_int}"]
        rx_array = rx_da.values[patch_inds_lat_thisVeg,patch_inds_lon_thisVeg]
        sim_array = ds_thisVeg[output_var].values
        sim_array_dims = ds_thisVeg[output_var].dims
        
        # Account for GDD harvest threshold minimum set in PlantCrop()
        if output_var=="GDDHARV_PERHARV":
            if gdd_min == None:
                raise RuntimeError(f"gdd_min must be provided when doing check_rx_obeyed() for GDDHARV_PERHARV")
            rx_array[rx_array < gdd_min] = gdd_min
        
        if np.any(sim_array != rx_array):
            diff_array = sim_array - rx_array
            
            # Allow negative GDDHARV values when harvest occurred because sowing was scheduled for the next day
            if output_var=="GDDHARV_PERHARV":
                diff_array = np.ma.masked_array(diff_array, mask= \
                    (diff_array < 0) & 
                    (ds_thisVeg["HARVEST_REASON_PERHARV"].values==5))
    
            if np.any(np.abs(diff_array[abs(diff_array) > 0]) > 0):
                min_diff, minLon, minLat, minGS = get_extreme_info(diff_array, np.min, sim_array_dims, dates_ds.gs, patch_lons_thisVeg, patch_lats_thisVeg)
                max_diff, maxLon, maxLat, maxGS = get_extreme_info(diff_array, np.max, sim_array_dims, dates_ds.gs, patch_lons_thisVeg, patch_lats_thisVeg)
                
                diffs_eg_txt = f"{vegtype_str} ({vegtype_int}): diffs range {min_diff} (lon {minLon}, lat {minLat}, year {minGS}) to {max_diff} (lon {maxLon}, lat {maxLat}, gs {maxGS})"
                if output_var=="GDDHARV_PERHARV" and np.max(abs(diff_array)) <= gdd_tolerance:
                    all_ok = 1
                    diff_str_list.append(f"   {diffs_eg_txt}")
                else:
                    all_ok = 0
                    break
    
    if all_ok == 2:
        print(f"✅ dates_ds{which_ds}: Prescribed {output_var} always obeyed")
    elif all_ok == 1:
        # print(f"🟨 dates_ds{which_ds}: Prescribed {output_var} *not* always obeyed, but acceptable:")
        # for x in diff_str_list: print(x)
        print(f"🟨 dates_ds{which_ds}: Prescribed {output_var} *not* always obeyed, but acceptable (diffs <= {gdd_tolerance})")
    else:
        print(f"❌ dates_ds{which_ds}: Prescribed {output_var} *not* always obeyed. E.g., {diffs_eg_txt}")
        

# Set up empty Dataset with time axis as "gs" (growing season) instead of what CLM puts out.
# Includes all the same variables as the input dataset, minus any that had dimensions mxsowings or mxharvests.
def set_up_ds_with_gs_axis(ds_in):
    # Get the data variables to include in the new dataset
    data_vars = dict()
    for v in ds_in.data_vars:
        if not any([x in ["mxsowings", "mxharvests"] for x in ds_in[v].dims]):
            data_vars[v] = ds_in[v]
    # Set up the new dataset
    gs_years = [t.year-1 for t in ds_in.time.values[:-1]]
    coords = ds_in.coords
    coords["gs"] = gs_years
    ds_out = xr.Dataset(data_vars=data_vars,
                        coords=coords,
                        attrs=ds_in.attrs)
    return ds_out


def convert_axis_time2gs(Ngs, this_ds, myVars):
    ### Checks ###
    # Relies on max harvests per year >= 2
    mxharvests = len(this_ds.mxharvests)
    if mxharvests < 2:
        raise RuntimeError(f"convert_axis_time2gs() assumes max harvests per year == 2, not {mxharvests}")
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
    for v in [x for x in myVars if x=="HDATES" or "PERHARV" in x]:
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
    for v in [x for x in myVars if x!="SDATES"]:
        new_ds_gs = extract_gs_timeseries(new_ds_gs, v, this_ds[v], hdate_included, Npatches, Ngs)
    
    ### Save additional variables ###
    new_ds_gs = new_ds_gs.assign(variables=data_vars)
    new_ds_gs.coords["lon"] = this_ds.coords["lon"]
    new_ds_gs.coords["lat"] = this_ds.coords["lat"]
    
    return new_ds_gs


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


def get_extreme_info(diff_array, rx_array, mxn, dims, gs, patches1d_lon, patches1d_lat):
    if mxn == np.min:
        diff_array = np.ma.masked_array(diff_array, mask=(np.abs(diff_array) == 0))
    themxn = mxn(diff_array)
    
    # Find the first patch-gs that has the mxn value
    matching_indices = np.where(diff_array == themxn)
    first_indices = [x[0] for x in matching_indices]
    
    # Get the lon, lat, and growing season of that patch-gs
    p = first_indices[dims.index("patch")]
    thisLon = patches1d_lon.values[p]
    thisLat = patches1d_lat.values[p]
    s = first_indices[dims.index("gs")]
    thisGS = gs.values[s]
    
    # Get the prescribed value for this patch-gs
    rx_array = rx_array.sel(lon=thisLon, lat=thisLat)
    if "gs" in rx_array.dims:
        rx_array
    
    return round(themxn, 3), round(thisLon, 3), round(thisLat,3), thisGS


# Get growing season lengths from a DataArray of hdate-sdate
def get_gs_len_da(this_da):
    tmp = this_da.values
    tmp[tmp < 0] = 365 + tmp[tmp < 0]
    this_da.values = tmp
    return this_da


def get_Nharv(array_in, these_dims):
    # Sum over time and mxevents to get number of events in time series for each patch
    sum_indices = tuple(these_dims.index(x) for x in ["time", "mxharvests"])
    Nevents_eachPatch = np.sum(array_in > 0, axis=sum_indices)
    return Nevents_eachPatch


def get_reason_freq_map(Ngs, thisCrop_gridded, reason):
    map_yx = np.sum(thisCrop_gridded==reason, axis=0, keepdims=False) / Ngs
    notnan_yx = np.bitwise_not(np.isnan(thisCrop_gridded.isel(gs=0, drop=True)))
    map_yx = map_yx.where(notnan_yx)
    return map_yx
 

# E.g. import_rx_dates("sdate", sdates_rx_file, dates_ds0_orig)
def import_rx_dates(var_prefix, date_inFile, dates_ds):
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
            thisVar = f"{var_prefix}{g+1}_{i}"
            date_varList = date_varList + [thisVar]

    ds = utils.import_ds(date_inFile, myVars=date_varList)
    
    for v in ds:
        ds = ds.rename({v: v.replace(var_prefix,"gs")})
    
    return ds


def make_axis(fig, ny, nx, n):
    ax = fig.add_subplot(ny,nx,n,projection=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False) # Turn off box outline
    return ax


def mask_immature(this_ds, this_vegtype, gridded_da):
    reason_gridded = utils.grid_one_variable(this_ds, "HARVEST_REASON_PERHARV", \
                vegtype=this_vegtype).squeeze(drop=True)
    gridded_da = gridded_da.where(reason_gridded == 1)
    return gridded_da


def set_firstharv_nan(this_ds, this_var, firstharv_nan_inds):
    this_da = this_ds[this_var]
    this_array = this_da.values
    this_array[0,0,firstharv_nan_inds] = np.nan
    this_da.values = this_array
    this_ds[this_var] = this_da
    return this_ds
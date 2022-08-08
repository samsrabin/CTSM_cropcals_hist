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


 # Make sure that, e.g., GDDACCUM_PERHARV is always <= HUI_PERHARV
def check_v0_le_v1(this_ds, vars, msg_txt=" ", both_nan_ok = False, throw_error=False):
    v0 = vars[0]
    v1 = vars[1]
    gdd_lt_hui = this_ds[v0] <= this_ds[v1]
    if both_nan_ok:
        gdd_lt_hui = gdd_lt_hui | (np.isnan(this_ds[v0]) & np.isnan(this_ds[v1]))
    if np.all(gdd_lt_hui):
        print(f"✅{msg_txt}{v0} always <= {v1}")
    else: 
        msg = f"❌{msg_txt}{v0} *not* always <= {v1}"
        gdd_lt_hui_vals = gdd_lt_hui.values
        p = np.where(~gdd_lt_hui_vals)[0][0]
        msg = msg + f"\ne.g., patch {p}: {this_ds.patches1d_itype_veg_str.values[p]}, lon {this_ds.patches1d_lon.values[p]} lat {this_ds.patches1d_lat.values[p]}:"
        msg = msg + f"\n{this_ds[v0].values[p,:]}"
        msg = msg + f"\n{this_ds[v1].values[p,:]}"
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


# Convert time*mxharvests axes to growingseason axis
def convert_axis_time2gs(this_ds, verbose=False, myVars=None, incl_orig=False):
    
    # For backwards compatibility.
    if "SDATES_PERHARV" not in this_ds:
        return convert_axis_time2gs_old(this_ds, myVars=myVars)
    # Otherwise...
    
    # How many non-NaN patch-seasons do we expect to have once we're done organizing things?
    Npatch = this_ds.dims["patch"]
    # Because some patches will be planted in the last year but not complete, we have to ignore any finalyear-planted seasons that do complete.
    Ngs = this_ds.dims["time"]-1
    expected_valid = Npatch*Ngs
    
    mxharvests =  this_ds.dims["mxharvests"]
    
    if verbose:
        print(f'Start: discrepancy of {np.sum(~np.isnan(this_ds.HDATES.values)) - expected_valid} patch-seasons')
    
    # Set all non-positive date values to NaN. These are seasons that were never harvested (or never started): "non-seasons."
    if this_ds.HDATES.dims != ("time", "mxharvests", "patch"):
        raise RuntimeError(f"This code relies on HDATES dims ('time', 'mxharvests', 'patch'), not {this_ds.HDATES.dims}")
    hdates_ymp = this_ds.HDATES.copy().where(this_ds.HDATES > 0).values
    hdates_pym = np.transpose(hdates_ymp.copy(), (2,0,1))
    sdates_ymp = this_ds.SDATES_PERHARV.copy().where(this_ds.SDATES_PERHARV > 0).values
    sdates_pym = np.transpose(sdates_ymp.copy(), (2,0,1))
    hdates_pym[hdates_pym <= 0] = np.nan
    
    # Find years where patch was inactive
    inactive_py = np.transpose(
        np.isnan(this_ds.HDATES).all(dim="mxharvests").values \
        & np.isnan(this_ds.SDATES_PERHARV).all(dim="mxharvests").values)
    # Find seasons that were planted while the patch was inactive
    sown_inactive_py = inactive_py[:,:-1] & (hdates_pym[:,1:,0] < sdates_pym[:,1:,0])
    sown_inactive_py = np.concatenate((np.full((Npatch, 1), False),
                                       sown_inactive_py),
                                      axis=1)

    # "Ignore harvests from seasons sown (a) before this output began or (b) when the crop was inactive"
    first_season_before_first_year_p = hdates_pym[:,0,0] < sdates_pym[:,0,0]
    first_season_before_first_year_py = np.full(hdates_pym.shape[:-1], fill_value=False)
    first_season_before_first_year_py[:,0] = first_season_before_first_year_p
    sown_prerun_or_inactive_py = first_season_before_first_year_py | sown_inactive_py
    sown_prerun_or_inactive_pym = np.concatenate((np.expand_dims(sown_prerun_or_inactive_py, axis=2),
                                            np.full((Npatch, Ngs+1, mxharvests-1), False)),
                                           axis=2)
    where_sown_prerun_or_inactive_pym = np.where(sown_prerun_or_inactive_pym)
    hdates_pym[where_sown_prerun_or_inactive_pym] = np.nan
    sdates_pym[where_sown_prerun_or_inactive_pym] = np.nan
    if verbose:
        print(f'After "Ignore harvests from before this output began: discrepancy of {np.sum(~np.isnan(hdates_pym)) - expected_valid} patch-seasons')
    
    # We need to keep some non-seasons---it's possible that "the yearY growing season" never happened (sowing conditions weren't met), but we still need something there so that we can make an array of dimension Npatch*Ngs. We do this by changing those non-seasons from NaN to -Inf before doing the filtering and reshaping, after which we'll convert them back to NaNs.
    
    # "In years with no sowing, pretend the first no-harvest is meaningful, unless that was intentionally ignored above."
    sdates_orig_ymp = this_ds.SDATES.copy().values
    sdates_orig_pym = np.transpose(sdates_orig_ymp.copy(), (2,0,1))
    if mxharvests > 2:
        print("Warning: Untested with mxharvests > 2")
    hdates_pym2 = hdates_pym.copy()
    sdates_pym2 = sdates_pym.copy()
    nosow_py = np.all(~(sdates_orig_pym > 0), axis=2)
    first_season_before_first_year = hdates_pym[:,0,0] < sdates_pym[:,0,0]
    nosow_py_1st = nosow_py & np.isnan(hdates_pym[:,:,0]) \
        & ~np.tile(np.expand_dims(first_season_before_first_year, axis=1),
                   (1,Ngs+1))
    where_nosow_py_1st = np.where(nosow_py_1st)
    hdates_pym2[where_nosow_py_1st[0], where_nosow_py_1st[1], 0] = -np.inf
    sdates_pym2[where_nosow_py_1st[0], where_nosow_py_1st[1], 0] = -np.inf
    for h in np.arange(mxharvests - 1):
        where_nosow_py = np.where(nosow_py & np.any(~np.isnan(hdates_pym[:,:,0:h]), axis=2) & np.isnan(hdates_pym[:,:,h]))
        hdates_pym2[where_nosow_py[0], where_nosow_py[1], 1] = -np.inf
        sdates_pym2[where_nosow_py[0], where_nosow_py[1], 1] = -np.inf
        
    # "In years with sowing that are followed by inactive years, check whether the last sowing was harvested before the patch was deactivated. If not, pretend the LAST [easier to implement!] no-harvest is meaningful."
    sdates_orig_masked_pym = sdates_orig_pym.copy()
    sdates_orig_masked_pym[np.where(sdates_orig_masked_pym <= 0)] = np.nan
    last_sdate_firstNgs_py = np.nanmax(sdates_orig_masked_pym[:,:-1,:], axis=2)
    last_hdate_firstNgs_py = np.nanmax(hdates_pym2[:,:-1,:], axis=2)
    last_sowing_not_harvested_sameyear_firstNgs_py = \
        (last_hdate_firstNgs_py < last_sdate_firstNgs_py) \
        | np.isnan(last_hdate_firstNgs_py)
    inactive_lastNgs_py = inactive_py[:,1:]
    last_sowing_never_harvested_firstNgs_py = last_sowing_not_harvested_sameyear_firstNgs_py & inactive_lastNgs_py
    last_sowing_never_harvested_py = np.concatenate((last_sowing_never_harvested_firstNgs_py,
                                                     np.full((Npatch,1), False)),
                                                    axis=1)
    last_sowing_never_harvested_pym = np.concatenate((np.full((Npatch, Ngs+1, mxharvests-1), False),
                                                      np.expand_dims(last_sowing_never_harvested_py, axis=2)),
                                                     axis=2)
    where_last_sowing_never_harvested_pym = last_sowing_never_harvested_pym
    hdates_pym3 = hdates_pym2.copy()
    sdates_pym3 = sdates_pym2.copy()
    hdates_pym3[where_last_sowing_never_harvested_pym] = -np.inf
    sdates_pym3[where_last_sowing_never_harvested_pym] = -np.inf
    
    # Convert to growingseason axis
    def pym_to_pg(pym, quiet=False):
        pg = np.reshape(pym, (pym.shape[0],-1))
        ok_pg = pg[~np.isnan(pg)]
        if not quiet:
            print(f"{ok_pg.size} included; unique N seasons = {np.unique(np.sum(~np.isnan(pg), axis=1))}")
        return pg
    hdates_pg = pym_to_pg(hdates_pym3.copy(), quiet=~verbose)
    sdates_pg = pym_to_pg(sdates_pym3.copy(), quiet=True)
    if verbose:
        print(f'After "In years with no sowing, pretend the first no-harvest is meaningful: discrepancy of {np.sum(~np.isnan(hdates_pg)) - expected_valid} patch-seasons')
    
    # "Ignore any harvests that were planted in the final year, because some cells will have incomplete growing seasons for the final year."
    lastyear_complete_season = (hdates_pg[:,-mxharvests:] >= sdates_pg[:,-mxharvests:]) | np.isinf(hdates_pg[:,-mxharvests:])
    def ignore_lastyear_complete_season(pg, excl, mxharvests):
        tmp_L = pg[:,:-mxharvests]
        tmp_R = pg[:,-mxharvests:]
        tmp_R[np.where(excl)] = np.nan
        pg = np.concatenate((tmp_L, tmp_R), axis=1)
        return pg
    hdates_pg2 = ignore_lastyear_complete_season(hdates_pg.copy(), lastyear_complete_season, mxharvests)
    sdates_pg2 = ignore_lastyear_complete_season(sdates_pg.copy(), lastyear_complete_season, mxharvests)
    is_valid = ~np.isnan(hdates_pg2)
    discrepancy = np.sum(is_valid) - expected_valid
    unique_Nseasons = np.unique(np.sum(is_valid, axis=1))
    if verbose:
        print(f'After "Ignore any harvests that were planted in the final year, because other cells will have incomplete growing seasons for the final year": discrepancy of {discrepancy} patch-seasons')
        try:
            import pandas as pd
            bc = np.bincount(np.sum(is_valid, axis=1))
            bc = bc[bc>0]
            df = pd.DataFrame({"Ngs": unique_Nseasons, "Count": bc})
            print(df)
        except:
            print(f"unique N seasons = {unique_Nseasons}")
        print(" ")
    
    # Create Dataset with time axis as "gs" (growing season) instead of what CLM puts out
    if discrepancy == 0:
        this_ds_gs = set_up_ds_with_gs_axis(this_ds)
        for v in this_ds.data_vars:
            if this_ds[v].dims != ('time', 'mxharvests', 'patch') and (not myVars or v in myVars): 
                continue
            
            # Set invalid values to NaN
            da_yhp = this_ds[v].copy()
            da_yhp = da_yhp.where(~np.isneginf(da_yhp))
            
            # Remove the nans and reshape to patches*growingseasons
            da_pyh = da_yhp.transpose("patch", "time", "mxharvests")
            ar_pg = np.reshape(da_pyh.values, (this_ds.dims["patch"], -1))
            ar_valid_pg = np.reshape(ar_pg[is_valid], (this_ds.dims["patch"], Ngs))
            # Change -infs to nans
            ar_valid_pg[np.isinf(ar_valid_pg)] = np.nan
            # Save as DataArray to new Dataset, stripping _PERHARV from variable name
            newname = v.replace("_PERHARV","")
            if newname in this_ds_gs:
                raise RuntimeError(f"{newname} already in dataset!")
            da_pg = xr.DataArray(data = ar_valid_pg, 
                                coords = [this_ds_gs.coords["patch"], this_ds_gs.coords["gs"]],
                                name = newname,
                                attrs = da_yhp.attrs)
            this_ds_gs[newname] = da_pg
    else:
        # Print details about example bad patch(es)
        if min(unique_Nseasons) < Ngs:
            print(f"Too few seasons (min {min(unique_Nseasons)} < {Ngs})")
            p = np.where(np.sum(~np.isnan(hdates_pg2), axis=1) == min(unique_Nseasons))[0][0]
            print_onepatch_wrongNgs(p, this_ds, sdates_ymp, hdates_ymp, sdates_pym, hdates_pym, sdates_pym2, hdates_pym2, sdates_pym3, hdates_pym3, sdates_pg, hdates_pg, sdates_pg2, hdates_pg2)
        if max(unique_Nseasons) > Ngs:
            print(f"Too many seasons (max {max(unique_Nseasons)} > {Ngs})")
            p = np.where(np.sum(~np.isnan(hdates_pg2), axis=1) == max(unique_Nseasons))[0][0]
            print_onepatch_wrongNgs(p, this_ds, sdates_ymp, hdates_ymp, sdates_pym, hdates_pym, sdates_pym2, hdates_pym2, sdates_pym3, hdates_pym3, sdates_pg, hdates_pg, sdates_pg2, hdates_pg2)
        raise RuntimeError(f"Can't convert time*mxharvests axes to growingseason axis: discrepancy of {discrepancy} patch-seasons")
    
    if incl_orig:
        return this_ds_gs, this_ds
    else:
        return this_ds_gs


# For backwards compatibility with files missing SDATES_PERHARV.
def convert_axis_time2gs_old(this_ds, myVars):
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
    # Because some patches will be planted in the last year but not complete, we have to ignore any finalyear-planted seasons that do complete.
    Ngs = this_ds.dims["time"]-1
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


# For backwards compatibility with files missing SDATES_PERHARV.
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


# For backwards compatibility with files missing SDATES_PERHARV.
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


def print_onepatch_wrongNgs(p, this_ds_orig, sdates_ymp, hdates_ymp, sdates_pym, hdates_pym, sdates_pym2, hdates_pym2, sdates_pym3, hdates_pym3, sdates_pg, hdates_pg, sdates_pg2, hdates_pg2):
    try:
        import pandas as pd
    except:
        print("Couldn't import pandas, so not displaying example bad patch ORIGINAL.")
    
    print(f"patch {p}: {this_ds_orig.patches1d_itype_veg_str.values[p]}, lon {this_ds_orig.patches1d_lon.values[p]} lat {this_ds_orig.patches1d_lat.values[p]}")
    
    print("Original SDATES (per sowing):")
    print(this_ds_orig.SDATES.values[:,:,p])
    
    print("Original HDATES (per harvest):")
    print(this_ds_orig.HDATES.values[:,:,p])
     
    if "pandas" in sys.modules:
        def print_pandas_ymp(msg, cols, arrs_tuple):
            print(f"{msg} ({np.sum(~np.isnan(arrs_tuple[0]))})")
            mxharvests = arrs_tuple[0].shape[1]
            arrs_list2 = []
            cols2 = []
            for h in np.arange(mxharvests):
                for i,a in enumerate(arrs_tuple):
                    arrs_list2.append(a[:,h])
                    cols2.append(cols[i] + str(h))
            arrs_tuple2 = tuple(arrs_list2)
            df = pd.DataFrame(np.stack(arrs_tuple2, axis=1))
            df.columns = cols2
            print(df)
        
        print_pandas_ymp("Original", ["sdate", "hdate"], 
                     (this_ds_orig.SDATES_PERHARV.values[:,:,p],
                      this_ds_orig.HDATES.values[:,:,p]
                      ))
        
        print_pandas_ymp("Masked", ["sdate", "hdate"], 
                     (sdates_ymp[:,:,p],
                      hdates_ymp[:,:,p]))
        
        print_pandas_ymp('After "Ignore harvests from before this output began"', ["sdate", "hdate"], 
                     (np.transpose(sdates_pym, (1,2,0))[:,:,p],
                      np.transpose(hdates_pym, (1,2,0))[:,:,p]))
        
        print_pandas_ymp('After "In years with no sowing, pretend the first no-harvest is meaningful"', ["sdate", "hdate"], 
                     (np.transpose(sdates_pym2, (1,2,0))[:,:,p] ,
                      np.transpose(hdates_pym2, (1,2,0))[:,:,p]))
        
        print_pandas_ymp('After "In years with sowing that are followed by inactive years, check whether the last sowing was harvested before the patch was deactivated. If not, pretend the LAST no-harvest is meaningful."', ["sdate", "hdate"], 
                     (np.transpose(sdates_pym3, (1,2,0))[:,:,p] ,
                      np.transpose(hdates_pym3, (1,2,0))[:,:,p]))
        
        def print_pandas_pg(msg, cols, arrs_tuple):
            print(f"{msg} ({np.sum(~np.isnan(arrs_tuple[0]))})")
            arrs_list = list(arrs_tuple)
            for i,a in enumerate(arrs_tuple):
                arrs_list[i] = np.reshape(a, (-1))
            arrs_tuple2 = tuple(arrs_list)
            df = pd.DataFrame(np.stack(arrs_tuple2, axis=1))
            df.columns = cols
            print(df)
        
        print_pandas_pg("Same, but converted to gs axis", ["sdate", "hdate"], 
                     (sdates_pg[p,:],
                      hdates_pg[p,:]))
        
        print_pandas_pg('After "Ignore any harvests that were planted in the final year, because some cells will have incomplete growing seasons for the final year"', ["sdate", "hdate"], 
                     (sdates_pg2[p,:],
                      hdates_pg2[p,:]))
    else:
        
        def print_nopandas(a1, a2, msg):
            print(msg)
            if a1.ndim==1:
                # I don't know why these aren't side-by-side!
                print(np.stack((a1, a2), axis=1))
            else:
                print(np.concatenate((a1, a2), axis=1))
    
        print_nopandas(sdates_ymp[:,:,p], hdates_ymp[:,:,p], "Masked:")
        
        print_nopandas(np.transpose(sdates_pym, (1,2,0))[:,:,p], np.transpose(hdates_pym, (1,2,0))[:,:,p], 'After "Ignore harvests from before this output began"')
        
        print_nopandas(np.transpose(sdates_pym2, (1,2,0))[:,:,p], np.transpose(hdates_pym2, (1,2,0))[:,:,p], 'After "In years with no sowing, pretend the first no-harvest is meaningful"')
        
        print_nopandas(np.transpose(sdates_pym3, (1,2,0))[:,:,p], np.transpose(hdates_pym3, (1,2,0))[:,:,p], 'After "In years with sowing that are followed by inactive years, check whether the last sowing was harvested before the patch was deactivated. If not, pretend the LAST [easier to implement!] no-harvest is meaningful."')
        
        print_nopandas(sdates_pg[p,:], hdates_pg[p,:], "Same, but converted to gs axis")
        
        print_nopandas(sdates_pg2[p,:], hdates_pg2[p,:], 'After "Ignore any harvests that were planted in the final year, because some cells will have incomplete growing seasons for the final year"')
    
    print("\n\n")
    


# For backwards compatibility with files missing SDATES_PERHARV.
def set_firstharv_nan(this_ds, this_var, firstharv_nan_inds):
    this_da = this_ds[this_var]
    this_array = this_da.values
    this_array[0,0,firstharv_nan_inds] = np.nan
    this_da.values = this_array
    this_ds[this_var] = this_da
    return this_ds
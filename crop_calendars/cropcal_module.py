# What system is the script running on?
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
import cartopy.crs as ccrs
from scipy import stats, signal
import warnings
import cftime
import pandas as pd
import os
import glob


def adjust_grainC(da_in, patches1d_itype_veg_str):
    # Parameters from Danica's 2020 paper
    fyield = 0.85 # 85% harvest efficiency (Kucharik & Brye, 2003)
    cgrain = 0.45 # 45% of dry biomass is C (Monfreda et al., 2008)
    
    # Dry matter fraction from Wirsenius (2000) Table A1.II except as noted
    drymatter_fractions = {
        'corn': 0.88,
        'cotton': 0.912, # Table A1.III, "Seed cotton", incl. lint, seed, and "other (ginning waste)"
        'miscanthus': 0.0,  # Not included in Wirsenius, but also not simulated, so I don't care
        'rice': 0.87,
        'soybean': 0.91,
        'sugarcane': 1-0.745, # Irvine, Cane Sugar Handbook, 10th ed., 1977, P. 16. (see sugarcane.py)
        'wheat': 0.88,
    }
    
    # Convert patches1d_itype_veg_str to needed format
    if isinstance(patches1d_itype_veg_str, xr.DataArray):
        patches1d_itype_veg_str = patches1d_itype_veg_str.values
    if not isinstance(patches1d_itype_veg_str[0], str):
        patches1d_itype_veg_int = patches1d_itype_veg_str
        patches1d_itype_veg_str = [utils.ivt_int2str(x) for x in patches1d_itype_veg_int]
    
    # Create new array with patch as the first dimension. This allows us to use Ellipsis when filling.
    patch_dim = da_in.dims.index('patch')
    wet_tp = np.full(da_in.shape, np.nan)
    wet_tp = np.moveaxis(wet_tp, patch_dim, 0)
    
    # Fill new array, increasing to include water weight
    drymatter_cropList = []
    da_in.load()
    for thisCrop, dm_frac in drymatter_fractions.items():
        drymatter_cropList.append(thisCrop)
        i_thisCrop = [i for i,x in enumerate(patches1d_itype_veg_str) if thisCrop in x]
        
        tmp = da_in.isel(patch=i_thisCrop).values
        if dm_frac != None:
            tmp[np.where(~np.isnan(tmp))] /= dm_frac
        elif np.any(tmp > 0):
            raise RuntimeError(f"You need to get a real dry-matter fraction for {thisCrop}")
        
        # For sugarcane, also account for the fact that sugar is only 43% of dry matter
        if thisCrop == "sugarcane":
            tmp /= 1 - 0.43
        
        wet_tp[i_thisCrop, ...] = np.moveaxis(tmp, patch_dim, 0)
    
    # Move patch dimension (now in 0th position) back to where it should be.
    wet_tp = np.moveaxis(wet_tp, 0, patch_dim)
    
    # Make sure NaN mask is unchanged
    if not np.array_equal(np.isnan(wet_tp), np.isnan(da_in.values)):
        missing_croptypes = [x for x in np.unique(patches1d_itype_veg_str) if x not in drymatter_cropList]
        raise RuntimeError(f'Failed to completely fill wet_tp. Missing crop types: {missing_croptypes}')

    # Save to output DataArray
    da_out = xr.DataArray(data = wet_tp * fyield / cgrain,
                          coords = da_in.coords,
                          attrs = da_in.attrs)
    return da_out


# Rounding errors can result in differences of up to lon/lat_tolerance. Tolerate those by replacing the value in the gridded dataset.
def adjust_gridded_lonlats(patches1d_lonlat, patches1d_ij, lu_dsg_lonlat_da, this_tolerance, i_or_j_str):
        missing_lonlats = np.unique(patches1d_lonlat.values[np.isnan(patches1d_ij)])
        new_gridded_lonlats = lu_dsg_lonlat_da.values
        for m in missing_lonlats:
            # Find closest value in gridded lonlat
            min_diff = np.inf
            for i, n in enumerate(np.sort(lu_dsg_lonlat_da)):
                this_diff = n - m
                if np.abs(this_diff) < min_diff:
                    min_diff = np.abs(this_diff)
                    closest_i = i
                    closest_n = n
                if this_diff > 0:
                    break
            if min_diff > this_tolerance:
                raise ValueError(f"NaN in patches1d_{i_or_j_str}xy; closest value is {min_diff}° off.")
            print(f"Replacing {m} with {closest_n}")
            new_gridded_lonlats[closest_i] = closest_n
            patches1d_ij[patches1d_lonlat.values == m] = closest_i + 1
        lu_dsg_lonlat_da = xr.DataArray(data = new_gridded_lonlats,
                                        coords = {"lat": new_gridded_lonlats})
        return lu_dsg_lonlat_da, patches1d_ij
   

def annual_mean_from_monthly(ds, var):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out


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


def check_constant_vars(this_ds, case, ignore_nan, constantGSs=None, verbose=True, throw_error=True):
    
    if isinstance(case, str):
        constantVars = [case]
    elif isinstance(case, list):
        constantVars = case
    elif isinstance(case, dict):
        constantVars = case['constantVars']
    else:
        raise TypeError(f'case must be str or dict, not {type(case)}')
    
    if not constantVars:
        return None
    
    if constantGSs:
        gs0 = this_ds.gs.values[0]
        gsN = this_ds.gs.values[-1]
        if constantGSs.start > gs0 or constantGSs.stop < gsN:
            print(f'❗ Only checking constantVars over {constantGSs.start}-{constantGSs.stop} (run includes {gs0}-{gsN})')
        this_ds = this_ds.sel(gs=constantGSs)
    
    any_bad = False
    any_bad_before_checking_rx = False
    if throw_error:
        emojus = '❌'
    else:
        emojus = '❗'
    if not isinstance(constantVars, list):
        constantVars = [constantVars]
    
    for v in constantVars:
        ok = True
        
        if "gs" in this_ds[v].dims:
            time_coord = "gs"
        elif "time" in this_ds[v].dims:
            time_coord = "time"
        else:
            raise RuntimeError(f"Which of these is the time coordinate? {this_ds[v].dims}")
        i_time_coord = this_ds[v].dims.index(time_coord)
        
        this_da = this_ds[v]
        ra_sp = np.moveaxis(this_da.copy().values, i_time_coord, 0)
        incl_patches = []
        bad_patches = np.array([])
        strList = []
        
        # Read prescription file, if needed
        rx_ds = None
        if isinstance(case, dict):
            if v == "GDDHARV" and 'rx_gdds_file' in case:
                rx_ds = import_rx_dates("gdd", case['rx_gdds_file'], this_ds, set_neg1_to_nan=False).squeeze()

        for t1 in np.arange(this_ds.dims[time_coord]-1):
            
            condn = ~np.isnan(ra_sp[t1,...])
            if t1 > 0:
                condn = np.bitwise_and(condn,
                                       np.all(np.isnan(ra_sp[:t1,...]), axis=0)
                                       )
            thesePatches = np.where(condn)[0]
            if thesePatches.size == 0:
                continue
            thesePatches = list(np.where(condn)[0])
            incl_patches += thesePatches
            # print(f't1 {t1}: {thesePatches}')
            
            t1_yr = this_ds[time_coord].values[t1]
            t1_vals = np.squeeze(this_da.isel({time_coord: t1, 'patch': thesePatches}).values)

            for t in np.arange(t1+1, this_ds.dims[time_coord]):
                t_yr = this_ds[time_coord].values[t]
                t_vals = np.squeeze(this_da.isel({time_coord: t, 'patch': thesePatches}).values)
                ok_p = t1_vals == t_vals
                
                # If allowed, ignore where either t or t1 is NaN. Should only be used for runs where land use varies over time.
                if ignore_nan:
                    ok_p = np.squeeze(np.bitwise_or(ok_p, np.isnan(t1_vals+t_vals)))
                
                if not np.all(ok_p):
                    any_bad_before_checking_rx = True
                    bad_patches_thisT = list(np.where(np.bitwise_not(ok_p))[0])
                    bad_patches = np.concatenate((bad_patches, np.array(thesePatches)[bad_patches_thisT]))
                    if rx_ds:
                        found_in_rx = np.array([False for x in bad_patches])
                    varyPatches = list(np.array(thesePatches)[bad_patches_thisT])
                    varyLons = this_ds.patches1d_lon.values[bad_patches_thisT]
                    varyLats = this_ds.patches1d_lat.values[bad_patches_thisT]
                    varyCrops = this_ds.patches1d_itype_veg_str.values[bad_patches_thisT]
                    varyCrops_int = this_ds.patches1d_itype_veg.values[bad_patches_thisT]
                    
                    any_bad_anyCrop = False
                    for c in np.unique(varyCrops_int):
                        rx_var = f'gs1_{c}'
                        varyLons_thisCrop = varyLons[np.where(varyCrops_int==c)]
                        varyLats_thisCrop = varyLats[np.where(varyCrops_int==c)]
                        theseRxVals = np.diag(rx_ds[rx_var].sel(lon=varyLons_thisCrop, lat=varyLats_thisCrop).values)
                        if len(theseRxVals) != len(varyLats_thisCrop):
                            raise RuntimeError(f"Expected {len(varyLats_thisCrop)} rx values; got {len(theseRxVals)}")
                        if not np.any(theseRxVals != -1):
                            continue
                        any_bad_anyCrop = True
                        break
                    if not any_bad_anyCrop:
                        continue
                    
                    # This bit is pretty inefficient, but I'm not going to optimize it until I actually need to use it.
                    for i, p in enumerate(bad_patches_thisT):
                        thisPatch = varyPatches[i]
                        thisLon = varyLons[i]
                        thisLat = varyLats[i]
                        thisCrop = varyCrops[i]
                        thisCrop_int = varyCrops_int[i]
                        
                        # If prescribed input had missing value (-1), it's fine for it to vary.
                        if rx_ds:
                            rx_var = f'gs1_{thisCrop_int}'
                            if thisLon in rx_ds.lon.values and thisLat in rx_ds.lat.values:
                                rx = rx_ds[rx_var].sel(lon=thisLon, lat=thisLat).values
                                Nunique = len(np.unique(rx))
                                if (Nunique == 1):
                                    found_in_rx[i] = True
                                    if rx == -1:
                                        continue
                                elif Nunique > 1:
                                    raise RuntimeError(f'How does lon {thisLon} lat {thisLat} {thisCrop} have time-varying {v}?')
                            else:
                                raise RuntimeError('lon {thisLon} lat {thisLat} {thisCrop} not in rx dataset?')
                        
                        # Print info (or save to print later)
                        any_bad = True
                        if verbose:
                            thisStr = f"   Patch {thisPatch} (lon {thisLon} lat {thisLat}) {thisCrop} ({thisCrop_int})"
                            if rx_ds and not found_in_rx[i]:
                                thisStr = thisStr.replace('(lon', '* (lon')
                            if not np.isnan(t1_vals[p]):
                                t1_val_print = int(t1_vals[p])
                            else:
                                t1_val_print = 'NaN'
                            if not np.isnan(t_vals[p]):
                                t_val_print = int(t_vals[p])
                            else:
                                t_val_print = 'NaN'
                            if v == "SDATES":
                                strList.append(f"{thisStr}: Sowing {t1_yr} jday {t1_val_print}, {t_yr} jday {t_val_print}")
                            else:
                                strList.append(f"{thisStr}: {t1_yr} {v} {t1_val_print}, {t_yr} {v} {t_val_print}")
                        else:
                            if ok:
                                print(f"{emojus} CLM output {v} unexpectedly vary over time:")
                                ok = False
                            print(f"{v} timestep {t} does not match timestep {t1}")
                            break
        if verbose and any_bad:
            print(f"{emojus} CLM output {v} unexpectedly vary over time:")
            strList.sort()
            if rx_ds and np.any(~found_in_rx):
                strList = ['*: Not found in prescribed input file (maybe minor lon/lat mismatch)'] + strList
            elif not rx_ds:
                strList = ['(No rx file checked)'] + strList
            print('\n'.join(strList))

        # Make sure every patch was checked once (or is all-NaN except possibly final season)
        incl_patches = np.sort(incl_patches)
        if not np.array_equal(incl_patches, np.unique(incl_patches)):
            raise RuntimeError('Patch(es) checked more than once!')
        incl_patches = list(incl_patches)
        incl_patches += list(np.where(np.all(np.isnan(ra_sp[:-1,]), axis=0))[0])
        incl_patches = np.sort(incl_patches)
        if not np.array_equal(incl_patches, np.unique(incl_patches)):
            raise RuntimeError('Patch(es) checked but also all-NaN??')
        if not np.array_equal(incl_patches, np.arange(this_ds.dims['patch'])):
            for p in np.arange(this_ds.dims['patch']):
                if p not in incl_patches:
                    break
            raise RuntimeError(f'Not all patches checked! E.g., {p}: {this_da.isel(patch=p).values}')
        
        if not any_bad:
            if any_bad_before_checking_rx:
                print(f"✅ CLM output {v} do not vary through {this_ds.dims[time_coord]} growing seasons of output (except for patch(es) with missing rx).")
            else:
                print(f"✅ CLM output {v} do not vary through {this_ds.dims[time_coord]} growing seasons of output.")

    if any_bad and throw_error:
        raise RuntimeError('Stopping due to failed check_constant_vars().')
    
    bad_patches = np.unique(bad_patches)
    return [int(p) for p in bad_patches]


def check_rx_obeyed(vegtype_list, rx_ds, dates_ds, which_ds, output_var, gdd_min=None, verbose=False):
    all_ok = 2
    diff_str_list = []
    gdd_tolerance = 1
    
    if "GDDHARV" in output_var and verbose:
        harvest_reason_da = dates_ds['HARVEST_REASON']
        unique_harvest_reasons = np.unique(harvest_reason_da.values[np.where(~np.isnan(harvest_reason_da.values))])
        pct_harv_at_mature = get_pct_harv_at_mature(harvest_reason_da)
        print(f"{which_ds} harvest reasons: {unique_harvest_reasons} ({pct_harv_at_mature}% harv at maturity)")
    
    for vegtype_str in vegtype_list:
        thisVeg_patches = np.where(dates_ds.patches1d_itype_veg_str == vegtype_str)[0]
        if thisVeg_patches.size == 0:
            continue
        ds_thisVeg = dates_ds.isel(patch=thisVeg_patches)
        patch_inds_lon_thisVeg = ds_thisVeg.patches1d_ixy.values.astype(int) - 1
        patch_inds_lat_thisVeg = ds_thisVeg.patches1d_jxy.values.astype(int) - 1
        patch_lons_thisVeg = ds_thisVeg.patches1d_lon
        patch_lats_thisVeg = ds_thisVeg.patches1d_lat
    
        vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
        rx_da = rx_ds[f"gs1_{vegtype_int}"]
        rx_array = rx_da.values[patch_inds_lat_thisVeg,patch_inds_lon_thisVeg]
        rx_array = np.expand_dims(rx_array, axis=1)
        sim_array = ds_thisVeg[output_var].values
        sim_array_dims = ds_thisVeg[output_var].dims
        
        # Ignore patches without prescribed value
        rx_array[np.where(rx_array < 0)] = np.nan
        
        # Account for...
        if "GDDHARV" in output_var:
            # ...GDD harvest threshold minimum set in PlantCrop()
            if gdd_min == None:
                raise RuntimeError(f"gdd_min must be provided when doing check_rx_obeyed() for {output_var}")
            rx_array[(rx_array >= 0) & (rx_array < gdd_min)] = gdd_min
            
            # ...harvest reason
            # 0: Should never happen in any simulation
            # 1: Harvesting at maturity
            # 2: Harvesting at max season length (mxmat)
            # 3: Crop was incorrectly planted in last time step of Dec. 31
            # 4: Today was supposed to be the planting day, but the previous crop still hasn't been harvested.
            # 5: Harvest the day before the next sowing date this year.
            # 6: Same as #5.
            # 7: Harvest the day before the next sowing date (today is Dec. 31 and the sowing date is Jan. 1)
            harvest_reason_da = ds_thisVeg['HARVEST_REASON']
            unique_harvest_reasons = np.unique(harvest_reason_da.values[np.where(~np.isnan(harvest_reason_da.values))])
            pct_harv_at_mature = get_pct_harv_at_mature(harvest_reason_da)
        
        if np.any(sim_array != rx_array):
            diff_array = sim_array - rx_array
            
            # Allow negative GDDHARV values when harvest occurred because sowing was scheduled for the next day
            if output_var=="GDDHARV_PERHARV":
                diff_array = np.ma.masked_array(diff_array, mask= \
                    (diff_array < 0) & 
                    (ds_thisVeg["HARVEST_REASON_PERHARV"].values==5))
            elif output_var=="GDDHARV":
                diff_array = np.ma.masked_array(diff_array, mask= \
                    (diff_array < 0) & 
                    (ds_thisVeg["HARVEST_REASON"].values==5))

            if np.any(np.abs(diff_array[abs(diff_array) > 0]) > 0):
                min_diff, minLon, minLat, minGS, minRx = get_extreme_info(diff_array, rx_array, np.nanmin, sim_array_dims, dates_ds.gs, patch_lons_thisVeg, patch_lats_thisVeg)
                max_diff, maxLon, maxLat, maxGS, maxRx = get_extreme_info(diff_array, rx_array, np.nanmax, sim_array_dims, dates_ds.gs, patch_lons_thisVeg, patch_lats_thisVeg)
                
                diffs_eg_txt = f"{vegtype_str} ({vegtype_int}): diffs range {min_diff} (lon {minLon}, lat {minLat}, gs {minGS}, rx ~{minRx}) to {max_diff} (lon {maxLon}, lat {maxLat}, gs {maxGS}, rx ~{maxRx})"
                if "GDDHARV" in output_var:
                    diffs_eg_txt += f"; harvest reasons: {unique_harvest_reasons} ({pct_harv_at_mature}% harvested at maturity)"
                if "GDDHARV" in output_var and np.nanmax(abs(diff_array)) <= gdd_tolerance:
                    if all_ok > 0:
                        all_ok = 1
                        diff_str_list.append(f"	  {diffs_eg_txt}")
                else:
                    all_ok = 0
                    if verbose:
                        print(f"❌ {which_ds}: Prescribed {output_var} *not* always obeyed. E.g., {diffs_eg_txt}")
                    else:
                        break
    
    if all_ok == 2:
        print(f"✅ {which_ds}: Prescribed {output_var} always obeyed")
    elif all_ok == 1:
        # print(f"🟨 {which_ds}: Prescribed {output_var} *not* always obeyed, but acceptable:")
        # for x in diff_str_list: print(x)
        print(f"🟨 {which_ds}: Prescribed {output_var} *not* always obeyed, but acceptable (diffs <= {gdd_tolerance})")
    elif not verbose:
        print(f"❌ {which_ds}: Prescribed {output_var} *not* always obeyed. E.g., {diffs_eg_txt}")


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


def christoph_detrend(x, w, center0=False):
    if w==0:
        raise RuntimeError("Don't use christoph_detrend() with w=0")

    moving_average = get_moving_average(x, w)

    r = get_window_radius(w)
    result = x[..., r:-r] - moving_average
    
    if not center0:
        ma_mean = np.mean(moving_average, axis=x.ndim-1)
        if x.ndim == 2:
            ma_mean = np.expand_dims(ma_mean, axis=1)
        result += ma_mean
    
    return result


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
    hdates_pym2 = hdates_pym.copy()
    sdates_pym2 = sdates_pym.copy()
    nosow_py = np.all(~(sdates_orig_pym > 0), axis=2)
    nosow_py_1st = nosow_py & np.isnan(hdates_pym[:,:,0])
    where_nosow_py_1st = np.where(nosow_py_1st)
    hdates_pym2[where_nosow_py_1st[0], where_nosow_py_1st[1], 0] = -np.inf
    sdates_pym2[where_nosow_py_1st[0], where_nosow_py_1st[1], 0] = -np.inf
    for h in np.arange(mxharvests - 1):
        if h == 0:
            continue
        elif h == 1:
            print("Warning: Untested with mxharvests > 2")
        where_nosow_py = np.where(nosow_py & ~np.any(np.isnan(hdates_pym[:,:,0:h]), axis=2) & np.isnan(hdates_pym[:,:,h]))
        hdates_pym2[where_nosow_py[0], where_nosow_py[1], h+1] = -np.inf
        sdates_pym2[where_nosow_py[0], where_nosow_py[1], h+1] = -np.inf
        
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
    is_fake = np.isneginf(hdates_pg2)
    is_fake = np.reshape(is_fake[is_valid], (this_ds.dims["patch"], Ngs))
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
            if this_ds[v].dims != ('time', 'mxharvests', 'patch') or (myVars and v not in myVars):
                continue
            
            # Set invalid values to NaN
            da_yhp = this_ds[v].copy()
            da_yhp = da_yhp.where(~np.isneginf(da_yhp))
            
            # Remove the nans and reshape to patches*growingseasons
            da_pyh = da_yhp.transpose("patch", "time", "mxharvests")
            ar_pg = np.reshape(da_pyh.values, (this_ds.dims["patch"], -1))
            ar_valid_pg = np.reshape(ar_pg[is_valid], (this_ds.dims["patch"], Ngs))
            # Change -infs to nans
            ar_valid_pg[is_fake] = np.nan
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


def cropnames_fao2clm(cropList_in):
    def convert1(thisCrop):
        thisCrop = thisCrop.lower().replace(' ', '')
        thisCrop = thisCrop.replace(',paddy', '')
        thisCrop = thisCrop.replace('seed', '')
        thisCrop = thisCrop.replace('beans', 'bean')
        thisCrop = thisCrop.replace('maize', 'corn')
        return thisCrop
    
    if isinstance(cropList_in, (list, np.ndarray)):
        cropList_out = [convert1(x) for x in cropList_in]
        if isinstance(cropList_in, np.ndarray):
            cropList_out = np.array(cropList_out)
    else:
        cropList_out = convert1(cropList_in)
    
    return cropList_out


def detrend(ps_in):
    # Can't detrend if NaNs are present, so...
    
    if isinstance(ps_in, xr.DataArray):
        ps_out = ps_in.values
    else:
        ps_out = ps_in
    
    unique_Nnans = np.unique(np.sum(np.isnan(ps_out), axis=1))
    Ngs = ps_in.shape[1]

    for n in unique_Nnans:
        
        # Don't need to detrend patches with <2 non-NaN seasons, and "detrending" 2 points will just set them both to zero.
        if n >= Ngs-2:
            continue
        
        # Get the patches with this number of NaN seasons
        ok = np.where(np.sum(np.isnan(ps_out), axis=1)==n)[0]
        Nok = len(ok)
        thisNok_ps = ps_out[ok,:]
        where_notnan = np.where(~np.isnan(thisNok_ps))
        
        # Get the non-NaN seasons of each such patch
        thisNok_notnan = thisNok_ps[where_notnan]
        thisNok_notnan_ps = np.reshape(thisNok_notnan, (Nok, Ngs-n))
        
        # Detrend these patches
        thisNok_notnan_dt_ps = signal.detrend(thisNok_notnan_ps, axis=1)
        
        # Save the detrended time series back to our output
        thisNok_dt_ps = np.copy(thisNok_ps)
        thisNok_dt_ps[where_notnan] = np.reshape(thisNok_notnan_dt_ps, (-1))
        ps_out[ok,:] = thisNok_dt_ps
    
    if isinstance(ps_in, xr.DataArray):
        ps_out = xr.DataArray(data = ps_out,
                              coords = ps_in.coords,
                              attrs = ps_in.attrs)
            
    return ps_out


def equalize_colorbars(ims, center0=False, this_var=None):
    vmin = np.inf
    vmax = -np.inf
    nims = len(ims)
    for im in ims:
        vmin = min(vmin, im.get_clim()[0])
        vmax = max(vmax, im.get_clim()[1])
    
    extend = "neither"
    if this_var == "HUIFRAC" and vmax > 1:
            vmax = 1
            extend = "max"
    
    if center0:
            v = np.max(np.abs([vmin, vmax]))
            vmin = -v
            vmax = v
    
    for i in np.arange(nims):
        ims[i].set_clim(vmin, vmax)
    
    return extend


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


def fao_data_get(fao_all, element, y1, yN, fao_to_clm_dict, cropList_combined_clm):
    fao_this = fao_all.copy().query(f"Element == '{element}'")

    # Convert t to Mt
    if element == 'Production':
        fao_this.Value *= 1e-6

    # Pivot and add Total column
    fao_this = fao_this.copy().pivot(index="Year", columns="Crop", values="Value")
    fao_this['Total'] = fao_this.sum(numeric_only=True, axis=1)

    # Remove unneeded years
    fao_this = fao_this.filter(items=np.arange(y1,yN+1), axis=0)
    
    # Reorder to match CLM
    if len(fao_to_clm_dict) != len(cropList_combined_clm):
        raise RuntimeError(f"fao_to_clm_dict and are different lengths ({len(cropList_combined_clm)} vs {len(cropList_combined_clm)})")
    if len(fao_this.columns) != len(cropList_combined_clm):
        raise RuntimeError(f"fao_this.columns and are different lengths ({len(fao_this.columns)} vs {len(cropList_combined_clm)})")
    new_order = [cropList_combined_clm.index(fao_to_clm_dict[x]) for x in fao_this.columns]
    new_cols = fao_this.columns[new_order]
    fao_this = fao_this[new_cols]

    # Make no-sugarcane version
    fao_this_nosgc = fao_this.drop(columns = ["Sugar cane", "Total"])
    fao_this_nosgc['Total'] = fao_this_nosgc.sum(numeric_only=True, axis=1)
    
    return fao_this, fao_this_nosgc


def fao_data_preproc(fao):
    
    # Because I always confuse Item vs. Element
    fao.rename(columns={"Item": "Crop"}, inplace=True, errors="raise")
    
    # Combine "Maize" and "Maize, green"
    fao.Crop.replace("Maize.*", "Maize", regex=True, inplace=True)
    fao = fao.groupby(by=["Crop","Year","Element","Area","Unit"], as_index=False).agg("sum")

    # Pick one rice
    rice_to_keep = "Rice, paddy"
    rice_to_drop = "Rice, paddy (rice milled equivalent)"
    drop_this = [x == rice_to_drop for x in fao.Crop]
    fao = fao.drop(fao[drop_this].index)
    
    # Filter out "China," which includes all Chinas
    if "China" in fao.Area.values:
        fao = fao.query('Area != "China"')
    
    return fao


def fullname_to_combinedCrop(fullnames, cropList_combined_clm):
    x = [strip_cropname(y).capitalize() for y in fullnames]
    z = [y if y in cropList_combined_clm else '' for y in x]
    return z


def get_caselist(which_cases):
    cases = {}
    if which_cases == "originalCLM":
        # A run that someone else did
        cases['cmip6'] = \
            {'filepath': '/Users/Shared/CESM_work/CropEvalData_ssr/danica_timeseries-cmip6_i.e21.IHIST.f09_g17/month_1/ssr_trimmed_annual.nc',
             'constantVars': None,
             'constantGSs': None,
             'res': 'f09_g17',
             'verbosename': 'cmip6: Old CESM run'}
    if "original" in which_cases:
        # My run with normal CLM code + my outputs
        cases['Original baseline'] = \
            {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.yield_perharv2.IHistClm50BgcCrop.1958-2014/cropcals3.f19-g17.yield_perharv2.IHistClm50BgcCrop.1958-2014.clm2.h1.1958-01-01-00000.nc',
             'constantVars': None,
             'constantGSs': None,
             'res': 'f19_g17',
             'verbosename': 'Original baseline: ctsm5.1.dev092 + my outvars'}
    # My run with rx_crop_calendars2 code but CLM calendars
    cases['CLM Default'] = \
        {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1958-2014/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1958-2014.clm2.h1.1958-01-01-00000.nc',
         'constantVars': None,
         'constantGSs': None,
         'res': 'f19_g17',
         'verbosename': 'CLM Default: my cropcal code, no Rx'}

    # My run with rx_crop_calendars2 code and GGCMI calendars
    # cases['Prescribed Calendars v2'] = \
    #     {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced3.clm2.h1.1958-01-01-00000.nc',
    #      'constantVars': ["SDATES", "GDDHARV"],
    #      'constantGSs': None, # 'None' with constantVars specified means all should be constant
    #      'res': 'f19_g17',
    #      'rx_sdates_file': "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
    #      'rx_hdates_file': "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
    #      'rx_gdds_file': "/Users/Shared/CESM_work/crop_dates/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1977-2014.gddgen/generate_gdds/gdds_20220927_174954.nc",
    #      'verbosename': 'Prescribed Calendars v2: unlim-gs sim and GDDgen'}
    # cases['Prescribed Calendars v3'] = \
    #     {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars3.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced3.mxmat/cropcals3.f19-g17.rx_crop_calendars3.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced3.mxmat.clm2.h1.1958-01-01-00000.nc',
    #      'constantVars': ["SDATES", "GDDHARV"],
    #      'constantGSs': None, # 'None' with constantVars specified means all should be constant
    #      'res': 'f19_g17',
    #      'rx_sdates_file': "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
    #      'rx_hdates_file': "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
    #      'rx_gdds_file': "/Users/Shared/CESM_work/crop_dates/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1977-2014.gddgen/generate_gdds/gdds_20220927_174954.nc",
    #      'verbosename': 'Prescribed Calendars v3: lim-gs sim, unlim-gs GDDgen'}
    cases['Prescribed Calendars'] = \
        {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars3.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced4.mxmat/cropcals3.f19-g17.rx_crop_calendars3.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced4.mxmat.clm2.h1.1958-01-01-00000.nc',
         'constantVars': ["SDATES", "GDDHARV"],
         'constantGSs': None, # 'None' with constantVars specified means all should be constant
         'res': 'f19_g17',
         'rx_sdates_file': "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
         'rx_hdates_file': "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
         'rx_gdds_file': "/Users/Shared/CESM_work/crop_dates/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1977-2014.gddgen/generate_gdds.mxmat.2022-10-26-171107/gdds_20221026_180012.nc",
         'verbosename': 'Prescribed Calendars v4: Rx sdates+GDDs, lim-gs sim and GDDgen'}
    if which_cases == "diagnose":
        # My run with rx_crop_calendars2 code and GGCMI sowing dates but CLM maturity reqts
        cases['Prescribed Sowing'] = \
            {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.sdateforced_not_gdd/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.sdateforced_not_gdd.clm2.h1.1958-01-01-00000.nc',
             'constantVars': ["SDATES"],
             'constantGSs': None, # 'None' with constantVars specified means all should be constant
             'res': 'f19_g17',
             'rx_sdates_file': "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
             'rx_hdates_file': None,
             'rx_gdds_file': None,
             'verbosename': 'Prescribed sowing: unlim-gs sim'}
        # My run with rx_crop_calendars2 code and CLM sowing dates but GGCMI maturity reqts
        cases['Prescribed Maturity'] = \
            {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced_not_sdate.mxmat/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced_not_sdate.mxmat.clm2.h1.1958-01-01-00000.nc',
             'constantVars': ["GDDHARV"],
             'constantGSs': None, # 'None' with constantVars specified means all should be constant
             'res': 'f19_g17',
             'rx_sdates_file': None,
             'rx_hdates_file': None,
             'rx_gdds_file': "/Users/Shared/CESM_work/crop_dates/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1977-2014.gddgen/generate_gdds/gdds_20220927_174954.nc",
             'verbosename': 'Prescribed maturity reqts.: unlim-gs sim and GDDs'}
                                               
    return cases


def get_earthstat_country_ts(earthstats, case, thisVar, countries_map, thisCrop_clm, i_theseYears_earthstat, country_id):
    tmp = earthstats[case['res']][thisVar]\
        .interp_like(countries_map)\
        .sel(crop=thisCrop_clm.title())\
        .isel(time=i_theseYears_earthstat)
    if country_id != None:
        tmp = tmp.where(countries_map == country_id)
    return tmp.sum(dim=['lon', 'lat']).values


def get_faostat_country_ts(fao_all_ctry, thisCrop_fao, top_y1, top_yN, country, element):
    tmp = fao_all_ctry.query(f'Crop == "{thisCrop_fao}" & Element == "{element}" & Year >= {top_y1} & Year <= {top_yN}')
    if country == "World":
        tmp = tmp.groupby(by="Year").agg("sum")
    else:
        tmp = tmp.query(f'Area == "{country}"')
    return tmp['Value'].values


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
    thisRx = rx_array[p][0]
    
    return round(themxn, 3), round(thisLon, 3), round(thisLat,3), thisGS, round(thisRx)


# Get growing season lengths from a DataArray of hdate-sdate
def get_gs_len_da(this_da):
    tmp = this_da.values
    tmp[tmp < 0] = 365 + tmp[tmp < 0]
    this_da.values = tmp
    return this_da


def get_mean_byCountry(fao, top_y1, top_yN):
   return fao.query(f'Year>={top_y1} & Year<={top_yN}').groupby(['Crop','Element','Area'])['Value'].mean()


# w is total window width (not radius)
def get_moving_average(x, w):
    if w==0:
        return x
    
    if x.ndim == 2:
        Ncases = x.shape[0]
        r = get_window_radius(w)
        result = np.full((Ncases, x.shape[1] - 2*r), fill_value=np.nan)
        for c in np.arange(0, (Ncases)):
            result[c,:] = get_moving_average(x[c,:], w)
    else:
        return np.convolve(x, np.ones(w), 'valid') / w
    
    return result


# For backwards compatibility with files missing SDATES_PERHARV.
def get_Nharv(array_in, these_dims):
    # Sum over time and mxevents to get number of events in time series for each patch
    sum_indices = tuple(these_dims.index(x) for x in ["time", "mxharvests"])
    Nevents_eachPatch = np.sum(array_in > 0, axis=sum_indices)
    return Nevents_eachPatch


def get_pct_harv_at_mature(harvest_reason_da):
    Nharv_at_mature = len(np.where(harvest_reason_da.values==1)[0])
    Nharv = len(np.where(harvest_reason_da.values>0)[0])
    if Nharv == 0:
        return np.nan
    pct_harv_at_mature = Nharv_at_mature / Nharv * 100
    pct_harv_at_mature = np.format_float_positional(pct_harv_at_mature, precision=2, unique=False, fractional=False, trim='k') # Round to 2 significant digits
    return pct_harv_at_mature


def get_reason_freq_map(Ngs, thisCrop_gridded, reason):
    map_yx = np.sum(thisCrop_gridded==reason, axis=0, keepdims=False) / Ngs
    notnan_yx = np.bitwise_not(np.isnan(thisCrop_gridded.isel(gs=0, drop=True)))
    map_yx = map_yx.where(notnan_yx)
    return map_yx


def get_reason_list_text():
    return [ \
        "???",					 # 0; should never actually be saved
        "Crop mature",			 # 1
        "Max CLM season length", # 2
        "Bad Dec31 sowing",		 # 3
        "Sowing today",			 # 4
        "Sowing tomorrow",		 # 5
        "Sown a yr ago tmrw.",	 # 6
        "Sowing tmrw. (Jan 1)"	 # 7
        ]


def get_timeseries_bias(sim, obs, fig_caselist, weights=None):
    
    weights_provided = weights is not None
    
    # The 1st dimension of sim array is case and the 2nd is year. Additional dimensions (e.g., country) should be collapsed into the second.
    if sim.ndim > 2:
        raise RuntimeError("Need to collapse all dimensions of sim after the first into the second.")
    if obs.ndim > 1:
        raise RuntimeError("Need to collapse all dimensions of obs into the first.")
    if weights_provided and weights.ndim > 1:
        raise RuntimeError("Need to collapse all dimensions of weights into the first.")
    
    # Simulations should be Ncases x Npoints
    if sim.ndim < 2:
        raise RuntimeError("sim array must have at least 2 dimensions (Ncases x Npoints)")    
    
    # Weights and obs should be the same shape
    if weights_provided and obs.shape != weights.shape:
        raise RuntimeError("weights array must be the same shape as obs array")
    
    # Weights must not be negative
    if weights_provided and np.any(weights < 0):
        raise RuntimeError("Negative weights not allowed")
    
    if weights_provided:
        weights = weights / np.sum(weights) # Ensure sum to 1
        bias = np.sum((sim - obs) * weights, axis=1)
    else:
        Npoints = sim.shape[1]
        bias = 1/Npoints * np.sum(sim - obs, axis=1)
    
    return bias


# Get yield dataset for top N countries (plus World)
def get_topN_ds(cases, reses, topYears, Ntop, thisCrop_fao, countries_key, fao_all_ctry, earthstats, min_viable_hui, mxmats):

    top_y1 = topYears[0]
    top_yN = topYears[-1]
    fao_mean_byCountry = get_mean_byCountry(fao_all_ctry, top_y1, top_yN)
    topN = fao_mean_byCountry[thisCrop_fao]['Production'].nlargest(Ntop)
    
    topN_countries = topN.keys().values
    topN_countries = np.concatenate((topN_countries, np.array(['World'])))
    Ntop += 1
    
    # Which countries are not found in our countries map?
    any_ctry_notfound = False
    for thisCountry in list(topN_countries):
        if thisCountry not in countries_key.name.values and thisCountry != "World":
            print(f'❗ {thisCountry} not in countries_key')
            any_ctry_notfound = True
    if any_ctry_notfound:
        raise RuntimeError('At least one country in FAO not found in key')

    NtopYears = len(topYears)
    
    thisCrop_clm = cropnames_fao2clm(thisCrop_fao)
    i_theseYears_earthstat = [i for i, x in enumerate(earthstats['f19_g17'].time.values) if (x.year >= top_y1) and (x.year <= top_yN)]
    caselist = [k for k,v in cases.items()]
    
    prod_ar = np.full((len(cases), NtopYears, Ntop), np.nan)
    area_ar = np.full((len(cases), NtopYears, Ntop), np.nan)
    prod_faostat_yc = np.full((NtopYears, Ntop), np.nan)
    area_faostat_yc = np.full((NtopYears, Ntop), np.nan)
    prod_earthstat_yc = np.full((NtopYears, Ntop), np.nan)
    area_earthstat_yc = np.full((NtopYears, Ntop), np.nan)
    
    for i_case, (casename, case) in enumerate(cases.items()):
        case_ds = case['ds']
        lu_ds = reses[case['res']]['ds']
        countries = lu_ds['countries'].load()
        countries_map = reses[case['res']]['dsg']['countries'].load()

        i_theseYears_case = [i for i, x in enumerate(case_ds.time.values) if (x.year >= top_y1) and (x.year <= top_yN)]
        i_theseYears_lu = [i for i, x in enumerate(lu_ds.time.values) if (x.year >= top_y1) and (x.year <= top_yN)]
        
        # Find this crop in production and area data
        i_thisCrop_case = [i for i, x in enumerate(case_ds.patches1d_itype_veg_str.values) if thisCrop_clm in x]
        if len(i_thisCrop_case) == 0:
            raise RuntimeError(f'No matches found for {thisCrop_fao} in case_ds.vegtype_str')
        i_thisCrop_lu = [i for i, x in enumerate(lu_ds.patches1d_itype_veg.values) if thisCrop_clm in utils.ivt_int2str(x)]
        if len(i_thisCrop_lu) == 0:
            raise RuntimeError(f'No matches found for {thisCrop_fao} in lu_ds.patches1d_itype_veg')
      
        # Yield...
        case_ds = get_yield_ann(case_ds, min_viable_hui=min_viable_hui, mxmats=mxmats)
        tmp_ds = case_ds.isel(patch=i_thisCrop_case, time=i_theseYears_case)
        yield_da = tmp_ds['YIELD_ANN']\
            .groupby(tmp_ds['patches1d_gi'])\
            .apply(xr.DataArray.sum, dim='patch', skipna=True)\
            .rename({'time': 'Year'})\
            * 1e-6 * 1e4 # g/m2 to tons/ha
        
        # Area...
        tmp_ds = lu_ds.isel(patch=i_thisCrop_lu, time=i_theseYears_lu)
        area_da = tmp_ds['AREA_CFT']\
            .groupby(tmp_ds['patches1d_gi'])\
            .apply(xr.DataArray.sum, dim='patch', skipna=True)\
            .rename({'time': 'Year'})\
            * 1e-4 # m2 to ha
        
        # Countries
        countries_da = countries.isel(patch=i_thisCrop_lu)\
            .groupby(tmp_ds['patches1d_gi'])\
            .apply(xr.DataArray.mean, dim='patch', skipna=True)
        
        # Get each top-N country's time series for this crop
        for c, country in enumerate(topN_countries):			  
            if country == "World":
                country_id = None
                yield_thisCountry_da = yield_da
                area_thisCountry_da = area_da
            else:
                country_id = countries_key.query(f'name == "{country}"')['num'].values
                if len(country_id) != 1:
                    raise RuntimeError(f'Expected 1 match of {country} in countries_key; got {len(country_id)}')
                yield_thisCountry_da = yield_da.where(countries_da == country_id)
                area_thisCountry_da = area_da.where(countries_da == country_id)
                
            area_ar[i_case,:,c] = area_thisCountry_da.sum(dim='patches1d_gi').values 
            
            # Production (tons)
            prod_ar[i_case,:,c] = (yield_thisCountry_da * area_thisCountry_da).sum(dim='patches1d_gi')
                
            # FAOSTAT production (tons) and area (ha)
            if i_case == 0:
                prod_faostat_yc[:,c] = get_faostat_country_ts(fao_all_ctry, thisCrop_fao, top_y1, top_yN, country, "Production")
                area_faostat_yc[:,c] = get_faostat_country_ts(fao_all_ctry, thisCrop_fao, top_y1, top_yN, country, "Area harvested")
            
            # EarthStat
            if np.all(np.isnan(prod_earthstat_yc[:,c])) and case['res']=='f09_g17':
                prod_earthstat_yc[:,c] = get_earthstat_country_ts(earthstats, case, 'Production', countries_map, thisCrop_clm, i_theseYears_earthstat, country_id)
                area_earthstat_yc[:,c] = get_earthstat_country_ts(earthstats, case, 'HarvestArea', countries_map, thisCrop_clm, i_theseYears_earthstat, country_id)
    
    if np.any(np.isnan(prod_ar)):
        raise RuntimeError("NaN in prod_ar")
    if np.any(np.isnan(area_ar)):
        raise RuntimeError("NaN in area_ar")
    if np.any(area_ar==0):
        raise RuntimeError("0 in area_ar")
    
    new_coords = {'Case': caselist,
                      'Year': topYears,
                      'Country': topN_countries}
    prod_da = xr.DataArray(data = prod_ar,
                                  coords = new_coords,
                                  attrs = {'units': 'tons'})
    area_da = xr.DataArray(data = area_ar,
                                  coords = new_coords,
                                  attrs = {'units': 'tons'})
    yield_da = prod_da / area_da
    yield_da = yield_da.assign_attrs({'units': 'tons/ha'})
    prod_faostat_da = xr.DataArray(data = prod_faostat_yc,
                                   coords = {'Year': topYears,
                                             'Country': topN_countries},
                                   attrs = {'units': 'tons'})
    area_faostat_da = xr.DataArray(data = area_faostat_yc,
                                   coords = {'Year': topYears,
                                             'Country': topN_countries},
                                   attrs = {'units': 'ha'})
    yield_faostat_da = prod_faostat_da / area_faostat_da
    yield_faostat_da = yield_faostat_da.assign_attrs({'units': 'tons/ha'})
    prod_earthstat_da = xr.DataArray(data = prod_earthstat_yc,
                                     coords = {'Year': topYears,
                                               'Country': topN_countries},)
    area_earthstat_da = xr.DataArray(data = area_earthstat_yc,
                                     coords = {'Year': topYears,
                                               'Country': topN_countries})
    yield_earthstat_da = prod_earthstat_da / area_earthstat_da
    
    topN_ds = xr.Dataset(data_vars = {'Production': prod_da,
                                      'Production (FAOSTAT)': prod_faostat_da,
                                      'Production (EarthStat)': prod_earthstat_da,
                                      'Area': area_da,
                                      'Area (FAOSTAT)': area_faostat_da,
                                      'Area (EarthStat)': area_earthstat_da,
                                      'Yield': yield_da,
                                      'Yield (FAOSTAT)': yield_faostat_da,
                                      'Yield (EarthStat)': yield_earthstat_da})
    
    # Detrend and get yield anomalies
    topN_dt_ds = xr.Dataset()
    topN_ya_ds = xr.Dataset()
    for i, v in enumerate(topN_ds):
        # Could make this cleaner by being smart in detrend()
        if "Case" in topN_ds[v].dims:
            tmp_dt_cyC = topN_ds[v].copy().values
            tmp_ya_cyC = topN_ds[v].copy().values
            for C, country in enumerate(topN_countries):
                tmp_dt_cy = tmp_dt_cyC[:,:,C]
                tmp_dt_cy = detrend(tmp_dt_cy)
                tmp_dt_cyC[:,:,C] = tmp_dt_cy
            topN_dt_ds[v] = xr.DataArray(data = tmp_dt_cyC,
                                                  coords = topN_ds[v].coords,
                                                  attrs = topN_ds[v].attrs)
            for C, country in enumerate(topN_countries):
                tmp_ya_cy = tmp_ya_cyC[:,:,C]
                tmp_ya_cy = yield_anomalies(tmp_ya_cy)
                tmp_ya_cyC[:,:,C] = tmp_ya_cy
            topN_ya_ds[v] = xr.DataArray(data = tmp_ya_cyC,
                                                  coords = topN_ds[v].coords,
                                                  attrs = topN_ds[v].attrs)
        else:
            tmp_dt_Cy = np.transpose(topN_ds[v].copy().values)
            tmp_dt_Cy = detrend(tmp_dt_Cy)
            topN_dt_ds[v] = xr.DataArray(data = np.transpose(tmp_dt_Cy),
                                                  coords = topN_ds[v].coords,
                                                  attrs = topN_ds[v].attrs)
            tmp_ya_Cy = np.transpose(topN_ds[v].copy().values)
            tmp_ya_Cy = yield_anomalies(tmp_ya_Cy)
            topN_ya_ds[v] = xr.DataArray(data = np.transpose(tmp_ya_Cy),
                                                  coords = topN_ds[v].coords,
                                                  attrs = topN_ds[v].attrs)
    
    topN_ya_ds[v].attrs['units'] = 'anomalies (unitless)'
    
    return topN_ds, topN_dt_ds, topN_ya_ds

 
def get_ts_prod_clm_yc_da(yield_gd, lu_ds, yearList, cropList_combined_clm):

    # Convert km2 to m2
    allCropArea = lu_ds.AREA*1e6 * lu_ds.LANDFRAC_PFT * lu_ds.PCT_CROP/100

    # Combined rainfed+irrigated
    
    cftList_str_clm = [] # Will fill during loop below
    cftList_int_clm = [] # Will fill during loop below
    ts_prod_clm_yc = np.full((len(yield_gd.time), len(cropList_combined_clm)), 0.0)
    for c, thisCrop in enumerate(cropList_combined_clm[:-1]):
        # print(f"{thisCrop}")
        for pft_str in yield_gd.ivt_str.values:
            if thisCrop.lower() not in pft_str:
                continue
            pft_int = utils.ivt_str2int(pft_str)
            cftList_str_clm.append(pft_str)
            cftList_int_clm.append(pft_int)
            # print(f"{pft_str}: {pft_int}")
            map_yield_thisCrop_clm = yield_gd.sel(ivt_str=pft_str)
            map_area_thisCrop_clm = allCropArea * lu_ds.PCT_CFT.sel(cft=pft_int)/100
            map_prod_thisCrop_clm = map_yield_thisCrop_clm * map_area_thisCrop_clm
            map_prod_thisCrop_clm = map_prod_thisCrop_clm * 1e-12
            if map_prod_thisCrop_clm.shape != map_yield_thisCrop_clm.shape:
                raise RuntimeError(f"Error getting production map {map_yield_thisCrop_clm.dims}: expected shape {map_yield_thisCrop_clm.shape} but got {map_prod_thisCrop_clm.shape}")
            ts_prod_thisCrop_clm = map_prod_thisCrop_clm.sum(dim=["lon","lat"])
            ts_prod_clm_yc[:,c] += ts_prod_thisCrop_clm.values
            # Total
            ts_prod_clm_yc[:,-1] += ts_prod_thisCrop_clm.values
            
    ts_prod_clm_yc_da = xr.DataArray(ts_prod_clm_yc,
                                                coords={"Year": yearList,
                                                        "Crop": cropList_combined_clm})
    return ts_prod_clm_yc_da

def get_ts_prod_clm_yc_da2(case_ds, lu_ds, yieldVar, cropList_combined_clm, quiet=False):
    
   # Get time dimension names.
   # To match time dimension on lu_ds, rename anything other than "time" to that.
   non_patch_dims = [x for x in case_ds[yieldVar].dims if x != "patch"]
   if len(non_patch_dims) != 1:
       raise RuntimeError(f"Expected one non-patch dimension of case_ds['{yieldVar}']; found {len(non_patch_dims)}: {non_patch_dims}")
   time_dim_in = non_patch_dims[0]
   if time_dim_in == "time":
       time_dim_out = "Year"
       yearList = [x.year for x in case_ds[yieldVar].time.values]
   elif time_dim_in == "gs":
       time_dim_out = "Growing season"
       yearList = case_ds[yieldVar].gs.values
   else:
       raise RuntimeError(f"Unknown time_dim_out for time_dim_in {time_dim_in}")
   if time_dim_in != "time":
       if not quiet:
           print(f"WARNING: Using calendar years from LU data with yield data of time dimension {time_dim_in}.")
       lu_ds = lu_ds.assign_coords({'time': case_ds[yieldVar].gs.values}).rename({'time': time_dim_in})
   
   prod_da = case_ds[yieldVar] * lu_ds['AREA_CFT']
   
   ts_prod_clm_yc_da = prod_da.groupby(case_ds['patches1d_itype_combinedCropCLM_str'])\
                       .apply(xr.DataArray.sum, dim='patch', skipna=True)\
                       .rename({time_dim_in: time_dim_out,
                                'patches1d_itype_combinedCropCLM_str': 'Crop'})\
                       .isel(Crop=slice(1,len(cropList_combined_clm)))\
                       * 1e-12
   ts_prod_clm_ySUM = ts_prod_clm_yc_da.sum(dim="Crop")\
                      .expand_dims(dim='Crop',
                                   axis=list(ts_prod_clm_yc_da.dims).index('Crop'))
   ts_prod_clm_yc_da = xr.concat((ts_prod_clm_yc_da,
                                  ts_prod_clm_ySUM),
                                  dim="Crop")\
                       .assign_coords({'Crop': cropList_combined_clm,
                                       time_dim_out: yearList})
   
   return ts_prod_clm_yc_da
   

def get_vegtype_str_figfile(vegtype_str_in):
    vegtype_str_out = vegtype_str_in
    if "soybean" in vegtype_str_in and "tropical" not in vegtype_str_in:
        vegtype_str_out = vegtype_str_out.replace("soybean", "temperate_soybean")
    for t in ["winter", "spring", "temperate", "tropical"]:
        if t in vegtype_str_out:
            vegtype_str_out = vegtype_str_out.replace(f"{t}_", "") + f"_{t}"
    if "irrigated" in vegtype_str_out:
        vegtype_str_out = vegtype_str_out.replace("irrigated_", "") + "_ir"
    else:
        vegtype_str_out = vegtype_str_out + "_rf"
    return vegtype_str_out


# Get vegtype str for figure titles
def get_vegtype_str_for_title(vegtype_str_in):
    vegtype_str_out = vegtype_str_in
    if "irrigated" in vegtype_str_in:
        vegtype_str_out = vegtype_str_out.replace("irrigated_", "") + " (ir)"
    else:
        vegtype_str_out = vegtype_str_out + " (rf)"
    if "temperate" in vegtype_str_in:
        vegtype_str_out = vegtype_str_out.replace("temperate_", "temp. ")
    if "tropical" in vegtype_str_in:
        vegtype_str_out = vegtype_str_out.replace("tropical_", "trop. ")
    elif "soybean" in vegtype_str_in:
        vegtype_str_out = "temp. " + vegtype_str_out
    if "soybean" in vegtype_str_in:
        vegtype_str_out = vegtype_str_out.replace("soybean", "soy")
    if "spring" in vegtype_str_in:
        vegtype_str_out = vegtype_str_out.replace("spring_", "spr. ")
    if "winter" in vegtype_str_in:
        vegtype_str_out = vegtype_str_out.replace("winter_", "win.")
    return vegtype_str_out


# Get vegtype str for figure titles
def get_vegtype_str_for_title_long(vegtype_str_in):
    vegtype_str_out = vegtype_str_in
    if "irrigated" in vegtype_str_in:
        vegtype_str_out = vegtype_str_out.replace("irrigated_", "Irrigated ")
    else:
        vegtype_str_out = "Rainfed " + vegtype_str_out
    vegtype_str_out = vegtype_str_out.replace("_", " ")
    if "soybean" in vegtype_str_in:
        vegtype_str_out = vegtype_str_out.replace("soybean", "soy")
    return vegtype_str_out


def get_vegtype_str_paramfile(vegtype_str_in):
    # Get vegtype str used in parameter file
    if vegtype_str_in == "soybean":
        vegtype_str_out = "temperate_soybean"
    elif vegtype_str_in == "irrigated_soybean":
        vegtype_str_out = "irrigated_temperate_soybean"
    else:
        vegtype_str_out = vegtype_str_in
    return vegtype_str_out


def get_window_radius(w):
    if not w % 2:
        raise RuntimeError("Window width must be odd")
    return int((w-1) / 2)


def get_yield(ds, **kwargs):
    
    # Fail if trying to get anything other than yield
    for key, selection in kwargs.items():
        if key=="out_var" and selection!="YIELD":
            raise RuntimeError("get_yield() may only be called with out_var='YIELD'. Did you mean to call zero_immatures() instead?")
    
    ds = zero_immatures(ds, out_var="YIELD", **kwargs)
    return ds


def get_yield_ann(ds, min_viable_hui=1.0, mxmats=None, force_update=False, lu_ds=None):
    
    mxmat_limited = bool(mxmats)
    
    def do_calculate(thisVar, ds, force_update, min_viable_hui, mxmat_limited):
        consider_skipping = (f'{thisVar}_ANN' in ds) and (not force_update)
        if consider_skipping:
            already_calculated = ds[f'{thisVar}_ANN'].attrs['min_viable_hui'] == min_viable_hui and ds[f'{thisVar}_ANN'].attrs['mxmat_limited'] == mxmat_limited
            is_locked = 'locked' in ds[f'{thisVar}_ANN'].attrs and ds[f'{thisVar}_ANN'].attrs['locked']
            do_calculate = not consider_skipping or not (already_calculated or is_locked)
        else:
            do_calculate = True
        return do_calculate
    
    #########
    # YIELD #
    #########
    
    if do_calculate("YIELD", ds, force_update, min_viable_hui, mxmat_limited):
    
        ds = get_yield(ds, min_viable_hui=min_viable_hui, mxmats=mxmats, forAnnual=True, force_update=force_update)
        
        tmp = ds["YIELD_PERHARV"].sum(dim='mxharvests', skipna=True).values
        grainc_to_food_ann_orig = ds["GRAINC_TO_FOOD_ANN"]
        ds["YIELD_ANN"] = xr.DataArray(data = tmp,
                                    attrs = grainc_to_food_ann_orig.attrs,
                                    coords = grainc_to_food_ann_orig.coords
                                    ).where(~np.isnan(grainc_to_food_ann_orig))
        
        # Save details
        ds["YIELD_ANN"].attrs['min_viable_hui'] = min_viable_hui
        ds["YIELD_ANN"].attrs['mxmat_limited'] = mxmat_limited
        if 'locked' in ds['YIELD_PERHARV'].attrs:
            ds["YIELD_ANN"].attrs['locked'] = True
    
    ##############
    # PRODUCTION #
    ##############
    
    if lu_ds is not None and do_calculate("PROD", ds, force_update, min_viable_hui, mxmat_limited):
        ds["PROD_ANN"] = ds["YIELD_ANN"] * lu_ds['AREA_CFT']
        ds['AREA_CFT'] = lu_ds['AREA_CFT']
        
        # Save details
        ds["PROD_ANN"].attrs['min_viable_hui'] = min_viable_hui
        ds["PROD_ANN"].attrs['mxmat_limited'] = mxmat_limited
        if 'locked' in ds['YIELD_PERHARV'].attrs:
            ds["PROD_ANN"].attrs['locked'] = True
    
    return ds


def import_max_gs_length(paramfile_dir, my_clm_ver, my_clm_subver):
    # Get parameter file
    pattern = os.path.join(paramfile_dir, f"*{my_clm_ver}_params.{my_clm_subver}.nc")
    paramfile = glob.glob(pattern)
    if len(paramfile) != 1:
        raise RuntimeError(f"Expected to find 1 match of {pattern}; found {len(paramfile)}")
    paramfile_ds = xr.open_dataset(paramfile[0])
    
    # Import max growing season length (stored in netCDF as nanoseconds!)
    paramfile_mxmats = paramfile_ds["mxmat"].values / np.timedelta64(1, 'D')
    
    # Import PFT name list
    paramfile_pftnames = [x.decode("UTF-8").replace(" ", "") for x in paramfile_ds["pftname"].values]
    
    # Build dict
    mxmat_dict = {}
    for i, pftname in enumerate(paramfile_pftnames):
        mxmat = paramfile_mxmats[i]
        if not np.isnan(mxmat):
            mxmat_dict[pftname] = int(mxmat)
        else:
            mxmat_dict[pftname] = np.inf
    
    return mxmat_dict

# E.g. import_rx_dates("sdate", sdates_rx_file, dates_ds0_orig)
def import_rx_dates(var_prefix, date_inFile, dates_ds, set_neg1_to_nan=True):
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
    
    did_warn = False
    for v in ds:
        v_new = v.replace(var_prefix,"gs")
        ds = ds.rename({v: v_new})
        
        # Set -1 prescribed GDD values to NaN. Only warn the first time.
        if set_neg1_to_nan and var_prefix == "gdd" and v_new != v and np.any(ds[v_new].values < 0):
            if np.any((ds[v_new].values < 0) & (ds[v_new].values != -1)):
                raise RuntimeError(f"Unexpected negative value in {v}")
            if not did_warn:
                print(f"Setting -1 rx GDD values to NaN")
                did_warn = True
            ds[v_new] = ds[v_new].where(ds[v_new] != -1)
    
    return ds


def import_output(filename, myVars, y1=None, yN=None, myVegtypes=utils.define_mgdcrop_list(), 
                        sdates_rx_ds=None, gdds_rx_ds=None, verbose=False):
    
    # Minimum harvest threshold allowed in PlantCrop()
    gdd_min = 50
    
    # Import
    this_ds = utils.import_ds(filename,
                                myVars=myVars,
                                myVegtypes=myVegtypes)
    
    # Trim to years of interest (do not include extra year needed for finishing last growing season)
    if y1 and yN:
        this_ds = check_and_trim_years(y1, yN, this_ds)
    else: # Assume including all growing seasons except last complete one are "of interest"
        y1 = this_ds.time.values[0].year
        yN = this_ds.time.values[-1].year - 2
        this_ds = check_and_trim_years(y1, yN, this_ds)
        
    # What vegetation types are included?
    vegtype_list = [x for x in this_ds.vegtype_str.values if x in this_ds.patches1d_itype_veg_str.values]
    
    # Check for consistency among sowing/harvest date/year info
    date_vars = ["SDATES_PERHARV", "SYEARS_PERHARV", "HDATES", "HYEARS"]
    all_nan = np.full(this_ds[date_vars[0]].shape, True)
    all_nonpos = np.full(this_ds[date_vars[0]].shape, True)
    all_pos = np.full(this_ds[date_vars[0]].shape, True)
    for v in date_vars:
        all_nan = all_nan & np.isnan(this_ds[v].values)
        all_nonpos = all_nonpos & (this_ds[v].values <= 0)
        all_pos = all_pos & (this_ds[v].values > 0)
    if np.any(np.bitwise_not(all_nan | all_nonpos | all_pos)):
        raise RuntimeError("Inconsistent missing/present values on mxharvests axis")
    
    # When doing transient runs, it's somehow possible for crops in newly-active patches to be *already alive*. They even have a sowing date (idop)! This will of course not show up in SDATES, but it does show up in SDATES_PERHARV. 
    # I could put the SDATES_PERHARV dates into where they "should" be, but instead I'm just going to invalidate those "seasons."
    #
    # In all but the last calendar year, which patches had no sowing?
    no_sowing_yp = np.all(np.isnan(this_ds.SDATES.values[:-1,:,:]), axis=1)
    # In all but the first calendar year, which harvests' jdays are < their sowings' jdays? (Indicates sowing the previous calendar year.)
    hsdate1_gt_hdate1_yp = this_ds.SDATES_PERHARV.values[1:,0,:] > this_ds.HDATES.values[1:,0,:]
    # Where both, we have the problem.
    falsely_alive_yp = no_sowing_yp & hsdate1_gt_hdate1_yp
    if np.any(falsely_alive_yp):
        print(f"Warning: {np.sum(falsely_alive_yp)} patch-seasons being ignored: Seemingly sown the year before harvest, but no sowings occurred that year.")
        falsely_alive_yp = np.concatenate((np.full((1, this_ds.dims["patch"]), False),
                                                    falsely_alive_yp),
                                                    axis=0)
        falsely_alive_y1p = np.expand_dims(falsely_alive_yp, axis=1)
        dummy_false_y1p = np.expand_dims(np.full_like(falsely_alive_yp, False), axis=1)
        falsely_alive_yhp = np.concatenate((falsely_alive_y1p, dummy_false_y1p), axis=1)
        for v in this_ds.data_vars:
            if this_ds[v].dims != ("time", "mxharvests", "patch"):
                continue
            this_ds[v] = this_ds[v].where(~falsely_alive_yhp)
            
    def check_no_negative(this_ds_in, varList_no_negative, which_file, verbose=False):
        tiny_negOK = 1e-12
        this_ds = this_ds_in.copy()
        for v in this_ds:
            if not any(x in v for x in varList_no_negative):
                continue
            the_min = np.nanmin(this_ds[v].values)
            if the_min < 0:
                if np.abs(the_min) <= tiny_negOK:
                    if verbose: print(f"Tiny negative value(s) in {v} (abs <= {tiny_negOK}) being set to 0 ({which_file})")
                else:
                    print(f"WARNING: Unexpected negative value(s) in {v}; minimum {the_min} ({which_file})")
                values = this_ds[v].copy().values
                values[np.where((values < 0) & (values >= -tiny_negOK))] = 0
                this_ds[v] = xr.DataArray(values,
                                          coords = this_ds[v].coords,
                                          dims = this_ds[v].dims,
                                          attrs = this_ds[v].attrs)

            elif verbose:
                print(f"No negative value(s) in {v}; min {the_min} ({which_file})")
        return this_ds
                
    def check_no_zeros(this_ds, varList_no_zero, which_file):
        for v in this_ds:
            if not any(x in v for x in varList_no_zero):
                continue
            if np.any(this_ds[v].values == 0):
                print(f"WARNING: Unexpected zero(s) in {v} ({which_file})")
            elif verbose:
                print(f"No zero value(s) in {v} ({which_file})")
            
    # Check for no zero values where there shouldn't be
    varList_no_zero = ["DATE", "YEAR"]
    check_no_zeros(this_ds, varList_no_zero, "original file")
    
    # Convert time*mxharvests axes to growingseason axis
    this_ds_gs = convert_axis_time2gs(this_ds, verbose=verbose, incl_orig=False)
    
    # These are needed for calculating yield later
    this_ds_gs['GRAINC_TO_FOOD_PERHARV'] = this_ds['GRAINC_TO_FOOD_PERHARV']
    this_ds_gs['GDDHARV_PERHARV'] = this_ds['GDDHARV_PERHARV']
    
    # Get growing season length
    this_ds["GSLEN_PERHARV"] = get_gs_len_da(this_ds["HDATES"] - this_ds["SDATES_PERHARV"])
    this_ds_gs["GSLEN"] = get_gs_len_da(this_ds_gs["HDATES"] - this_ds_gs["SDATES"])
    this_ds_gs["GSLEN_PERHARV"] = this_ds["GSLEN_PERHARV"]
    
    # Get HUI accumulation as fraction of required
    this_ds_gs["HUIFRAC"] = this_ds_gs["HUI"] / this_ds_gs["GDDHARV"]
    this_ds_gs["HUIFRAC_PERHARV"] = this_ds["HUI_PERHARV"] / this_ds["GDDHARV_PERHARV"]
    
    # Avoid tiny negative values
    varList_no_negative = ["GRAIN", "REASON", "GDD", "HUI", "YEAR", "DATE", "GSLEN"]
    this_ds_gs = check_no_negative(this_ds_gs, varList_no_negative, "new file", verbose=verbose)
    
    # Check for no zero values where there shouldn't be
    varList_no_zero = ["REASON", "DATE"]
    check_no_zeros(this_ds_gs, varList_no_zero, "new file")
    
    # Check that e.g., GDDACCUM <= HUI
    for vars in [["GDDACCUM", "HUI"],
                 ["SYEARS", "HYEARS"]]:
        if all(v in this_ds_gs for v in vars):
            check_v0_le_v1(this_ds_gs, vars, both_nan_ok=True, throw_error=True)
        
    # Check that prescribed calendars were obeyed
    if sdates_rx_ds:
        check_rx_obeyed(vegtype_list, sdates_rx_ds, this_ds, "this_ds", "SDATES")
    if gdds_rx_ds:
        check_rx_obeyed(vegtype_list, gdds_rx_ds, this_ds, "this_ds", "SDATES", "GDDHARV", gdd_min=gdd_min)

    # Convert time axis to integer year, saving original as 'cftime'
    this_ds_gs = this_ds_gs.assign_coords({'cftime': this_ds['time_bounds'].isel({'hist_interval': 0})})
    this_ds_gs = this_ds_gs.assign_coords({"time": [t.year for t in this_ds_gs['cftime'].values]})
    
    # Import monthly use and demand, if present
    pattern = os.path.join(os.path.dirname(filename), "*.h2.*.nc")
    irrig_file_patches = glob.glob(pattern)
    if irrig_file_patches:
        if len(irrig_file_patches) > 1:
            raise RuntimeError(f"Expected at most 1 *.h2.*.nc; found {len(irrig_file_patches)}")
        irrig_file_patches = irrig_file_patches[0]
        irrig_ds_patches = utils.import_ds(irrig_file_patches, myVegtypes=myVegtypes, chunks={'time': 12},
                                           myVars = ['QIRRIG_DEMAND', 'QIRRIG_DRIP', 'QIRRIG_SPRINKLER'])
        
        # Trim to years of interest
        # Time axis shows END of time period; fix this.
        new_time = [x[0] for x in irrig_ds_patches.time_bounds.values]
        irrig_ds_patches = irrig_ds_patches
        irrig_ds_patches = irrig_ds_patches.assign_coords(time=new_time)
        irrig_ds_patches = irrig_ds_patches.sel(time=slice(f"{y1}-01-01", f"{yN+1}-12-31"))
        
        time_mth_dimname = "time_mth"
        this_ds_gs["QIRRIG_DEMAND_PATCH_MTH"] = irrig_ds_patches["QIRRIG_DEMAND"].rename({'time': time_mth_dimname})
        this_ds_gs["QIRRIG_APPLIED_PATCH_MTH"] = irrig_ds_patches["QIRRIG_DRIP"].rename({'time': time_mth_dimname}) \
                                               + irrig_ds_patches["QIRRIG_SPRINKLER"].rename({'time': time_mth_dimname})
        this_ds_gs['QIRRIG_APPLIED_PATCH_MTH'].attrs = this_ds_gs['QIRRIG_DEMAND_PATCH_MTH'].attrs
        this_ds_gs["QIRRIG_APPLIED_PATCH_MTH"].attrs['long_name'] = 'water added via drip or sprinkler irrigation'
                
        # Calculate unfulfilled demand
        this_ds_gs['QIRRIG_UNFULFILLED_PATCH_MTH'] = this_ds_gs['QIRRIG_DEMAND_PATCH_MTH'] - this_ds_gs['QIRRIG_APPLIED_PATCH_MTH']
        this_ds_gs['QIRRIG_UNFULFILLED_PATCH_MTH'].attrs = this_ds_gs['QIRRIG_DEMAND_PATCH_MTH'].attrs
        this_ds_gs['QIRRIG_UNFULFILLED_PATCH_MTH'].attrs['long_name'] = 'irrigation demand unable to be filled'
                
        # Calculate yearly means
        for v in ["QIRRIG_DEMAND_PATCH_MTH", "QIRRIG_APPLIED_PATCH_MTH", "QIRRIG_UNFULFILLED_PATCH_MTH"]:
            v2 = v.replace('MTH', 'ANN')
            v2_da = utils.weighted_annual_mean(this_ds_gs[v], time_in='time_mth')
            this_ds_gs[v2] = v2_da.assign_coords({'time': this_ds_gs.time})
            if np.any(~np.isnan(this_ds_gs[v].values)) and not np.any(~np.isnan(this_ds_gs[v2].values)):
                raise RuntimeError("Annual mean turned into all NaN")
            this_ds_gs[v2].attrs = this_ds_gs[v].attrs
        
    return this_ds_gs


def make_axis(fig, ny, nx, n):
    ax = fig.add_subplot(ny,nx,n,projection=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False) # Turn off box outline
    return ax


def mask_immature(this_ds, this_vegtype, gridded_da):
    if "HARVEST_REASON_PERHARV" in this_ds:
        thisVar = "HARVEST_REASON_PERHARV"
    elif "HARVEST_REASON" in this_ds:
        thisVar = "HARVEST_REASON"
    else:
        raise RuntimeError('Neither HARVEST_REASON nor HARVEST_REASON_PERHARV found in this_ds.')
    reason_gridded = utils.grid_one_variable(this_ds, thisVar, vegtype=this_vegtype).squeeze(drop=True)
    gridded_da = gridded_da.where(reason_gridded == 1)
    return gridded_da


def open_lu_ds(filename, y1, yN, existing_ds, ungrid=True):
    # Open and trim to years of interest
    dsg = xr.open_dataset(filename).sel(time=slice(y1,yN))
    
    # Assign actual lon/lat coordinates
    dsg = dsg.assign_coords(lon=("lsmlon", existing_ds.lon.values),
                            lat=("lsmlat", existing_ds.lat.values))
    dsg = dsg.swap_dims({"lsmlon": "lon",
                         "lsmlat": "lat"})
    
    dsg['AREA_CFT'] = dsg.AREA*1e6 * dsg.LANDFRAC_PFT * dsg.PCT_CROP/100 * dsg.PCT_CFT/100
    dsg['AREA_CFT'].attrs = {'units': 'm2'}
    dsg['AREA_CFT'].load()
    
    if not ungrid:
        return dsg
    
    # Un-grid
    query_ilons = [int(x)-1 for x in existing_ds['patches1d_ixy'].values]
    query_ilats = [int(x)-1 for x in existing_ds['patches1d_jxy'].values]
    query_ivts = [list(dsg.cft.values).index(x) for x in existing_ds['patches1d_itype_veg'].values]
    
    ds = xr.Dataset(attrs=dsg.attrs)
    for v in ["AREA", "LANDFRAC_PFT", "PCT_CFT", "PCT_CROP", "AREA_CFT"]:
        if 'time' in dsg[v].dims:
            new_coords = existing_ds['GRAINC_TO_FOOD_ANN'].coords
        else:
            new_coords = existing_ds['patches1d_lon'].coords
        if 'cft' in dsg[v].dims:
            ds[v] = dsg[v].isel(lon=xr.DataArray(query_ilons, dims='patch'),
                                lat=xr.DataArray(query_ilats, dims='patch'),
                                cft=xr.DataArray(query_ivts, dims='patch'),
                                drop=True)\
                            .assign_coords(new_coords)
        else:
            ds[v] = dsg[v].isel(lon=xr.DataArray(query_ilons, dims='patch'),
                                lat=xr.DataArray(query_ilats, dims='patch'),
                                drop=True)\
                            .assign_coords(new_coords)
    for v in existing_ds:
        if "patches1d_" in v:
            ds[v] = existing_ds[v]
    ds['lon'] = dsg['lon']
    ds['lat'] = dsg['lat']
    return ds
    

def print_gs_table(ds):
    if "patch" in ds.dims:
        raise RuntimeError('Input Dataset must have no patch dimension')
    data = {}
    for v in ['gs', 'SDATES', 'GDDHARV', 'HDATES', 'HARVEST_REASON']:
        if ds[v].dims != ("gs",):
            raise RuntimeError(f'{v} dims must be (\'gs\'), not {ds[v].dims}')
        data[v] = ds[v].values
    print(f'Lon {ds.patches1d_lon.values} lat {ds.patches1d_lat.values}, {ds.patches1d_itype_veg_str.values}')
    maxrows = pd.get_option('display.max_rows')
    pd.set_option('display.max_rows', None)
    print(pd.DataFrame(data = data))
    pd.set_option('display.max_rows', maxrows)


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


def round_lonlats_to_match_da(ds_a, varname_a, tolerance, ds_b=None, varname_b=None):
    if (ds_b and varname_b) or (not ds_b and not varname_b):
        raise RuntimeError(f"You must provide one of ds_b or varname_b")
    elif ds_b:
        da_a = ds_a[varname_a]
        da_b = ds_b[varname_a]
    elif varname_b:
        da_a = ds_a[varname_a]
        da_b = ds_a[varname_b]
        
    unique_vals_a = np.unique(da_a)
    unique_vals_b = np.unique(da_b)
    
    # Depends on unique values being sorted ascending, which they should be from np.unique().
    # If calling with one of these being unique values from a coordinate axis, having that be unique_vals_b should avoid large differences due to some coordinate values not being represented in data.
    def get_diffs(unique_vals_a, unique_vals_b):
        max_diff = 0
        y0 = 0
        diffs = []
        for x in unique_vals_a:
            min_diff_fromx = np.inf
            for i,y in enumerate(unique_vals_b[y0:]):
                y_minus_x = y-x
                min_diff_fromx = min(min_diff_fromx, np.abs(y_minus_x))
                if y_minus_x >= 0:
                    break
            max_diff = max(max_diff, min_diff_fromx)
            if min_diff_fromx > 0:
                diffs.append(min_diff_fromx)
        return max_diff, diffs
    
    max_diff, diffs = get_diffs(unique_vals_a, unique_vals_b)
    
    if max_diff > tolerance:
        new_tolerance = np.ceil(np.log10(max_diff))
        if new_tolerance > 0:
            print(len(unique_vals_a))
            print(unique_vals_a)
            print(len(unique_vals_b))
            print(unique_vals_b)
            print(diffs)
            raise RuntimeError(f"Maximum disagreement in {varname_a} ({max_diff}) is intolerably large")
        new_tolerance = 10**new_tolerance
        if varname_b:
            print(f"Maximum disagreement between {varname_a} and {varname_b} ({max_diff}) is larger than default tolerance of {tolerance}. Increasing tolerance to {new_tolerance}.")
        else:
            print(f"Maximum disagreement in {varname_a} ({max_diff}) is larger than default tolerance of {tolerance}. Increasing tolerance to {new_tolerance}.")
        tolerance = new_tolerance
    neglog_tolerance = int(-np.log10(tolerance))
    
    da_a = np.round(da_a, neglog_tolerance)
    da_b = np.round(da_b, neglog_tolerance)
    
    def replace_vals(ds, new_da, varname):
        if len(ds[varname]) != len(new_da):
            raise RuntimeError(f"Not allowing you to change length of {varname} from {len(ds[varname])} to {len(new_da)}")
        if varname in ds.coords:
            ds = ds.assign_coords({varname: new_da.values})
        elif varname in ds.data_vars:
            ds[varname] = new_da
        else:
            raise TypeError(f"{varname} is not a coordinate or data variable; update round_lonlats_to_match_da()->replace_vals() to handle it")
        return ds
    
    ds_a = replace_vals(ds_a, da_a, varname_a)
    if ds_b:
        ds_b = replace_vals(ds_b, da_b, varname_a)
        return ds_a, ds_b, tolerance
    else:
        ds_a = replace_vals(ds_a, da_b, varname_b)
        return ds_a, tolerance


def round_lonlats_to_match_ds(ds_a, ds_b, which_coord, tolerance):
    tolerance_orig = np.inf
    i=0
    max_Nloops = 10
    while tolerance != tolerance_orig:
        if i > max_Nloops:
            raise RuntimeError(f"More than {max_Nloops} loops required in round_lonlats_to_match_ds()")
        tolerance_orig = tolerance
        patches_var = "patches1d_" + which_coord
        if patches_var in ds_a and which_coord in ds_a:
            ds_a, tolerance = round_lonlats_to_match_da(ds_a, patches_var, tolerance, varname_b=which_coord)
        if patches_var in ds_b and which_coord in ds_b:
            ds_b, tolerance = round_lonlats_to_match_da(ds_b, patches_var, tolerance, varname_b=which_coord)
        if patches_var in ds_a and patches_var in ds_b:
            ds_a, ds_b, tolerance = round_lonlats_to_match_da(ds_a, patches_var, tolerance, ds_b=ds_b)
        if which_coord in ds_a and which_coord in ds_b:
            ds_a, ds_b, tolerance = round_lonlats_to_match_da(ds_a, which_coord, tolerance, ds_b=ds_b)
    return ds_a, ds_b, tolerance


# For backwards compatibility with files missing SDATES_PERHARV.
def set_firstharv_nan(this_ds, this_var, firstharv_nan_inds):
    this_da = this_ds[this_var]
    this_array = this_da.values
    this_array[0,0,firstharv_nan_inds] = np.nan
    this_da.values = this_array
    this_ds[this_var] = this_da
    return this_ds


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


def shift_sim_timeseries(array_in_cy, whichWay, inds_obs, inds_sim):
    array_out_cy = np.full((array_in_cy.shape[0], array_in_cy.shape[1]-2), fill_value=np.nan)
    array_out_cy[inds_obs,:] = array_in_cy[inds_obs,1:-1]
    if whichWay == "L":
        array_out_cy[inds_sim,:] = array_in_cy[inds_sim,:-2]
    elif whichWay == "R":
        array_out_cy[inds_sim,:] = array_in_cy[inds_sim,2:]
    else:
        raise RuntimeError("whichWay {whichWay} not recognized. Use L or R.")
    return array_out_cy


def strip_cropname(x):
    for y in ['irrigated', 'temperate', 'tropical', 'spring', 'winter']:
        x = x.replace(y+"_", "")
    return x


def subtract_mean(in_ps):
   warnings.filterwarnings("ignore", message="Mean of empty slice") # Happens when you do np.nanmean() of an all-NaN 
   mean_p = np.nanmean(in_ps, axis=1)
   warnings.filterwarnings("always", message="Mean of empty slice")
   out_ps = in_ps - np.expand_dims(mean_p, axis=1)
   return out_ps


def time_units_and_trim(ds, y1, yN, dt_type):
    
    # Convert to dt_type
    time0 = ds.time.values[0]
    time0_type = type(time0)
    if not isinstance(time0, cftime.datetime):
        if not isinstance(time0, np.integer):
            raise TypeError(f"Unsure how to convert time of type {time0_type} to cftime.datetime")
        print(f"Converting integer time to {dt_type}, assuming values are years and each time step is Jan. 1.")
        ds = ds.assign_coords({"time": [dt_type(x, 1, 1) for x in ds.time.values]})
    elif not isinstance(time0, dt_type):
        raise TypeError(f"Expected time axis to be type {dt_type} but got {time0_type}")
    
    # Trim
    ds = ds.sel(time=slice(f"{y1}-01-01", f"{yN}-01-01"))
        
    return ds

# kwargs like gridded_ds_dim='ungridded_target_ds_dim'
# e.g.: lon='patches1d_ixy', lat='patches1d_jxy'
def ungrid(gridded_xr, ungridded_target_ds, coords_var, **kwargs):
    
    # Remove any empties from ungridded_target_ds
    for key, selection in kwargs.items():
        if isinstance(ungridded_target_ds[selection].values[0], str):
            # print(f"Removing ungridded_target_ds patches where {selection} is empty")
            isel_list = [i for i, x in enumerate(ungridded_target_ds[selection]) if x != ""]
            ungridded_target_ds = ungridded_target_ds.copy().isel({'patch': isel_list})
            
            ### I don't think this is necessary
            # unique_ungridded_selection = ungridded_target_ds[selection].values
            # isel_list = [i for i, x in enumerate(gridded_xr[key].values) if x in unique_ungridded_selection]
            # gridded_xr = gridded_xr.isel({key: isel_list})
    
    new_coords = ungridded_target_ds[coords_var].coords
        
    isel_dict = {}
    for key, selection in kwargs.items():
        if key in ['lon', 'lat']:
            isel_dict[key] = xr.DataArray([int(x)-1 for x in ungridded_target_ds[selection].values],
                                          dims = 'patch')
        else:
            values_list = list(gridded_xr[key].values)
            isel_dict[key] = xr.DataArray([values_list.index(x) for x in ungridded_target_ds[selection].values],
                                          dims = 'patch')
    ungridded_da = gridded_xr.isel(isel_dict, drop=True).assign_coords(new_coords)
    return ungridded_da


# After https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/weigcorr.htm
def weighted_cov(a, b, w):
    ma = np.average(a, weights=w)
    mb = np.average(b, weights=w)
    return np.sum(w*(a-ma)*(b-mb)) / np.sum(w)
    
def weighted_pearsons_r(x, y, w):
    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w)*weighted_cov(y, y, w))

def yield_anomalies(ps_in):
    if isinstance(ps_in, xr.DataArray):
        ps_out = ps_in.values
    else:
        ps_out = ps_in
    
    # After Jägermeyr & Frieler (2018, Sci. Adv.)
    ps_out = detrend(ps_out)
    ps_out /= np.expand_dims(np.std(ps_out, axis=1), axis=1)
    
    if isinstance(ps_in, xr.DataArray):
        ps_out = xr.DataArray(data = ps_out,
                              coords = ps_in.coords,
                              attrs = ps_in.attrs)
            
    return ps_out

def zero_immatures(ds, out_var="YIELD", min_viable_hui=None, mxmats=None, forAnnual=False, force_update=False):
    
    mxmat_limited = bool(mxmats)
    
    if any(x in out_var for x in ["YIELD", "MATURE"]):
        in_var = "GRAINC_TO_FOOD"
    else:
        raise RuntimeError(f"out_var {out_var} not recognized. Accepted values: *YIELD*, *MATURE*")
    
    huifrac_var = "HUIFRAC"
    gddharv_var = "GDDHARV"
    gslen_var = "GSLEN"
    if forAnnual:
        out_var += "_PERHARV"
        in_var += "_PERHARV"
        huifrac_var += "_PERHARV"
        gddharv_var += "_PERHARV"
        gslen_var += "_PERHARV"
    
    if out_var in ds and not force_update and all([x in ds[out_var].attrs for x in ['min_viable_hui', 'mxmat_limited']]):
        if ds[out_var].attrs['min_viable_hui'] == min_viable_hui and ds[out_var].attrs['mxmat_limited'] == mxmat_limited:
            return ds
        elif 'locked' in ds[out_var].attrs and ds[out_var].attrs['locked']:
            return ds
    
    ds[out_var] = ds[in_var].copy()
    
    # Set yield to zero where minimum viable HUI wasn't reached
    if min_viable_hui is not None:
        
        huifrac = ds[huifrac_var].copy().values
        huifrac[np.where(ds[gddharv_var].values==0)] = 1
        if min_viable_hui == "isimip3" or min_viable_hui == "ggcmi3":
            corn_value = 0.8
            other_value = 0.9
            min_viable_hui_touse = np.full_like(huifrac, fill_value=other_value)
            for veg_str in np.unique(ds.patches1d_itype_veg_str.values):
                if "corn" not in veg_str:
                    continue
                is_thistype = np.where((ds.patches1d_itype_veg_str.values == veg_str))[0]
                patch_index = list(ds[huifrac_var].dims).index("patch")
                if patch_index == 0:
                    min_viable_hui_touse[is_thistype,...] = corn_value
                elif patch_index == ds[huifrac_var].ndim - 1:
                    min_viable_hui_touse[...,is_thistype] = corn_value
                else:
                    # Need patch to be either first or last dimension to allow use of ellipses
                    raise RuntimeError(f"Temporarily rearrange min_viable_hui_touse so that patch dimension is first (0) or last ({ds[huifrac_var].ndim - 1}), instead of {patch_index}.")
        elif isinstance(min_viable_hui, str):
            raise RuntimeError(f"min_viable_hui {min_viable_hui} not recognized. Accepted strings are ggcmi3 or isimip3")
        else:
            min_viable_hui_touse = min_viable_hui
        if np.any(huifrac < min_viable_hui_touse):
            print(f"Setting {out_var} to zero where minimum viable HUI ({min_viable_hui}) wasn't reached")
            tmp_da = ds[in_var]
            tmp = tmp_da.copy().values
            dont_include = (huifrac < min_viable_hui_touse) & (tmp > 0)
            tmp[np.where(dont_include)] = 0
            if "MATURE" in out_var:
                tmp[np.where(~dont_include & ~np.isnan(tmp))] = 1
                tmp_da.attrs['units'] = 'fraction'
            ds[out_var] = xr.DataArray(data = tmp,
                                       attrs = tmp_da.attrs,
                                       coords = tmp_da.coords)
    
    # Get variants with values set to 0 if season was longer than CLM PFT parameter mxmat
    if mxmat_limited:
        tmp_da = ds[out_var]
        tmp_ra = tmp_da.copy().values
        for veg_str in np.unique(ds.patches1d_itype_veg_str.values):
            mxmat_veg_str = veg_str.replace("soybean", "temperate_soybean").replace("tropical_temperate", "tropical")
            mxmat = mxmats[mxmat_veg_str]
            tmp_ra[np.where((ds.patches1d_itype_veg_str.values == veg_str) & (ds[gslen_var].values > mxmat))] = 0
        ds[out_var] = xr.DataArray(data = tmp_ra,
                                   coords = tmp_da.coords,
                                   attrs = tmp_da.attrs)
    
    # Get *biomass* *actually harvested*
    if "YIELD" in out_var:
        ds[out_var] = adjust_grainC(ds[out_var], ds.patches1d_itype_veg_str)
    
    # Save details
    ds[out_var].attrs['min_viable_hui'] = min_viable_hui
    ds[out_var].attrs['mxmat_limited'] = mxmat_limited
    
    # Get dimensions in expected order (time/gs, patch)
    if not forAnnual:
        ds[out_var] = ds[out_var].transpose("gs", "patch")
    
    return ds

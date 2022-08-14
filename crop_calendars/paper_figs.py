# %% Setup

# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from cropcal_module import *

# Import general CTSM Python utilities
sys.path.append(my_ctsm_python_gallery)
import utils

import numpy as np
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import cartopy.crs as ccrs
from matplotlib import cm
import datetime as dt
import re
import importlib
from cropcal_module import *
import pandas as pd
import cftime

# Ignore these annoying warnings
import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


# %% Define functions

def adjust_grainC(da):
   # Parameters from Danica's 2020 paper
   fyield = 0.85 # 85% harvest efficiency (Kucharik & Brye, 2003)
   cgrain = 0.45 # 45% of biomass is C (Monfreda et al., 2008)
   da = da * fyield / cgrain
   return da


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


def get_ts_prod_clm_yc_da(yield_gd, lu_ds, yearList, cropList_combined_clm):

   # Convert km2 to m2
   allCropArea = lu_ds.AREA*1e6 * lu_ds.LANDFRAC_PFT * lu_ds.PCT_CROP/100

   # Combined rainfed+irrigated
   
   cftList_str_clm = [] # Will fill during loop below
   cftList_int_clm = [] # Will fill during loop below
   ts_prod_clm_yc = np.full((Nyears, len(cropList_combined_clm)), 0.0)
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


def import_output(filename, myVars, y1=None, yN=None, constantVars=None, myVegtypes=utils.define_mgdcrop_list(), 
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
   
   # Check that some things are constant across years
   if constantVars:
      t1 = 0 # 0-indexed
      for v in constantVars:
         ok = True

         t1_yr = this_ds.gs.values[t1]
         t1_vals = np.squeeze(this_ds[v].isel(gs=t1).values)

         for t in np.arange(t1+1, this_ds.dims["gs"]):
            t_yr = this_ds.gs.values[t]
            t_vals = np.squeeze(this_ds[v].isel(gs=t).values)
            ok_p = np.squeeze(t1_vals == t_vals)
            if not np.all(ok_p):
                  if ok:
                     print(f"❌ CLM output {v} unexpectedly vary over time:")
                  ok = False
                  if verbose:
                     for thisPatch in np.where(np.bitwise_not(ok_p))[0]:
                        thisLon = this_ds.patches1d_lon.values[thisPatch]
                        thisLat = this_ds.patches1d_lat.values[thisPatch]
                        thisCrop = this_ds.patches1d_itype_veg_str.values[thisPatch]
                        thisCrop_int = this_ds.patches1d_itype_veg.values[thisPatch]
                        thisStr = f"   Patch {thisPatch} (lon {thisLon} lat {thisLat}) {thisCrop} ({thisCrop_int})"
                        if v == "SDATES":
                              print(f"{thisStr}: Sowing {t1_yr} jday {int(t1_vals[thisPatch])}, {t_yr} jday {int(t_vals[thisPatch])}")
                        else:
                              print(f"{thisStr}: {t1_yr} {v} {int(t1_vals[thisPatch])}, {t_yr} {v} {int(t_vals[thisPatch])}")
                  else:
                     print(f"{v} timestep {t} does not match timestep {t1}")

         if ok:
            print(f"✅ CLM output {v} do not vary through {this_ds.dims['gs'] - t1} growing seasons of output.")
   
   # Convert time*mxharvests axes to growingseason axis
   this_ds_gs = convert_axis_time2gs(this_ds, verbose=verbose, incl_orig=False)
   
   # Get growing season length
   this_ds["GSLEN_PERHARV"] = get_gs_len_da(this_ds["HDATES"] - this_ds["SDATES_PERHARV"])
   this_ds_gs["GSLEN"] = get_gs_len_da(this_ds_gs["HDATES"] - this_ds_gs["SDATES"])
   
   # Get *biomass* *actually harvested*
   this_ds["GRAIN_HARV_TOFOOD_ANN"] = adjust_grainC(this_ds["GRAINC_TO_FOOD_ANN"])
   this_ds["GRAINC_TO_FOOD_PERHARV"] = adjust_grainC(this_ds["GRAINC_TO_FOOD_PERHARV"])
   this_ds_gs["GRAIN_HARV_TOFOOD_ANN"] = adjust_grainC(this_ds_gs["GRAINC_TO_FOOD_ANN"])
   this_ds_gs["GRAIN_HARV_TOFOOD"] = adjust_grainC(this_ds_gs["GRAINC_TO_FOOD"])
   
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
      check_rx_obeyed(vegtype_list, sdates_rx_ds, this_ds, 0, "SDATES")
   if gdds_rx_ds:
      check_rx_obeyed(vegtype_list, gdds_rx_ds, this_ds, 0, "SDATES", "GDDHARV", gdd_min=gdd_min)
   
   # Convert time axis to integer year
   this_ds_gs = this_ds_gs.assign_coords({"time": [t.year for t in this_ds.time_bounds.values[:,0]]})
   
   return this_ds_gs


def open_lu_ds(filename, y1, yN, existing_ds):
   # Open and trim to years of interest
   ds = xr.open_dataset(filename).sel(time=slice(y1,yN))

   # Assign actual lon/lat coordinates
   ds = ds.assign_coords(lon=("lsmlon", existing_ds.lon.values),
                              lat=("lsmlat", existing_ds.lat.values))
   ds = ds.swap_dims({"lsmlon": "lon",
                           "lsmlat": "lat"})
   return ds   


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


# %% Import model output

y1 = 1961
yN = 2010
yearList = np.arange(y1,yN+1)
Nyears = len(yearList)

# Define cases
cases = {}
# A run that someone else did
cases['cmip6_i.e21.IHIST.f09_g17'] = {'filepath': '/Users/Shared/CESM_work/CropEvalData_ssr/danica_timeseries-cmip6_i.e21.IHIST.f09_g17/month_1/ssr_trimmed_annual.nc',
                                      'constantVars': None,
                                      'res': 'f09_g17'}
# My run with normal CLM code + my outputs
cases['ctsm5.1.dev092'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013/2022-08-08/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013.clm2.h1.1950-01-01-00000.nc',
                    'constantVars': None,
                    'res': 'f19_g17'}

# Note that _PERHARV will be stripped off upon import
myVars = ['GRAINC_TO_FOOD_PERHARV', 'GRAINC_TO_FOOD_ANN', 'SDATES', 'SDATES_PERHARV', 'SYEARS_PERHARV', 'HDATES', 'HYEARS', 'GDDHARV_PERHARV', 'GDDACCUM_PERHARV', 'HUI_PERHARV', 'SOWING_REASON_PERHARV', 'HARVEST_REASON_PERHARV']

verbose_import = False
for i, (casename, case) in enumerate(cases.items()):
   print(f"Importing {casename}...")
   
   if casename == 'cmip6_i.e21.IHIST.f09_g17':
      cmip6_dir = "/Users/Shared/CESM_work/CropEvalData_ssr/danica_timeseries-cmip6_i.e21.IHIST.f09_g17/month_1/"
      cmip6_file = cmip6_dir + "ssr_trimmed_annual.nc"
      this_ds = xr.open_dataset(cmip6_file)

      # Convert gC/m2 to g/m2 actually harvested
      this_ds["GRAIN_HARV_TOFOOD_ANN"] = adjust_grainC(this_ds.GRAINC_TO_FOOD)

      # Rework to match what we already have
      this_ds = this_ds.assign_coords({"ivt": np.arange(np.min(this_ds.patches1d_itype_veg.values),
                                                        np.max(this_ds.patches1d_itype_veg.values)+1)})
      
      # Add vegtype_str
      this_ds['vegtype_str'] = xr.DataArray(data=[utils.ivt_int2str(x) for x in this_ds.ivt.values],
                                            dims = {"ivt": this_ds.ivt})

   else:
      this_ds = import_output(case['filepath'], myVars=myVars, constantVars=case['constantVars'], 
                              y1=y1, yN=yN, verbose=verbose_import)
   
   case["ds"] = this_ds




# %% Import LU data

# Define resolutions
reses = {}
# f09_g17 ("""1-degree"""; i.e., 1.25 lon x 0.9 lat)
reses["f09_g17"] = {"lu_path": "/Users/Shared/CESM_work/CropEvalData_ssr/landuse.timeseries_0.9x1.25_hist_78pfts_CMIP6_simyr1850-2015_c170824.nc4"}
# f19_g17 ("""2-degree"""; i.e., 2.5 lon x 1.9 lat)
reses["f19_g17"] = {"lu_path": "/Users/Shared/CESM_work/CropEvalData_ssr/landuse.timeseries_1.9x2.5_hist_78pfts_CMIP6_simyr1850-2015_c170824.nc4"}

# Import land use to reses dicts
for i, (resname, res) in enumerate(reses.items()):
   
   print(f"Importing {resname}...")
   
   # Find a matching case
   for (_, case) in cases.items():
      if case["res"] == resname:
         break
   if case["res"] != resname:
      raise RuntimeError(f"No case found with res {resname}")
   
   # Can avoid saving to res dict once I'm confident the ungridded Dataset works
   res['dsg'] = open_lu_ds(res['lu_path'], y1, yN, case['ds'])
   res['dsg'] = res['dsg'].assign_coords({"time": [cftime.DatetimeNoLeap(y, 1, 1, 0, 0, 0, 0, has_year_zero=True) for y in res['dsg'].time.values]})
      
print("Done importing land use.")

# Copy LU to cases dict
print("Copying LU data to case datasets:")
for i, (casename, case) in enumerate(cases.items()):
   print(casename + "...")
   case_ds = case['ds']
   
   # Remove missing vegetation types (???)
   bad_veg_types = [i for i,x in enumerate(case_ds.vegtype_str.values) if isinstance(x, float)]
   if np.any(np.in1d(case_ds.ivt.values[bad_veg_types], case_ds.ivt)):
      raise RuntimeError("Didn't expect to find any of these...")
   if len(bad_veg_types) > 0:
      break
      
   # Harmonize lon/lat (round to a high precision but one where float weirdness won't be an issue)
   initial_tolerance = 1e-6
   lu_dsg = reses[case['res']]['dsg'].copy()
   case_ds, lu_dsg, lon_tolerance = round_lonlats_to_match_ds(case_ds, lu_dsg, "lon", initial_tolerance)
   case_ds, lu_dsg, lat_tolerance = round_lonlats_to_match_ds(case_ds, lu_dsg, "lat", initial_tolerance)
      
   # Ensure that time axes are formatted the same
   case_ds = time_units_and_trim(case_ds, y1, yN, cftime.DatetimeNoLeap)
   lu_dsg = time_units_and_trim(lu_dsg, y1, yN, cftime.DatetimeNoLeap)

   # Save
   case['ds'] = case_ds
   case['ds'].load()
   reses[case['res']]['dsg'] = lu_dsg
   

print("Done.")


# %% Get FAO data from CSV

fao_all = pd.read_csv("/Users/sam/Documents/git_repos/CTSM_cropcals_hist/crop_calendar_MATLAB/FAOSTAT_data_6-15-2022.csv")
fao_all.rename(columns={"Item": "Crop"}, inplace=True, errors="raise") # Because I always confuse Item vs. Element

fao_prod = fao_all.copy().query("Element == 'Production'")

# Combine "Maize" and "Maize, green"
fao_prod.Crop.replace("Maize.*", "Maize", regex=True, inplace=True)
fao_prod = fao_prod.groupby(by=["Crop","Year","Element","Area","Unit"], as_index=False).agg("sum")

# Pick one rice
rice_to_keep = "Rice, paddy"
rice_to_drop = "Rice, paddy (rice milled equivalent)"
drop_this = [x == rice_to_drop for x in fao_prod.Crop]
fao_prod = fao_prod.drop(fao_prod[drop_this].index)

# Convert t to Mt
fao_prod.Value *= 1e-6

# Pivot and add Total column
fao_prod = fao_prod.copy().pivot(index="Year", columns="Crop", values="Value")
fao_prod['Total'] = fao_prod.sum(numeric_only=True, axis=1)

# Remove unneeded years
fao_prod = fao_prod.filter(items=np.arange(y1,yN+1), axis=0)

# Make no-sugarcane version
fao_prod_nosgc = fao_prod.drop(columns = ["Sugar cane", "Total"])
fao_prod_nosgc['Total'] = fao_prod_nosgc.sum(numeric_only=True, axis=1)



# %% Get CLM crop production

cropList_combined_clm = ["Corn", "Rice", "Cotton", "Soybean", "Sugarcane", "Wheat", "Total"]

for i, (casename, case) in enumerate(cases.items()):
   print(f"Gridding {casename}...")
   case_ds = case['ds']
   yield_gd = utils.grid_one_variable(case_ds.sel(time=case_ds.time.values), "GRAIN_HARV_TOFOOD_ANN")
   
   lu_dsg = reses[case['res']]['dsg']
   yield_gd = yield_gd.assign_coords({"lon": lu_dsg.lon,
                                      "lat": lu_dsg.lat})
   case['ds']['GRAIN_HARV_TOFOOD_ANN_GD'] = yield_gd
   
   case['ds']['ts_prod_yc'] = get_ts_prod_clm_yc_da(yield_gd, lu_dsg, yearList, cropList_combined_clm)


# %% Compare total production to FAO

# Set up figure
ny = 1
nx = 3
# figsize = (14, 7.5)
figsize = (18, 7.5)

# All crops
f, axes = plt.subplots(ny, nx, sharey="row", figsize=figsize)
fao_prod.plot(ax=axes[0])
axes[0].title.set_text("FAO")
axes[0].set_ylabel("Mt")
for i, (casename, case) in enumerate(cases.items()):
   case['ds'].ts_prod_yc.plot.line(x="Year", ax=axes[i+1])
   axes[i+1].title.set_text(casename)

# Ignoring sugarcane
f, axes = plt.subplots(ny, nx, sharey="row", figsize=figsize)
fao_prod_nosgc.plot(ax=axes[0])
axes[0].title.set_text("FAO")
axes[0].set_ylabel("Mt")
for i, (casename, case) in enumerate(cases.items()):
   case['ds'].ts_prod_yc.sel({"Crop": [c for c in cropList_combined_clm if "Sugar" not in c]}).plot.line(x="Year", ax=axes[i+1])
   axes[i+1].title.set_text(casename)









# %%
cropList = yield_gd.ivt_str.values



ts_total_production_clm = None
for pft_str in cropList:
   pft_int = utils.ivt_str2int(pft_str)
   print(f"{pft_str}: {pft_int}")
   map_yield_thisCrop_clm = yield_gd.sel(ivt_str=pft_str)
   map_area_thisCrop_clm = allCropArea * lu_ds.PCT_CFT.sel(cft=pft_int)/100
   map_prod_thisCrop_clm = map_yield_thisCrop_clm * map_area_thisCrop_clm
   map_prod_thisCrop_clm = map_prod_thisCrop_clm * 1e-12 # Convert g to million tons
   if not isinstance(ts_total_production_clm, xr.DataArray):
      ts_total_production_clm = map_prod_thisCrop_clm.sum(dim=["lon","lat"])
   else:
      ts_total_production_clm = map_prod_thisCrop_clm.sum(dim=["lon","lat"]) + ts_total_production_clm
      
ts_total_production_clm.plot()

# %%






# %%

gridded = utils.grid_one_variable(this_ds, "GRAINC_TO_FOOD_ANN")
this_map = gridded.mean(dim="time").sel(ivt_str="spring_wheat")
this_map.squeeze() + lu_ds.PFTDATA_MASK


# %%

this_map = utils.grid_one_variable(this_ds, "GRAINC_TO_FOOD_ANN")

# layout = "3x1"
layout = "2x2"
bin_width = 15
lat_bin_edges = np.arange(0, 91, bin_width)

fontsize_titles = 18
fontsize_axislabels = 15
fontsize_ticklabels = 15

def make_map(ax, this_map, this_title, bin_width, fontsize_ticklabels, fontsize_titles): 
    im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
            this_map, shading="auto")
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

# %%    
    
# Set up figure and first subplot
ny = 1
nx = 2
fig = plt.figure(figsize=(14, 7.5))
ax = fig.add_subplot(ny,nx,1,projection=ccrs.PlateCarree())

make_map(ax, this_map.mean(dim="time").sel(ivt_str="spring_wheat"), "spring wheat", bin_width, fontsize_ticklabels, fontsize_titles)
ax = fig.add_subplot(ny,nx,2,projection=ccrs.PlateCarree())
make_map(ax, this_map.mean(dim="time").sel(ivt_str="irrigated_spring_wheat"), "irrigated spring wheat", bin_width, fontsize_ticklabels, fontsize_titles)
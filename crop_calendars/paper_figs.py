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

# Ignore these annoying warnings
import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


# %% Define functions

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
   
   # Avoid tiny negative values
   varList_no_negative = ["GRAINC", "REASON", "GDD", "HUI", "YEAR", "DATE", "GSLEN"]
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




# %% Import

# thisfile = "/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013/2022-08-05_test/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013.clm2.h1.1950-01-01-00000.nc"
# thisfile = "/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013/2022-08-07_test/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013.clm2.h1.1950-01-01-00000.nc"
thisfile = "/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013/2022-08-07_test2/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013.clm2.h1.1950-01-01-00000.nc"

# Note that _PERHARV will be stripped off upon import
myVars = ['GRAINC_TO_FOOD_PERHARV', 'GRAINC_TO_FOOD_ANN', 'SDATES', 'SDATES_PERHARV', 'SYEARS_PERHARV', 'HDATES', 'HYEARS', 'GDDHARV_PERHARV', 'GDDACCUM_PERHARV', 'HUI_PERHARV', 'SOWING_REASON_PERHARV', 'HARVEST_REASON_PERHARV']

this_ds = import_output(thisfile, myVars=myVars, verbose=False)

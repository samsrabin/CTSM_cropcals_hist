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

# Ignore these annoying warnings
import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


# %% Define functions

def import_output(filename, myVars, y1=None, yN=None, constantVars=None, myVegtypes=utils.define_mgdcrop_list(), 
                  sdates_rx_ds=None, gdds_rx_ds=None):
   
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
   
   # How many growing seasons can we use? Ignore last season because it can be incomplete for some gridcells.
   Ngs = this_ds.dims['time'] - 1
   
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
   
   return this_ds
         
   # Align output sowing and harvest dates/etc.
   this_ds = convert_axis_time2gs(Ngs, this_ds, myVars)
   
   # Get growing season length
   this_ds["GSLEN"] = get_gs_len_da(this_ds["HDATES"] - this_ds["SDATES"])
   
   # Check that some things are constant across years
   if constantVars:
      verbose = True
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
   
   # Check that GDDACCUM_PERHARV <= HUI_PERHARV
   if all(v in this_ds for v in ["GDDACCUM_PERHARV", "HUI_PERHARV"]):
      check_gddaccum_le_hui(this_ds)
      
   # Check that prescribed calendars were obeyed
   if sdates_rx_ds:
      check_rx_obeyed(vegtype_list, sdates_rx_ds, this_ds, 0, "SDATES")
   if gdds_rx_ds:
      check_rx_obeyed(vegtype_list, gdds_rx_ds, this_ds, 0, "SDATES", "GDDHARV_PERHARV", gdd_min=gdd_min)
   
   return this_ds




# %% Import
from cropcal_module import *

thisfile = "/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013/2022-08-05_test/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013.clm2.h1.1950-01-01-00000.nc"

myVars = ['GRAINC_TO_FOOD_PERHARV', 'GRAINC_TO_FOOD_ANN', 'SDATES', 'SDATES_PERHARV', 'SYEARS_PERHARV', 'HDATES', 'HYEARS', 'GDDHARV_PERHARV', 'GDDACCUM_PERHARV', 'HUI_PERHARV', 'SOWING_REASON_PERHARV', 'HARVEST_REASON_PERHARV']

this_ds = import_output(thisfile, myVars=myVars)


# %% Convert time*mxharvests axes to growingseason axis

def pym_to_pg(pym, quiet=False):
   pg = np.reshape(pym, (pym.shape[0],-1))
   ok_pg = pg[~np.isnan(pg)]
   if not quiet:
      print(f"{ok_pg.size} included; unique N seasons = {np.unique(np.sum(~np.isnan(pg), axis=1))}")
   return pg

# We expect "1 fewer than Nyears" valid values, because we have to ignore any finalyear-planted seasons that complete
Npatch = this_ds.dims["patch"]
Ngs = this_ds.dims["time"]-1
expected_valid = Npatch*Ngs

print(f'Start: discrepancy of {np.sum(~np.isnan(this_ds.HDATES.values)) - expected_valid} patch-seasons')

# Set all non-positive values to NaN
hdates_ymp = this_ds.HDATES.where(this_ds.HDATES > 0).values.copy()
hdates_pym = np.transpose(hdates_ymp.copy(), (2,0,1))
sdates_ymp = this_ds.SDATES_PERHARV.where(this_ds.SDATES_PERHARV > 0).values.copy()
sdates_pym = np.transpose(sdates_ymp.copy(), (2,0,1))
print(hdates_pym.shape)
hdates_pym[hdates_pym <= 0] = np.nan

# "Ignore harvests from before this output began"
first_season_before_first_year = hdates_pym[:,0,0] < sdates_pym[:,0,0]
hdates_pym[first_season_before_first_year,0,0] = np.nan
sdates_pym[first_season_before_first_year,0,0] = np.nan
print(f'After "Ignore harvests from before this output began: discrepancy of {np.sum(~np.isnan(hdates_pym)) - expected_valid} patch-seasons')

# "In years with no sowing, pretend the first no-harvest is meaningful, unless that was intentionally ignored above."
hdates_pym2 = hdates_pym.copy()
sdates_pym2 = sdates_pym.copy()
nosow_py = np.transpose(np.all(np.bitwise_not(this_ds.SDATES.values > 0),axis=1))
# Need to generalize for mxharvests > 2
where_nosow_py_1st = np.where(nosow_py & np.isnan(hdates_pym[:,:,0]) & ~np.tile(np.expand_dims(first_season_before_first_year, axis=1), (1,Ngs+1)))
hdates_pym2[where_nosow_py_1st[0], where_nosow_py_1st[1], 0] = -np.inf
sdates_pym2[where_nosow_py_1st[0], where_nosow_py_1st[1], 0] = -np.inf
where_nosow_py_2nd = np.where(nosow_py & ~np.isnan(hdates_pym[:,:,0]) & np.isnan(hdates_pym[:,:,1]))
hdates_pym2[where_nosow_py_2nd[0], where_nosow_py_2nd[1], 1] = -np.inf
sdates_pym2[where_nosow_py_2nd[0], where_nosow_py_2nd[1], 1] = -np.inf
hdates_pg = pym_to_pg(hdates_pym2.copy())
sdates_pg = pym_to_pg(sdates_pym2.copy(), quiet=True)
print(f'After "In years with no sowing, pretend the first no-harvest is meaningful: discrepancy of {np.sum(~np.isnan(hdates_pg)) - expected_valid} patch-seasons')

# "Ignore any harvests that were planted in the final year, because some cells will have incomplete growing seasons for the final year."
lastyear_complete_season = (hdates_pg[:,-2:] >= sdates_pg[:,-2:]) | np.isinf(hdates_pg[:,-2:])
def ignore_lastyear_complete_season(pg, excl):
   tmp_L = pg[:,:-2]
   tmp_R = pg[:,-2:]
   tmp_R[np.where(excl)] = np.nan
   pg = np.concatenate((tmp_L, tmp_R), axis=1)
   return pg
hdates_pg2 = ignore_lastyear_complete_season(hdates_pg.copy(), lastyear_complete_season)
sdates_pg2 = ignore_lastyear_complete_season(sdates_pg.copy(), lastyear_complete_season)
discrepancy = np.sum(~np.isnan(hdates_pg2)) - expected_valid
print(f'After "Ignore any harvests that were planted in the final year, because other cells will have incomplete growing seasons for the final year": discrepancy of {discrepancy} patch-seasons')
print(f"unique N seasons = {np.unique(np.sum(~np.isnan(hdates_pg2), axis=1))}")

if discrepancy == 0:
   # Remove the nans
   # Change -infs to nans
   # For now:
   pass
else:
   raise RuntimeError(f"Can't convert time*mxharvests axes to growingseason axis: discrepancy of {discrepancy} patch-seasons")




# %% Query a patch with bad # seasons
# p = np.random.choice(np.where(np.sum(~np.isnan(hdates_pg2), axis=1)>4)[0], 1)[0]
# p=47577
# p = np.random.choice(np.where(np.sum(~np.isnan(hdates_pg2), axis=1)<4)[0], 1)[0]
# p = 38250

nan_firstsow = np.where(np.isnan(this_ds.SDATES.values[0,0,:]))[0]
too_many_gs = np.where(np.sum(~np.isnan(hdates_pg2), axis=1)>4)[0]
[x for x in too_many_gs if x not in nan_firstsow] # if empty: all cells with too many growing seasons had NaN in first row
nan_firstsow_but_ok_Ngs = [x for x in nan_firstsow if x not in too_many_gs]
np.sum(~np.isnan(this_ds.SDATES.values[:,:,nan_firstsow_but_ok_Ngs])) # if >0: some nan_firstsow_but_ok_Ngs patches had non-nan sowings at some point
# ps = np.random.choice(too_many_gs, 1)
# ps = [47637] # Harvested y2d18 but (a) now sowing y1 and (b) sowing y2 not until d263
ps = too_many_gs

import pandas as pd
for p in ps:
   print(p)
   print("Original:")
   df = pd.DataFrame(np.concatenate((this_ds.SDATES.values[:,:,p], 
                                    this_ds.SDATES_PERHARV.values[:,:,p], 
                                    this_ds.HDATES.values[:,:,p],
                                    ), 
                                    axis=1))
   df.columns = ["sdates", "hsdate1", "hsdate2", "hdate1", "hdate2"]
   print(df)

# print("Masked:")
# print(sdates_ymp[:,:,p])
# print(hdates_ymp[:,:,p])
# print('After "Ignore harvests from before this output began"')
# print(np.transpose(sdates_pym, (1,2,0))[:,:,p])
# print(np.transpose(hdates_pym, (1,2,0))[:,:,p])
# print('After "In years with no sowing, pretend the first no-harvest is meaningful"')
# print(np.transpose(sdates_pym2, (1,2,0))[:,:,p])
# print(np.transpose(hdates_pym2, (1,2,0))[:,:,p])
# print(sdates_pg[p,:])
# print(hdates_pg[p,:])
# print('After "Ignore any harvests that were planted in the final year, because some cells will have incomplete growing seasons for the final year"')
# print(sdates_pg2[p,:])
# print(hdates_pg2[p,:])

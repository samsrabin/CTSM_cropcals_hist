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

outDir_figs = "/Users/sam/Documents/Dropbox/2021_Rutgers/CropCalendars/Figures/"

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
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")


# %% Define functions

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
         tmp /= dm_frac
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


def equalize_colorbars(ims):
   vmin = np.inf
   vmax = -np.inf
   nims = len(ims)
   for im in ims:
      vmin = min(vmin, im.get_clim()[0])
      vmax = max(vmax, im.get_clim()[1])
   for i in np.arange(nims):
      ims[i].set_clim(vmin, vmax)


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

         t1_yr = this_ds.time.values[t1]
         t1_vals = np.squeeze(this_ds[v].isel(time=t1).values)

         for t in np.arange(t1+1, this_ds.dims["time"]):
            t_yr = this_ds.time.values[t]
            t_vals = np.squeeze(this_ds[v].isel(time=t).values)
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
            print(f"✅ CLM output {v} do not vary through {this_ds.dims['time'] - t1} years of output.")
   
   # Convert time*mxharvests axes to growingseason axis
   this_ds_gs = convert_axis_time2gs(this_ds, verbose=verbose, incl_orig=False)
   
   # Get growing season length
   this_ds["GSLEN_PERHARV"] = get_gs_len_da(this_ds["HDATES"] - this_ds["SDATES_PERHARV"])
   this_ds_gs["GSLEN"] = get_gs_len_da(this_ds_gs["HDATES"] - this_ds_gs["SDATES"])
   
   # Get *biomass* *actually harvested*
   this_ds["GRAIN_HARV_TOFOOD_ANN"] = adjust_grainC(this_ds["GRAINC_TO_FOOD_ANN"], this_ds.patches1d_itype_veg_str)
   this_ds["GRAINC_TO_FOOD_PERHARV"] = adjust_grainC(this_ds["GRAINC_TO_FOOD_PERHARV"], this_ds.patches1d_itype_veg_str)
   this_ds_gs["GRAIN_HARV_TOFOOD_ANN"] = adjust_grainC(this_ds_gs["GRAINC_TO_FOOD_ANN"], this_ds.patches1d_itype_veg_str)
   this_ds_gs["GRAIN_HARV_TOFOOD"] = adjust_grainC(this_ds_gs["GRAINC_TO_FOOD"], this_ds.patches1d_itype_veg_str)
   
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


def make_map(ax, this_map, fontsize, lonlat_bin_width=None, units=None, cmap='viridis', vrange=None, linewidth=1.0, this_title=None, show_cbar=False): 
   im = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
                       this_map, shading="auto",
                       cmap=cmap)
   if vrange:
      im.set_clim(vrange[0], vrange[1])
   ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
   ax.coastlines(linewidth=linewidth)
   if this_title:
      ax.set_title(this_title, fontsize=fontsize['titles'])
   if show_cbar:
      cbar = plt.colorbar(im, orientation="horizontal", fraction=0.1, pad=0.02)
      cbar.set_label(label=units, fontsize=fontsize['axislabels'])
      cbar.ax.tick_params(labelsize=fontsize['ticklabels'])
    
   def set_ticks(lonlat_bin_width, fontsize, x_or_y):
      
      if x_or_y == "x":
         ticks = np.arange(-180, 181, lonlat_bin_width)
      else:
         ticks = np.arange(-60, 91, lonlat_bin_width)
         
      ticklabels = [str(x) for x in ticks]
      for i,x in enumerate(ticks):
         if x%2:
               ticklabels[i] = ''
      
      if x_or_y == "x":
         plt.xticks(ticks, labels=ticklabels,
                     fontsize=fontsize['ticklabels'])
      else:
         plt.yticks(ticks, labels=ticklabels,
                     fontsize=fontsize['ticklabels'])
   
   if lonlat_bin_width:
      set_ticks(lonlat_bin_width, fontsize, "y")
      # set_ticks(lonlat_bin_width, fontsize, "x")
   else:
      # Need to do this for subplot row labels
      set_ticks(-1, fontsize, "y")
      plt.yticks([])
   for x in ax.spines:
      ax.spines[x].set_visible(False)
   
   if show_cbar:
      return im, cbar
   else:
      return im, None


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
cases['cmip6'] = {'filepath': '/Users/Shared/CESM_work/CropEvalData_ssr/danica_timeseries-cmip6_i.e21.IHIST.f09_g17/month_1/ssr_trimmed_annual.nc',
                  'constantVars': None,
                  'res': 'f09_g17'}
# My run with normal CLM code + my outputs
cases['ctsm5.1.dev092'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013/2022-08-08/cropcals.f19-g17.sdates_perharv.IHistClm50BgcCrop.1950-2013.clm2.h1.1950-01-01-00000.nc',
                    'constantVars': None,
                    'res': 'f19_g17'}
# My run with rx_crop_calendars2 code but CLM calendars
cases['mycode_clmcals'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1950-2013.clm/2022-08-09/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1950-2013.clm.clm2.h1.1950-01-01-00000.nc',
                    'constantVars': None,
                    'res': 'f19_g17'}
# My run with rx_crop_calendars2 code and GGCMI calendars
cases['mycode_ggcmicals'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1976-2013_gddforced/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1976-2013_gddforced.clm2.h1.1950-01-01-00000.nc',
                    'constantVars': None,  # Would be SDATES, but land use changes over time
                    'res': 'f19_g17'}

# Note that _PERHARV will be stripped off upon import
myVars = ['GRAINC_TO_FOOD_PERHARV', 'GRAINC_TO_FOOD_ANN', 'SDATES', 'SDATES_PERHARV', 'SYEARS_PERHARV', 'HDATES', 'HYEARS', 'GDDHARV_PERHARV', 'GDDACCUM_PERHARV', 'HUI_PERHARV', 'SOWING_REASON_PERHARV', 'HARVEST_REASON_PERHARV']

verbose_import = False
caselist = []
for i, (casename, case) in enumerate(cases.items()):
   print(f"Importing {casename}...")
   caselist.append(casename)
   
   if casename == 'cmip6':
      this_ds = xr.open_dataset(case['filepath'])

      # Convert gC/m2 to g/m2 actually harvested
      this_ds["GRAIN_HARV_TOFOOD_ANN"] = adjust_grainC(this_ds.GRAINC_TO_FOOD, this_ds.patches1d_itype_veg)

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
   
   res['dsg'] = open_lu_ds(res['lu_path'], y1, yN, case['ds'])
   res['dsg'] = res['dsg'].assign_coords({"time": [cftime.DatetimeNoLeap(y, 1, 1, 0, 0, 0, 0, has_year_zero=True) for y in res['dsg'].time.values]})
   
   res['dsg']['AREA_CFT'] = res['dsg'].AREA*1e6 * res['dsg'].LANDFRAC_PFT * res['dsg'].PCT_CROP/100 * res['dsg'].PCT_CFT/100
   res['dsg']['AREA_CFT'].attrs = {'units': 'm2'}
      
print("Done importing land use.")

# Harmonize LU and cases
print("Harmonizing LU data and case datasets:")
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

def get_fao_data(fao_all, element):
   fao_this = fao_all.copy().query(f"Element == '{element}'")

   # Combine "Maize" and "Maize, green"
   fao_this.Crop.replace("Maize.*", "Maize", regex=True, inplace=True)
   fao_this = fao_this.groupby(by=["Crop","Year","Element","Area","Unit"], as_index=False).agg("sum")

   # Pick one rice
   rice_to_keep = "Rice, paddy"
   rice_to_drop = "Rice, paddy (rice milled equivalent)"
   drop_this = [x == rice_to_drop for x in fao_this.Crop]
   fao_this = fao_this.drop(fao_this[drop_this].index)

   # Convert t to Mt
   if element == 'Production':
      fao_this.Value *= 1e-6

   # Pivot and add Total column
   fao_this = fao_this.copy().pivot(index="Year", columns="Crop", values="Value")
   fao_this['Total'] = fao_this.sum(numeric_only=True, axis=1)

   # Remove unneeded years
   fao_this = fao_this.filter(items=np.arange(y1,yN+1), axis=0)

   # Make no-sugarcane version
   fao_this_nosgc = fao_this.drop(columns = ["Sugar cane", "Total"])
   fao_this_nosgc['Total'] = fao_this_nosgc.sum(numeric_only=True, axis=1)
   
   return fao_this, fao_this_nosgc

fao_prod, fao_prod_nosgc = get_fao_data(fao_all, 'Production')
fao_area, fao_area_nosgc = get_fao_data(fao_all, 'Area harvested')


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
print("Done gridding.")


# %% Import FAO Earthstat (gridded FAO data)

print("Importing FAO EarthStat...")
earthstats = {}

# Import high res
earthstats['f09_g17'] = xr.open_dataset('/Users/Shared/CESM_work/CropEvalData_ssr/FAO-EarthStatYields/EARTHSTATMIRCAUNFAOCropDataDeg09.nc')

# Include just crops we care about
cropList_fao_gd_all = ['Wheat', 'Maize', 'Rice', 'Barley', 'Rye', 'Millet', 'Sorghum', 'Soybeans', 'Sunflower', 'Potatoes', 'Cassava', 'Sugar cane', 'Sugar beet', 'Oil palm', 'Rape seed / Canola', 'Groundnuts / Peanuts', 'Pulses', 'Citrus', 'Date palm', 'Grapes / Vine', 'Cotton', 'Cocoa', 'Coffee', 'Others perennial', 'Fodder grasses', 'Others annual', 'Fibre crops', 'All crops']
cropList_fao_gd = ["Maize", "Rice", "Cotton", "Soybeans", "Sugar cane", "Wheat"]
earthstats['f09_g17'] = earthstats['f09_g17'].isel(crop=[cropList_fao_gd_all.index(x) for x in cropList_fao_gd]).assign_coords({'crop': cropList_combined_clm[:-1]})

# Include just years we care about
earthstats['f09_g17'] = earthstats['f09_g17'].isel(time=[i for i,x in enumerate(earthstats['f09_g17'].time.values) if x.year in np.arange(y1,yN+1)])

# Interpolate to lower res, starting with just variables that won't get messed up by the interpolation
print("Interpolating to f19_g17...")
interp_lons = reses['f19_g17']['dsg'].lon.values
interp_lats = reses['f19_g17']['dsg'].lat.values
drop_vars = ['Area', 'HarvestArea', 'IrrigatedArea', 'PhysicalArea', 'RainfedArea', 'Production']
earthstats['f19_g17'] = earthstats['f09_g17']\
   .drop(labels=drop_vars)\
   .interp(lon=interp_lons, lat=interp_lats)
   
# Disallow negative interpolated values
for v in earthstats['f19_g17']:
   vals = earthstats['f19_g17'][v].values
   vals[vals < 0] = 0
   earthstats['f19_g17'][v] = xr.DataArray(data = vals,
                                           coords = earthstats['f19_g17'][v].coords,
                                           attrs = earthstats['f19_g17'][v].attrs)
   
# These are not exact, but it seems like:
#    PhysicalArea = RainfedArea + IrrigatedArea (max diff ~51 ha, max rel diff ~0.01%)
#    PhysicalFraction = RainfedFraction + IrrigatedFraction (max rel diff ~0.01%)
#    Production = HarvestArea * Yield (max diff 4 tons, max rel diff ~0.000012%)
# But!!!
#    PhysicalArea ≠ Area * LandFraction * PhysicalFraction (max rel diff 100%)

# Re-calculate variables that were dropped
f19_g17_cellarea = reses['f19_g17']['dsg']['AREA']
f19_g17_landarea = f19_g17_cellarea * earthstats['f19_g17']['LandFraction']
f19_g17_croparea_ha = f19_g17_landarea * earthstats['f19_g17']['PhysicalFraction']*100
recalc_ds = xr.Dataset(data_vars = \
   {'Area': f19_g17_cellarea,
    'PhysicalArea': f19_g17_croparea_ha,
    'HarvestArea': f19_g17_croparea_ha * earthstats['f19_g17']['HarvestFraction'],
    'IrrigatedArea': f19_g17_croparea_ha * earthstats['f19_g17']['IrrigatedFraction'],
    'RainfedArea': f19_g17_croparea_ha * earthstats['f19_g17']['RainfedFraction'],
    'Production': f19_g17_croparea_ha * earthstats['f19_g17']['HarvestFraction']*earthstats['f19_g17']['Yield']})
for v in recalc_ds:
   recalc_ds[v].attrs = earthstats['f09_g17'].attrs
   if recalc_ds[v].dims == ('lat', 'lon', 'crop', 'time'):
      recalc_ds[v] = recalc_ds[v].transpose('crop', 'time', 'lat', 'lon')
   discrep_sum_rel = 100*(np.sum(recalc_ds[v].values) - np.sum(earthstats['f09_g17'][v].values)) / np.sum(earthstats['f09_g17'][v].values)
   print(f"Discrepancy in {v} f19_g17 rel to f09_g17: {discrep_sum_rel}%")
earthstats['f19_g17'] = earthstats['f19_g17'].merge(recalc_ds)

# Check consistency of non-dropped variables
for v in earthstats['f19_g17']:
   if "Area" in v or v == "Production":
      continue
   discrep_sum_rel = 100*(np.mean(earthstats['f19_g17'][v].values) - np.mean(earthstats['f09_g17'][v].values)) / np.mean(earthstats['f09_g17'][v].values)
   print(f"Discrepancy in {v} f19_g17 rel to f09_g17: {discrep_sum_rel}%")


print("Done importing FAO EarthStat.")


# %% Compare area, production, and yield of individual crops

# Set up figure
ny = 2
nx = 4
# figsize = (14, 7.5)
figsize = (20, 10)
f_area, axes_area = plt.subplots(ny, nx, figsize=figsize)
axes_area = axes_area.flatten()
f_prod, axes_prod = plt.subplots(ny, nx, figsize=figsize)
axes_prod = axes_prod.flatten()
f_yield, axes_yield = plt.subplots(ny, nx, figsize=figsize)
axes_yield = axes_yield.flatten()

fig_caselist = ["FAOSTAT"]
this_earthstat_res = "f09_g17"
fig_caselist += [f"FAO EarthStat ({this_earthstat_res})"]
for (casename, case) in cases.items():
   fig_caselist.append(casename)
   
def make_1crop_plot(ax_this, ydata_this, caselist, thisCrop_clm, units, y1, yN):
   da = xr.DataArray(data = ydata_this,
                     coords = {'Case': caselist,
                               'Year': np.arange(y1,yN+1)})
   da.plot.line(x="Year", ax=ax_this)
   ax_this.title.set_text(thisCrop_clm)
   ax_this.set_xlabel("")
   ax_this.set_ylabel(units)
   ax_this.get_legend().remove()

def finishup_allcrops_plot(c, ny, nx, axes_this, f_this, suptitle, outDir_figs):
   # Delete unused axes, if any
   for a in np.arange(c+1, ny*nx):
      f_this.delaxes(axes_this[a])
      
   f_this.suptitle(suptitle,
                   x = 0.1, horizontalalignment = 'left',
                   fontsize=24)
   
   f_this.legend(handles = axes_this[0].lines,
                 labels = fig_caselist,
                 loc = "upper center",
                 ncol = int(np.floor(len(fig_caselist)/4))+1)

   f_this.savefig(outDir_figs + suptitle + " by crop.pdf",
                  bbox_inches='tight')

for c, thisCrop_clm in enumerate(cropList_combined_clm + ["Total (no sgc)"]):
   ax_area = axes_area[c]
   ax_prod = axes_prod[c]
   ax_yield = axes_yield[c]
   
   # FAOSTAT
   if thisCrop_clm == "Total (no sgc)":
      thisCrop_fao = fao_area_nosgc.columns[-1]
      ydata_area = np.array(fao_area_nosgc[thisCrop_fao])
      ydata_prod = np.array(fao_prod_nosgc[thisCrop_fao])
   else:
      thisCrop_fao = fao_area.columns[c]
      ydata_area = np.array(fao_area[thisCrop_fao])
      ydata_prod = np.array(fao_prod[thisCrop_fao])
   
   # FAO EarthStat
   if thisCrop_clm == "Total":
      area_tyx = earthstats[this_earthstat_res].HarvestArea.sum(dim="crop").copy()
      prod_tyx = earthstats[this_earthstat_res].Production.sum(dim="crop").copy()
   elif thisCrop_clm == "Total (no sgc)":
      area_tyx = earthstats[this_earthstat_res].drop_sel(crop=['Sugarcane']).HarvestArea.sum(dim="crop").copy()
      prod_tyx = earthstats[this_earthstat_res].drop_sel(crop=['Sugarcane']).Production.sum(dim="crop").copy()
   else:
      area_tyx = earthstats[this_earthstat_res].HarvestArea.sel(crop=thisCrop_clm).copy()
      prod_tyx = earthstats[this_earthstat_res].Production.sel(crop=thisCrop_clm).copy()
   ts_area_y = area_tyx.sum(dim=["lat","lon"]).values
   ydata_area = np.stack((ydata_area,
                          ts_area_y))
   ts_prod_y = 1e-6*prod_tyx.sum(dim=["lat","lon"]).values
   ydata_prod = np.stack((ydata_prod,
                          ts_prod_y))
      
   # CLM outputs
   for i, (casename, case) in enumerate(cases.items()):
      
      # Area
      lu_dsg = reses[case['res']]['dsg']
      dummy_tyx_da = lu_dsg.AREA_CFT.isel(cft=0, drop=True)
      area_tyx = np.full(dummy_tyx_da.shape, 0.0)
      if "Total" in thisCrop_clm:
         incl_crops = [x for x in case['ds'].ivt_str.values if x.replace('irrigated_','').replace('temperate_','').replace('tropical_','') in [y.lower() for y in cropList_combined_clm]]
         if thisCrop_clm == "Total (no sgc)":
            incl_crops = [x for x in incl_crops if "sugarcane" not in x]
         elif thisCrop_clm == "Total":
            pass
         else:
            raise RuntimeError("???")
         area_tyx = lu_dsg.AREA_CFT.sel(cft=[utils.ivt_str2int(x) for x in incl_crops]).sum(dim="cft").values
      else:
         for pft_str in case['ds'].ivt_str.values:
            if thisCrop_clm.lower() not in pft_str:
               continue
            pft_int = utils.ivt_str2int(pft_str)
            area_tyx += lu_dsg.AREA_CFT.sel(cft=pft_int).values
      area_tyx *= 1e-4 # m2 to ha
      area_tyx_da = xr.DataArray(data=area_tyx, coords=dummy_tyx_da.coords)
      ts_area_y = area_tyx_da.sum(dim=["lat", "lon"])
      ydata_area = np.concatenate((ydata_area,
                                   np.expand_dims(ts_area_y.values, axis=0)),
                                  axis=0)
      
      # Production
      if thisCrop_clm == "Total (no sgc)":
         ts_prod_y = case['ds'].drop_sel(Crop=['Sugarcane', 'Total']).ts_prod_yc.sum(dim="Crop").copy()
      else:
         ts_prod_y = case['ds'].ts_prod_yc.sel(Crop=thisCrop_clm).copy()
      ydata_prod = np.concatenate((ydata_prod,
                                 np.expand_dims(ts_prod_y.values, axis=0)),
                                 axis=0)
   # Convert ha to Mha
   ydata_area *= 1e-6
   
   # Calculate FAO* yields
   ydata_yield = ydata_prod / ydata_area
   
   # Make plots for this crop
   make_1crop_plot(ax_area, ydata_area, fig_caselist, thisCrop_clm, "Mha", y1, yN)
   make_1crop_plot(ax_prod, ydata_prod, fig_caselist, thisCrop_clm, "Mt", y1, yN)
   make_1crop_plot(ax_yield, ydata_yield, fig_caselist, thisCrop_clm, "t/ha", y1, yN)
   
# Finish up and save
finishup_allcrops_plot(c, ny, nx, axes_area, f_area, "Global crop area", outDir_figs)
finishup_allcrops_plot(c, ny, nx, axes_prod, f_prod, "Global crop production", outDir_figs)
finishup_allcrops_plot(c, ny, nx, axes_yield, f_yield, "Global crop yield", outDir_figs)   


# %% Make maps of individual crops (rainfed, irrigated)

# Define reference case, if you want to plot differences
# ref_casename = None
ref_casename = 'mycode_clmcals'

overwrite = False

varList = {
   'GDDHARV': {
      'suptitle':   'Mean harvest reqt',
      'time_dim':   'gs',
      'units':      'GDD',
      'multiplier': 1},
   'GRAIN_HARV_TOFOOD_ANN_GD': {
      'suptitle':   'Mean annual yield',
      'time_dim':   'time',
      'units':      't/ha',
      'multiplier': 1e-6 * 1e4}, # g/m2 to tons/ha
   'GSLEN': {
      'suptitle':   'Mean growing season length',
      'time_dim':   'gs',
      'units':      'days',
      'multiplier': 1},
   'HDATES': {
      'suptitle':   'Mean harvest date',
      'time_dim':   'gs',
      'units':      'day of year',
      'multiplier': 1},
   'HUI': {
      'suptitle':   'Mean HUI accum',
      'time_dim':   'gs',
      'units':      'GDD',
      'multiplier': 1},
   'SDATES': {
      'suptitle':   'Mean sowing date',
      'time_dim':   'gs',
      'units':      'day of year',
      'multiplier': 1},
}

nx = 2
dpi = 150

fontsize = {}
if ref_casename:
   fontsize['titles'] = 14
   fontsize['axislabels'] = 12
   fontsize['ticklabels'] = 8
   fontsize['suptitle'] = 16
else:
   fontsize['titles'] = 18
   fontsize['axislabels'] = 15
   fontsize['ticklabels'] = 15
   fontsize['suptitle'] = 24

for (this_var, var_info) in varList.items():
   
   if var_info['time_dim'] == "time":
      yrange_str = f'{y1}-{yN}'
   else:
      yrange_str = f'{y1}-{yN-1} gs'
   suptitle = var_info['suptitle'] + f' ({yrange_str})'
   
   print(f'Mapping {this_var}...')

   # First, determine how many cases have this variable
   ny = 0
   fig_caselist = []
   for i, (casename, case) in enumerate(cases.items()):
      if ref_casename and cases[ref_casename]['res'] != case['res']:
         # Not bothering with regridding (for now?)
         pass
      elif this_var in case['ds']:
         ny += 1
         fig_caselist += [casename]
      elif casename == ref_casename:
         raise RuntimeError(f'ref_case {ref_casename} is missing {this_var}')
   
   # Rearrange caselist for this figure so that reference case is first
   if ref_casename:
      if len(fig_caselist) <= 1:
         raise RuntimeError(f"Only ref case {ref_casename} has {this_var}")
      fig_caselist = [ref_casename] + [x for x in fig_caselist if x != ref_casename]
   
   # Now set some figure parameters based on # cases
   if ny == 1:
      print("WARNING: Check that the layout looks good for ny == 1")
      figsize = (24, 7.5)    # width, height
      suptitle_ypos = 0.85
   elif ny == 2:
      figsize = (12, 7)    # width, height
      if ref_casename:
         suptitle_xpos = 0.515
         suptitle_ypos = 0.95
      else:
         suptitle_xpos = 0.55
         suptitle_ypos = 1
      cbar_pos = [0.17, 0.05, 0.725, 0.025]  # left edge, bottom edge, width, height
      new_sp_bottom = 0.11
      new_sp_left = None
   elif ny == 3:
      figsize = (14, 10)    # width, height
      suptitle_xpos = 0.55
      suptitle_ypos = 1
      cbar_pos = [0.2, 0.05, 0.725, 0.025]  # left edge, bottom edge, width, height
      new_sp_bottom = 0.11 # default: 0.1
      new_sp_left = 0.125
   elif ny == 4:
      figsize = (22, 16)    # width, height
      suptitle_xpos = 0.55
      suptitle_ypos = 1
      cbar_pos = [0.2, 0.05, 0.725, 0.025]  # left edge, bottom edge, width, height
      new_sp_bottom = 0.11 # default: 0.1
      new_sp_left = 0.125
   else:
      raise ValueError(f"Set up for ny = {ny}")

   # Get all crops we care about
   clm_types_main = [x.lower().replace('wheat','spring_wheat') for x in cropList_combined_clm[:-1]]
   clm_types_rfir = []
   for x in clm_types_main:
      for y in cases[list(cases.keys())[0]]['ds'].vegtype_str.values:
         if x in y:
            clm_types_rfir.append(y)
   clm_types = np.unique([x.replace('irrigated_', '') for x in clm_types_rfir])

   for thisCrop_main in clm_types:
      
      # Get the name we'll use in output text/filenames
      thisCrop_out = thisCrop_main
      if "soybean" in thisCrop_out and "tropical" not in thisCrop_out:
         thisCrop_out = thisCrop_out.replace("soy", "temperate_soy")
      
      # Skip if file exists and we're not overwriting
      diff_txt = ""
      if ref_casename:
         diff_txt = f" Diff {ref_casename}"
      fig_outfile = outDir_figs + "Map " + suptitle + diff_txt + f" {thisCrop_out}.png"
      if os.path.exists(fig_outfile) and not overwrite:
         print(f'   Skipping {thisCrop_out} (file exists).')
         continue
      
      print(f'   {thisCrop_out}...')
      c = -1
      fig = plt.figure(figsize=figsize)
      ims = []
      axes = []
      cbs = []
      for i, casename in enumerate(fig_caselist):

         case = cases[casename]
         this_ds = case['ds']
         if this_var not in case['ds']:
            continue
         elif ref_casename and cases[ref_casename]['res'] != case['res']:
            # Not bothering with regridding (for now?)
            continue
         c += 1
         
         plotting_diffs = ref_casename and casename != ref_casename
         
         this_map = this_ds[this_var]
         
         found_types = [x for x in this_ds.vegtype_str.values if thisCrop_main in x]

         # Grid, if needed
         if "lon" not in this_map.dims:
            this_map = utils.grid_one_variable(this_ds, this_var, vegtype=found_types)
            
         # Get mean, set colormap
         units = var_info['units']
         if units == "day of year":
            ar = stats.circmean(this_map, high=365, low=1, axis=this_map.dims.index(var_info['time_dim']), nan_policy='omit')
            dummy_map = this_map.isel({var_info['time_dim']: 0}, drop=True)
            this_map = xr.DataArray(data = ar,
                                    coords = dummy_map.coords,
                                    attrs = dummy_map.attrs)
            if plotting_diffs:
               this_map_vals = (this_map - refcase_map).values
               this_map_vals[this_map_vals > 365/2] -= 365
               this_map = xr.DataArray(data = this_map_vals,
                                       coords = this_map.coords,
                                       attrs = this_map.attrs)
               cmap = 'RdBu'
               vrange = list(np.nanmax(np.abs(this_map.values)) * np.array([-1,1]))
               units = "days"
            else:
               cmap = 'twilight'
               vrange = [1, 365]
         else:
            this_map = this_map.mean(dim=var_info['time_dim'])
            this_map *= var_info['multiplier']
            if plotting_diffs:
               this_map = this_map - refcase_map
               cmap = 'RdBu'
               vrange = list(np.nanmax(np.abs(this_map.values)) * np.array([-1,1]))
            else:
               cmap = 'viridis'
               vrange = None
         
         if casename == ref_casename:
            refcase_map = this_map.copy()
            
         cbar_units = units
         if plotting_diffs:
            cbar_units = f"Diff. from {ref_casename} ({units})"
            if not np.any(np.abs(this_map) > 0):
               print(f'      {casename} identical to {ref_casename}!')
               cbar_units += ": None!"
         
         rainfed_types = [x for x in found_types if "irrigated" not in x]
         ax = fig.add_subplot(ny,nx,nx*c+1,projection=ccrs.PlateCarree(), ylabel="mirntnt")
         axes.append(ax)
         thisCrop = thisCrop_main
         im, cb = make_map(ax, this_map.sel(ivt_str=thisCrop), fontsize, units=cbar_units, cmap=cmap, vrange=vrange, linewidth=0.5, show_cbar=bool(ref_casename))
         ims.append(im)
         cbs.append(cb)

         irrigated_types = [x for x in found_types if "irrigated" in x]
         ax = fig.add_subplot(ny,nx,nx*c+2,projection=ccrs.PlateCarree())
         axes.append(ax)
         thisCrop = "irrigated_" + thisCrop_main
         im, cb = make_map(ax, this_map.sel(ivt_str=thisCrop), fontsize, units=cbar_units, cmap=cmap, vrange=vrange, linewidth=0.5, show_cbar=bool(ref_casename))
         ims.append(im)
         cbs.append(cb)

      if ref_casename:
         equalize_colorbars(ims[:nx])
         equalize_colorbars(ims[nx:])
      elif not vrange:
         equalize_colorbars(ims)

      fig.suptitle(suptitle,
                  x = suptitle_xpos,
                  y = suptitle_ypos,
                  fontsize = fontsize['suptitle'])

      # Add row labels
      leftmost = np.arange(0, nx*ny, nx)
      for a, ax in enumerate(axes):
         if a not in leftmost:
            nearest_leftmost = np.max(leftmost[leftmost < a])
            axes[a].sharey(axes[nearest_leftmost])
      for i, a in enumerate(leftmost):
         axes[a].set_ylabel(fig_caselist[i], fontsize=fontsize['titles'])
         axes[a].yaxis.set_label_coords(-0.05, 0.5)

      # Add column labels
      topmost = np.arange(nx)
      column_labels = ['rainfed', 'irrigated']
      for a, ax in enumerate(axes):
         if a not in topmost:
            nearest_topmost = a % nx
            axes[a].sharex(axes[nearest_topmost])
      for i, a in enumerate(topmost):
         axes[a].set_title(f"{thisCrop_out} ({column_labels[i]})",
                           fontsize=fontsize['titles'],
                           y=1.1)
      
      if not ref_casename:
         cbar_ax = fig.add_axes(cbar_pos)
         fig.tight_layout()
         cb = fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal', label=units)
         cb.ax.tick_params(labelsize=fontsize['ticklabels'])
         cb.set_label(units, fontsize=fontsize['titles'])
      
      plt.subplots_adjust(bottom=new_sp_bottom, left=new_sp_left)
      
      fig.savefig(fig_outfile,
                  bbox_inches='tight', facecolor='white', dpi=dpi)
      plt.close()
   
print('Done making maps.')
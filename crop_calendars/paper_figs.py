# %% Setup

which_cases = "main2"
# which_cases = "originalCLM"
# which_cases = "originalBaseline" # As originalCLM, but without cmip6
# which_cases = "diagnose"

# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import cropcal_module as cc

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

outDir_figs = "/Users/sam/Documents/Dropbox/2021_Rutgers/CropCalendars/Figures_" \
   + which_cases + "/"
if not os.path.exists(outDir_figs):
   os.mkdir(outDir_figs)

import numpy as np
from scipy import stats, signal
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import cartopy.crs as ccrs
from matplotlib import cm
import datetime as dt
import re
import importlib
import pandas as pd
import cftime

# Ignore these annoying warnings
import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")


# %% Define functions etc.

cropList_combined_clm = ["Corn", "Rice", "Cotton", "Soybean", "Sugarcane", "Wheat", "Total"]

plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

def get_non_rx_map(var_info, cases, casename, this_var, thisCrop_main, found_types):
   time_dim = var_info['time_dim']
   case = cases[casename]
   this_ds = case['ds']
   if this_var not in case['ds']:
      return xr.DataArray(), "continue"
   elif ref_casename and ref_casename!="rx" and cases[ref_casename]['res'] != case['res']:
      # Not bothering with regridding (for now?)
      return xr.DataArray(), "continue"
   this_map = this_ds[this_var]
   
   # Grid the included vegetation types, if needed
   if "lon" not in this_map.dims:
      this_map = utils.grid_one_variable(this_ds, this_var, vegtype=found_types)
   # If not, select the included vegetation types
   else:
      this_map = this_map.sel(ivt_str=found_types)
   
   return this_map, time_dim


def make_map(ax, this_map, fontsize, lonlat_bin_width=None, units=None, cmap='viridis', vrange=None, linewidth=1.0, this_title=None, show_cbar=False, bounds=None, extend='both'): 
   
   if bounds:
      norm = mcolors.BoundaryNorm(bounds, cmap.N, extend=extend)
      im = ax.pcolormesh(this_map.lon.values, this_map.lat.values,
                         this_map, shading="auto",
                         norm=norm,
                         cmap=cmap)
   else:
      im = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
                         this_map, shading="auto",
                         cmap=cmap)
      if vrange:
         im.set_clim(vrange[0], vrange[1])
   ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
   ax.coastlines(linewidth=linewidth, color="white")
   ax.coastlines(linewidth=linewidth*0.6)
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


# %% Import model output

y1 = 1961
yN = 2010

gs1 = y1
gsN = yN - 1
yearList = np.arange(y1,yN+1)
Nyears = len(yearList)

# Define cases
cases = {}
if which_cases == "originalCLM":
   # A run that someone else did
   cases['cmip6'] = {'filepath': '/Users/Shared/CESM_work/CropEvalData_ssr/danica_timeseries-cmip6_i.e21.IHIST.f09_g17/month_1/ssr_trimmed_annual.nc',
                     'constantVars': None,
                     'constantGSs': None,
                     'res': 'f09_g17'}
if "original" in which_cases:
   # My run with normal CLM code + my outputs
   cases['Original baseline'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.yield_perharv2.IHistClm50BgcCrop.1958-2014/cropcals3.f19-g17.yield_perharv2.IHistClm50BgcCrop.1958-2014.clm2.h1.1958-01-01-00000.nc',
                                 'constantVars': None,
                                 'constantGSs': None,
                                 'res': 'f19_g17'}
# My run with rx_crop_calendars2 code but CLM calendars
cases['New baseline'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1958-2014/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1958-2014.clm2.h1.1958-01-01-00000.nc',
                         'constantVars': None,
                         'constantGSs': None,
                         'res': 'f19_g17'}
# My run with rx_crop_calendars2 code and GGCMI calendars
cases['Prescribed calendars'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced3.clm2.h1.1958-01-01-00000.nc',
                             'constantVars': ["SDATES", "GDDHARV"],
                             'constantGSs': None, # 'None' with constantVars specified means all should be constant
                             'res': 'f19_g17',
                             'rx_sdates_file': "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
                             'rx_hdates_file': "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
                             'rx_gdds_file': "/Users/Shared/CESM_work/crop_dates/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1977-2014.gddgen/generate_gdds/gdds_20220927_174954.nc"}
if which_cases == "diagnose":
   # My run with rx_crop_calendars2 code and GGCMI sowing dates but CLM maturity reqts
   cases['Prescribed sowing'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.sdateforced_not_gdd/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.sdateforced_not_gdd.clm2.h1.1958-01-01-00000.nc',
                              'constantVars': ["SDATES"],
                              'constantGSs': None, # 'None' with constantVars specified means all should be constant
                              'res': 'f19_g17',
                              'rx_sdates_file': "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc",
                              'rx_hdates_file': None,
                              'rx_gdds_file': None}
   # My run with rx_crop_calendars2 code and CLM sowing dates but GGCMI maturity reqts
   cases['Prescribed maturity reqts.'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced_not_sdate/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced_not_sdate.clm2.h1.1958-01-01-00000.nc',
                              'constantVars': ["GDDHARV"],
                              'constantGSs': None, # 'None' with constantVars specified means all should be constant
                              'res': 'f19_g17',
                              'rx_sdates_file': None,
                              'rx_hdates_file': None,
                              'rx_gdds_file': "/Users/Shared/CESM_work/crop_dates/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1977-2014.gddgen/generate_gdds/gdds_20220927_174954.nc"}


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
      this_ds["GRAIN_HARV_TOFOOD_ANN"] = cc.adjust_grainC(this_ds.GRAINC_TO_FOOD, this_ds.patches1d_itype_veg)

      # Rework to match what we already have
      this_ds = this_ds.assign_coords({"ivt": np.arange(np.min(this_ds.patches1d_itype_veg.values),
                                                        np.max(this_ds.patches1d_itype_veg.values)+1)})
      
      # Add vegtype_str
      this_ds['vegtype_str'] = xr.DataArray(data=[utils.ivt_int2str(x) for x in this_ds.ivt.values],
                                            dims = {"ivt": this_ds.ivt})

   else:
      this_ds = cc.import_output(case['filepath'], myVars=myVars,
                                 y1=y1, yN=yN, verbose=verbose_import)
      
      bad_patches = cc.check_constant_vars(this_ds, case, ignore_nan=True, constantGSs=case['constantGSs'], verbose=True, throw_error=False)
      # for p in bad_patches:
      #    cc.print_gs_table(this_ds.isel(patch=p))
   
   case["ds"] = this_ds

# Get growing season set info   
gs_values = this_ds.gs.values
Ngs = len(gs_values)

# Get list of simulated vegetation types
clm_sim_veg_types = []
for casename, case in cases.items():
   case_ds = case['ds']
   if "SDATES" not in case_ds:
      continue
   ever_active = np.where(np.any(case_ds['SDATES'] > 0, axis=1))[0]
   ivt_str_ever_active = np.unique(case_ds.isel(patch=ever_active)['patches1d_itype_veg_str'].values)
   clm_sim_veg_types += list(ivt_str_ever_active)
clm_sim_veg_types = np.unique(clm_sim_veg_types)

# Get all crops we care about
clm_types_main = [x.lower().replace('wheat','spring_wheat') for x in cropList_combined_clm[:-1]]
clm_types_rfir = []
for x in clm_types_main:
   for y in cases[list(cases.keys())[0]]['ds'].vegtype_str.values:
      if x in y:
         clm_types_rfir.append(y)
clm_types = np.unique([x.replace('irrigated_', '') for x in clm_types_rfir])


# %% Import LU data

# Define resolutions
reses = {}
# f09_g17 ("""1-degree"""; i.e., 1.25 lon x 0.9 lat)
reses["f09_g17"] = {"lu_path": "/Users/Shared/CESM_work/CropEvalData_ssr/landuse.timeseries_0.9x1.25_hist_78pfts_CMIP6_simyr1850-2015_c170824.nc4"}
# f19_g17 ("""2-degree"""; i.e., 2.5 lon x 1.9 lat)
reses["f19_g17"] = {"lu_path": "/Users/Shared/CESM_work/CropEvalData_ssr/landuse.timeseries_1.9x2.5_hist_78pfts_CMIP6_simyr1850-2015_c170824.nc4"}

# Import land use to reses dicts
for i, (resname, res) in enumerate(reses.items()):
   
   # Find a matching case
   for (_, case) in cases.items():
      if case["res"] == resname:
         break
   if case["res"] != resname:
      continue
   print(f"Importing {resname}...")
   
   res['dsg'] = cc.open_lu_ds(res['lu_path'], y1, yN, case['ds'])
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
   case_ds, lu_dsg, lon_tolerance = cc.round_lonlats_to_match_ds(case_ds, lu_dsg, "lon", initial_tolerance)
   case_ds, lu_dsg, lat_tolerance = cc.round_lonlats_to_match_ds(case_ds, lu_dsg, "lat", initial_tolerance)
      
   # Ensure that time axes are formatted the same
   case_ds = cc.time_units_and_trim(case_ds, y1, yN, cftime.DatetimeNoLeap)
   lu_dsg = cc.time_units_and_trim(lu_dsg, y1, yN, cftime.DatetimeNoLeap)

   # Save
   case['ds'] = case_ds
   case['ds'].load()
   reses[case['res']]['dsg'] = lu_dsg
   

print("Done.")


# %% Import GGCMI sowing and harvest dates, and check sims
# Minimum harvest threshold allowed in PlantCrop()
gdd_min = 50

for i, (casename, case) in enumerate(cases.items()):
   
   if 'rx_sdates_file' in case:
      if case['rx_sdates_file']:
         case['rx_sdates_ds'] = cc.import_rx_dates("sdate", case['rx_sdates_file'], case['ds'])
      if case['rx_hdates_file']:
         case['rx_hdates_ds'] = cc.import_rx_dates("hdate", case['rx_hdates_file'], case['ds'])
      if case['rx_gdds_file']:
         case['rx_gdds_ds'] = cc.import_rx_dates("gdd", case['rx_gdds_file'], case['ds'])
      
      # Equalize lons/lats
      lonlat_tol = 1e-4
      for v in ['rx_sdates_ds', 'rx_hdates_ds', 'rx_gdds_ds']:
         if v in case:
            for l in ['lon', 'lat']:
               max_diff_orig = np.max(np.abs(case[v][l].values - case['ds'][l].values))
               if max_diff_orig > lonlat_tol:
                  raise RuntimeError(f'{v} {l} values differ too much from {casename} ({max_diff_orig} > {lonlat_tol})')
               elif max_diff_orig > 0:
                  case[v] = case[v].assign_coords({l: case['ds'][l].values})
                  max_diff = np.max(np.abs(case[v][l].values - case['ds'][l].values))
                  print(f'{v} {l} max_diff {max_diff_orig} → {max_diff}')
               else:
                  print(f'{v} {l} max_diff {max_diff_orig}')
      
      if case['rx_sdates_file'] and case['rx_hdates_file']:
         case['rx_gslen_ds'] = case['rx_hdates_ds'].copy()
         for v in case['rx_gslen_ds']:
            if v == "time_bounds":
               continue
            case['rx_gslen_ds'][v] = cc.get_gs_len_da(case['rx_hdates_ds'][v] - case['rx_sdates_ds'][v])
         
      # Check
      if case['rx_sdates_file']:
         cc.check_rx_obeyed(case['ds'].vegtype_str.values, case['rx_sdates_ds'].isel(time=0), case['ds'], casename, "SDATES")
      if case['rx_gdds_file']:
         cc.check_rx_obeyed(case['ds'].vegtype_str.values, case['rx_gdds_ds'].isel(time=0), case['ds'], casename, "GDDHARV", gdd_min=gdd_min)
      

# %% Get FAO data from CSV

fao_all = pd.read_csv("/Users/sam/Documents/git_repos/CTSM_cropcals_hist/crop_calendar_MATLAB/FAOSTAT_data_6-15-2022.csv")

fao_all = cc.fao_data_preproc(fao_all)
fao_prod, fao_prod_nosgc = cc.fao_data_get(fao_all, 'Production', y1, yN)
fao_area, fao_area_nosgc = cc.fao_data_get(fao_all, 'Area harvested', y1, yN)


# %% Get CLM crop production

for i, (casename, case) in enumerate(cases.items()):
   print(f"Gridding {casename}...")
   case_ds = case['ds']
   yield_gd = utils.grid_one_variable(case_ds.sel(time=case_ds.time.values), "GRAIN_HARV_TOFOOD_ANN")
   
   lu_dsg = reses[case['res']]['dsg']
   yield_gd = yield_gd.assign_coords({"lon": lu_dsg.lon,
                                      "lat": lu_dsg.lat})
   case['ds']['GRAIN_HARV_TOFOOD_ANN_GD'] = yield_gd
   
   case['ds']['ts_prod_yc'] = cc.get_ts_prod_clm_yc_da(yield_gd, lu_dsg, yearList, cropList_combined_clm)
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


# %% Import country map and key

# Half-degree countries from Brendan
countries = xr.open_dataset('/Users/sam/Documents/Dropbox/2021_Rutgers/CropCalendars/countries_brendan/gadm0.mask.nc4')

# Nearest-neighbor remap countries to LU resolutions
for resname, res in reses.items():
   if 'dsg' not in res:
      continue
   res['dsg']['countries'] = utils.lon_idl2pm(countries).interp_like(res['dsg']['AREA'], method='nearest')['gadm0']
   
countries_key = pd.read_csv('/Users/sam/Documents/Dropbox/2021_Rutgers/CropCalendars/countries_brendan/Nation_ID.csv',
                               header=None,
                               names=['num', 'name'])

fao_all_ctry = pd.read_csv("/Users/sam/Documents/git_repos/CTSM_cropcals_hist/crop_calendar_MATLAB/FAOSTAT_data_en_8-21-2022_byCountry.csv")
fao_all_ctry = cc.fao_data_preproc(fao_all_ctry)

# Replace some countries' names in key to match FAO data
countries_key = countries_key.replace({'Turkey': 'Türkiye',
                                       'Vietnam': 'Viet Nam'})
# Replace some countries' names in FAO data to match key (to keep them short for figs)
fao_all_ctry = fao_all_ctry.replace({'Bolivia (Plurinational State of)': 'Bolivia',
                                     'Russian Federation': 'Russia',
                                     'Syrian Arab Republic': 'Syria'})
# Rename some countries to keep them short for figs
countries_key = countries_key.replace({'China': 'China, mld.',   # Because it also has Taiwan
                                       'United States': 'USA',
                                       'South Africa': 'S. Africa'})
fao_all_ctry = fao_all_ctry.replace({'China, mainland': 'China, mld.',
                                     'United States of America': 'USA',
                                     'South Africa': 'S. Africa'})

# Make sure every country in map is in key
for i, x in enumerate(np.unique(countries.gadm0.values)):
   if not np.isnan(x) and not np.any(countries_key.num.values == x):
      print(f'❗ {x} not found in key')


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
f_yield_dt, axes_yield_dt = plt.subplots(ny, nx, figsize=figsize)
axes_yield_dt = axes_yield_dt.flatten()

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
   plt.close(f_this)

for c, thisCrop_clm in enumerate(cropList_combined_clm + ["Total (no sgc)"]):
   ax_area = axes_area[c]
   ax_prod = axes_prod[c]
   ax_yield = axes_yield[c]
   ax_yield_dt = axes_yield_dt[c]
   
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
   make_1crop_plot(ax_yield_dt, cc.detrend(ydata_yield), fig_caselist, thisCrop_clm, "t/ha", y1, yN)
   
# Finish up and save
finishup_allcrops_plot(c, ny, nx, axes_area, f_area, "Global crop area", outDir_figs)
finishup_allcrops_plot(c, ny, nx, axes_prod, f_prod, "Global crop production", outDir_figs)
finishup_allcrops_plot(c, ny, nx, axes_yield, f_yield, "Global crop yield", outDir_figs)
finishup_allcrops_plot(c, ny, nx, axes_yield_dt, f_yield_dt, "Global crop yield (detrended)", outDir_figs)


# %% Make maps of individual crops (rainfed, irrigated)

# Define reference case, if you want to plot differences
ref_casename = None
# ref_casename = 'New baseline'
# ref_casename = 'rx'

overwrite = True

varList = {
   'GDDHARV': {
      'suptitle':   'Mean harvest requirement',
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
      'suptitle':   'Mean HUI at harvest',
      'time_dim':   'gs',
      'units':      'GDD',
      'multiplier': 1},
   'HUIFRAC': {
      'suptitle':   'Mean HUI at harvest (fraction of required)',
      'time_dim':   'gs',
      'units':      'Fraction of required',
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
      yrange_str = f'{y1}-{yN-1} growing seasons'
   suptitle = var_info['suptitle'] + f' ({yrange_str})'
   
   print(f'Mapping {this_var}...')

   # First, determine how many cases have this variable
   ny = 0
   fig_caselist = []
   for i, (casename, case) in enumerate(cases.items()):
      if ref_casename and ref_casename != "rx" and cases[ref_casename]['res'] != case['res']:
         # Not bothering with regridding (for now?)
         pass
      elif this_var in case['ds']:
         ny += 1
         fig_caselist += [casename]
      elif casename == ref_casename:
         raise RuntimeError(f'ref_case {ref_casename} is missing {this_var}')
   if ny == 0:
      print(f"No cases contain {this_var}; skipping.")
      continue
   
   # Add "prescribed" "case," if relevant
   if this_var in ["GDDHARV", "GSLEN", "HDATES", "SDATES"]:
      if this_var == "GDDHARV":
         rx_ds_key = "rx_gdds_ds"
      elif this_var == "GSLEN":
         rx_ds_key = "rx_gslen_ds"
      elif this_var == "HDATES":
         rx_ds_key = "rx_hdates_ds"
      elif this_var == "SDATES":
         rx_ds_key = "rx_sdates_ds"
      else:
         raise RuntimeError(f"What rx_ds_key should I use for {this_var}?")
      if this_var in ["GSLEN", "HDATES", "SDATES"]:
         rx_row_label = "ISIMIP3"
      elif this_var == "GDDHARV":
         rx_row_label = "ISIMIP3-derived"
      else:
         raise RuntimeError(f"What row label should be used instead of 'rx' for {this_var}?")
      rx_parent_found = False
      for i, (casename, case) in enumerate(cases.items()):
         # For now, we're just assuming all runs with a given prescribed variable use the same input file
         if rx_ds_key in case:
            rx_parent_casename = casename
            rx_ds = case[rx_ds_key]
            fig_caselist += ["rx"]
            ny += 1
            rx_parent_found = True
            break
      if not rx_parent_found:
         raise RuntimeError(f"No case found with {rx_ds_key}")
   elif ref_casename == "rx":
      print(f"Skipping {this_var} because it has no rx dataset against which to compare simulations")
   
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

   for thisCrop_main in clm_types:
      
      # Get the name we'll use in output text/filenames
      thisCrop_out = thisCrop_main
      if "soybean" in thisCrop_out and "tropical" not in thisCrop_out:
         thisCrop_out = thisCrop_out.replace("soy", "temperate_soy")
      
      # Skip if file exists and we're not overwriting
      diff_txt = ""
      if ref_casename == "rx":
         diff_txt = f" Diff {rx_row_label}"
      elif ref_casename:
         diff_txt = f" Diff {ref_casename}"
      fig_outfile = outDir_figs + "Map " + suptitle + diff_txt + f" {thisCrop_out}.png"
      if os.path.exists(fig_outfile) and not overwrite:
         print(f'   Skipping {thisCrop_out} (file exists).')
         continue
      
      print(f'   {thisCrop_out}...')
      found_types = [x for x in clm_types_rfir if thisCrop_main in x]
      
      c = -1
      fig = plt.figure(figsize=figsize)
      ims = []
      axes = []
      cbs = []
      for i, casename in enumerate(fig_caselist):
         if casename == "rx":
            time_dim = "time"
            these_rx_vars = ["gs1_" + str(x) for x in utils.vegtype_str2int(found_types)]
            this_map = xr.concat((rx_ds[x].assign_coords({'ivt_str': found_types[i]}) for i, x in enumerate(these_rx_vars)), dim="ivt_str")
            this_map = this_map.squeeze(drop=True)
            if "lon" not in this_map.dims:
               this_ds = xr.Dataset(data_vars={'tmp': this_map})
               this_map = utils.grid_one_variable(this_ds, 'tmp')
            
            # Apply LU mask
            parent_map, parent_time_dim = get_non_rx_map(var_info, cases, rx_parent_casename, this_var, thisCrop_main, found_types)
            if parent_time_dim == "continue":
               raise RuntimeError("What should I do here?")
            this_map = this_map.where(~np.isnan(parent_map.mean(dim=parent_time_dim)))
         else:
            this_map, time_dim = get_non_rx_map(var_info, cases, casename, this_var, thisCrop_main, found_types)
            if time_dim == "continue":
               continue
         c += 1
         
         plotting_diffs = ref_casename and casename != ref_casename
         
         # Get mean, set colormap
         units = var_info['units']
         if units == "day of year":
            if time_dim in this_map.dims:
               ar = stats.circmean(this_map, high=365, low=1, axis=this_map.dims.index(time_dim), nan_policy='omit')
               dummy_map = this_map.isel({time_dim: 0}, drop=True)
            else:
               ar = this_map.copy()
               dummy_map = this_map.copy()
            this_map = xr.DataArray(data = ar,
                                    coords = dummy_map.coords,
                                    attrs = dummy_map.attrs)
            if plotting_diffs:
               this_map_vals = (this_map - refcase_map).values
               this_map_vals[this_map_vals > 365/2] -= 365
               this_map_vals[this_map_vals < -365/2] += 365
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
            if time_dim in this_map.dims:
               this_map = this_map.mean(dim=time_dim)
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
            if ref_casename == "rx":
               cbar_units = f"Diff. from {rx_row_label} ({units})"
            else:
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
         cc.equalize_colorbars(ims[:nx])
         cc.equalize_colorbars(ims[nx:])
      elif not vrange:
         cc.equalize_colorbars(ims)

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
         
         if fig_caselist[i] == "rx":
            row_label = rx_row_label
         else:
            row_label = fig_caselist[i]
         
         axes[a].set_ylabel(row_label, fontsize=fontsize['titles'])
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
      
      # plt.show()
      # break
      
      fig.savefig(fig_outfile,
                  bbox_inches='tight', facecolor='white', dpi=dpi)
      plt.close()
   
print('Done making maps.')


# %% Make maps of harvest reasons
importlib.reload(cc)

thisVar = "HARVEST_REASON"
reason_list_text_all = cc.get_reason_list_text()

for i, (casename, case) in enumerate(cases.items()):
   if i == 0:
      reason_list = np.unique(case['ds'][thisVar].values)
   else:
      reason_list = np.unique(np.concatenate( \
         (reason_list, \
         np.unique(case['ds'][thisVar].values))))
reason_list = [int(x) for x in reason_list if not np.isnan(x)]
reason_list_text = [reason_list_text_all[x] for x in reason_list]

ny = 2
nx = len(reason_list)

epsilon = np.nextafter(0, 1)
# bounds = [epsilon] + list(np.arange(0.2, 1.001, 0.2))
bounds = [epsilon] + list(np.arange(0.1, 1, 0.1)) + [1-epsilon]
extend = 'both'

figsize = (8, 4)
cbar_adj_bottom = 0.15
cbar_ax_rect = [0.15, 0.05, 0.7, 0.05]
cmap = plt.cm.jet
wspace = None
hspace = None
fontsize = {}
fontsize['titles'] = 8
fontsize['axislabels'] = 8
fontsize['ticklabels'] = 8
if nx == 3:
   figsize = (8, 3)
   cbar_adj_bottom = 0.1
   cbar_ax_rect = [0.15, 0.05, 0.7, 0.035]
   wspace = 0.1
   hspace = -0.1
elif nx != 2:
   print(f"Since nx = {nx}, you may need to rework some parameters")

for v, vegtype_str in enumerate(clm_types_rfir):
   if 'winter' in vegtype_str or 'miscanthus' in vegtype_str:
      continue
   print(f"{thisVar}: {vegtype_str}...")
   vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
   
   # Get variations on vegtype string
   vegtype_str_title = cc.get_vegtype_str_for_title_long(vegtype_str)
   
   # Set up figure
   fig = plt.figure(figsize=figsize)
   axes = []
   
   for i, (casename, case) in enumerate(cases.items()):
      # Grid
      thisCrop_gridded = utils.grid_one_variable(case['ds'], thisVar, \
         vegtype=vegtype_int).squeeze(drop=True)
      
      # Map each reason's frequency
      for f, reason in enumerate(reason_list):
         reason_text = reason_list_text[f]
         
         map_yx = cc.get_reason_freq_map(Ngs, thisCrop_gridded, reason)
         ax = cc.make_axis(fig, ny, nx, i*nx + f+1)
         axes.append(ax)
         im0 = make_map(ax, map_yx, fontsize, cmap=cmap, bounds=bounds, extend=extend, linewidth=0.3)
            
   # Add column labels
   topmost = np.arange(0, nx)
   for a, ax in enumerate(axes):
      if a not in topmost:
         nearest_topmost = a % nx
         axes[a].sharex(axes[nearest_topmost])
   for i, a in enumerate(topmost):
      axes[a].set_title(f"{reason_list_text[i]}",
                        fontsize=fontsize['titles'],
                        y=1.05)

   # Add row labels
   leftmost = np.arange(0, ny*nx, nx)
   for a, ax in enumerate(axes):
      if a not in leftmost:
         nearest_leftmost = a % ny
         axes[a].sharey(axes[nearest_leftmost])
      # I don't know why this is necessary, but otherwise the labels won't appear.
      ax.set_yticks([])
   for i, a in enumerate(leftmost):
      axes[a].set_ylabel(caselist[i], fontsize=fontsize['titles'])
      axes[a].yaxis.set_label_coords(-0.05, 0.5)

   suptitle = f"Harvest reason: {vegtype_str_title} ({gs1}-{gsN} growing seasons)"
   fig.suptitle(suptitle, fontsize=fontsize['titles']*1.2, fontweight="bold")
   fig.subplots_adjust(bottom=cbar_adj_bottom)
    
   cbar_ax = fig.add_axes(cbar_ax_rect)
   norm = mcolors.BoundaryNorm(bounds, cmap.N, extend=extend)
   cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                       orientation='horizontal',
                       spacing='proportional',
                       cax=cbar_ax)
   cbar_ax.tick_params(labelsize=fontsize['titles'])
   
   plt.xlabel("Fraction of growing seasons", fontsize=fontsize['titles'])
   if wspace != None:
      plt.subplots_adjust(wspace=wspace)
   if hspace != None:
      plt.subplots_adjust(hspace=hspace)
   
   # plt.show()
   # break
   
   # Save
   filename = suptitle.replace(":", "").replace("growing seasons", "gs")
   outfile = os.path.join(outDir_figs, f"{filename} {vegtype_str}.png")
   plt.savefig(outfile, dpi=300, transparent=False, facecolor='white', \
               bbox_inches='tight')
   plt.close()
   # break
    

# %% Make scatter plots, FAOSTAT vs. CLM, of top 10 countries for each crop

Ntop = 10
# top_y1 = 1961 # First year of FAO data
top_y1 = 1992 # Pre-1992, you start getting USSR, which isn't in map
top_yN = 2009
overwrite = True
portrait = False

which_to_plot = "Yield"
# which_to_plot = "Detrended yield"
# which_to_plot = "Yield anomaly"

if portrait:
   ny = 4
   nx = 3
   figsize = (10, 16)
   suptitle_ypos = 0.91
else:
   ny = 3
   nx = 4
   figsize = (16, 13)
   suptitle_ypos = 0.92

topYears = np.arange(top_y1, top_yN+1)
NtopYears = len(topYears)


fao_crops = np.unique(fao_all_ctry.Crop.values)
for c, thisCrop in enumerate(fao_crops):
      
   thisCrop_clm = cc.cropnames_fao2clm(thisCrop)
   
   suptitle = f"{thisCrop}, FAOSTAT vs CLM ({top_y1}-{top_yN})"
   file_prefix = which_to_plot.replace('aly','')
   fig_outfile = outDir_figs + f"{file_prefix} scatter top 10 " + suptitle + ".pdf"
   if os.path.exists(fig_outfile) and not overwrite:
      print(f'   Skipping {thisCrop_out} (file exists).')
      continue
   
   # Get yield datasets
   topN_ds, topN_dt_ds, topN_ya_ds = cc.get_topN_ds(cases, reses, topYears, Ntop, thisCrop, countries_key, fao_all_ctry, earthstats)
   Ntop_global = Ntop + 1
   
   if which_to_plot == "Yield":
      plot_ds = topN_ds
   elif which_to_plot == "Detrended yield":
      plot_ds = topN_dt_ds
   elif which_to_plot == "Yield anomaly":
      plot_ds = topN_ya_ds
   else:
      raise RuntimeError(f"Which dataset should be used for '{which_to_plot}'?")

   f, axes = plt.subplots(ny, nx, figsize=figsize)
   axes = axes.flatten()

   # New baseline will have hollow circles if Original baseline is included
   i_h = caselist.index('New baseline')

   for c, country in enumerate(plot_ds.Country.values):
      
      # Text describing R-squared changes for each country
      r2_change_text = ""
      
      ax = axes[c]
      sc = xr.plot.scatter(plot_ds.sel(Country=country),
                           x='Yield (FAOSTAT)',
                           y='Yield',
                           hue='Case',
                           ax=ax)
      
      for case in caselist:
         lr = stats.linregress(x = plot_ds['Yield (FAOSTAT)'].sel(Country=country),
                               y = plot_ds['Yield'].sel(Country=country, Case=case))
         if case == "New baseline":
            t = "{r1:.3g} $\\rightarrow$ "
            r2_change_text += t.format(r1=lr.rvalue**2)
         elif case == "Prescribed calendars":
            r2_change_text += "{r2:.3g}".format(r2=lr.rvalue**2)
            
      # Set title
      country_bf = ''
      for w in country.split(' '):
         country_bf += r'$\bf{' + w + r'}$' + ' '
      ax.set_title(country_bf + f'($R^2$ {r2_change_text})')
      
      # Set New baseline to have hollow circles if Original baseline is included
      if "Original baseline" in caselist:
         color = sc[i_h].get_facecolor()
         sc[i_h].set_facecolor('none')
         sc[i_h].set_edgecolor(color)
      
      xlims = list(ax.get_xlim())
      ylims = list(ax.get_ylim())
      newlims = [min(xlims[0], ylims[0]), max(xlims[1], ylims[1])]
      ax.set_xlim(newlims)
      ax.set_ylim(newlims)
      # ax.set_aspect('equal')
      ax.get_legend().remove()
      ax.set_xlabel(None)
      ax.set_ylabel(None)
      if c == Ntop_global-1:
         ax.legend(bbox_to_anchor=(1.5,0.5), loc='center')

   # Delete unused axes, if any
   for a in np.arange(c+1, ny*nx):
      f.delaxes(axes[a])
      
   # Add row labels
   leftmost = np.arange(0, nx*ny, nx)
   for i, a in enumerate(leftmost):
      axes[a].set_ylabel(f"{which_to_plot} (CLM)", fontsize=12, fontweight='bold')
      axes[a].yaxis.set_label_coords(-0.15, 0.5)

   # Add column labels
   bottommost = np.arange(Ntop_global)[-nx:]
   bottommost_cols = bottommost % nx
   for i, a in enumerate(bottommost):
      ax = axes[a]
      x = np.mean(ax.get_xlim())
      y = ax.get_ylim()[0] - 0.2*np.diff(ax.get_ylim()) # 20% below the bottom edge
      ax.text(x=x, y=y,
              s=f"{which_to_plot} (FAOSTAT)",
              fontsize=12,
              ha='center',
              fontweight='bold')

   # Add figure title
   suptitle_xpos = 0.5
   f.suptitle(suptitle,
              x = suptitle_xpos,
              y = suptitle_ypos,
              fontsize = 18,
              fontweight='bold')

   # Make square plots
   for a, ax in enumerate(axes):
      xlims = list(ax.get_xlim())
      ylims = list(ax.get_ylim())
      newlims = [min(xlims[0], ylims[0]), max(xlims[1], ylims[1])]
      ax.set_xlim(newlims)
      ax.set_ylim(newlims)
      ax.set_aspect('equal')
   
   # Add 1:1 lines
   for ax in axes:
      xlims = list(ax.get_xlim())
      ylims = list(ax.get_ylim())
      p1 = min(xlims[0], ylims[0])
      p2 = min(xlims[1], ylims[1])
      ax.plot([p1, p2], [p1, p2], 'k--', lw=0.5, alpha=0.5)
   
   f.savefig(fig_outfile,
             bbox_inches='tight', facecolor='white')
   plt.close()
   
   
# %% Make line plots, FAOSTAT vs. CLM, of top 10 countries for each crop

Ntop = 10
# top_y1 = 1961 # First year of FAO data
top_y1 = 1992 # Pre-1992, you start getting USSR, which isn't in map
top_yN = 2009
overwrite = True
portrait = False

# which_to_plot = "Yield"
# which_to_plot = "Detrended yield"
which_to_plot = "Yield anomaly"

if portrait:
   ny = 4
   nx = 3
   figsize = (10, 18)
   suptitle_ypos = 0.91
else:
   ny = 3
   nx = 4
   figsize = (18, 14)
   suptitle_ypos = 0.93

topYears = np.arange(top_y1, top_yN+1)
NtopYears = len(topYears)

legend_members = caselist + ['FAOSTAT']

fao_crops = np.unique(fao_all_ctry.Crop.values)
for c, thisCrop in enumerate(fao_crops):
      
   thisCrop_clm = cc.cropnames_fao2clm(thisCrop)
   
   suptitle = f"{thisCrop}, FAOSTAT vs CLM ({top_y1}-{top_yN})"
   file_prefix = which_to_plot.replace('aly','')
   fig_outfile = outDir_figs + f"{file_prefix} timeseries top 10 " + suptitle + ".pdf"
   if os.path.exists(fig_outfile) and not overwrite:
      print(f'   Skipping {thisCrop_out} (file exists).')
      continue
   
   # Get yield datasets
   topN_ds, topN_dt_ds, topN_ya_ds = cc.get_topN_ds(cases, reses, topYears, Ntop, thisCrop, countries_key, fao_all_ctry, earthstats)
   Ntop_global = Ntop + 1
   
   if which_to_plot == "Yield":
      plot_ds = topN_ds
   elif which_to_plot == "Detrended yield":
      plot_ds = topN_dt_ds
   elif which_to_plot == "Yield anomaly":
      plot_ds = topN_ya_ds
   else:
      raise RuntimeError(f"Which dataset should be used for '{which_to_plot}'?")

   f, axes = plt.subplots(ny, nx, figsize=figsize)
   axes = axes.flatten()

   # New baseline will have hollow circles if Original baseline is included
   i_h = caselist.index('New baseline')

   for c, country in enumerate(plot_ds.Country.values):
      
      # Text describing R-squared changes for each country
      r2_change_text = ""
            
      ax = axes[c]
      xr.plot.line(plot_ds['Yield'].sel(Country=country),
                   hue='Case',
                   ax=ax)
      xr.plot.line(plot_ds['Yield (FAOSTAT)'].sel(Country=country),
                   'k--', ax=ax)
      
      for case in caselist:
         lr = stats.linregress(x = plot_ds['Yield (FAOSTAT)'].sel(Country=country),
                               y = plot_ds['Yield'].sel(Country=country, Case=case))
         if case == "New baseline":
            t = "{r1:.3g} $\\rightarrow$ "
            r2_change_text += t.format(r1=lr.rvalue**2)
         elif case == "Prescribed calendars":
            r2_change_text += "{r2:.3g}".format(r2=lr.rvalue**2)
            
      # Set title
      country_bf = ''
      for w in country.split(' '):
         country_bf += r'$\bf{' + w + r'}$' + ' '
      ax.set_title(country_bf + f'($R^2$ {r2_change_text})')
      
      # Set New baseline to be dashed line if Original baseline is included
      if "Original baseline" in caselist:
         print('Set New baseline to be dashed line if Original baseline is included')
         # color = sc[i_h].get_facecolor()
         # sc[i_h].set_facecolor('none')
         # sc[i_h].set_edgecolor(color)
      
      # ax.set_aspect('equal')
      ax.get_legend().remove()
      ax.set_xlabel(None)
      ax.set_ylabel(None)
      if c == Ntop_global-1:
         ax.legend(legend_members, bbox_to_anchor=(1.5,0.5), loc='center')
   
   # Delete unused axes, if any
   for a in np.arange(c+1, ny*nx):
      f.delaxes(axes[a])
      
   # Add row labels
   leftmost = np.arange(0, nx*ny, nx)
   for i, a in enumerate(leftmost):
      axes[a].set_ylabel(f"{which_to_plot} (CLM)", fontsize=12, fontweight='bold')
      axes[a].yaxis.set_label_coords(-0.15, 0.5)

   # Add figure title
   suptitle_xpos = 0.5
   f.suptitle(suptitle,
              x = suptitle_xpos,
              y = suptitle_ypos,
              fontsize = 18,
              fontweight='bold')
   
   f.savefig(fig_outfile,
             bbox_inches='tight', facecolor='white')
   plt.close()
   
   
# %% Compare mean growing season length to GGCMI models

# varList = ["GSLEN", "GSLEN.onlyMature", "GSLEN.onlyMature.noOutliers", "GSLEN.onlyMature.useMedian"]
# varList = ["GSLEN"]
# varList = ["GSLEN.onlyMature"]
# varList = ["GSLEN.onlyMature.diffExpected"]
# varList = ["GSLEN.onlyMature.diffExpected.noOutliers"]
varList = ["GSLEN.onlyMature.diffExpected.useMedian"]
# varList = ["GSLEN", "GSLEN.onlyMature"]
# varList = ["GSLEN.onlyMature.noOutliers"]
# varList = ["GSLEN.onlyMature.useMedian"]

ggcmi_out_topdir = "/Users/Shared/GGCMI/AgMIP.output"
ggcmi_cropcal_dir = "/Users/Shared/GGCMI/AgMIP.input/phase3/ISIMIP3/crop_calendar"

importlib.reload(cc)

y1_ggcmi = 1980
yN_ggcmi = 2009
verbose = False

ggcmi_models_orig = ["ACEA", "CROVER", "CYGMA1p74", "DSSAT-Pythia", "EPIC-IIASA", "ISAM", "LDNDC", "LPJ-GUESS", "LPJmL", "pDSSAT", "PEPIC", "PROMET", "SIMPLACE-LINTUL5"]
Nggcmi_models_orig = len(ggcmi_models_orig)

fontsize = {}
fontsize['titles'] = 14
fontsize['axislabels'] = 12
fontsize['ticklabels'] = 8
fontsize['suptitle'] = 16
# bin_width = 30
bin_width = None

def get_new_filename(pattern):
   thisFile = glob.glob(pattern)
   if len(thisFile) > 1:
      raise RuntimeError(f"Expected at most 1 match of {pattern}; found {len(thisFile)}")
   return thisFile

def trim_years_ggcmi(y1, yN, ds_in):
   Ngs = yN - y1 + 1
   time_units = ds_in.time.attrs["units"]
   match = re.search("growing seasons since \d+-01-01, 00:00:00", time_units)
   if not match:
      raise RuntimeError(f"Can't process time axis '{time_units}'")
   sinceyear = int(re.search("since \d+", match.group()).group().replace("since ", ""))
   thisDS_years = ds_in.time.values + sinceyear - 1
   ds_in = ds_in.isel(time=np.nonzero(np.bitwise_and(thisDS_years>=y1, thisDS_years <= yN))[0])
   if ds_in.dims["time"] != Ngs:
      tmp = ds_in.dims["time"]
      raise RuntimeError(f"Expected {Ngs} matching growing seasons in GGCMI dataset; found {tmp}")
   return ds_in

ggcmiDS_started = False

Ngs_ggcmi = yN_ggcmi - y1_ggcmi + 1
gs_ggcmi = np.arange(y1_ggcmi, yN_ggcmi + 1)

for thisVar_orig in varList:
   thisVar = thisVar_orig
   
   # Processing options
   title_prefix = ""
   filename_prefix = ""
   onlyMature = "onlyMature" in thisVar
   if onlyMature:
      thisVar = thisVar.replace(".onlyMature", "")
      title_prefix = title_prefix + " (if mat.)"
      filename_prefix = filename_prefix + "_ifmature"
   noOutliers = "noOutliers" in thisVar
   if noOutliers:
      thisVar = thisVar.replace(".noOutliers", "")
      title_prefix = title_prefix + " (no outl.)"
      filename_prefix = filename_prefix + "_nooutliers"
   useMedian = "useMedian" in thisVar
   if useMedian:
      thisVar = thisVar.replace(".useMedian", "")
      title_prefix = title_prefix + " (median)"
      filename_prefix = filename_prefix + "_median"
   diffExpected = "diffExpected" in thisVar
   if diffExpected:
      thisVar = thisVar.replace(".diffExpected", "")
      filename_prefix = filename_prefix + "_diffExpected"

   ny = 4
   nx = 5
   if Nggcmi_models_orig + len(cases) + 2 > ny*nx:
      raise RuntimeError(f"2 + {Nggcmi_models_orig} GGCMI models + CLM cases ({len(cases)}) > ny*nx ({ny*nx})")
   vmin = 0.0
   title_prefix = "Seas. length" + title_prefix
   filename_prefix = "seas_length_compGGCMI" + filename_prefix
   if diffExpected:
      units = "Season length minus expected"
      cmap = plt.cm.RdBu
   else:
      units = "Days"
      cmap = plt.cm.viridis
   vmin = None
   
   figsize = (16, 8)
   cbar_adj_bottom = 0.15
   cbar_ax_rect = [0.15, 0.05, 0.7, 0.025]
   if nx != 4 or ny != 4:
      print(f"Since (nx,ny) = ({nx},{ny}), you may need to rework some parameters")

   for v, vegtype_str in enumerate(clm_sim_veg_types):
      
      if "corn" in vegtype_str:
         vegtype_str_ggcmi = "mai"
      elif "rice" in vegtype_str:
         vegtype_str_ggcmi = "ri1" # Ignoring ri2, which isn't simulated in CLM yet
      elif "soybean" in vegtype_str:
         vegtype_str_ggcmi = "soy"
      elif "spring_wheat" in vegtype_str:
         vegtype_str_ggcmi = "swh"
      # elif "winter_wheat" in vegtype_str:
      #    vegtype_str_ggcmi = "wwh"
      else:
         print(f"{thisVar}: Skipping {vegtype_str}.")
         continue
      print(f"{thisVar}: {vegtype_str}...")
      if "irrigated" in vegtype_str:
         irrtype_str_ggcmi = "firr"
      else:
         irrtype_str_ggcmi = "noirr"
      ncvar = f"matyday-{vegtype_str_ggcmi}-{irrtype_str_ggcmi}"
      vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
      
      # Get variations on vegtype string
      vegtype_str_paramfile = cc.get_vegtype_str_paramfile(vegtype_str)
      vegtype_str_title = cc.get_vegtype_str_for_title(vegtype_str)
      vegtype_str_figfile = cc.get_vegtype_str_figfile(vegtype_str)
      
      # Import GGCMI outputs
      ggcmi_models_bool = np.full((Nggcmi_models_orig,), False)
      for g, thisModel in enumerate(ggcmi_models_orig):
         
         # Only need to import each variable once
         if ggcmiDS_started and ncvar in ggcmiDS:
               did_read = False
               break
         did_read = True
         
         # Open file
         pattern = os.path.join(ggcmi_out_topdir, thisModel, "phase3a", "gswp3-w5e5", "obsclim", vegtype_str_ggcmi, f"*{ncvar}*")
         thisFile = glob.glob(pattern)
         if not thisFile:
               if verbose:
                  print(f"{ncvar}: Skipping {thisModel}")
               continue
         elif len(thisFile) != 1:
               raise RuntimeError(f"Expected 1 match of {pattern}; found {len(thisFile)}")
         thisDS = xr.open_dataset(thisFile[0], decode_times=False)
         ggcmi_models_bool[g] = True
         
         # Set up GGCMI Dataset
         if not ggcmiDS_started:
               ggcmiDS = xr.Dataset(coords={"gs": gs_values,
                                            "lat": thisDS.lat,
                                            "lon": thisDS.lon,
                                            "model": ggcmi_models_orig,
                                            "cft": clm_sim_veg_types})
               ggcmiDS_started = True
         
         # Set up DataArray for this crop-irr
         if g==0:
            matyday_da = xr.DataArray(data=np.full((Ngs_ggcmi,
                                                   thisDS.dims["lat"],
                                                   thisDS.dims["lon"],
                                                   Nggcmi_models_orig
                                                   ),
                                                   fill_value=np.nan),
                                      coords=[(gs_ggcmi)] + [ggcmiDS.coords[x] for x in ["lat", "lon", "model"]])
         
         # Get just the seasons you need
         thisDS = trim_years_ggcmi(y1_ggcmi, yN_ggcmi, thisDS)
         thisDA = thisDS[ncvar]
         
         # Pre-filtering
         thisMax = np.nanmax(thisDA.values)
         if thisMax > 10**19:
            if verbose:
               print(f"Warning: {ncvar}: {thisModel}: Max {thisMax} (before filtering); setting values >1e19 to NaN")
            thisDA.values[np.where(thisDA.values > 10**19)] = np.nan
         thisMax = np.nanmax(thisDA.values)
         highMax = thisMax > 366
         if highMax and verbose:
            print(f"Warning: {ncvar}: {thisModel}: Max {thisMax} (before filtering)")
         
         # Figure out which seasons to include
         if highMax:
            filterVar = "maturityindex"
            thisFile = get_new_filename(pattern.replace("matyday", filterVar))
            filter_str = None
            if thisFile:
               filterDS = xr.open_dataset(thisFile[0], decode_times=False)
               filterDS = trim_years_ggcmi(y1_ggcmi, yN_ggcmi, filterDS)
               filter_str = f"(after filtering by {filterVar} == 1)"
               thisDA = thisDA.where(filterDS[ncvar.replace("matyday", filterVar)] == 1)
            else:
               filterVar = "maturitystatus"
               thisFile = get_new_filename(pattern.replace("matyday", filterVar))
               if thisFile:
                  filterDS = xr.open_dataset(thisFile[0], decode_times=False)
                  filterDS = trim_years_ggcmi(y1_ggcmi, yN_ggcmi, filterDS)
                  filter_str = f"(after filtering by {filterVar} >= 1)"
                  thisDA = thisDA.where(filterDS[ncvar.replace("matyday", filterVar)] >= 1)
               else:
                  filterVar = "yield"
                  thisFile = get_new_filename(pattern.replace("matyday", filterVar))
                  if thisFile:
                     filterDS = xr.open_dataset(thisFile[0], decode_times=False)
                     filterDS = trim_years_ggcmi(y1_ggcmi, yN_ggcmi, filterDS)
                     filter_str = f"(after filtering by {filterVar} > 0)"
                     thisDA = thisDA.where(filterDS[ncvar.replace("matyday", filterVar)] > 0)
            if not filter_str:
               filter_str = "(after no filtering)"
            thisMax = np.nanmax(thisDA.values)
            if thisMax > 366:
               if verbose:
                  print(f"Warning: {ncvar}: {thisModel}: Max {thisMax} {filter_str}; setting values > 364 to NaN")
               thisDA.values[np.where(thisDA.values > 364)] = np.nan
                  
         # Only include cell-seasons with positive yield
         filterVar = "yield"
         thisFile = get_new_filename(pattern.replace("matyday", filterVar))
         if thisFile:
            filterDS = xr.open_dataset(thisFile[0], decode_times=False)
            filterDS = trim_years_ggcmi(y1_ggcmi, yN_ggcmi, filterDS)
            thisDA = thisDA.where(filterDS[ncvar.replace("matyday", filterVar)] > 0)
               
         # Don't include cell-years with growing season length < 50 (how Jonas does his: https://ebi-forecast.igb.illinois.edu/ggcmi/issues/421#note-5)
         this_matyday_array = thisDA.values
         this_matyday_array[np.where(this_matyday_array < 50)] = np.nan
         
         # Rework time axis
         thisMin = np.nanmin(this_matyday_array)
         if thisMin < 0:
            if verbose:
               print(f"{thisModel}: {ncvar}: Setting negative matyday values (min = {thisMin}) to NaN")
            this_matyday_array[np.where(this_matyday_array < 0)] = np.nan
         matyday_da[:,:,:,g] = this_matyday_array
      
      if did_read:
         ggcmiDS[ncvar] = matyday_da
         ggcmiDS[f"{ncvar}-inclmodels"] = matyday_da = xr.DataArray( \
            data=ggcmi_models_bool,
            coords={"model": ggcmiDS.coords["model"]})
      ggcmiDA = ggcmiDS[ncvar].copy()
      
      # If needed, remove outliers
      if noOutliers:
         ggcmiDA = cc.remove_outliers(ggcmiDA)
         
      # Get summary statistic
      if useMedian:
         ggcmiDA_mn = ggcmiDA.median(axis=0)
      else:
         ggcmiDA_mn = np.mean(ggcmiDA, axis=0)
      
      # If you want to remove models that didn't actually simulate this crop-irr, do that here.
      # For now, it just uses the entire list.
      Nggcmi_models = Nggcmi_models_orig
      ggcmi_models = ggcmi_models_orig
      
      # Get GGCMI expected
      if irrtype_str_ggcmi=="noirr":
         tmp_rfir_token = "rf"
      else:
         tmp_rfir_token = "ir"
      thisFile = os.path.join(ggcmi_cropcal_dir, f"{vegtype_str_ggcmi}_{tmp_rfir_token}_ggcmi_crop_calendar_phase3_v1.01.nc4")
      ggcmiExpDS = xr.open_dataset(thisFile)
      ggcmiExp_gslen_yx = ggcmiExpDS["growing_season_length"] / np.timedelta64(1, 'D')
      
      # Get maps from CLM sims
      gslen_ds = xr.Dataset()
      clmExp_gslen_ds = xr.Dataset()
      for casename, case in cases.items():
         case_ds = case['ds']
         
         thisCrop1_gridded = utils.grid_one_variable(case_ds, thisVar, vegtype=vegtype_int).squeeze(drop=True)
         
         # If needed, only include seasons where crop reached maturity
         if onlyMature:
            thisCrop1_gridded = cc.mask_immature(case_ds, vegtype_int, thisCrop1_gridded)
               
         # If needed, remove outliers
         if noOutliers:
            thisCrop1_gridded = cc.remove_outliers(thisCrop1_gridded)
               
         # Get summary statistic
         if useMedian:
            gslen_yx = thisCrop1_gridded.median(axis=0)
         else:
            gslen_yx = np.mean(thisCrop1_gridded, axis=0)
         
         # Save Dataset
         gslen_ds[casename] = gslen_yx
         
         # Save prescribed crops
         if 'rx_gslen_ds' in case:
            if len(clmExp_gslen_ds.dims) > 0:
               raise RuntimeError('Need to amend figure code to allow for multiple prescribed GS lengths')
            clmExp_gslen_yx = case['rx_gslen_ds'][f"gs1_{vegtype_int}"].isel(time=0, drop=True)
            clmExp_gslen_yx = clmExp_gslen_yx.where(np.bitwise_not(np.isnan(gslen_yx)))
            clmExp_gslen_ds['map2'] = clmExp_gslen_yx
      
      
      # Set up figure
      importlib.reload(cc)
      fig = plt.figure(figsize=figsize)
      subplot_title_suffixes = ["", ""]
      ims = []
      
      # Plot CLM cases
      for i, (casename, case) in enumerate(cases.items()):
         gslen_yx = gslen_ds[casename]
         
         if diffExpected:
            map1_yx = gslen_yx - clmExp_gslen_ds['map2']
         else:
            map1_yx = gslen_yx
                  
         ax = cc.make_axis(fig, ny, nx, i+1)
         im = make_map(ax, map1_yx, fontsize, this_title=casename, lonlat_bin_width=bin_width, cmap=cmap)
         ims.append(im[0])
      
      if not diffExpected:
         ax = cc.make_axis(fig, ny, nx, i+2)
         im = make_map(ax, clmExp_gslen_yx, fontsize, this_title="CLM expected", lonlat_bin_width=bin_width, cmap=cmap)
         ims.append(im[0])
         
         ax = cc.make_axis(fig, ny, nx, i+3)
         im = make_map(ax, clmExp_gslen_yx, fontsize, this_title="GGCMI expected", lonlat_bin_width=bin_width, cmap=cmap)
         ims.append(im[0])
      
      # Plot GGCMI outputs
      if diffExpected:
         ggcmiDA_mn_plot = ggcmiDA_mn - ggcmiExp_gslen_yx
      else:
         ggcmiDA_mn_plot = ggcmiDA_mn
      for g in np.arange(Nggcmi_models):
         ggcmi_yx = ggcmiDA_mn_plot.isel(model=g, drop=True)
         if verbose: print(f'{ggcmi_models[g]} map ranges {np.nanmin(ggcmi_yx.values)} to {np.nanmax(ggcmi_yx.values)}')
         ax = cc.make_axis(fig, ny, nx, 3+g+len(caselist))
         if not np.any(~np.isnan(ggcmi_yx.values)):
            ax.set_title(ggcmi_models[g], fontsize=fontsize['titles'])
            continue
         im = make_map(ax, ggcmi_yx, fontsize, this_title=ggcmi_models[g], lonlat_bin_width=bin_width, cmap=cmap)
         ims.append(im[0])
      
      cc.equalize_colorbars(ims, center0=diffExpected)
      
      suptitle = f"{title_prefix}: {vegtype_str_title} ({y1_ggcmi}-{yN_ggcmi})"
      fig.suptitle(suptitle, y=0.95, fontsize=fontsize['titles']*2.2)
      fig.subplots_adjust(bottom=cbar_adj_bottom)
      
      cbar_ax = fig.add_axes(cbar_ax_rect)
      cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
      cbar_ax.tick_params(labelsize=fontsize['ticklabels']*2)
      
      plt.xlabel(units, fontsize=fontsize['titles']*2)
      
      plt.subplots_adjust(wspace=0, hspace=0.3)
      
      # weinrueruebr
      # plt.show()
      # break
      
      # Save
      outfile = os.path.join(outDir_figs, f"{filename_prefix}_{vegtype_str_figfile}.png")
      plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
               bbox_inches='tight')
      plt.close()

# %%

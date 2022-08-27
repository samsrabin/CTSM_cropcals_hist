# %% Setup

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

outDir_figs = "/Users/sam/Documents/Dropbox/2021_Rutgers/CropCalendars/Figures/"

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


# %% Define functions

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
cases['mycode_ggcmicals'] = {'filepath': '/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1976-2013_gddforced2/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1976-2013_gddforced2.clm2.h1.1950-01-01-00000.nc',
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
      this_ds["GRAIN_HARV_TOFOOD_ANN"] = cc.adjust_grainC(this_ds.GRAINC_TO_FOOD, this_ds.patches1d_itype_veg)

      # Rework to match what we already have
      this_ds = this_ds.assign_coords({"ivt": np.arange(np.min(this_ds.patches1d_itype_veg.values),
                                                        np.max(this_ds.patches1d_itype_veg.values)+1)})
      
      # Add vegtype_str
      this_ds['vegtype_str'] = xr.DataArray(data=[utils.ivt_int2str(x) for x in this_ds.ivt.values],
                                            dims = {"ivt": this_ds.ivt})

   else:
      this_ds = cc.import_output(case['filepath'], myVars=myVars, constantVars=case['constantVars'], 
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


# %% Get FAO data from CSV

fao_all = pd.read_csv("/Users/sam/Documents/git_repos/CTSM_cropcals_hist/crop_calendar_MATLAB/FAOSTAT_data_6-15-2022.csv")

fao_all = cc.fao_data_preproc(fao_all)
fao_prod, fao_prod_nosgc = cc.fao_data_get(fao_all, 'Production', y1, yN)
fao_area, fao_area_nosgc = cc.fao_data_get(fao_all, 'Area harvested', y1, yN)


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
# ref_casename = None
ref_casename = 'mycode_clmcals'

overwrite = True

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

         # Grid the included vegetation types, if needed
         if "lon" not in this_map.dims:
            this_map = utils.grid_one_variable(this_ds, this_var, vegtype=found_types)
         # If not, select the included vegetation types
         else:
            this_map = this_map.sel(ivt_str=found_types)
            
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


# %% Import country map and key

# Half-degree countries from Brendan
countries = xr.open_dataset('/Users/sam/Documents/Dropbox/2021_Rutgers/CropCalendars/countries_brendan/gadm0.mask.nc4')

# Nearest-neighbor remap countries to LU resolutions
for resname, res in reses.items():
   res['dsg']['countries'] = utils.lon_idl2pm(countries).interp_like(res['dsg']['AREA'], method='nearest')['gadm0']
   
countries_key = pd.read_csv('/Users/sam/Documents/Dropbox/2021_Rutgers/CropCalendars/countries_brendan/Nation_ID.csv',
                               header=None,
                               names=['num', 'name'])

fao_all_ctry = pd.read_csv("/Users/sam/Documents/git_repos/CTSM_cropcals_hist/crop_calendar_MATLAB/FAOSTAT_data_en_8-21-2022_byCountry.csv")
fao_all_ctry = cc.fao_data_preproc(fao_all_ctry)

# Replace some countries' names in key to match FAO data
countries_key = countries_key.replace({'China': 'China, mainland',   # Because it also has Taiwan
                                       'Turkey': 'Türkiye',
                                       'Vietnam': 'Viet Nam'})
# Replace some countries' names in FAO data to match key (to keep them short for figs)
fao_all_ctry = fao_all_ctry.replace({'Bolivia (Plurinational State of)': 'Bolivia',
                                     'Russian Federation': 'Russia',
                                     'Syrian Arab Republic': 'Syria',
                                     'United States of America': 'United States'})

# Make sure every country in map is in key
for i, x in enumerate(np.unique(countries.gadm0.values)):
   if not np.isnan(x) and not np.any(countries_key.num.values == x):
      print(f'❗ {x} not found in key')


# %% Make scatter plots, FAOSTAT vs. CLM, of top 10 countries for each crop

# top_y1 = 1961 # First year of FAO data
top_y1 = 1992 # Pre-1992, you start getting USSR, which isn't in map
top_yN = 2009
overwrite = True

topYears = np.arange(top_y1, top_yN+1)
NtopYears = len(topYears)
fao_mean_byCountry = cc.get_mean_byCountry(fao_all_ctry, top_y1, top_yN)

fao_crops = np.unique(fao_all_ctry.Crop.values)
Ntop = 10
topN_list = []
countries_in_topNs = []
for thisCrop in fao_crops:
   topN = fao_mean_byCountry[thisCrop]['Production'].nlargest(Ntop)
   topN_list.append(topN)
   countries_in_topNs = countries_in_topNs + list(topN.keys())

# Which countries in our top 10 lists are not found in our countries key?
countries_in_topNs = np.unique(countries_in_topNs)
any_ctry_notfound = False
for thisCountry in countries_in_topNs:
   if thisCountry not in countries_key.name.values:
      print(f'❗ {thisCountry} not in countries_key')
      any_ctry_notfound = True
if any_ctry_notfound:
   raise RuntimeError('At least one country in FAO not found in key')
      
i_theseYears_earthstat = [i for i, x in enumerate(earthstats['f09_g17'].time.values) if (x.year >= top_y1) and (x.year <= top_yN)]

for c, thisCrop in enumerate(fao_crops):
      
   thisCrop_clm = cc.cropnames_fao2clm(thisCrop)
   topN = topN_list[c]
   
   suptitle = f"{thisCrop}, FAOSTAT vs CLM ({top_y1}-{top_yN})"
   fig_outfile = outDir_figs + "Yield anom scatter top 10 " + suptitle + ".pdf"
   if os.path.exists(fig_outfile) and not overwrite:
      print(f'   Skipping {thisCrop_out} (file exists).')
      continue
   
   prod_ar = np.full((len(cases), NtopYears, Ntop), np.nan)
   area_ar = np.full((len(cases), NtopYears, Ntop), np.nan)
   prod_faostat_yc = np.full((NtopYears, Ntop), np.nan)
   area_faostat_yc = np.full((NtopYears, Ntop), np.nan)
   prod_earthstat_yc = np.full((NtopYears, Ntop), np.nan)
   area_earthstat_yc = np.full((NtopYears, Ntop), np.nan)
   
   for i_case, (casename, case) in enumerate(cases.items()):
      case_ds = case['ds']
      lu_ds = reses[case['res']]['dsg']
      countries_map = lu_ds['countries'].load()

      i_theseYears_case = [i for i, x in enumerate(case_ds.time.values) if (x.year >= top_y1) and (x.year <= top_yN)]
      i_theseYears_lu = [i for i, x in enumerate(lu_ds.time.values) if (x.year >= top_y1) and (x.year <= top_yN)]
      
      # Find this crop in production and area data
      i_thisCrop_case = [i for i, x in enumerate(case_ds.vegtype_str.values) if thisCrop_clm in x]
      if len(i_thisCrop_case) == 0:
         raise RuntimeError(f'No matches found for {thisCrop} in case_ds.vegtype_str')
      i_thisCrop_lu = [i for i, x in enumerate(lu_ds.cft.values) if thisCrop_clm in utils.ivt_int2str(x)]
      if len(i_thisCrop_lu) == 0:
         raise RuntimeError(f'No matches found for {thisCrop} in lu_ds.cft')
      
      # Get each top-N country's time series for this crop
      for c, country in enumerate(topN.keys()):
         country_id = countries_key.query(f'name == "{country}"')['num'].values
         if len(country_id) != 1:
            raise RuntimeError(f'Expected 1 match of {country} in countries_key; got {len(country_id)}')
         
         # Yield...
         yield_da = case_ds['GRAIN_HARV_TOFOOD_ANN_GD']\
            .isel(ivt_str=i_thisCrop_case, time=i_theseYears_case)\
            .sum(dim='ivt_str')\
            .where(countries_map == country_id)\
            * 1e-6 * 1e4 # g/m2 to tons/ha
                     
         # Area...
         area_da = lu_ds['AREA_CFT']\
            .isel(cft=i_thisCrop_lu, time=i_theseYears_lu)\
            .sum(dim='cft')\
            .where(countries_map == country_id)\
            * 1e-4 # m2 to ha
         area_ar[i_case,:,c] = area_da.sum(dim=['lon', 'lat']).values 
         
         # Production (tons)
         prod_ar[i_case,:,c] = (yield_da * area_da).sum(dim=['lon', 'lat'])
            
         # FAOSTAT
         if i_case == 0:
            # Production (tons)
            prod_faostat_yc[:,c] = fao_all_ctry.query(f'Area == "{country}" & Crop == "{thisCrop}" & Element == "Production" & Year >= {top_y1} & Year <= {top_yN}')['Value'].values
            # Area (ha)
            area_faostat_yc[:,c] = fao_all_ctry.query(f'Area == "{country}" & Crop == "{thisCrop}" & Element == "Area harvested" & Year >= {top_y1} & Year <= {top_yN}')['Value'].values
         
         # EarthStat
         if np.all(np.isnan(prod_earthstat_yc[:,c])) and case['res']=='f09_g17':
            prod_earthstat_yc[:,c] = earthstats[case['res']]['Production']\
               .interp_like(countries_map)\
               .sel(crop=thisCrop_clm.title())\
               .isel(time=i_theseYears_earthstat)\
               .where(countries_map == country_id)\
               .sum(dim=['lon', 'lat'])\
               .values
            area_earthstat_yc[:,c] = earthstats[case['res']]['HarvestArea']\
               .interp_like(countries_map)\
               .sel(crop=thisCrop_clm.title())\
               .isel(time=i_theseYears_earthstat)\
               .where(countries_map == country_id)\
               .sum(dim=['lon', 'lat'])\
               .values

   new_coords = {'Case': caselist,
                 'Year': topYears,
                 'Country': topN.keys().values}
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
                                           'Country': topN.keys().values},
                                  attrs = {'units': 'tons'})
   area_faostat_da = xr.DataArray(data = area_faostat_yc,
                                  coords = {'Year': topYears,
                                           'Country': topN.keys().values},
                                  attrs = {'units': 'ha'})
   yield_faostat_da = prod_faostat_da / area_faostat_da
   yield_faostat_da = yield_faostat_da.assign_attrs({'units': 'tons/ha'})
   prod_earthstat_da = xr.DataArray(data = prod_earthstat_yc,
                                    coords = {'Year': topYears,
                                              'Country': topN.keys().values},)
   area_earthstat_da = xr.DataArray(data = area_earthstat_yc,
                                    coords = {'Year': topYears,
                                              'Country': topN.keys().values})
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
         for C, country in enumerate(topN.keys()):
            tmp_dt_cy = tmp_dt_cyC[:,:,C]
            tmp_dt_cy = cc.detrend(tmp_dt_cy)
            tmp_dt_cyC[:,:,C] = tmp_dt_cy
         topN_dt_ds[v] = xr.DataArray(data = tmp_dt_cyC,
                                      coords = topN_ds[v].coords,
                                      attrs = topN_ds[v].attrs)
         for C, country in enumerate(topN.keys()):
            tmp_ya_cy = tmp_ya_cyC[:,:,C]
            tmp_ya_cy = cc.yield_anomalies(tmp_ya_cy)
            tmp_ya_cyC[:,:,C] = tmp_ya_cy
         topN_ya_ds[v] = xr.DataArray(data = tmp_ya_cyC,
                                      coords = topN_ds[v].coords,
                                      attrs = topN_ds[v].attrs)
      else:
         tmp_dt_Cy = np.transpose(topN_ds[v].copy().values)
         tmp_dt_Cy = cc.detrend(tmp_dt_Cy)
         topN_dt_ds[v] = xr.DataArray(data = np.transpose(tmp_dt_Cy),
                                      coords = topN_ds[v].coords,
                                      attrs = topN_ds[v].attrs)
         tmp_ya_Cy = np.transpose(topN_ds[v].copy().values)
         tmp_ya_Cy = cc.yield_anomalies(tmp_ya_Cy)
         topN_ya_ds[v] = xr.DataArray(data = np.transpose(tmp_ya_Cy),
                                      coords = topN_ds[v].coords,
                                      attrs = topN_ds[v].attrs)
      topN_ya_ds[v].attrs['units'] = 'anomalies (unitless)'
         

   # plot_ds = topN_ds
   # plot_ds = topN_dt_ds
   plot_ds = topN_ya_ds

   ny = 4
   nx = 3

   figsize = (10, 16)
   f, axes = plt.subplots(ny, nx, figsize=figsize)
   axes = axes.flatten()

   # This one will have hollow circles
   i_h = caselist.index('mycode_clmcals')

   # Text describing R2 changes for each country
   r2_change_text = ""

   for c, country in enumerate(plot_ds.Country.values):
      ax = axes[c]
      sc = xr.plot.scatter(plot_ds.sel(Country=country),
                           x='Yield (FAOSTAT)',
                           y='Yield',
                           hue='Case',
                           ax=ax)
      
      for case in caselist:
         lr = stats.linregress(x = plot_ds['Yield (FAOSTAT)'].sel(Country=country),
                              y = plot_ds['Yield'].sel(Country=country, Case=case))
         if case == "mycode_clmcals":
            t = country + ": " + "{r1:.3g} $\\rightarrow$ "
            r2_change_text += t.format(r1=lr.rvalue**2)
         elif case == "mycode_ggcmicals":
            r2_change_text += "{r2:.3g}\n".format(r2=lr.rvalue**2)
      # Set this one to have hollow circles
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
      if c == Ntop-1:
         ax.legend(bbox_to_anchor=(3,0.5), loc='center')

   # Delete unused axes, if any
   for a in np.arange(c+1, ny*nx):
      f.delaxes(axes[a])
      
   # Add row labels
   leftmost = np.arange(0, nx*ny, nx)
   for i, a in enumerate(leftmost):
      axes[a].set_ylabel("Yield anomaly (CLM)", fontsize=12, fontweight='bold')
      axes[a].yaxis.set_label_coords(-0.15, 0.5)

   # Add column labels
   bottommost = np.arange(Ntop)[-3:]
   bottommost_cols = bottommost % nx
   for i, a in enumerate(bottommost):
      ax = axes[a]
      x = np.mean(ax.get_xlim())
      y = ax.get_ylim()[0] - 0.2*np.diff(ax.get_ylim()) # 20% below the bottom edge
      ax.text(x=x, y=y,
              s="Yield anomaly (FAOSTAT)",
              fontsize=12,
              ha='center',
              fontweight='bold')

   # Add figure title
   suptitle_xpos = 0.5
   suptitle_ypos = 0.91
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
      
   # Add "title" to text
   r2_change_text = r"$\bf{R^2\/changes}\/CLM \rightarrow GGCMI}$" + "\n" + r2_change_text
   # Add text to plot
   ax = axes[Ntop-1]
   x = ax.get_xlim()[1] + 0.2*np.diff(ax.get_xlim()) # 20% past the right edge
   y = ax.get_ylim()[1]
   ax.text(x, y, r2_change_text[:-1], va='top');
   tmp = ax.get_children()[4]
   
   f.savefig(fig_outfile,
             bbox_inches='tight', facecolor='white')
   plt.close()
   
   
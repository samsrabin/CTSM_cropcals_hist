# %% Setup

# which_cases = "ctsm_lu_5.0_vs_5.2"
# which_cases = "diagnose.52"
# which_cases = "diagnose.2022"
# which_cases = "diagnose"
which_cases = "main2.52"
# which_cases = "main2.2022"
# which_cases = "main2"
# which_cases = "originalBaseline" # As originalCLM, but without cmip6
# which_cases = "originalCLM"
# which_cases = "test"


# Include irrigation?
incl_irrig = True

# Yields will be set to zero unless HUI at harvest is ≥ min_viable_hui
min_viable_hui = 1

# Include diagnostic info in figures?
diagnostics = True

# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import cropcal_module as cc
import cropcal_figs_module as ccf

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

import numpy as np
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
from matplotlib import cm
import datetime as dt
import re
import importlib
import pandas as pd
import cftime
from fig_global_timeseries import global_timeseries_yieldetc, global_timeseries_irrig_inclcrops, global_timeseries_irrig_allcrops
from fig_maps_allCrops import *
from fig_maps_eachCrop import maps_eachCrop
from fig_maps_grid import maps_gridlevel_vars

# Ignore these annoying warnings
import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")


# %% Define functions etc.

cropList_combined_clm_nototal = ["Corn", "Rice", "Cotton", "Soybean", "Sugarcane", "Wheat"]
cropList_combined_clm_nototal.sort()
cropList_combined_clm = cropList_combined_clm_nototal.copy()
cropList_combined_clm.append("Total")
# cropList_combined_faostat = ["Maize", "Rice, paddy", "Seed cotton", "Soybeans", "Sugar cane", "Wheat"]

fao_to_clm_dict = {"Maize": "Corn",
                   "Rice, paddy": "Rice",
                   "Seed cotton": "Cotton",
                   "Soybeans": "Soybean",
                   "Sugar cane": "Sugarcane",
                   "Wheat": "Wheat",
                   "Total": "Total"}

plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

# Add GRAINC variants with too-long seasons set to 0
# CLM max growing season length, mxmat, is stored in the following files:
#	 * clm5_1: lnd/clm2/paramdata/ctsm51_params.c211112.nc
#	 * clm5_0: lnd/clm2/paramdata/clm50_params.c211112.nc
#	 * clm4_5: lnd/clm2/paramdata/clm45_params.c211112.nc
paramfile_dir = "/Users/Shared/CESM_inputdata/lnd/clm2/paramdata/"
my_clm_ver = 50
my_clm_subver = "c211112"
pattern = os.path.join(paramfile_dir, f"*{my_clm_ver}_params.{my_clm_subver}.nc")
paramfile = glob.glob(pattern)
if len(paramfile) != 1:
    raise RuntimeError(f"Expected to find 1 match of {pattern}; found {len(paramfile)}")
paramfile_ds = xr.open_dataset(paramfile[0])
# Import max growing season length (stored in netCDF as nanoseconds!)
paramfile_mxmats = paramfile_ds["mxmat"].values / np.timedelta64(1, 'D')
# Import PFT name list
paramfile_pftnames = [x.decode("UTF-8").replace(" ", "") for x in paramfile_ds["pftname"].values]
# Save as dict
mxmats = {}
for i, mxmat in enumerate(paramfile_mxmats):
    mxmats[paramfile_pftnames[i]] = mxmat


# %% Import model output
importlib.reload(cc)
importlib.reload(utils)

y1 = 1961
yN = 2013

gs1 = y1
gsN = yN - 1
yearList = np.arange(y1,yN+1)
Nyears = len(yearList)

# Define cases
cases = cc.get_caselist(which_cases)

outDir_figs = "/Users/sam/Documents/Dropbox/2021_Rutgers/CropCalendars/Figures/"
for i, (casename, case) in enumerate(cases.items()):
    outDir_figs += case['verbosename'].replace(": ", " | ") + "/"
outDir_figs += "figs/"
if not os.path.exists(outDir_figs):
    os.makedirs(outDir_figs)

# Note that _PERHARV will be stripped off upon import
myVars = ['GRAINC_TO_FOOD_PERHARV', 'GRAINC_TO_FOOD_ANN', 'SDATES', 'SDATES_PERHARV', 'SYEARS_PERHARV', 'HDATES', 'HYEARS', 'GDDHARV_PERHARV', 'GDDACCUM_PERHARV', 'HUI_PERHARV', 'SOWING_REASON_PERHARV', 'HARVEST_REASON_PERHARV']

verbose_import = False
caselist = []
for i, (casename, case) in enumerate(cases.items()):
    print(f"Importing {casename}...")
    caselist.append(casename)
    
    irrig_ds = None
    if casename == 'cmip6':
        this_ds = xr.open_dataset(case['filepath'])

        # Convert gC/m2 to g/m2 actually harvested
        this_ds["YIELD"] = cc.adjust_grainC(this_ds.GRAINC_TO_FOOD, this_ds.patches1d_itype_veg)
        this_ds["YIELD"].attrs['min_viable_hui'] = 0.0
        this_ds["YIELD"].attrs['mxmat_limited'] = True
        this_ds["YIELD"].attrs['locked'] = True
        
        # Rework to match what we already have
        this_ds = this_ds.assign_coords({"ivt": np.arange(np.min(this_ds.patches1d_itype_veg.values),
                                                          np.max(this_ds.patches1d_itype_veg.values)+1)})
        
        # Add vegtype_str
        this_ds['vegtype_str'] = xr.DataArray(data=[utils.ivt_int2str(x) for x in this_ds.ivt.values],
                                              dims = {"ivt": this_ds.ivt})

    else:
        this_ds = cc.import_output(case['filepath'], myVars=myVars,
                                   y1=y1, yN=yN, verbose=verbose_import,
                                   incl_irrig=incl_irrig)
        bad_patches = cc.check_constant_vars(this_ds, case, ignore_nan=True, constantGSs=case['constantGSs'], verbose=True, throw_error=False)
        # for p in bad_patches:
        #	  cc.print_gs_table(this_ds.isel(patch=p))
    
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

# Get names for CLM crop production
for _, case in cases.items():
    case['ds']['patches1d_itype_combinedCropCLM_str'] = \
        xr.DataArray(cc.fullname_to_combinedCrop(case['ds']['patches1d_itype_veg_str'].values, cropList_combined_clm),
                     coords = case['ds']['patches1d_itype_veg_str'].coords)

print("Done importing model output.")


# %% Import LU data
importlib.reload(cc)

# Define resolutions
reses = {}
# f09_g17 ("""1-degree"""; i.e., 1.25 lon x 0.9 lat)
reses["f09_g17"] = {"lu_path": "/Users/Shared/CESM_work/CropEvalData_ssr/landuse.timeseries_0.9x1.25_hist_78pfts_CMIP6_simyr1850-2015_c170824.nc4"}
# f19_g17 ("""2-degree"""; i.e., 2.5 lon x 1.9 lat), ctsm5.0
reses["f19_g17"] = {"lu_path": "/Users/Shared/CESM_work/CropEvalData_ssr/landuse.timeseries_1.9x2.5_hist_78pfts_CMIP6_simyr1850-2015_c170824.nc4"}
# f19_g17 ("""2-degree"""; i.e., 2.5 lon x 1.9 lat), ctsm5.2
reses["f19_g17_ctsm5.2"] = {"lu_path": "/Users/Shared/CESM_work/CropEvalData_ssr/landuse.timeseries_1.9x2.5_SSP5-8.5_78_CMIP6_1850-2015_c230227.nc"}

# Import land use to reses dicts
for i, (resname, res) in enumerate(reses.items()):
    
    # Find a matching case
    for (_, case) in cases.items():
        if case["res"] == resname:
            break
    if case["res"] != resname:
        continue
    print(f"Importing {resname}...")
    
    res['ds'] = cc.open_lu_ds(res['lu_path'], y1, yN+1, case['ds'].sel(time=slice(y1,yN+1)))
    res['ds'] = res['ds'].assign_coords({"time": [cftime.DatetimeNoLeap(y, 1, 1, 0, 0, 0, 0, has_year_zero=True) for y in res['ds'].time.values]})
    res['dsg'] = xr.Dataset(data_vars={'AREA': utils.grid_one_variable(res['ds'], 'AREA')})
        
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
    lu_ds = reses[case['res']]['ds'].copy()
    case_ds, lu_ds, lon_tolerance = cc.round_lonlats_to_match_ds(case_ds, lu_ds, "lon", initial_tolerance)
    case_ds, lu_ds, lat_tolerance = cc.round_lonlats_to_match_ds(case_ds, lu_ds, "lat", initial_tolerance)
        
    # Ensure that time axes are formatted the same
    case_ds = cc.time_units_and_trim(case_ds, y1, yN+1, cftime.DatetimeNoLeap)
    lu_ds = cc.time_units_and_trim(lu_ds, y1, yN+1, cftime.DatetimeNoLeap)

    # Save
    case['ds'] = case_ds
    case['ds'].load()
    reses[case['res']]['ds'] = lu_ds

print("Generating crop area masks...")
for i, (casename, case) in enumerate(cases.items()):
    print(casename + "...")
    lu_ds = reses[case['res']]['ds']
    
    # Ensure that 'time' axes line up
    if not np.array_equal(case['ds']['time'].values, lu_ds['time'].values):
        raise RuntimeError(f"Time axis mismatch between {casename} outputs and land use file")
    
    # Save crop area to case Dataset
    case['ds']['AREA_CFT'] = lu_ds['AREA_CFT']
    
    # Where was there crop area at sowing?
    # Straightforward: 'time' axis lines up for model outputs and LU
    case['ds']['croparea_positive_sowing'] = lu_ds['AREA_CFT'] > 0
    
    # Where was there crop area throughout the season? This is needed for variables with time axis 'gs' (growing season). Where harvest happened the same year as planting, we can just use croparea_positive_sowing. Elsewhere: Mask out cells where either sowing OR harvest year had 0 area.
    # First, make sure the time/gs axes align correctly.
    yearY_as_gs = lu_ds['time'].isel(time=slice(0, lu_ds.dims['time'] - 1)).dt.year.values
    if not np.array_equal(yearY_as_gs, case['ds']['gs'].values):
        raise RuntimeError("Growing season mismatch")
    # Start with where there was crop area at sowing.
    croparea_positive_wholeseason = case['ds']['croparea_positive_sowing'].isel(time=slice(0, lu_ds.dims['time'] - 1)).copy().values
    # Where was crop area positive in years Y and Y+1? Need to use .values because time coordinates won't match.
    croparea_positive_bothyears = ((lu_ds['AREA_CFT'].isel(time=slice(0, lu_ds.dims['time'] - 1)) > 0).values # sowing year
                                 & (lu_ds['AREA_CFT'].isel(time=slice(1, lu_ds.dims['time'])) > 0).values) # harvest year
    # In patch-growingseasons where harvest year is different from sowing year, use croparea_positive_bothyears instead of croparea_positive_sowing.
    where_harvyear_not_sowyear = np.where((case['ds']['SYEARS'] != case['ds']['HYEARS']).values)
    croparea_positive_wholeseason[where_harvyear_not_sowyear] = croparea_positive_bothyears[where_harvyear_not_sowyear]
    # Convert that to DataArray, using time coordinate of year Y as growing season
    case['ds']['croparea_positive_wholeseason'] = xr.DataArray(data=croparea_positive_wholeseason,
                                                           coords={'patch': lu_ds['patch'],
                                                                   'gs': yearY_as_gs})

    # NOW we can trim the last year from the time axis, at least in this case's Dataset
    case['ds'] = cc.time_units_and_trim(case['ds'], y1, yN, cftime.DatetimeNoLeap)
    

# And now that all that's done, trim the last year from land use data.
for _, res in reses.items():
    if 'ds' in res:
        res['ds'] = cc.time_units_and_trim(res['ds'], y1, yN, cftime.DatetimeNoLeap)

print("Done.")
 

# %% Calculate irrigation totals
importlib.reload(cc)

if incl_irrig:
    mms_to_mmd = 60*60*24
    print("Calculating irrigation totals...")
    for i, (casename, case) in enumerate(cases.items()):
        mmd_to_m3d = 1e-3
        area_cft = reses[case['res']]['ds']['AREA_CFT']
        area_cft_timemax = area_cft.max(dim="time")
        area_irrig_grid_timemax =  reses[case['res']]['ds']['IRRIGATED_AREA_GRID'].max(dim="time")
        mmd_to_m3d_patch = area_cft * mmd_to_m3d
        mmd_to_m3d_grid = case['ds']['AREA_GRID'] * mmd_to_m3d
        for v in case['ds']:
            if "QIRRIG" not in v or "_FRAC_" in v:
                continue
            
            # Mask where no area
            if "patch" in case['ds'][v].dims:
                case['ds'][v] = case['ds'][v].where(area_cft_timemax > 0)
            elif "gridcell" in case['ds'][v].dims:
                case['ds'][v] = case['ds'][v].where(area_irrig_grid_timemax > 0)
            else:
                print(f"Unable to mask no-area members of {v}")
                        
            # Convert mm/s to mm/day
            v_da = case['ds'][v]
            if v_da.attrs['units'] == 'mm/s':
                v_da *= mms_to_mmd
                v_da.attrs['units'] = "mm/d"
            else:
                print(f"Not calculating total for {v} (units {v_da.attrs['units']}, not mm/s)")
                continue
            
            # Calculate total patch-level irrigation (mm/s → m3)
            v2 = v.replace('QIRRIG', 'IRRIG')
            v2_da = case['ds'][v].copy()
            if v2_da.attrs['units'] != 'mm/d':
                raise RuntimeError(f"How are {v} units {v2_da.attrs['units']} not mm/d?")
            elif "time_mth" in case['ds'][v].dims:
                print(f"Not calculating total for {v} (monthly-fying areas too memory-intensive)")
                continue
                days_in_month = v2_da['time_mth'].dt.days_in_month
                v2_da *= days_in_month * mms_to_m3d_patch
            
            if "ANN" in v:
                if "PATCH" in v:
                    v2_da *= mmd_to_m3d_patch * 365
                elif "GRID" in v:
                    v2_da *= mmd_to_m3d_grid * 365
                else:
                    raise RuntimeError(f"Unsure how to get total for {v}")
            else:
                raise RuntimeError(f"Decide how/whether to calculate total for {v}")
            v2_da.attrs['units'] = "m^3"
            case['ds'][v2] = v2_da
            
            # Calculate total gridcell-level irrigation for included crops
            if "PATCH" in v2:
                v3 = v2.replace("PATCH", "GRID")
                grid_index_var = 'patches1d_gi'
                case['ds'][v3] = case['ds'][v2].groupby(case['ds'][grid_index_var]).sum().rename({grid_index_var: 'grid'})
                case['ds'][v3].attrs = case['ds'][v2].attrs
        

# %% Import GGCMI sowing and harvest dates, and check sims
importlib.reload(cc)

for i, (casename, case) in enumerate(cases.items()):
    
    if 'gdd_min' in case:
        gdd_min = case['gdd_min']
    else:
        gdd_min = None
    
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
fao_prod, fao_prod_nosgc = cc.fao_data_get(fao_all, 'Production', y1, yN, fao_to_clm_dict, cropList_combined_clm)
fao_area, fao_area_nosgc = cc.fao_data_get(fao_all, 'Area harvested', y1, yN, fao_to_clm_dict, cropList_combined_clm)


# %% Import FAO Earthstat (gridded FAO data)

print("Importing FAO EarthStat...")
earthstats_gd = {}
earthstats_gd = {}

# Import high res
earthstats_gd['f09_g17'] = xr.open_dataset('/Users/Shared/CESM_work/CropEvalData_ssr/FAO-EarthStatYields/EARTHSTATMIRCAUNFAOCropDataDeg09.nc')

# Include just crops we care about
cropList_fao_gd_all = ['Wheat', 'Maize', 'Rice', 'Barley', 'Rye', 'Millet', 'Sorghum', 'Soybeans', 'Sunflower', 'Potatoes', 'Cassava', 'Sugar cane', 'Sugar beet', 'Oil palm', 'Rape seed / Canola', 'Groundnuts / Peanuts', 'Pulses', 'Citrus', 'Date palm', 'Grapes / Vine', 'Cotton', 'Cocoa', 'Coffee', 'Others perennial', 'Fodder grasses', 'Others annual', 'Fibre crops', 'All crops']
cropList_fao_gd = ["Maize", "Cotton", "Rice", "Soybeans", "Sugar cane", "Wheat"]
earthstats_gd['f09_g17'] = earthstats_gd['f09_g17'].isel(crop=[cropList_fao_gd_all.index(x) for x in cropList_fao_gd]).assign_coords({'crop': cropList_combined_clm[:-1]})

# Include just years we care about
earthstats_gd['f09_g17'] = earthstats_gd['f09_g17'].isel(time=[i for i,x in enumerate(earthstats_gd['f09_g17'].time.values) if x.year in np.arange(y1,yN+1)])

# Interpolate to lower res, starting with just variables that won't get messed up by the interpolation
print("Interpolating to f19_g17...")
res_ds = None
for r in reses.keys():
    if "f19_g17" in r and "ds" in reses[r]:
        res_ds = reses[r]["ds"]
        break
if res_ds is None:
    raise RuntimeError(f"No f19_g17 Dataset found in reses! Keys checked: {reses.keys()}")
interp_lons = res_ds['lon']
interp_lats = res_ds['lat']
drop_vars = ['Area', 'HarvestArea', 'IrrigatedArea', 'PhysicalArea', 'RainfedArea', 'Production']
earthstats_gd['f19_g17'] = earthstats_gd['f09_g17']\
    .drop(labels=drop_vars)\
    .interp(lon=interp_lons, lat=interp_lats)

# Disallow negative interpolated values
for v in earthstats_gd['f19_g17']:
    vals = earthstats_gd['f19_g17'][v].values
    vals[vals < 0] = 0
    earthstats_gd['f19_g17'][v] = xr.DataArray(data = vals,
                                               coords = earthstats_gd['f19_g17'][v].coords,
                                               attrs = earthstats_gd['f19_g17'][v].attrs)
    
# These are not exact, but it seems like:
#	  PhysicalArea = RainfedArea + IrrigatedArea (max diff ~51 ha, max rel diff ~0.01%)
#	  PhysicalFraction = RainfedFraction + IrrigatedFraction (max rel diff ~0.01%)
#	  Production = HarvestArea * Yield (max diff 4 tons, max rel diff ~0.000012%)
# But!!!
#	  PhysicalArea ≠ Area * LandFraction * PhysicalFraction (max rel diff 100%)

# Re-calculate variables that were dropped
f19_g17_cellarea = utils.grid_one_variable(res_ds, 'AREA', fillValue=0)
f19_g17_landarea = f19_g17_cellarea * earthstats_gd['f19_g17']['LandFraction']
f19_g17_croparea_ha = f19_g17_landarea * earthstats_gd['f19_g17']['PhysicalFraction']*100
recalc_ds = xr.Dataset(data_vars = \
    {'Area': f19_g17_cellarea,
     'PhysicalArea': f19_g17_croparea_ha,
     'HarvestArea': f19_g17_croparea_ha * earthstats_gd['f19_g17']['HarvestFraction'],
     'IrrigatedArea': f19_g17_croparea_ha * earthstats_gd['f19_g17']['IrrigatedFraction'],
     'RainfedArea': f19_g17_croparea_ha * earthstats_gd['f19_g17']['RainfedFraction'],
     'Production': f19_g17_croparea_ha * earthstats_gd['f19_g17']['HarvestFraction']*earthstats_gd['f19_g17']['Yield']})
for v in recalc_ds:
    recalc_ds[v].attrs = earthstats_gd['f09_g17'].attrs
    if recalc_ds[v].dims == ('lat', 'lon', 'crop', 'time'):
        recalc_ds[v] = recalc_ds[v].transpose('crop', 'time', 'lat', 'lon')
    discrep_sum_rel = 100*(np.nansum(recalc_ds[v].values) - np.sum(earthstats_gd['f09_g17'][v].values)) / np.sum(earthstats_gd['f09_g17'][v].values)
    print(f"Discrepancy in {v} f19_g17 rel to f09_g17: {discrep_sum_rel}%")
earthstats_gd['f19_g17'] = earthstats_gd['f19_g17'].merge(recalc_ds)

# Check consistency of non-dropped variables
for v in earthstats_gd['f19_g17']:
    if "Area" in v or v == "Production":
        continue
    discrep_sum_rel = 100*(np.mean(earthstats_gd['f19_g17'][v].values) - np.mean(earthstats_gd['f09_g17'][v].values)) / np.mean(earthstats_gd['f09_g17'][v].values)
    print(f"Discrepancy in {v} f19_g17 rel to f09_g17: {discrep_sum_rel}%")

# Ungrid
importlib.reload(cc)
earthstats={}
earthstats['f19_g17'] = cc.ungrid(earthstats_gd['f19_g17'],
                                  cases[[x for x in cases.keys()][0]]['ds'], 'GRAINC_TO_FOOD_ANN',
                                  lon='patches1d_ixy',
                                  lat='patches1d_jxy',
                                  crop='patches1d_itype_combinedCropCLM_str')

earthstats['f19_g17_ctsm5.2'] = earthstats['f19_g17']

print("Done importing FAO EarthStat.")


# %% Import country map and key

# Half-degree countries from Brendan
countries = xr.open_dataset('/Users/sam/Documents/Dropbox/2021_Rutgers/CropCalendars/countries_brendan/gadm0.mask.nc4')

# Nearest-neighbor remap countries to gridspecs we have EarthStats data for
for gridspec, dsg in earthstats_gd.items():
    dsg['countries'] = utils.lon_idl2pm(countries).interp_like(dsg['Area'], method='nearest')['gadm0']
    if gridspec in earthstats:
        ds = earthstats[gridspec]
        ds['countries'] = cc.ungrid(dsg['countries'], ds, 'Area', lon='patches1d_ixy', lat='patches1d_jxy')

# Save those to LU Datasets as well
for resname, res in reses.items():
    if 'dsg' not in res:
        continue
    res['dsg']['countries'] = utils.lon_idl2pm(countries).interp_like(res['dsg']['AREA'], method='nearest')['gadm0']
    res['ds']['countries'] = cc.ungrid(res['dsg']['countries'], res['ds'], 'AREA', lon='patches1d_ixy', lat='patches1d_jxy')
    
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
countries_key = countries_key.replace({'China': 'China, mld.',	  # Because it also has Taiwan
                                       'United States': 'USA',
                                       'South Africa': 'S. Africa'})
fao_all_ctry = fao_all_ctry.replace({'China, mainland': 'China, mld.',
                                     'United States of America': 'USA',
                                     'South Africa': 'S. Africa'})

# Make sure every country in map is in key
for i, x in enumerate(np.unique(countries.gadm0.values)):
    if not np.isnan(x) and not np.any(countries_key.num.values == x):
        print(f'❗ {x} not found in key')
        
        
        
        
# END OF IMPORT AND SETUP
print('Done with import and setup.')





# %% * Compare area, production, yield, and irrigation of individual crops
importlib.reload(cc)
importlib.reload(sys.modules['fig_global_timeseries'])
from fig_global_timeseries import global_timeseries_yieldetc, global_timeseries_irrig_inclcrops, global_timeseries_irrig_allcrops

# # GGCMI setup, for reference:
# min_viable_hui = "ggcmi3"
# w = 5
# use_annual_yields = False
# plot_y1 = 1980
# plot_yN = 2010

### min_viable_hui = 1.0
min_viable_hui = "ggcmi3"
include_scatter = True
# min_viable_hui = ["ggcmi3", 0, 1]
# include_scatter = False

include_shiftsens = False

# Window width for detrending (0 for signal.detrend())
w = 5

# Use annual yields? If not, uses growing-season yields.
use_annual_yields = False

mxmat_limited = False
# mxmat_limited = True

# Equalize axes of scatter plots?
equalize_scatter_axes = True

# Remove bias from scatter plots?
remove_scatter_bias = True

# extra = "Total (no sgc)"
extra = "Total (grains)"

# Rounding precision for stats
bias_round = 1
corrcoef_round = 3

# Which observation dataset should be included on figure?
obs_for_fig = "FAOSTAT"

# Which years to include?
plot_y1 = 1980
plot_yN = 2009

# Do not actually make figures
noFigs = False

# Set up figure
# ny = 2
# nx = 4
# figsize = (35, 18)
ny = 3
nx = 3
figsize = (25, 18)


if mxmat_limited:
    mxmats_tmp = mxmats
else:
    mxmats_tmp = None
    
cases = global_timeseries_yieldetc(cases, cropList_combined_clm, earthstats_gd, fao_area, fao_area_nosgc, fao_prod, fao_prod_nosgc, outDir_figs, reses, yearList, equalize_scatter_axes=equalize_scatter_axes, extra=extra, figsize=figsize, include_scatter=include_scatter, include_shiftsens=include_shiftsens, min_viable_hui_list=min_viable_hui, mxmats=mxmats_tmp, noFigs=noFigs, ny=ny, nx=nx, obs_for_fig=obs_for_fig, plot_y1=plot_y1, plot_yN=plot_yN, remove_scatter_bias=remove_scatter_bias, bias_round=bias_round, corrcoef_round=corrcoef_round, use_annual_yields=use_annual_yields, w=w)

if incl_irrig:
    global_timeseries_irrig_inclcrops("IRRIG_DEMAND_PATCH_ANN", cases, reses, cropList_combined_clm, outDir_figs, extra="Total (grains)", figsize=figsize, noFigs=noFigs, ny=ny, nx=nx, plot_y1=plot_y1, plot_yN=plot_yN)
    global_timeseries_irrig_inclcrops("IRRIG_APPLIED_PATCH_ANN", cases, reses, cropList_combined_clm, outDir_figs, extra="Total (all land)", figsize=figsize, noFigs=noFigs, ny=ny, nx=nx, plot_y1=plot_y1, plot_yN=plot_yN)
    global_timeseries_irrig_allcrops("IRRIG_FROM_SURFACE_GRID_ANN", cases, outDir_figs, figsize=(16,10), noFigs=noFigs, plot_y1=plot_y1, plot_yN=plot_yN)


# %% Make maps of individual crops (rainfed, irrigated)
importlib.reload(cc)
importlib.reload(sys.modules['fig_maps_eachCrop'])
from fig_maps_eachCrop import maps_eachCrop
importlib.reload(sys.modules['cropcal_figs_module'])
from cropcal_figs_module import *

# Yield settings
min_viable_hui = "ggcmi3"
mxmat_limited = False

# Define reference case, if you want to plot differences
# ref_casename = None
ref_casename = 'CLM Default'
# ref_casename = 'rx'

overwrite = True
save_netcdfs = False

plot_y1 = 1980
plot_yN = 2010

if ref_casename=="rx" or len(caselist)==4:
    skip_SDATES = ['Prescribed Calendars']
else:
    skip_SDATES = None

varList = {
    'GDDHARV': {
        'suptitle':   'Mean harvest requirement',
        'time_dim':   'gs',
        'units':      'GDD'},
    'YIELD_ANN': {
        'suptitle':   'Mean annual yield',
        'time_dim':   'time',
        'units':      't/ha'},
    'PROD_ANN': {
        'suptitle':   'Mean annual production',
        'time_dim':   'time',
        'units':      'Mt'},
    'GSLEN': {
        'suptitle':   'Mean growing season length',
        'time_dim':   'gs',
        'units':      'days',
        'chunk_colorbar': True},
    'HDATES': {
        'suptitle':   'Mean harvest date',
        'time_dim':   'gs',
        'units':      'day of year',
        'chunk_colorbar': True},
    'HUI': {
        'suptitle':   'Mean HUI at harvest',
        'time_dim':   'gs',
        'units':      'GDD'},
    'HUIFRAC': {
        'suptitle':   'Mean HUI at harvest (fraction of required)',
        'time_dim':   'gs',
        'units':      'Fraction of required'},
    'SDATES': {
        'suptitle':   'Mean sowing date',
        'time_dim':   'gs',
        'units':      'day of year',
        'chunk_colorbar': True,
        'skip_cases': skip_SDATES},
    'MATURE': {
        'suptitle':   'Mature harvests',
        'time_dim':   'gs',
        'units':      'fraction'},
    'IRRIG_DEMAND_PATCH_ANN': {
        'suptitle':   'Mean irrigation water demand',
        'time_dim':   'time',
        'units':      'km$^3$'},
    'QIRRIG_DEMAND_PATCH_PKMTH': {
        'suptitle':   'Mean irrigation water demand: Peak month',
        'time_dim':   "time", # Only used to set label "Year"
        'units':      'month'},
}

nx = 2
dpi = 300
    
if mxmat_limited:
    mxmats_tmp = mxmats
else:
    mxmats_tmp = None

maps_eachCrop(cases, clm_types, clm_types_rfir, dpi, lu_ds, min_viable_hui, mxmats_tmp, nx, outDir_figs, overwrite, plot_y1, plot_yN, ref_casename, varList)
    
print('Done making maps.')


# %% Make gridcell-level maps
importlib.reload(sys.modules['fig_maps_grid'])
from fig_maps_grid import *
importlib.reload(cc)
importlib.reload(sys.modules['cropcal_figs_module'])
from cropcal_figs_module import *

nx = 1
custom_figname = None
varList = {
    'IRRIG_FROM_SURFACE_GRID_ANN': {
        'suptitle':   'Irrigation from surface water',
        'units':      'm3',
        'multiplier': 1},
    'QIRRIG_FROM_SURFACE_GRID_PKMTH': {
        'suptitle':   'Mean peak month of irrigation',
        'units':      'month',
        'multiplier': 1},
    'QIRRIG_FROM_SURFACE_GRID_PKMTH_DIFF': {
        'suptitle':   'Difference in mean peak month of irrigation',
        'units':      'months',
        'multiplier': 1},
    'IRRIG_WITHDRAWAL_FRAC_SUPPLY_VALPKMTHWITHDRAWAL_ANN': {
        'suptitle':   'Mean irrigation use as frac. supply in annual peak month',
        'units':      None,
        'multiplier': 1},
    'IRRIG_WITHDRAWAL_FRAC_SUPPLY_VALPKMTHWITHDRAWAL_ANN_DIFF': {
        'suptitle':   'Difference in mean irrigation use as frac. supply in annual peak month',
        'units':      None,
        'multiplier': 1},
}

# nx = 2
# custom_figname = "Irrigation pk month diffs"
# varList = {
#     'QIRRIG_FROM_SURFACE_GRID_PKMTH_DIFF': {
#         'suptitle':   'Difference in mean peak month of irrigation',
#         'units':      'Months',
#         'multiplier': 1,
#         'suppress_difftext': True},
#     'IRRIG_WITHDRAWAL_FRAC_SUPPLY_VALPKMTHWITHDRAWAL_ANN_DIFF': {
#         'suptitle':   'Difference in mean irrigation use as\nfrac. supply in annual peak month',
#         'units':      "Percentage points",
#         'multiplier': 100,
#         'suppress_difftext': True},
# }

plot_y1 = 1980
plot_yN = 2010

maps_gridlevel_vars(cases, varList, outDir_figs=outDir_figs, y1=plot_y1, yN=plot_yN, nx=nx, custom_figname=custom_figname)


# %% Make maps of harvest reasons
importlib.reload(cc)

plot_gs1 = 1980
plot_gsN = 2009

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

plot_Ngs = plot_gsN - plot_gs1 + 1

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
        thisCrop_gridded = utils.grid_one_variable(case['ds'].sel(gs=slice(plot_gs1, plot_gsN)), thisVar, \
            vegtype=vegtype_int).squeeze(drop=True)
        
        # Map each reason's frequency
        for f, reason in enumerate(reason_list):
            reason_text = reason_list_text[f]
            
            map_yx = cc.get_reason_freq_map(plot_Ngs, thisCrop_gridded, reason)
            ax = cc.make_axis(fig, ny, nx, i*nx + f+1)
            axes.append(ax)
            im0 = make_map(ax, map_yx, fontsize, cmap=cmap, bounds=bounds, extend_bounds=extend, linewidth=0.3)
                
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

    suptitle = f"Harvest reason: {vegtype_str_title} ({plot_gs1}-{plot_gsN} growing seasons)"
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
importlib.reload(cc)

# Yield settings
min_viable_hui = 1.0
mxmat_limited = True

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
    print(f"{thisCrop}...")
    
    thisCrop_clm = cc.cropnames_fao2clm(thisCrop)
    
    suptitle = f"{thisCrop}, FAOSTAT vs CLM ({top_y1}-{top_yN})"
    file_prefix = which_to_plot.replace('aly','')
    fig_outfile = outDir_figs + f"{file_prefix} scatter top 10 " + suptitle + ".pdf"
    if os.path.exists(fig_outfile) and not overwrite:
        print(f'    Skipping {thisCrop_out} (file exists).')
        continue
    
    # Get yield datasets
    print("    Analyzing...")
    if mxmat_limited:
        mxmats_tmp = mxmats
    else:
        mxmats_tmp = None
    topN_ds, topN_dt_ds, topN_anom_ds = cc.get_topN_ds(cases, reses, topYears, Ntop, thisCrop, countries_key, fao_all_ctry, earthstats, min_viable_hui, mxmats_tmp)
    Ntop_global = Ntop + 1
    
    print("    Plotting...")
    if which_to_plot == "Yield":
        plot_ds = topN_ds
    elif which_to_plot == "Detrended yield":
        plot_ds = topN_dt_ds
    elif which_to_plot == "Yield anomaly":
        plot_ds = topN_anom_ds
    else:
        raise RuntimeError(f"Which dataset should be used for '{which_to_plot}'?")

    f, axes = plt.subplots(ny, nx, figsize=figsize)
    axes = axes.flatten()

    # CLM Default will have hollow circles if Original baseline is included
    i_h = caselist.index('CLM Default')

    for c, country in enumerate(plot_ds.Country.values):
        
        # Text describing R-squared changes for each country
        stat_change_text = ""
        
        ax = axes[c]
        sc = xr.plot.scatter(plot_ds.sel(Country=country),
                             x='Yield (FAOSTAT)',
                             y='Yield',
                             hue='Case',
                             ax=ax)
        
        for case in caselist:
            lr = stats.linregress(x = plot_ds['Yield (FAOSTAT)'].sel(Country=country),
                                         y = plot_ds['Yield'].sel(Country=country, Case=case))
            if case == "CLM Default":
                t = "{r1:.3g} $\\rightarrow$ "
                stat_change_text += t.format(r1=lr.rvalue**2)
            elif case == "Prescribed Calendars":
                stat_change_text += "{r2:.3g}".format(r2=lr.rvalue**2)
                
        # Set title
        country_bf = ''
        for w in country.split(' '):
            country_bf += r'$\bf{' + w + r'}$' + ' '
        ax.set_title(country_bf + f'($R^2$ {stat_change_text})')
        
        # Set CLM Default to have hollow circles if Original baseline is included
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
 
print("Done.")
 
    
# %% Make line plots, FAOSTAT vs. CLM, of top 10 countries for each crop
importlib.reload(cc)
importlib.reload(sys.modules['cropcal_figs_module'])
from cropcal_figs_module import *

# Yield settings
min_viable_hui = "ggcmi3"
mxmat_limited = False

Ntop = 10

# The years with which to determine the "top" countries
# top_y1 = 1961 # First year of FAO data
top_y1 = 1992 # Pre-1992, you start getting USSR, which isn't in map
top_yN = 2009

# The years to show on the plot
plot_y1 = 1980
plot_yN = 2009

overwrite = True
portrait = False

which_to_plot = "Production"
# which_to_plot = "Yield"
# which_to_plot = "Yield, detrended"
# which_to_plot = "Yield, anomaly"

# ref_dataset = "FAOSTAT"
ref_dataset = "EarthStat"

if mxmat_limited:
    mxmats_tmp = mxmats
else:
    mxmats_tmp = None

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
plotYears = np.arange(plot_y1, plot_yN+1)

legend_members = [ref_dataset] + caselist

fao_crops = np.unique(fao_all_ctry.Crop.values)
for thisCrop in fao_crops:
    print(f"{thisCrop}...")
        
    thisCrop_clm = cc.cropnames_fao2clm(thisCrop)
    
    suptitle = f"{thisCrop}, {ref_dataset} vs CLM (top countries {top_y1}-{top_yN})"
    file_prefix = which_to_plot.replace('aly','')
    fig_outfile = outDir_figs + f"{file_prefix} timeseries {plot_y1}-{plot_yN} top 10 " + suptitle + ".pdf"
    if os.path.exists(fig_outfile) and not overwrite:
        print(f'    Skipping {thisCrop} (file exists).')
        continue
    
    # Get yield datasets
    print("    Analyzing...")
    topN_ds, topN_dt_ds, topN_ya_ds = cc.get_topN_ds(cases, reses, plotYears, topYears, Ntop, thisCrop, countries_key, fao_all_ctry, earthstats_gd, min_viable_hui, mxmats_tmp)
    Ntop_global = Ntop + 1
    
    print("    Plotting...")
    if "detrended" in which_to_plot:
        plot_ds = topN_dt_ds
    elif "anomaly" in which_to_plot:
        plot_ds = topN_anom_ds
    else:
        plot_ds = topN_ds
    thisVar = which_to_plot.split(',')[0]
    if thisVar == "Yield":
        units = "t/ha"
        multiplier = 1
    elif thisVar == "Production":
        units = "Mt"
        multiplier = 1e-6 # t to Mt
    else:
        raise RuntimeError(f"Unknown units for thisVar {thisVar}")
    plot_ds *= multiplier

    f, axes = plt.subplots(ny, nx, figsize=figsize)
    axes = axes.flatten()

    # CLM Default will have hollow circles if Original baseline is included
    i_h = caselist.index('CLM Default')

    for c, country in enumerate(plot_ds.Country.values):
        
        # Text describing statistic changes for each country
        stat_change_text = ""
                
        ax = axes[c]
        xr.plot.line(plot_ds[f'{thisVar} ({ref_dataset})'].sel(Country=country),
                     'k--', ax=ax)
        
        for ca, case in enumerate(caselist):
            
            color = cropcal_colors_cases(case)
            if color is None:
                color = cm.Dark2(ca)
            xr.plot.line(plot_ds[thisVar].sel(Country=country, Case=case),
                         color=color,
                         ax=ax)
        
            if c==0 and case==caselist[0]:
                print("    WARNING: No shifting allowed in calculation of bias!")
            xdata = plot_ds[f'{thisVar} ({ref_dataset})'].sel(Country=country).values
            ydata = plot_ds[thisVar].sel(Country=country, Case=case).values
            where_x_and_y_notnan = np.where(~(np.isnan(xdata) | np.isnan(ydata)))
            if np.any(where_x_and_y_notnan):
                xdata = xdata[where_x_and_y_notnan]
                ydata = ydata[where_x_and_y_notnan]
                bias = cc.get_timeseries_bias(ydata, xdata)
                if np.isnan(bias):
                    raise RuntimeError("Bias is NaN")
                if case == "CLM Default":
                    t = "{r1:.3g} $\\rightarrow$ "
                    stat_change_text += t.format(r1=bias)
                elif case == "Prescribed Calendars":
                    stat_change_text += "{r2:.3g}".format(r2=bias)
                
        # Set title
        thisTitle = ''
        for w in country.split(' '):
            thisTitle += r'$\bf{' + w + r'}$' + ' '
        if stat_change_text:
            thisTitle += f'({stat_change_text})'
        ax.set_title(thisTitle)
        
        # Set CLM Default to be dashed line if Original baseline is included
        if "Original baseline" in caselist:
            print('Set CLM Default to be dashed line if Original baseline is included')
            # color = sc[i_h].get_facecolor()
            # sc[i_h].set_facecolor('none')
            # sc[i_h].set_edgecolor(color)
        
        # ax.set_aspect('equal')
        if ax.get_legend() is not None:
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
        axes[a].set_ylabel(f"{which_to_plot} (CLM, {units})", fontsize=12, fontweight='bold')
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

print("Done.")


# %% Get top countries contributing to production difference

def get_country_data(ds, clm_crop, countries_thiscrop):
    da_thiscrop = (ds
                   .sel(time=slice("1980-01-01", "2009-01-01"))
                   .isel(patch=np.where(case_ds['patches1d_itype_combinedCropCLM_str']==clm_crop)[0])
                   .groupby(countries_thiscrop)
                   .sum(dim='patch')
                   .mean(dim='time')
                   )
    return da_thiscrop

topN = 10
gridline_alpha = 0.2
wspace = 0.8
hspace = 0.25

# Get FAOSTAT means for each country over 1992-2009
fao_all_ctry['Area'] = [x.replace("'", "") for x in fao_all_ctry['Area']]
fao80_meanProd_byCountry = cc.get_mean_byCountry(fao_all_ctry[fao_all_ctry['Element']=='Production'], 1980, 2009)
fao80_meanArea_byCountry = cc.get_mean_byCountry(fao_all_ctry[fao_all_ctry['Element']=='Area harvested'], 1980, 2009)
fao92_meanProd_byCountry = cc.get_mean_byCountry(fao_all_ctry[fao_all_ctry['Element']=='Production'], 1992, 2009)
fao92_meanArea_byCountry = cc.get_mean_byCountry(fao_all_ctry[fao_all_ctry['Element']=='Area harvested'], 1992, 2009)

fig_simProdDiff, axes_simProdDiff = plt.subplots(nrows=3, ncols=2, figsize=(10, 13))
fig_absBiasDiff, axes_absBiasDiff = plt.subplots(nrows=3, ncols=2, figsize=(10, 13))
fig_faoProd, axes_faoProd = plt.subplots(nrows=3, ncols=2, figsize=(10, 13))

# fao_crops = np.unique(fao_all_ctry.Crop.values)
fao_crops = ['Maize', 'Seed cotton', 'Rice, paddy', 'Soybeans', 'Sugar cane', 'Wheat']
for c, fao_crop in enumerate(fao_crops):
    
    # Get data info
    clm_crop = cc.cropnames_fao2clm(fao_crop).capitalize()
    print(f"{fao_crop} ({clm_crop})")
    countries_thiscrop = lu_ds['countries'].isel(patch=np.where(case_ds['patches1d_itype_combinedCropCLM_str']==clm_crop)[0])
    
    # Get plot info
    spx = c%2
    spy = int(np.floor(c/2))
    ax_simProdDiff = axes_simProdDiff[spy][spx]
    ax_absBiasDiff = axes_absBiasDiff[spy][spx]
    ax_faoProd = axes_faoProd[spy][spx]
    
    # Change in production
    prod0 = get_country_data(cases['CLM Default']['ds']['PROD_ANN'], clm_crop, countries_thiscrop)
    prod1 = get_country_data(cases['Prescribed Calendars']['ds']['PROD_ANN'], clm_crop, countries_thiscrop)
    prod0_pd = prod0.to_pandas()
    prod1_pd = prod1.to_pandas()
    da_thiscrop = prod1 - prod0
    pd_thiscrop = da_thiscrop.to_pandas()
    pd_thiscrop_topNneg = pd_thiscrop.nsmallest(topN)
    pd_thiscrop_topNpos = pd_thiscrop.nlargest(topN)
    countries_topNneg_int = [int(x) for x in pd_thiscrop_topNneg.axes[0]]
    countries_topNpos_int = [int(x) for x in pd_thiscrop_topNpos.axes[0]]
    countries_topNneg = [countries_key.name.values[countries_key.num == x][0] for x in countries_topNneg_int]
    countries_topNpos = [countries_key.name.values[countries_key.num == x][0] for x in countries_topNpos_int]
    topNcountries_int = np.flip(np.concatenate((countries_topNpos_int, np.flip(countries_topNneg_int))))
    topNcountries = np.flip(np.concatenate((countries_topNpos, np.flip(countries_topNneg))))
    ylabels = [x.replace("Democratic Republic of the", "D.R.") for x in topNcountries]
    # Convert units
    units = 'Mt'
    xdata = np.flip(1e-12*np.concatenate((pd_thiscrop_topNpos, np.flip(pd_thiscrop_topNneg))))
    # Make plot
    ax_simProdDiff.barh(np.arange(2*topN), xdata,
                        tick_label=ylabels)
    ax_simProdDiff.set_title(clm_crop, weight="bold")
    ax_simProdDiff.set_xlabel(units)
    fig_simProdDiff.suptitle("Production (1980-2009):\nPrescribed Calendars minus CLM Default",
                             y=0.94, weight="bold")
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    ax_simProdDiff.grid(True, axis='y', alpha=gridline_alpha)
    
    # Of those countries, how much is absolute yield bias changed?
    area0 = get_country_data(cases['CLM Default']['ds']['AREA_CFT'], clm_crop, countries_thiscrop)
    area1 = get_country_data(cases['Prescribed Calendars']['ds']['AREA_CFT'], clm_crop, countries_thiscrop)
    yield0 = prod0 / area0
    yield1 = prod1 / area1
    units = "t/ha"
    yield0 *= 1e-6 * 1e4
    yield1 *= 1e-6 * 1e4
    yield0_pd = pd.DataFrame(yield0.to_pandas())
    yield1_pd = pd.DataFrame(yield1.to_pandas())
    fao80_meanProd_byCountry_thisCrop = pd.DataFrame(fao80_meanProd_byCountry[fao_crop])
    fao80_meanArea_byCountry_thisCrop = pd.DataFrame(fao80_meanArea_byCountry[fao_crop])
    yield0_list = []
    yield1_list = []
    yieldFao_list = []
    for country in topNcountries:
        country_int = countries_key['num'][countries_key['name']==country].values[0]
        yield0_list.append(yield0_pd.query(f"countries == {country_int}").values[0][0])
        yield1_list.append(yield1_pd.query(f"countries == {country_int}").values[0][0])
        if country == "Venezuela":
            country += " (Bolivarian Republic of)"
        elif country == "Iran":
            country += " (Islamic Republic of)"
        elif country == "Taiwan":
            country = "China, Taiwan Province of"
        elif country == "Brunei":
            country += " Darussalam"
        elif country == "South Korea":
            country = "Republic of Korea"
        elif country == "North Korea":
            country = "Democratic Peoples Republic of Korea"
        elif country == "Republic of Congo":
            country = "Congo"
        elif country == "United Kingdom":
            country = "United Kingdom of Great Britain and Northern Ireland"
        prod = fao80_meanProd_byCountry_thisCrop.query(f"Area == '{country}'").values[0][0]
        area = fao80_meanArea_byCountry_thisCrop.query(f"Area == '{country}'").values[0][0]
        yieldFao_list.append(prod / area)
    bias0 = np.array(yield0_list) - np.array(yieldFao_list)
    bias1 = np.array(yield1_list) - np.array(yieldFao_list)
    absBiasDiff = np.abs(bias1) - np.abs(bias0)
    # Make plot
    ax_absBiasDiff.barh(np.arange(2*topN), absBiasDiff,
                        tick_label=ylabels)
    ax_absBiasDiff.set_title(clm_crop, weight="bold")
    ax_absBiasDiff.set_xlabel(units)
    fig_absBiasDiff.suptitle("Absolute yield bias (1980-2009):\nPrescribed Calendars minus CLM Default",
                             y=0.94, weight="bold")
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    ax_absBiasDiff.grid(True, axis='y', alpha=gridline_alpha)
    
    # The top FAOSTAT producing countries in 1992-2009
    fao92_meanProd_byCountry_thisCrop = pd.DataFrame(fao92_meanProd_byCountry[fao_crop])
    fao92_meanProd_byCountry_thisCrop_topN = fao92_meanProd_byCountry_thisCrop.nlargest(topN, columns="Value")
    fao92_meanProd_byCountry_thisCrop_other = fao92_meanProd_byCountry_thisCrop.nlargest(topN, columns="Value")
    # fao_cumul_contributions_topN = 100*fao92_meanProd_byCountry_thisCrop_other.cumsum() / fao92_meanProd_byCountry_thisCrop.sum()
    # Make plot
    units = "Mt"
    ylabels = [x[1] for x in fao92_meanProd_byCountry_thisCrop_topN.axes[0]]
    ylabels = np.flip(ylabels, axis=0)
    xdata = 1e-6*np.flip(np.squeeze(fao92_meanProd_byCountry_thisCrop_topN.values), axis=0)
    ax_faoProd.barh(np.arange(topN), xdata,
                    tick_label=ylabels)
    ax_faoProd.set_title(clm_crop, weight="bold")
    ax_faoProd.set_xlabel(units)
    fig_faoProd.suptitle("Production (1980-2009): FAOSTAT",
                         y=0.94, weight="bold")
    
fig_simProdDiff.subplots_adjust(wspace=wspace, hspace=hspace)
outfile = os.path.join(outDir_figs, f"Top {topN} countries, bars, ∆ Production.pdf")
fig_simProdDiff.savefig(outfile, dpi='figure', bbox_inches='tight')

fig_absBiasDiff.subplots_adjust(wspace=wspace, hspace=hspace)
outfile = os.path.join(outDir_figs, f"Top {topN} countries from ∆ production, bars, ∆ Yield.pdf")
fig_absBiasDiff.savefig(outfile, dpi='figure', bbox_inches='tight')

fig_faoProd.subplots_adjust(wspace=wspace, hspace=hspace)
outfile = os.path.join(outDir_figs, f"Top {topN} countries, bars, FAO production.pdf")
fig_faoProd.savefig(outfile, dpi='figure', bbox_inches='tight')


# %% Get country totals

# country = "World"
# country = "India"
country = 'China, mld.'

# thisCrop_str = None
thisCrop_str = "sugarcane"

stat_y1 = 1980
stat_yN = 2009

varList = {
    'IRRIG_DEMAND_PATCH_ANN': {
        'suptitle':   'Mean irrigation water demand',
        'time_dim':   'time',
        'units':      'km3/yr',
        'multiplier': 1e-9,
        'operation': 'sum'},
    'PROD_ANN': {
        'suptitle':   'Mean annual production',
        'time_dim':   'time',
        'units':      'Mt',
        'multiplier': 1e-12, # g to Mt
        'operation': 'sum'},
    'AREA_ANN': {
        'suptitle':   'Mean annual area',
        'time_dim':   'time',
        'units':      'Mha',
        'multiplier': 1e-6 * 1e-4, # m2 to Mha
        'operation': 'sum'},
    'YIELD_ANN': {
        'suptitle':   'Mean annual yield',
        'time_dim':   'time',
        'units':      't/ha',
        'multiplier': 1e-6 * 1e4, # g/m2 to tons/ha
        'operation': 'area-weighted mean'},
    'NHARVEST_DISCREP': {
        'suptitle':   'Mean N harvests that would be missed if only seeing max 1 per calendar year',
        'time_dim':   'time',
        'units':      'count',
        'multiplier': 1,
        'operation': 'sum'},
}

if country != "World":
    if country not in countries_key.name.values:
        raise RuntimeError(f"{country} not found in countries key")
    country_id = int(countries_key.num[countries_key.name==country])
print(f"{country}:")

for (this_var, var_info) in varList.items():
    print(f"{stat_y1}-{stat_yN} {var_info['suptitle']} ({var_info['units']}):")
    for i, (casename, case) in enumerate(cases.items()):
        if this_var == "AREA_ANN":
            case_da = reses[case['res']]['ds']['AREA_CFT']
        else:
            case_da = case['ds'][this_var]
        case_da = case_da.sel(time=slice(f"{stat_y1}-01-01", f"{stat_yN}-12-31"))
        
        # Get weights
        if "patch" not in case_da.dims:
            raise RuntimeError(f"Set up weights for dims {case_da.dims}")
        Nyears = stat_yN - stat_y1 + 1
        weights_time = xr.DataArray(data = np.full_like(case['ds']['patches1d_lon'], 1/Nyears),
                                    coords = case['ds']['patches1d_lon'].coords)
        if var_info['operation'] == "area-weighted mean":
            weight_numerators_time = reses[case['res']]['ds']['AREA_CFT'].sel(time=slice(f"{stat_y1}-01-01", f"{stat_yN}-12-31"))
            # print("You may not want to area-weight across years within each gridcell")
            # weights_time = (weight_numerators_time / weight_numerators_time.sum(dim="time")).values
            # weights_time[np.where(weight_numerators_time.sum(dim="time")==0)] = 0
            # weights_time = xr.DataArray(data = weights_time,
            #                             coords = weight_numerators_time.coords)
            weight_numerators = weight_numerators_time.mean(dim="time")
        else:
            weight_numerators = None
            
        # Get time-averaged mean
        case_da = (case_da * weights_time).sum(dim="time")
        
        # Mask not-included countries
        if country != "World":
            if "patch" not in case_da.dims:
                raise RuntimeError(f"No country DataArray compatible with dims {case_da.dims}")
            case_countries = reses[case['res']]['ds']['countries']
            case_da = case_da.where(case_countries==country_id)
            if var_info['operation'] == "area-weighted mean":
                weight_numerators = weight_numerators.where(case_countries==country_id)
        
        # Mask not-included crops
        if thisCrop_str is not None:
            if "patch" not in case_da.dims:
                raise RuntimeError(f"You can't distinguish crops in a DataArray with dims {case_da.dims}")
            is_thisCrop = [True if thisCrop_str.lower() in x.lower() else False for x in case['ds']['patches1d_itype_combinedCropCLM_str'].values]
            case_da = case_da.where(is_thisCrop)
            if var_info['operation'] == "area-weighted mean":
                weight_numerators = weight_numerators.where(is_thisCrop)
        
        # Perform the operation to get statistic
        if var_info['operation'] == "sum":
            value = case_da.sum().values
        elif var_info['operation'] == "area-weighted mean":
            weights = weight_numerators / weight_numerators.sum()
            value = (case_da * weights).sum().values
        else:
            raise RuntimeError(f"Unsure how to process operation {var_info['operation']}")
        
        value *= var_info['multiplier']
        print(f"   {casename}: {value}")
    
    
    

    
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
        cmap = cropcal_colors['div_other_norm']
    else:
        units = "Days"
        cmap = cropcal_colors['seq_other']
    cmap = plt.get_cmap(cmap)
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
        #	  vegtype_str_ggcmi = "wwh"
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
        plt.savefig(outfile, dpi=300, transparent=False, facecolor='white', \
                    bbox_inches='tight')
        plt.close()
print("Done.")


# %% Make maps for all 6 crops
importlib.reload(sys.modules['cropcal_figs_module'])
from cropcal_figs_module import *
importlib.reload(sys.modules['fig_maps_allCrops'])
from fig_maps_allCrops import *

these_cases = ['CLM Default', 'Prescribed Calendars']

crop_subset = None; ny = 3; nx = 2; figsize = (14,16); croptitle_side="top"

varList = {
    'PROD_ANN': {
        'suptitle':   'Mean annual production',
        'units':      'Mt'},
    'PROD_ANN_DIFF': {
        'suptitle':   'Mean annual production',
        'units':      'Mt',
        'maskcolorbar_near0': "percentile|5|cumulative"},
    'YIELD_ANN': {
        'suptitle':   'Mean annual yield',
        'units':      't/ha'},
    'YIELD_ANN_DIFF': {
        'suptitle':   'Mean annual yield',
        'units':      't/ha'},
    'YIELD_ANN_BIASEARTHSTAT': {
        'suptitle':   'Mean annual yield difference rel. EarthStat',
        'units':      't/ha'},
    'YIELD_ANN_ABSBIASEARTHSTAT_DIFF': {
        'suptitle':   'Mean annual yield difference rel. EarthStat',
        'units':      '∆ abs.\nbias (t/ha)'},
    'PROD_ANN_BIASEARTHSTAT': {
        'suptitle':   'Mean annual production bias rel. EarthStat',
        'units':      'Mt'},
    'PROD_ANN_ABSBIASEARTHSTAT_DIFF': {
        'suptitle':   'Mean annual production bias rel. EarthStat',
        'units':      '∆ abs.\nbias (Mt)',
        'maskcolorbar_near0': "percentile|5|cumulative"},
    'IRRIG_DEMAND_PATCH_ANN': {
        'suptitle':   'Mean annual irrigation',
        'units':      'km$^3$',},
    'IRRIG_DEMAND_PATCH_ANN_DIFF': {
        'suptitle':   'Mean annual irrigation difference',
        'units':      'km$^3$',
        'maskcolorbar_near0': "percentile|5|cumulative"},
    'IRRIG_DEMAND_PATCH_ANN_DIFF_ALLEXPSIMCROPS': {
        'suptitle':   'Mean annual irrigation difference',
        'units':      'km$^3$',
        'maskcolorbar_near0': "percentile|5|cumulative"},
    'IRRIG_DEMAND_PATCH_ANN_DIFFPOSNEG': {
        'suptitle':   'Components of total mean annual irrigation difference',
        'units':      'km$^3$',
        'maskcolorbar_near0': "percentile|5|cumulative",
        'suppress_difftext': True},
    'GSLEN': {
        'suptitle':   'Mean growing season length',
        'time_dim':   'gs',
        'units':      'days'},
    'GSLEN_DIFF': {
        'suptitle':   'Mean growing season length difference',
        'time_dim':   'gs',
        'units':      'days'},
    'GSLEN_BIAS': {
        'suptitle':   'Mean growing season length bias',
        'time_dim':   'gs',
        'units':      'days'},
}

# Yield settings
min_viable_hui = "ggcmi3"


for (this_var, var_info) in varList.items():
    cases = maps_allCrops(cases, these_cases, reses, this_var, var_info, outDir_figs, cropList_combined_clm_nototal, figsize, earthstats=earthstats, min_viable_hui=min_viable_hui, ny=ny, nx=nx, croptitle_side=croptitle_side, crop_subset=crop_subset)

print("Done.")


# %% Make maps for 3 crops (rows) and 2 variables (columns)
importlib.reload(cc)
importlib.reload(sys.modules['cropcal_figs_module'])
from cropcal_figs_module import *
importlib.reload(sys.modules['fig_maps_allCrops'])
from fig_maps_allCrops import *
importlib.reload(utils)

crop_subset = ['Cotton', 'Rice', 'Sugarcane']
# crop_subset = ['Corn', 'Soybean', 'Wheat']; 

these_cases = ['CLM Default', 'Prescribed Calendars']
ny = 3
nx = 1
figsize = (14,16)
croptitle_side="left"

varList = {
    'PROD_ANN_DIFF.PROD_ANN_ABSBIASEARTHSTAT_DIFF': {
        'suptitle':   ['Change in production', 'Change in abs. production bias'],
        'units':      ['Mt', 'Mt'],
        'multiplier': [1e-12, 1e-6], # g to Mt, t to Mt
        'maskcolorbar_near0': ["percentile|5|cumulative", "percentile|5|cumulative"],
        'suppress_difftext': [True, True]},
    'PROD_ANN_DIFF.YIELD_ANN_ABSBIASEARTHSTAT_DIFF': {
        'suptitle':   ['Change in production', 'Change in abs. yield bias'],
        'units':      ['Mt', 't/ha'],
        'multiplier': [1e-12, 1e-6 * 1e4], # g to Mt, g/m2 to tons/ha
        'maskcolorbar_near0': ["percentile|5|cumulative", False],
        'suppress_difftext': [True, True]},
    'YIELD_ANN_DIFF.YIELD_ANN_ABSBIASEARTHSTAT_DIFF': {
        'suptitle':   ['Change in yield', 'Change in abs. yield bias'],
        'units':      ['t/ha', 't/ha'],
        'multiplier': [1e-6 * 1e4, 1e-6 * 1e4],  # g/m2 to tons/ha
        'maskcolorbar_near0': [False, False],
        'suppress_difftext': [True, True]},
}

# Yield settings
min_viable_hui = "ggcmi3"


for (this_var, var_info) in varList.items():
    cases = maps_allCrops(cases, these_cases, reses, this_var, var_info, outDir_figs, cropList_combined_clm_nototal, figsize, earthstats=earthstats, min_viable_hui=min_viable_hui, ny=ny, nx=nx, croptitle_side=croptitle_side, crop_subset=crop_subset)

print("Done.")



# %% 

def color_algo(maxval_in):
    Ncolors = 9
    
    maxval = maxval_in
    factor = 1
    while maxval < Ncolors:
        factor *= 10
        maxval *= 10
    if maxval % Ncolors > 0:
        maxval = np.ceil(maxval / Ncolors)*Ncolors
    vmin = -maxval / factor
    vmax = maxval / factor
    binwidth = 2*vmax / Ncolors
    try:
        bounds = np.arange(vmin, vmax + maxval_in/100, binwidth)
    except:
        print(f"maxval_in: {maxval_in}")
        print(f"factor: {factor}")
        print(f"maxval: {maxval}")
        print(f"binwidth: {binwidth}")
        stop
    if bounds[-2] >= maxval_in:
        print(f"maxval_in: {maxval_in}")
        print(f"vmin: {vmin}")
        print(f"vmax: {vmax}")
        print(f"binwidth: {binwidth}")
        print(f"factor: {factor}")
        print(f"maxval: {maxval}")
        print(f"bounds: {bounds}")
        raise RuntimeError(f"maxval_in {maxval_in} not in top bin ({bounds[-2]},{bounds[-1]})")
 
 
for i,x in enumerate(np.arange(0.01,19,0.07)):
    if i>0 and i % 20 == 0:
        print(i)
    color_algo(x)
print("All good")
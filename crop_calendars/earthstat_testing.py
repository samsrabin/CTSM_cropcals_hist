# %% Setup

# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
# from cropcal_module import *

# # Import general CTSM Python utilities
# sys.path.append(my_ctsm_python_gallery)
# import utils

import numpy as np
import xarray as xr



# %% Import FAO Earthstat QUARTER DEGREE (gridded FAO data)

earthstats = {}

# Import
earthstats['qd'] = xr.open_dataset('/Users/Shared/CESM_work/CropEvalData_ssr/FAO-EarthStatYields/EARTHSTATMIRCAUNFAOCropDataDeg025.nc')
earthstats['f09_g17'] = xr.open_dataset('/Users/Shared/CESM_work/CropEvalData_ssr/FAO-EarthStatYields/EARTHSTATMIRCAUNFAOCropDataDeg09.nc')

# Include just crops we care about
cropList_fao_gd_all = ['Wheat', 'Maize', 'Rice', 'Barley', 'Rye', 'Millet', 'Sorghum', 'Soybeans', 'Sunflower', 'Potatoes', 'Cassava', 'Sugar cane', 'Sugar beet', 'Oil palm', 'Rape seed / Canola', 'Groundnuts / Peanuts', 'Pulses', 'Citrus', 'Date palm', 'Grapes / Vine', 'Cotton', 'Cocoa', 'Coffee', 'Others perennial', 'Fodder grasses', 'Others annual', 'Fibre crops', 'All crops']
cropList_fao_gd = ["Maize", "Rice", "Cotton", "Soybeans", "Sugar cane", "Wheat"]
earthstats['qd'] = earthstats['qd'].isel(crop=[cropList_fao_gd_all.index(x) for x in cropList_fao_gd]).assign_coords({'crop': cropList_fao_gd})
earthstats['f09_g17'] = earthstats['f09_g17'].isel(crop=[cropList_fao_gd_all.index(x) for x in cropList_fao_gd]).assign_coords({'crop': cropList_fao_gd})
   

# %% Test Peter's interpolation

for v in earthstats['f09_g17']:
   if "Area" in v or v=="Production":
      discrep_sum_rel = 100*(np.nansum(earthstats['f09_g17'][v].values) - np.nansum(earthstats['qd'][v].values)) / np.nansum(earthstats['qd'][v].values)
   else:
      discrep_sum_rel = 100*(np.nanmean(earthstats['f09_g17'][v].values) - np.nanmean(earthstats['qd'][v].values)) / np.nanmean(earthstats['qd'][v].values)
   print(f"Discrepancy in {v} f09_g17 rel to qd: {discrep_sum_rel}%")


# %% Test calculations

# These are not exact, but it seems like:
#    PhysicalArea = RainfedArea + IrrigatedArea (max diff ~51 ha, max rel diff ~0.01%)
#    PhysicalFraction = RainfedFraction + IrrigatedFraction (max rel diff ~0.01%)
#    Production = HarvestArea * Yield (max diff 4 tons, max rel diff ~0.000012%)
# But!!!
#    PhysicalArea ≠ Area * LandFraction * PhysicalFraction (max rel diff 100%)

def test_physical(earthstats, res, area_or_fraction):
   diff = (earthstats[res]["Rainfed"+area_or_fraction] + earthstats[res]["Irrigated"+area_or_fraction] - earthstats[res]["Physical"+area_or_fraction]).sum()
   return 100*(diff / earthstats[res]["Physical"+area_or_fraction].sum()).values

area_or_fraction = 'Area'
res = 'qd'
print(f"Physical{area_or_fraction}: {res} Rainfed{area_or_fraction} + Irrigated{area_or_fraction} off by {test_physical(earthstats, res, area_or_fraction)}%")
res = 'f09_g17'
print(f"Physical{area_or_fraction}: {res} Rainfed{area_or_fraction} + Irrigated{area_or_fraction} off by {test_physical(earthstats, res, area_or_fraction)}%")

area_or_fraction = 'Fraction'
res = 'qd'
print(f"Physical{area_or_fraction}: {res} Rainfed{area_or_fraction} + Irrigated{area_or_fraction} off by {test_physical(earthstats, res, area_or_fraction)}%")
res = 'f09_g17'
print(f"Physical{area_or_fraction}: {res} Rainfed{area_or_fraction} + Irrigated{area_or_fraction} off by {test_physical(earthstats, res, area_or_fraction)}%")

def test_physical2(earthstats, res):
   diff = (100*earthstats[res]["Area"]*earthstats[res]["LandFraction"]*earthstats[res]["PhysicalFraction"] - earthstats[res]["PhysicalArea"]).sum()
   return 100*(diff / earthstats[res]["PhysicalArea"].sum()).values
res = 'qd'
print(f"PhysicalArea: {res} Area*LandFraction*PhysicalFraction off by {test_physical2(earthstats, res)}%")
res = 'f09_g17'
print(f"PhysicalArea: {res} Area*LandFraction*PhysicalFraction off by {test_physical2(earthstats, res)}%")

# %%
def test_harvestarea(earthstats, res):
   diff = (earthstats[res]["PhysicalArea"]*earthstats[res]["HarvestFraction"] - earthstats[res]["HarvestArea"]).sum()
   return 100*(diff / earthstats[res]["HarvestArea"].sum()).values
res = 'qd'
print(f"HarvestArea: {res} PhysicalArea*HarvestFraction off by {test_physical2(earthstats, res)}%")
res = 'f09_g17'
print(f"HarvestArea: {res} PhysicalArea*HarvestFraction off by {test_physical2(earthstats, res)}%")
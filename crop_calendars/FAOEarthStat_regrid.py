# %% Options

# Years to include (set to None to leave that edge unbounded)
y1 = 2000
yN = 2009

# Directory where outputs are downloaded
indir = "/Users/Shared/CESM_runs/yield_perharv_f10_f10_mg37/lnd.291a1d5/hist/"

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"


# %% Imports

from re import I
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import shutil
import os
import time
import datetime as dt
import cftime
import warnings
import glob
import cartopy.crs as ccrs
from matplotlib import cm
import xesmf as xe

import sys
sys.path.append(my_ctsm_python_gallery)
import utils

import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")



# %% Import/process annual outputs

import importlib
importlib.reload(utils)

h2_file = indir + "yield_perharv_f10_f10_mg37.clm2.h2.2000-01-01-00000.nc"

clm = utils.import_ds(h2_file, \
    myVars=["GRAINC_TO_FOOD_ANN"], 
    myVegtypes=utils.define_mgdcrop_list())

# Align dates correctly
clm["time"] = clm["time"] - dt.timedelta(days=1)

clm = clm.sel(time=slice("2000-01-01","2009-12-31"))

# clm = utils.lon_pm2idl(clm)

# Get resolution
def get_res(lonorlat):
   res = np.mod(lonorlat.values[1:] - lonorlat.values[:-1], 360)
   res = np.unique(res)
   if len(res) > 1:
      raise RuntimeError(f"Expected 1 unique resolution, found {len(res)}: {res}")
   return res[0]
clm_xres = get_res(clm.lon)
clm_yres = get_res(clm.lat)
print(f"CLM resolution: {clm_xres} x {clm_yres}")

# # Chop & screw grid to center coordinates
# print("WARNING: CENTERING LONGITUDES ASSUMING FILE VALUES ARE LEFT EDGES")
# clm2 = clm.assign_coords(lon=clm.lon.values + clm_xres/2)
# print("WARNING: CENTERING LATITUDES ASSUMING FILE VALUES ARE LOWER EDGES AND TOSSING NORTHERNMOST ROW")
# clm2 = clm2.sel(lat=slice(-90,90-clm_yres))
# clm2 = clm2.assign_coords(lat=clm2.lat.values + clm_yres/2)


# % Make map

grainc_to_food = utils.grid_one_variable(clm, "GRAINC_TO_FOOD_ANN").sum(dim=["time", "ivt_str"], min_count=1)
print(grainc_to_food)

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
    ax.gridlines()
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
   #  plt.axis('off')
    
fig = plt.figure(figsize=(7.5,14))
ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
make_map(ax, grainc_to_food, "grainc_to_food", bin_width, fontsize_ticklabels, fontsize_titles)


# %% Import/process FAO EarthStat

fao = xr.open_dataset("/Users/Shared/CESM_work/CropEvalData_ssr/FAO-EarthStatYields/EARTHSTATMIRCAUNFAOCropDataDeg025.nc")

# Get resolution
fao_xres = get_res(fao.lon)
fao_yres = get_res(fao.lat)
print(f"FAO resolution: {fao_xres} x {fao_yres}")

# Regrid summable variables to CLM resolution
def get_scale(clm, fao):
   scale_out = clm/fao
   if scale_out != int(scale_out):
      raise RuntimeError(f"Can't scale by {scale_out} because it's not an integer")
   return int(scale_out)
fao2 = fao.drop(labels=["LandFraction", "RainfedFraction", "IrrigatedFraction", "HarvestFraction", "PhysicalFraction", "Yield", "CropIntensity", "LandMask"])
fao2 = fao2.coarsen(lon=get_scale(clm_xres,fao_xres), lat=get_scale(clm_yres,fao_yres)).sum()

# Check results
for v in fao2:
   total = np.sum(fao[v].values)
   total2 = np.sum(fao2[v].values)
   diff = total2 - total
   print(f"Difference in total {v}: {diff} ({diff/total*100}%)")
   
   
# %% Try direct regidding of FAO EarthStat to weird CLM grid


regridder = xe.Regridder(fao, clm, "bilinear")
fao3 = regridder(fao)

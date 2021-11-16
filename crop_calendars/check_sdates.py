# %% Setup

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Directory where input file(s) can be found
# indir = "/Volumes/Reacher/CESM_runs/numa_20211014_rx/normal/"
# generate_gdds = False
# indir = "/Volumes/Reacher/CESM_runs/numa_20211014_rx/"
indir = "/Volumes/Reacher/CESM_runs/f10_f10_mg37/"
# indir = "/Volumes/Reacher/CESM_runs/f10_f10_mg37/2021-11-10/"

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import glob
import cftime
import cartopy.crs as ccrs
from matplotlib import cm

import os

import sys
sys.path.append(my_ctsm_python_gallery)
import utils

import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")


# %% Import realized sowing and harvest dates

# Either the name of a file within $indir, or a pattern that will return a list of files.
pattern = "*h2.*-01-01-00000.nc"

# Get list of all files in $indir matching $pattern
filelist = glob.glob(indir + pattern)

# Import
this_ds = utils.import_ds(filelist, myVars=["SDATES", "HDATES"], myVegtypes=utils.define_mgdcrop_list())
sdates_gridded = utils.grid_one_variable(\
    this_ds, 
    "SDATES", 
    time=1)


# %% Import expected sowing dates


sdate_inFile = "/Volumes/Reacher/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01.2000-2000.nc"

# Get run info:
# Max number of growing seasons per year
if "mxgrowseas" in this_ds:
    mxgrowseas = this_ds.dims["mxgrowseas"]
else:
    mxgrowseas = 1
    
# Which vegetation types were simulated?
itype_veg_toImport = np.unique(this_ds.patches1d_itype_veg)

sdate_varList = []
for i in itype_veg_toImport:
    for g in np.arange(mxgrowseas):
        thisVar = f"sdate{g+1}_{i}"
        sdate_varList = sdate_varList + [thisVar]

sdates_rx = utils.import_ds(sdate_inFile, myVars=sdate_varList)

# # Convert to match CESM output longitude (0-360, centered on prime meridian)
# sdates_rx = utils.lon_idl2pm(sdates_rx)


# # %%
# sdates_rx.sdate1_17.interp( \
#     lat=this_ds.patches1d_lat.values[0], 
#     lon=this_ds.patches1d_lon.values[0], 
#     method="nearest")

# %% Compare simulated to prescribed sdates

# import importlib
# importlib.reload(utils)

outdir = f"{indir}/sdates"
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
def make_map(ax, this_map, this_title): 
    new_cmap = cm.get_cmap('hsv', 365)
    new_cmap.set_under((0.5,0.5,0.5,1.0))    
    new_cmap.set_over("k")       
    im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
            this_map, cmap=new_cmap, shading="auto",
            vmin=1, vmax=365)
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(this_title)
    plt.colorbar(im1, orientation="horizontal", pad=0.0)

ny = 2
nx = 1
for i, vt_str in enumerate(this_ds.vegtype_str.values):
    
    print(vt_str)
    fig = plt.figure(figsize=(8,10))
        
    # Input
    vt = this_ds.ivt.values[i]
    thisVar = f"sdate1_{vt}"
    if thisVar in sdates_rx:
        ax = fig.add_subplot(ny,nx,1,projection=ccrs.PlateCarree())
        make_map(ax, sdates_rx[thisVar].squeeze(drop=True), f"Input {vt_str}")
    
    # Output
    ax = fig.add_subplot(ny,nx,2,projection=ccrs.PlateCarree())
    out_map = sdates_gridded.sel(ivt_str=vt_str).squeeze(drop=True)
    make_map(ax, out_map, f"Output {vt_str}")
        
    # Prepend filename with "_" if output map empty
    if not np.any(np.bitwise_not(np.isnan(out_map))): 
        vt_str = "_" + vt_str
        
    print("Saving...")
    outfile = f"{outdir}/sdates_{vt_str}.png"
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
        bbox_inches='tight')
    plt.close()
        
print("Done!")



# %% Map harvest and sowing dates

# import importlib
# importlib.reload(utils)

hdates00_gridded = utils.grid_one_variable(\
    this_ds, 
    "HDATES", 
    time=1)

sdates00_gridded = utils.grid_one_variable(\
    this_ds, 
    "SDATES", 
    time=1)

hdates01_gridded = utils.grid_one_variable(\
    this_ds, 
    "HDATES", 
    time=2)

sdates01_gridded = utils.grid_one_variable(\
    this_ds, 
    "SDATES", 
    time=2)

outdir = f"{indir}/sowing_and_harvest"
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
def make_map(ax, this_map, this_title): 
    new_cmap = cm.get_cmap('hsv', 365)
    new_cmap.set_under((0.5,0.5,0.5,1.0))    
    new_cmap.set_over("k")       
    im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
            this_map, cmap=new_cmap, shading="auto",
            vmin=1, vmax=365)
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(this_title)
    plt.colorbar(im1, orientation="horizontal", pad=0.0)

ny = 2
nx = 2
for i, vt_str in enumerate(this_ds.vegtype_str.values):
    
    print(vt_str)
    fig = plt.figure(figsize=(12,8))
            
    # Output
    ax = fig.add_subplot(ny,nx,1,projection=ccrs.PlateCarree())
    out_map = hdates00_gridded.sel(ivt_str=vt_str).squeeze(drop=True)
    make_map(ax, out_map, f"Sim harv {vt_str} '00")
    ax = fig.add_subplot(ny,nx,2,projection=ccrs.PlateCarree())
    out_map = sdates00_gridded.sel(ivt_str=vt_str).squeeze(drop=True)
    make_map(ax, out_map, f"Sim sow {vt_str} '00")
    ax = fig.add_subplot(ny,nx,3,projection=ccrs.PlateCarree())
    out_map = hdates01_gridded.sel(ivt_str=vt_str).squeeze(drop=True)
    make_map(ax, out_map, f"Sim harv {vt_str} '01")
    ax = fig.add_subplot(ny,nx,4,projection=ccrs.PlateCarree())
    out_map = sdates01_gridded.sel(ivt_str=vt_str).squeeze(drop=True)
    make_map(ax, out_map, f"Sim sow {vt_str} '01")
                    
    # Prepend filename with "_" if output map empty
    if not np.any(np.bitwise_not(np.isnan(out_map))): 
        vt_str = "_" + vt_str
        
    print("Saving...")
    outfile = f"{outdir}/hshsdates_{vt_str}.png"
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
        bbox_inches='tight')
    plt.close()
        
print("Done!")

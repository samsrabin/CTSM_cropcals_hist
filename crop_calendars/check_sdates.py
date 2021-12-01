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

def make_map(ax, this_map, this_title, use_new_cmap=True, vmin=1, vmax=365): 
    if use_new_cmap:
        new_cmap = cm.get_cmap('hsv', 365)
        new_cmap.set_under((0.5,0.5,0.5,1.0))    
        new_cmap.set_over("k")
        im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
                this_map, cmap=new_cmap, shading="auto",
                vmin=vmin, vmax=vmax)
    else:
        im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
                this_map, shading="auto",
                vmin=vmin, vmax=vmax)
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(this_title)
    plt.colorbar(im1, orientation="horizontal", pad=0.0)
 

# %% Import sowing and harvest dates
dates_ds = utils.import_ds(glob.glob(indir + "*h2.*-01-01-00000.nc"), \
    myVars=["SDATES", "HDATES"], 
    myVegtypes=utils.define_mgdcrop_list())


# %% Check that sowing and harvest dates were the same in 2000 and 2001

print(dates_ds.SDATES.isel(time=2).values[dates_ds.SDATES.isel(time=1).values
                                          != dates_ds.SDATES.isel(time=2).values])

print(dates_ds.HDATES.isel(time=2).values[dates_ds.HDATES.isel(time=1).values
                                          != dates_ds.HDATES.isel(time=2).values])


# %% Import expected sowing dates

sdate_inFile = "/Volumes/Reacher/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp.2000-2000.nc"

# Get run info:
# Max number of growing seasons per year
if "mxgrowseas" in dates_ds:
    mxgrowseas = dates_ds.dims["mxgrowseas"]
else:
    mxgrowseas = 1
    
# Which vegetation types were simulated?
itype_veg_toImport = np.unique(dates_ds.patches1d_itype_veg)

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

# %% Make maps of simulated vs. prescribed sdates

# import importlib
# importlib.reload(utils)

sdates_grid = utils.grid_one_variable(\
    dates_ds, 
    "SDATES", 
    time__values="2001")

outdir = f"{indir}/sdates"
if not os.path.exists(outdir):
    os.makedirs(outdir)

ny = 2
nx = 1
for i, vt_str in enumerate(dates_ds.vegtype_str.values):
    
    print(vt_str)
    fig = plt.figure(figsize=(8,10))
        
    # Input
    vt = dates_ds.ivt.values[i]
    thisVar = f"sdate1_{vt}"
    if thisVar in sdates_rx:
        ax = fig.add_subplot(ny,nx,1,projection=ccrs.PlateCarree())
        make_map(ax, sdates_rx[thisVar].squeeze(drop=True), f"Input {vt_str}")
    
    # Output
    ax = fig.add_subplot(ny,nx,2,projection=ccrs.PlateCarree())
    out_map = sdates_grid.sel(ivt_str=vt_str).squeeze(drop=True)
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


# %% Compare simulated sdates to those from Python's NN selection

year = 2001

outdir = f"{indir}/sdates_check"
if not os.path.exists(outdir):
    os.makedirs(outdir)

sdates_grid = utils.grid_one_variable(\
    dates_ds, 
    "SDATES", 
    time__values=f"{year}")
sdates_grid = utils.lon_pm2idl(sdates_grid)

rx_interp = sdates_rx.interp(lon=sdates_grid.lon,
                             lat=sdates_grid.lat, 
                             method="nearest",
                             kwargs={"fill_value": "extrapolate"})

ny = 2
nx = 2
for i, vt_str in enumerate(dates_ds.vegtype_str.values):
    
    # Input
    vt = dates_ds.ivt.values[i]
    thisVar = f"sdate1_{vt}"
    if not thisVar in sdates_rx:
        continue
    print(vt_str)
    fig = plt.figure(figsize=(15,9))
    rx_thiscrop = sdates_rx[thisVar]
    ax = fig.add_subplot(ny,nx,2,projection=ccrs.PlateCarree())
    make_map(ax, rx_thiscrop.squeeze(drop=True), f"Input {vt_str}")
    # Extrapolating here means that any CTSM coordinates outside the limits of the prescribed 
    # dates' axes will just use the value at the corresponding edge of the corresponding prescribed 
    # dates' axis. So for example, grid resolution f10_f10_mg37 has a cell centered at lon -180° 
    # lat Y°, but the prescribed dates' longitude axis has its first cell centered at -179.75°. The 
    # extrapolated value will be taken from that first column (i.e., -179.75°).
    rx_thiscrop_interp = rx_interp[thisVar].squeeze(drop=True)
    ax = fig.add_subplot(ny,nx,1,projection=ccrs.PlateCarree())
    make_map(ax, rx_thiscrop_interp, f"Input {vt_str} (CTSM res.)")
    
    # Output
    ax = fig.add_subplot(ny,nx,3,projection=ccrs.PlateCarree())
    out_thiscrop = sdates_grid.sel(ivt_str=vt_str).squeeze(drop=True)
    make_map(ax, out_thiscrop, f"Output {vt_str} ({year})")
    
    # Difference
    ax = fig.add_subplot(ny,nx,4,projection=ccrs.PlateCarree())
    diff_map = out_thiscrop - rx_thiscrop_interp
    diff_map.values[diff_map.values==0] = np.nan
    new_caxis_lim = max(abs(np.nanmin(diff_map)), abs(np.nanmax(diff_map)))      
    make_map(ax, diff_map, "Difference (out – in)", 
             use_new_cmap=False,
             vmin=-new_caxis_lim, vmax=new_caxis_lim)

    print("Saving...")
    outfile = f"{outdir}/sdatescheck_{vt_str}.png"
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
        bbox_inches='tight')
    plt.close()
        

# %% Map harvest and sowing dates

# import importlib
# importlib.reload(utils)

hdates00_gridded = utils.grid_one_variable(\
    dates_ds, 
    "HDATES", 
    time__indices=1,
    mxharvests__indices=0)

sdates00_gridded = utils.grid_one_variable(\
    dates_ds, 
    "SDATES", 
    time__indices=1)

hdates01_gridded = utils.grid_one_variable(\
    dates_ds, 
    "HDATES", 
    time__indices=2,
    mxharvests__indices=0)

sdates01_gridded = utils.grid_one_variable(\
    dates_ds, 
    "SDATES", 
    time__indices=2)

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
for i, vt_str in enumerate(dates_ds.vegtype_str.values):
    
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


# %% Inspecting one gridcell's daily outputs

thisVar = "CPHASE"

daily_ds = utils.import_ds(glob.glob(indir + "*h1.*-01-01-00000.nc"), \
    myVegtypes=utils.define_mgdcrop_list())

dates1_ds = dates_ds.isel(time=1, mxgrowseas=0).squeeze(drop=True)

tmp = (dates1_ds.SDATES.values > 50) & (dates1_ds.SDATES.values < 75) & (dates1_ds.patches1d_itype_veg_str.values=="temperate_corn")
if tmp.sum() != 1:
    raise RuntimeError(f"Expected 1 match of condition; found {tmp.sum()}")
thisPatch = dates_ds.patch.values[np.where(tmp)][0]

# Get dates in a format that matplotlib can use
with warnings.catch_warnings():
    # Ignore this warning in this with-block
    warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar, 'noleap', to a pandas.DatetimeIndex, which uses dates from the standard calendar.  This may lead to subtle errors in operations that depend on the length of time between dates.")
    datetime_vals = daily_ds.indexes["time"].to_datetimeindex()

this_ds = daily_ds.sel(patch=thisPatch)

thisvar_da = daily_ds[thisVar].sel(patch=thisPatch)
thisLon = daily_ds.lon.values[int(daily_ds.patches1d_ixy.sel(patch=thisPatch).values.item())]
thisLat = daily_ds.lat.values[int(daily_ds.patches1d_jxy.sel(patch=thisPatch).values.item())]

fig = plt.figure(figsize=(12,8))
plt.plot(datetime_vals, thisvar_da.values)
plt.title(f"{thisVar} ({thisLon} lon, {thisLat} lat)")
plt.ylabel(daily_ds.variables[thisVar].attrs['units'])
plt.show()




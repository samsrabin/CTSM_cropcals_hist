# %% Setup

# Years of interest
y1 = 1980
yN = 2009

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Directory where input file(s) can be found
indir = "/Users/Shared/CESM_runs/f10_f10_mg37_1850/"

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

def import_rx_dates(s_or_h, date_inFile, dates_ds):
    # Get run info:
    # Max number of growing seasons per year
    if "mxgrowseas" in dates_ds:
        mxgrowseas = dates_ds.dims["mxgrowseas"]
    else:
        mxgrowseas = 1
        
    # Which vegetation types were simulated?
    itype_veg_toImport = np.unique(dates_ds.patches1d_itype_veg)

    date_varList = []
    for i in itype_veg_toImport:
        for g in np.arange(mxgrowseas):
            thisVar = f"{s_or_h}date{g+1}_{i}"
            date_varList = date_varList + [thisVar]

    ds = utils.import_ds(date_inFile, myVars=date_varList)
    
    for v in ds:
        ds = ds.rename({v: v.replace(f"{s_or_h}date","gs")})
    
    return ds

def thisCrop_map_to_patches(lon_points, lat_points, map_ds, vegtype_int):
    # xarray pointwise indexing; see https://xarray.pydata.org/en/stable/user-guide/indexing.html#more-advanced-indexing
    return map_ds[f"gs1_{vegtype_int}"].sel( \
        lon=xr.DataArray(lon_points, dims="patch"),
        lat=xr.DataArray(lat_points, dims="patch")).squeeze(drop=True)
    


# %% Import output sowing and harvest dates

# Keep 1 extra year to avoid incomplete final growing season for crops harvested after Dec. 31.
y1_import_str = f"{y1+1}-01-01"
yN_import_str = f"{yN+2}-01-01"

print(f"Importing netCDF time steps {y1_import_str} through {yN_import_str}")

dates_ds = utils.import_ds(glob.glob(indir + "*h2.*"), \
    myVars=["SDATES", "HDATES"], 
    myVegtypes=utils.define_mgdcrop_list())

dates_ds = utils.xr_flexsel(dates_ds, \
    time__values=slice(y1_import_str,
                       yN_import_str))


# %% Check that simulated sowing and harvest dates do not vary between years

for v in ["SDATES", "HDATES"]:
    for t in np.arange(dates_ds.dims["time"]):
        if t==0:
            continue
        if not np.all((dates_ds[v].isel(time=0) == dates_ds[v].isel(time=t)).values):
            raise RuntimeError(f"{v} timestep {t} does not match timestep 0")
print("✅ CLM output sowing and harvest dates do not vary between years.")


# %% Import expected sowing dates

sdate_inFile = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"

sdates_rx = import_rx_dates("s", sdate_inFile, dates_ds)


# %% Check that input and output sdates match

sdates_grid = utils.grid_one_variable(\
    dates_ds, 
    "SDATES")

all_ok = True
for i, vt_str in enumerate(dates_ds.vegtype_str.values):
    
    # Input
    vt = dates_ds.ivt.values[i]
    thisVar = f"sdate1_{vt}"
    if thisVar not in sdates_rx:
        continue
    in_map = sdates_rx[thisVar].squeeze(drop=True)
    
    # Output
    out_map = sdates_grid.sel(ivt_str=vt_str).squeeze(drop=True)
    
    # Check for differences
    diff_map = out_map - in_map
    if np.any(diff_map.values[np.invert(np.isnan(diff_map.values))]):
        print(f"Difference(s) found in {vt_str}")
        all_ok = False
        
if all_ok:
    print("✅ Input and output sdates match!")


# %% Import prescribed harvest dates

hdate_inFile = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"

hdates_rx = import_rx_dates("h", hdate_inFile, dates_ds)

# Determine cells where growing season crosses new year
grows_across_newyear = hdates_rx < sdates_rx


# %% Import accumulated GDDs

# Keep 1 extra year to avoid incomplete final growing season for crops harvested after Dec. 31.
y1_import_str = f"{y1}-01-01"
yN_import_str = f"{yN+1}-12-31"

print(f"Importing netCDF time steps {y1_import_str} through {yN_import_str}")

accumGDD_ds = utils.import_ds(glob.glob(indir + "*h1.*"), \
    myVars=["GDDPLANT"], 
    myVegtypes=utils.define_mgdcrop_list(),
    timeSlice=slice(y1_import_str, yN_import_str))

if not np.any(accumGDD_ds.GDDPLANT.values != 0):
    raise RuntimeError("All GDDPLANT values are zero!")


# %% Get mean GDDs in GGCMI growing season

import cftime

# Get day of year for each day in time axis
# doy = [t.timetuple().tm_yday for t in accumGDD_ds.time.values]
doy = np.array([t.timetuple().tm_yday for t in accumGDD_ds.time.values])

# Get standard datetime axis for outputs
t1 = accumGDD_ds.time.values[0]
Nyears = yN - y1 + 1
new_dt_axis = np.array([cftime.datetime(y, 1, 1, 
                               calendar=t1.calendar,
                               has_year_zero=t1.has_year_zero)
               for y in np.arange(y1, yN+1)])
time_indsP1 = np.arange(Nyears + 1)

# Set up output Dataset
gdds_ds = accumGDD_ds.isel(time=np.arange(Nyears))
gdds_ds = gdds_ds.assign_coords(time=new_dt_axis)
del gdds_ds["GDDPLANT"]

incl_vegtype_indices = []
for v, vegtype_str in enumerate(accumGDD_ds.vegtype_str.values):
    vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
    newVar = f"gdd1_{vegtype_int}"
    
    # Get time series for each patch of this type
    thisCrop_ds = utils.xr_flexsel(accumGDD_ds, vegtype=vegtype_str)
    thisCrop_da = thisCrop_ds.GDDPLANT
    if not thisCrop_da.size:
        continue
    print(f"{vegtype_str}...")
    incl_vegtype_indices = incl_vegtype_indices + [v]
    
    # Get prescribed harvest dates for these patches
    lon_points = thisCrop_ds.patches1d_lon.values
    lat_points = thisCrop_ds.patches1d_lat.values
    thisCrop_hdates_rx = thisCrop_map_to_patches(lon_points, lat_points, hdates_rx, vegtype_int)
    # Get "grows across new year?" for these patches
    thisCrop_gany = thisCrop_map_to_patches(lon_points, lat_points, grows_across_newyear, vegtype_int)
    
    # Get the accumulated GDDs at each prescribed harvest date
    # There's almost certainly a more efficient way to do this than looping through patches!
    for p in np.arange(thisCrop_hdates_rx.size):
        thisPatch_da = thisCrop_da.isel(patch=p)
        thisCell_gdds_da = thisPatch_da.isel(time=np.where(doy==thisCrop_hdates_rx.sel(patch=p).values)[0])
        if thisCrop_gany[p]:
            thisCell_gdds_da = thisCell_gdds_da.isel(time=time_indsP1[1:])
        else:
            thisCell_gdds_da = thisCell_gdds_da.isel(time=time_indsP1[:-1])
        
        # Set to standard datetime axis for outputs
        thisCell_gdds_da = thisCell_gdds_da.assign_coords(time=new_dt_axis)
        
        # Add to new DataArray
        if p==0:
            gdds_da = thisCell_gdds_da
            gdds_da = gdds_da.rename(newVar)
        else:
            gdds_da = xr.concat([gdds_da, thisCell_gdds_da], dim="patch")
        
    # Change attributes of gdds_da
    gdds_da = gdds_da.assign_attrs({"long_name": f"GDD harvest target for {vegtype_str}"})
    del gdds_da.attrs["cell_methods"]
    
    # Add to gdds_ds
    gdds_ds[newVar] = gdds_da
    
# Fill NAs with dummy values
dummy_fill = -1
gdds_ds = gdds_ds.fillna(dummy_fill)

# Remove unused vegetation types
gdds_ds = gdds_ds.isel(ivt=incl_vegtype_indices)

# Take mean
gdds_mean_ds = gdds_ds.mean(dim="time", keep_attrs=True)


# %% Grid

# Save map figures to files?
save_figs = False

def make_map(ax, this_map, this_title): 
    im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
            this_map, shading="auto",
            vmin=0)
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(this_title)
    plt.colorbar(im1, orientation="horizontal", pad=0.0)
if save_figs:
    outdir = indir + "maps/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

for v, vegtype_str in enumerate(gdds_mean_ds.vegtype_str.values):
    vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
    thisVar = f"gdd1_{vegtype_int}"
    
    # Grid
    thisCrop_gridded = utils.grid_one_variable(gdds_mean_ds, thisVar, fillValue=-1, vegtype=vegtype_int)
    thisCrop_gridded = thisCrop_gridded.squeeze(drop=True)
    
    # Add to Dataset
    if v==0:
        gdd_maps_ds = thisCrop_gridded.to_dataset()
    gdd_maps_ds[thisVar] = thisCrop_gridded
    gdd_maps_ds[thisVar] = thisCrop_gridded
    
    # Make figure    
    if save_figs:
        ax = plt.axes(projection=ccrs.PlateCarree())
        make_map(ax, thisCrop_gridded, vegtype_str)
        outfile = f"{outdir}/{thisVar}_{vegtype_str}.png"
        plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
            bbox_inches='tight')
        plt.close()

# %% Setup

# Years of interest (do not include extra year needed for finishing last growing season)
y1 = 1980
yN = 2009 # 2009

# Save map figures to files?
save_figs = True

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Directory where input file(s) can be found (figure files will be saved in subdir here)
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_1850/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-29/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/tmp/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-30/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601.02.72441c4e"
indir = "/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601.03.ba902039"

# Directory to save output netCDF
outdir = "/Users/Shared/CESM_work/crop_dates/"
if save_figs:
    outdir_figs = indir + "figs/"
    if not os.path.exists(outdir_figs):
        os.makedirs(outdir_figs)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import glob
import cftime
import cartopy.crs as ccrs
from matplotlib import cm
import datetime as dt

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
    if "mxsowings" in dates_ds:
        mxsowings = dates_ds.dims["mxsowings"]
    else:
        mxsowings = 1
        
    # Which vegetation types were simulated?
    itype_veg_toImport = np.unique(dates_ds.patches1d_itype_veg)

    date_varList = []
    for i in itype_veg_toImport:
        for g in np.arange(mxsowings):
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

# Get file list
if indir[-1] != os.path.sep:
    indir = indir + os.path.sep
h2_pattern = indir + "*h2.*"
h2_filelist = glob.glob(h2_pattern)
if not h2_filelist:
    raise RuntimeError(f"No files found matching pattern: {h2_pattern}")

dates_ds = utils.import_ds(h2_filelist, \
    myVars=["SDATES", "HDATES"], 
    myVegtypes=utils.define_mgdcrop_list())

dates_ds = utils.xr_flexsel(dates_ds, \
    time__values=slice(y1_import_str,
                       yN_import_str))

patchList = dates_ds.patch.values


# %%
# Check that, during period of interest, simulated harvest always happens the day before sowing
# Could vectorize this, but it gets complicated because some cells are sown Jan. 1 and some aren't.
verbose = True

ok_p = np.full((dates_ds.dims["patch"]), True)

for p, thisPatch in enumerate(patchList):
        
    thisLon = dates_ds.patches1d_lon.values[p]
    thisLat = dates_ds.patches1d_lat.values[p]
    # thisLon = np.round(dates_ds.patches1d_lon.values[p], decimals=2)
    # thisLat = np.round(dates_ds.patches1d_lat.values[p], decimals=2)
    thisCrop = dates_ds.patches1d_itype_veg_str.values[p]
    thisIVT = dates_ds.patches1d_itype_veg.values[p]
    thisStr = f"Patch {thisPatch} (lon {thisLon} lat {thisLat}) {thisCrop} ({thisIVT})"
    sim_sp = dates_ds["SDATES"].sel(patch=thisPatch).values
    sim_hp = dates_ds["HDATES"].sel(patch=thisPatch).values
    
    # There should be no missing sowings
    if any(sim_sp < 1):
        ok_p[p] = False
        if verbose:
            if np.all(sim_sp < 1):
                print(f"{thisStr}: Sowing never happened")
            else:
                print(f"{thisStr}: Sowing didn't happen some year(s); first {y1+int(np.argwhere(sim_sp < 1)[0])}")
        continue

    # Should only need to consider one sowing and one harvest
    if sim_sp.shape[1] > 1:
        ok_p[p] = False
        if verbose: print(f"{thisStr}: Expected mxsowings 1 but found {sim_sp.shape[1]}")
        continue
    sim_sp = sim_sp[:,0]
    if np.any(sim_hp[:,1:] > 0):
        ok_p[p] = False
        if verbose: print(f"{thisStr}: More than 1 harvest found in some year(s)")
        continue
    sim_hp = sim_hp[:,0]

    # Align
    if sim_sp[0] > 1:
        sim_hp = sim_hp[1:]
    else:
        sim_hp = sim_hp[0:-1]
    sim_sp = sim_sp[0:-1]

    # We're going to be comparing each harvest to the sowing that FOLLOWS it.
    sim_sp = sim_sp[1:]
    sim_hp = sim_hp[0:-1]
        
    # There should no longer be any missing harvests
    if any(sim_hp < 1):
        ok_p[p] = False
        if verbose:
            if np.all(sim_hp < 1):
                print(f"{thisStr}: Harvest never happened")
            else:
                print(f"{thisStr}: Harvest didn't happen some growing season(s); first {y1+int(np.argwhere(sim_hp < 1)[0])}")
        continue

    # Harvest should always happen the day before the next sowing.
    exp_hp = ((sim_sp - 2)%365) + 1
    if not np.array_equal(sim_hp, exp_hp):
        ok_p[p] = False
        if verbose: print(f"{thisStr}: Not every harvest happens the day before next sowing")
        continue
    
if np.all(ok_p):
    print("✅ CLM output sowing and harvest dates look good.")
else:
    print(f"❌ {sum(np.bitwise_not(ok_p))} patch(es) had problem(s) with CLM output sowing and/or harvest dates.")


# %% Import expected sowing dates. This will be used as our template output file.

sdate_inFile = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"

sdates_rx = import_rx_dates("s", sdate_inFile, dates_ds)


# %% Check that input and output sdates match

sdates_grid = utils.grid_one_variable(\
    dates_ds, 
    "SDATES")

all_ok = True
any_found = False
vegtypes_skipped = []
vegtypes_included = []
print("Checking for matching input and output sdates:")
for i, vt_str in enumerate(dates_ds.vegtype_str.values):
    
    # Input
    vt = dates_ds.ivt.values[i]
    thisVar = f"gs1_{vt}"
    if thisVar not in sdates_rx:
        vegtypes_skipped = vegtypes_skipped + [vt_str]
        # print(f"    {vt_str} ({vt}) SKIPPED...")
        continue
    vegtypes_included = vegtypes_included + [vt_str]
    any_found = True
    print(f"    {vt_str} ({vt})...")
    in_map = sdates_rx[thisVar].squeeze(drop=True)
    
    # Output
    out_map = sdates_grid.sel(ivt_str=vt_str).squeeze(drop=True)
    
    # Check for differences
    diff_map = out_map - in_map
    if np.any(diff_map.values[np.invert(np.isnan(diff_map.values))]):
        print(f"Difference(s) found in {vt_str}")
        all_ok = False

if not (any_found):
    raise RuntimeError("No matching variables found in sdates_rx!")

# Sanity checks for included vegetation types
vegtypes_skipped = np.unique([x.replace("irrigated_","") for x in vegtypes_skipped])
vegtypes_skipped_weird = [x for x in vegtypes_skipped if x in vegtypes_included]
if np.array_equal(vegtypes_included, [x.replace("irrigated_","") for x in vegtypes_included]):
    print("\nWARNING: No irrigated crops included!!!\n")
elif vegtypes_skipped_weird:
    print(f"\nWarning: Some crop types had output rainfed patches but no irrigated patches: {vegtypes_skipped_weird}")
       
if all_ok:
    print("✅ Input and output sdates match!")


# %% Import prescribed harvest dates

# hdate_inFile = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"
hdate_inFile = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.20220602_230029.nc"

hdates_rx = import_rx_dates("h", hdate_inFile, dates_ds)

# Determine cells where growing season crosses new year
grows_across_newyear = hdates_rx < sdates_rx


# %% Import accumulated GDDs

clm_gdd_var = "GDDACCUM"

# Keep 1 extra year to avoid incomplete final growing season for crops harvested after Dec. 31.
y1_import_str = f"{y1}-01-01"
yN_import_str = f"{yN+1}-12-31"

print(f"Importing netCDF time steps {y1_import_str} through {yN_import_str}")

myVars = [clm_gdd_var]
if save_figs:
    myVars = myVars.append("GDDHARV")

h1_ds = utils.import_ds(glob.glob(indir + "*h1.*"), \
    myVars=myVars, 
    myVegtypes=utils.define_mgdcrop_list(),
    timeSlice=slice(y1_import_str, yN_import_str))

if not np.any(h1_ds[clm_gdd_var].values != 0):
    raise RuntimeError(f"All {clm_gdd_var} values are zero!")


# %% Get mean GDDs in GGCMI growing season

import cftime

# Get day of year for each day in time axis
# doy = [t.timetuple().tm_yday for t in accumGDD_ds.time.values]
doy = np.array([t.timetuple().tm_yday for t in h1_ds.time.values])

# Get standard datetime axis for outputs
t1 = h1_ds.time.values[0]
Nyears = yN - y1 + 1
new_dt_axis = np.array([cftime.datetime(y, 1, 1, 
                               calendar=t1.calendar,
                               has_year_zero=t1.has_year_zero)
               for y in np.arange(y1, yN+1)])
time_indsP1 = np.arange(Nyears + 1)

# Set up output Dataset(s)
def setup_output_ds(in_ds, thisVar, Nyears, new_dt_axis):
    out_ds = in_ds.isel(time=np.arange(Nyears))
    out_ds = out_ds.assign_coords(time=new_dt_axis)
    del out_ds[thisVar]
    return out_ds
gddaccum_ds = setup_output_ds(h1_ds, clm_gdd_var, Nyears, new_dt_axis)
longname_prefix = "GDD harvest target for "
if save_figs:
    gddharv_ds = setup_output_ds(h1_ds, "GDDHARV", Nyears, new_dt_axis)
    
def get_values_at_harvest(thisCrop_hdates_rx, in_da, time_indsP1, new_dt_axis, newVar):
    # There's almost certainly a more efficient way to do this than looping through patches!
    for p in np.arange(thisCrop_hdates_rx.size):
        thisPatch_da = in_da.isel(patch=p)
        
        # Extract time range of interest plus extra year for cells where growing season crosses the new year
        thisCell_gdds_da = thisPatch_da.isel(time=np.where(doy==thisCrop_hdates_rx.sel(patch=p).values)[0])
        
        # Extract the actual time range of interest for this cell, depending on whether its growing season crosses the new year
        if thisCrop_gany[p]:
            thisCell_gdds_da = thisCell_gdds_da.isel(time=time_indsP1[1:])
        else:
            thisCell_gdds_da = thisCell_gdds_da.isel(time=time_indsP1[:-1])
        
        # Set to standard datetime axis for outputs
        thisCell_gdds_da = thisCell_gdds_da.assign_coords(time=new_dt_axis)
        
        # Add to new DataArray
        if p==0:
            out_da = thisCell_gdds_da
            out_da = out_da.rename(newVar)
        else:
            out_da = xr.concat([out_da, thisCell_gdds_da], dim="patch")
        
    return out_da

incl_vegtype_indices = []
for v, vegtype_str in enumerate(h1_ds.vegtype_str.values):
    vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
    newVar = f"gdd1_{vegtype_int}"
    
    # Get time series for each patch of this type
    thisCrop_ds = utils.xr_flexsel(h1_ds, vegtype=vegtype_str)
    thisCrop_da = thisCrop_ds[clm_gdd_var]
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
    gdds_da = get_values_at_harvest(thisCrop_hdates_rx, thisCrop_da, time_indsP1, new_dt_axis, newVar)
    
    # Import previous GDD requirements for harvest
    if save_figs:
        gddharv_da = get_values_at_harvest(thisCrop_hdates_rx, thisCrop_ds["GDDHARV"], time_indsP1, new_dt_axis, newVar)
        
    # Change attributes of gdds_da
    gdds_da = gdds_da.assign_attrs({"long_name": f"{longname_prefix}{vegtype_str}"})
    del gdds_da.attrs["cell_methods"]
    
    # Add to gdds_ds
    warnings.filterwarnings("ignore", message="Increasing number of chunks by factor of 30")
    with warnings.catch_warnings():
        gddaccum_ds[newVar] = gdds_da
        if save_figs:
            gddharv_ds[newVar] = gddharv_da
    
# Fill NAs with dummy values
dummy_fill = -1
gdds_fill0_ds = gddaccum_ds.fillna(0)
gddaccum_ds = gddaccum_ds.fillna(dummy_fill)

# Remove unused vegetation types
gddaccum_ds = gddaccum_ds.isel(ivt=incl_vegtype_indices)
gdds_fill0_ds = gdds_fill0_ds.isel(ivt=incl_vegtype_indices)

# Take mean
gdds_mean_ds = gddaccum_ds.mean(dim="time", keep_attrs=True)
gdds_fill0_mean_ds = gdds_fill0_ds.mean(dim="time", keep_attrs=True)
if save_figs:
    gddharv_mean_ds = gddharv_ds.mean(dim="time", keep_attrs=True)
    
print("Done getting means")


# %% Grid

# Fill value
fillValue = -1

for v, vegtype_str in enumerate(gdds_mean_ds.vegtype_str.values):
    vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
    thisVar = f"gdd1_{vegtype_int}"
    print(f"Gridding {vegtype_str} ({vegtype_int})...")
    
    # Grid
    thisCrop_gridded = utils.grid_one_variable(gdds_mean_ds, thisVar, \
        fillValue=fillValue, vegtype=vegtype_int).squeeze(drop=True)
    thisCrop_fill0_gridded = utils.grid_one_variable(gdds_fill0_mean_ds, thisVar, \
        vegtype=vegtype_int).squeeze(drop=True)
    thisCrop_fill0_gridded = thisCrop_fill0_gridded.fillna(0)
    thisCrop_fill0_gridded.attrs["_FillValue"] = fillValue
    if save_figs:
        gddharv_gridded = utils.grid_one_variable(gddharv_mean_ds, thisVar, \
            fillValue=fillValue, vegtype=vegtype_int).squeeze(drop=True)
    
    # Add singleton time dimension
    thisCrop_gridded = thisCrop_gridded.expand_dims(time = sdates_rx.time)
    thisCrop_fill0_gridded = thisCrop_fill0_gridded.expand_dims(time = sdates_rx.time)
    
    # Add to Dataset
    if v==0:
        gdd_maps_ds = thisCrop_gridded.to_dataset()
        gdd_fill0_maps_ds = thisCrop_fill0_gridded.to_dataset()
        if save_figs:
            gddharv_maps_ds = gddharv_gridded.to_dataset()
    gdd_maps_ds[thisVar] = thisCrop_gridded
    gdd_fill0_maps_ds[thisVar] = thisCrop_fill0_gridded
    if save_figs:
        gddharv_maps_ds[thisVar] = gddharv_gridded
    
# Add dummy variables for crops not actually simulated
# Unnecessary?
template_ds = xr.open_dataset(sdate_inFile, decode_times=True)
all_vars = [v.replace("sdate","gdd") for v in template_ds if "sdate" in v]
all_longnames = [template_ds[v].attrs["long_name"].replace("Planting day ", longname_prefix) + " (dummy)" for v in template_ds if "sdate" in v]
dummy_vars = [v for v in all_vars if v not in gdd_maps_ds]
def make_dummy(thisCrop_gridded, addend):
    dummy_gridded = thisCrop_gridded
    dummy_gridded.values = dummy_gridded.values*0 + addend
    return dummy_gridded
dummy_gridded = make_dummy(thisCrop_gridded, -1)
dummy_gridded0 = make_dummy(thisCrop_fill0_gridded, 0)

for v, thisVar in enumerate(dummy_vars):
    dummy_gridded.name = thisVar
    dummy_gridded.attrs["long_name"] = all_longnames[v]
    gdd_maps_ds[thisVar] = dummy_gridded
    dummy_gridded0.name = thisVar
    dummy_gridded0.attrs["long_name"] = all_longnames[v]
    gdd_fill0_maps_ds[thisVar] = dummy_gridded0

# Add lon/lat attributes
def add_lonlat_attrs(ds):
    ds.lon.attrs = {\
        "long_name": "coordinate_longitude",
        "units": "degrees_east"}
    ds.lat.attrs = {\
        "long_name": "coordinate_latitude",
        "units": "degrees_north"}
    return ds
gdd_maps_ds = add_lonlat_attrs(gdd_maps_ds)
gdd_fill0_maps_ds = add_lonlat_attrs(gdd_fill0_maps_ds)
gddharv_maps_ds = add_lonlat_attrs(gddharv_maps_ds)

print("Done.")


# %% 
# Save before/after map and boxplot figures, if doing so

# layout = "3x1"
layout = "2x2"
bin_width = 15
lat_bin_edges = np.arange(0, 91, bin_width)

fontsize_titles = 8
fontsize_axislabels = 8
fontsize_ticklabels = 7

def make_map(ax, this_map, this_title, vmax, bin_width, fontsize_ticklabels, fontsize_titles): 
    im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
            this_map, shading="auto",
            vmin=0, vmax=vmax)
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(this_title, fontsize=fontsize_titles)
    cbar = plt.colorbar(im1, orientation="horizontal", fraction=0.1, pad=0.02)
    cbar.ax.tick_params(labelsize=fontsize_ticklabels)
    
    ticks = np.arange(-90, 91, bin_width)
    ticklabels = [str(x) for x in ticks]
    for i,x in enumerate(ticks):
        if x%2:
            ticklabels[i] = ''
    plt.yticks(np.arange(-90,91,15), labels=ticklabels,
               fontsize=fontsize_ticklabels)
    
def get_non_nans(in_da, fillValue):
    in_da = in_da.where(in_da != fillValue)
    return in_da.values[~np.isnan(in_da.values)]

def set_boxplot_props(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], markeredgecolor=color, markersize=3)

Nbins = len(lat_bin_edges)-1
bin_names = ["All"]
for b in np.arange(Nbins):
    lower = lat_bin_edges[b]
    upper = lat_bin_edges[b+1]
    bin_names.append(f"{lower}–{upper}")
    
color_old = '#beaed4'
color_new = '#7fc97f'
def make_plot(data, offset):
    linewidth = 1.5
    offset = 0.4*offset
    bpl = plt.boxplot(data, positions=np.array(range(len(data)))*2.0+offset, widths=0.6, 
                      boxprops=dict(linewidth=linewidth), whiskerprops=dict(linewidth=linewidth), 
                      capprops=dict(linewidth=linewidth), medianprops=dict(linewidth=linewidth),
                      flierprops=dict(markeredgewidth=0.5))
    return bpl

if save_figs:
    
    # Maps
    ny = 3
    nx = 1
    print("Making before/after maps...")
    for v, vegtype_str in enumerate(gdds_mean_ds.vegtype_str.values):
        vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
        thisVar = f"gdd1_{vegtype_int}"
        print(f"   {vegtype_str} ({vegtype_int})...")
        
        
        # Maps #####################
        
        gdd_map = gdd_maps_ds[thisVar].isel(time=0, drop=True)
        gdd_map_yx = gdd_map.where(gdd_map != fillValue)
        gddharv_map = gddharv_maps_ds[thisVar]
        gddharv_map_yx = gddharv_map.where(gddharv_map != fillValue)
                
        vmax = max(np.max(gdd_map_yx), np.max(gddharv_map_yx))
        
        # Set up figure and first subplot
        if layout == "3x1":
            fig = plt.figure(figsize=(7.5,14))
            ax = fig.add_subplot(ny,nx,1,projection=ccrs.PlateCarree())
        elif layout == "2x2":
            fig = plt.figure(figsize=(10,5))
            spec = fig.add_gridspec(nrows=2, ncols=2,
                                    width_ratios=[0.4,0.6])
            ax = fig.add_subplot(spec[0,0],projection=ccrs.PlateCarree())
        else:
            raise RuntimeError(f"layout {layout} not recognized")
        
        thisMin = int(np.round(np.min(gddharv_map_yx)))
        thisMax = int(np.round(np.max(gddharv_map_yx)))
        thisTitle = f"{vegtype_str}: Old (range {thisMin}–{thisMax})"
        make_map(ax, gddharv_map_yx, thisTitle, vmax, bin_width,
                 fontsize_ticklabels, fontsize_titles)
        
        if layout == "3x1":
            ax = fig.add_subplot(ny,nx,2,projection=ccrs.PlateCarree())
        elif layout == "2x2":
            ax = fig.add_subplot(spec[1,0],projection=ccrs.PlateCarree())
        else:
            raise RuntimeError(f"layout {layout} not recognized")
        thisMin = int(np.round(np.min(gdd_map_yx)))
        thisMax = int(np.round(np.max(gdd_map_yx)))
        thisTitle = f"{vegtype_str}: New (range {thisMin}–{thisMax})"
        make_map(ax, gdd_map_yx, thisTitle, vmax, bin_width,
                 fontsize_ticklabels, fontsize_titles)
        
        # Boxplots #####################
        
        gdd_vector = get_non_nans(gdd_map, fillValue)
        gddharv_vector = get_non_nans(gddharv_map, fillValue)
        
        lat_abs = np.abs(gdd_map.lat.values)
        gdd_bybin_old = [gddharv_vector]
        gdd_bybin_new = [gdd_vector]
        for b in np.arange(Nbins):
            lower = lat_bin_edges[b]
            upper = lat_bin_edges[b+1]
            lat_inds = np.where((lat_abs>=lower) & (lat_abs<upper))[0]
            # gdd_map_thisBin = gdd_map.where(gdd_map.lat>=lower )
            gdd_vector_thisBin = get_non_nans(gdd_map[lat_inds,:], fillValue)
            gddharv_vector_thisBin = get_non_nans(gddharv_map[lat_inds,:], fillValue)
            gdd_bybin_old.append(gddharv_vector_thisBin)
            gdd_bybin_new.append(gdd_vector_thisBin)
                
        if layout == "3x1":
            ax = fig.add_subplot(ny,nx,3)
        elif layout == "2x2":
            ax = fig.add_subplot(spec[:,1])
        else:
            raise RuntimeError(f"layout {layout} not recognized")

        bpl = make_plot(gdd_bybin_old, -1)
        bpr = make_plot(gdd_bybin_new, 1)
        set_boxplot_props(bpl, color_old)
        set_boxplot_props(bpr, color_new)
        
        # draw temporary lines to create a legend
        plt.plot([], c=color_old, label='Old')
        plt.plot([], c=color_new, label='New')
        plt.legend(fontsize=fontsize_titles)
        
        plt.xticks(range(0, len(bin_names) * 2, 2), bin_names,
                   fontsize=fontsize_ticklabels)
        plt.yticks(fontsize=fontsize_ticklabels)
        plt.xlabel("|latitude| zone", fontsize=fontsize_axislabels)
        plt.ylabel("Growing degree-days", fontsize=fontsize_axislabels)
        plt.title(f"Zonal changes: {vegtype_str}", fontsize=fontsize_titles)

        outfile = f"{outdir_figs}/{thisVar}_{vegtype_str}_gs{y1}-{yN}.png"
        plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
            bbox_inches='tight')
        plt.close()

    print("Done.")
    

# %% Check that all cells that had sdates are represented

all_ok = True
for vt_str in vegtypes_included:
    vt_int = utils.ivt_str2int(vt_str)
    # print(f"{vt_int}: {vt_str}")
    
    map_gdd = gdd_maps_ds[f"gdd1_{vt_int}"].isel(time=0, drop=True)
    map_sdate = sdates_grid.isel(time=0, mxsowings=0, drop=True).sel(ivt_str=vt_str, drop=True)
    
    ok_gdd = map_gdd.where(map_gdd >= 0).notnull()
    ok_sdate = map_sdate.where(map_sdate > 0).notnull()
    missing_both = np.bitwise_and(np.bitwise_not(ok_gdd), np.bitwise_not(ok_sdate))
    ok_both = np.bitwise_and(ok_gdd, ok_sdate)
    ok_both = np.bitwise_or(ok_both, missing_both)
    if np.any(np.bitwise_not(ok_both)):
        all_ok = False
        gdd_butnot_sdate = np.bitwise_and(ok_gdd, np.bitwise_not(ok_sdate))
        sdate_butnot_gdd = np.bitwise_and(ok_sdate, np.bitwise_not(ok_gdd))
        if np.any(gdd_butnot_sdate):
            print(f"{vt_int} {vt_str}: {np.sum(gdd_butnot_sdate)} cells in GDD but not sdate")
        if np.any(sdate_butnot_gdd):
            print(f"{vt_int} {vt_str}: {np.sum(sdate_butnot_gdd)} cells in sdate but not GDD")

if not all_ok:
    print("❌ Mismatch between sdate and GDD outputs")
else:
    print("✅ All sdates have GDD and vice versa")


# %% Save to netCDF

# Get output file path
if not os.path.exists(outdir):
    os.makedirs(outdir)
datestr = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = outdir + "gdds_" + datestr + ".nc"
outfile_fill0 = outdir + "gdds_fill0_" + datestr + ".nc"

def save_gdds(sdate_inFile, hdate_inFile, outfile, gdd_maps_ds, sdates_rx):
    # Set up output file from template (i.e., prescribed sowing dates).
    template_ds = xr.open_dataset(sdate_inFile, decode_times=True)
    for v in template_ds:
        if "sdate" in v:
            template_ds = template_ds.drop(v)
    template_ds.to_netcdf(path=outfile, format="NETCDF3_CLASSIC")
    template_ds.close()

    # Add global attributes
    comment = f"Derived from CLM run plus crop calendar input files {os.path.basename(sdate_inFile) and {os.path.basename(hdate_inFile)}}."
    gdd_maps_ds.attrs = {\
        "author": "Sam Rabin (sam.rabin@gmail.com)",
        "comment": comment,
        "created": dt.datetime.now().astimezone().isoformat()
        }

    # Add time_bounds
    gdd_maps_ds["time_bounds"] = sdates_rx.time_bounds

    # Save cultivar GDDs
    gdd_maps_ds.to_netcdf(outfile, mode="a", format="NETCDF3_CLASSIC")

save_gdds(sdate_inFile, hdate_inFile, outfile, gdd_maps_ds, sdates_rx)
save_gdds(sdate_inFile, hdate_inFile, outfile_fill0, gdd_fill0_maps_ds, sdates_rx)


# %% Misc.

# # %% View GDD for a single crop and cell

# tmp = accumGDD_ds[clm_gdd_var].values
# incl = np.bitwise_and(accumGDD_ds.patches1d_lon.values==270, accumGDD_ds.patches1d_lat.values==40)
# tmp2 = tmp[:,incl]
# plt.plot(tmp2)



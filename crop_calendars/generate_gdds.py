# %% Setup

# Years of interest (do not include extra year needed for finishing last growing season)
y1 = 1980
yN = 2009 # 2009
# y1 = 1951
# yN = 1952 # 2009

# Save map figures to files?
save_figs = True

# Where is the script running?
import socket
hostname = socket.gethostname()

# Import the CTSM Python utilities
import sys
if hostname == "Sams-2021-MacBook-Pro.local":
    sys.path.append("/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/")
    import utils
else:
    # Only possible because I have export PYTHONPATH=$HOME in my .bash_profile
    from ctsm_python_gallery_myfork.ctsm_py import utils

# Import the generate_gdds functions
import os
if hostname == "Sams-2021-MacBook-Pro.local":
    sys.path.append("/Users/sam/Documents/git_repos/CTSM_cropcals_hist/crop_calendars")
    import generate_gdds_functions as gddfn
else:
    # Only possible because I have export PYTHONPATH=$HOME in my .bash_profile
    from CTSM_cropcals_hist.crop_calendars import generate_gdds_functions as gddfn


# Directory where input file(s) can be found (figure files will be saved in subdir here)
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_1850/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-29/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/tmp/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37/2022-03-30/"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601.02.72441c4e"
# indir = "/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601.03.ba902039"
# indir = "/glade/scratch/samrabin/archive/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1950-2013.ggcmi2/lnd/hist"
# indir = "/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1950-2013.ggcmi2"
indir = "/glade/scratch/samrabin/archive/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1976-2013_gddgen/lnd/hist"
# indir = "/Users/Shared/CESM_runs/cropcals_2deg/cropcals.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.1976-2013_gddgen"

# sdate_inFile = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"
# hdate_inFile = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.20220602_230029.nc"
sdate_inFile = "/glade/u/home/samrabin/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"
hdate_inFile = "/glade/u/home/samrabin/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"
# sdate_inFile = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"
# hdate_inFile = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"


# Directory to save output netCDF
import os
if hostname == "Sams-2021-MacBook-Pro.local":
    outdir = "/Users/Shared/CESM_work/crop_dates/"
else:
    outdir = "/glade/u/home/samrabin/crop_dates/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
if save_figs:
    outdir_figs = indir + "figs/"
    if not os.path.exists(outdir_figs):
        os.makedirs(outdir_figs)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import cartopy.crs as ccrs
import datetime as dt
import pickle

import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")






# %% Import and process

# Keep 1 extra year to avoid incomplete final growing season for crops harvested after Dec. 31.
y1_import_str = f"{y1+1}-01-01"
yN_import_str = f"{yN+2}-01-01"

print(f"Importing netCDF time steps {y1_import_str} through {yN_import_str} (years are +1 because of CTSM output naming)")

import importlib
importlib.reload(gddfn)

pickle_file = indir + f'/{y1}-{yN}.pickle'
h1_ds_file = indir + f'/{y1}-{yN}.h1_ds.nc'
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        y1, yN, pickle_year, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices, incorrectly_daily, gddharv_in_h3, save_figs, incl_vegtypes_str = pickle.load(f)
    print(f'Will resume import at {pickle_year+1}')
    h1_ds = None
else:
    incorrectly_daily = False
    skip_patches_for_isel_nan_lastyear = np.ndarray([])
    gddharv_in_h3 = False
    pickle_year = -np.inf
    gddaccum_yp_list = []
    gddharv_yp_list = []
    incl_vegtypes_str = None
    lastYear_active_patch_indices = None
sdates_rx = sdate_inFile
hdates_rx = hdate_inFile

for y, thisYear in enumerate(np.arange(y1+1,yN+3)):
    
    if thisYear <= pickle_year:
        continue
    
    h1_ds, sdates_rx, hdates_rx, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices, incorrectly_daily, gddharv_in_h3, incl_vegtypes_str = gddfn.import_and_process_1yr(y1, yN, y, thisYear, sdates_rx, hdates_rx, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices, incorrectly_daily, gddharv_in_h3, save_figs, indir, incl_vegtypes_str, h1_ds_file)
     
    print(f'   Saving pickle file ({pickle_file})...')
    with open(pickle_file, 'wb') as f:
        pickle.dump([y1, yN, thisYear, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices, incorrectly_daily, gddharv_in_h3, save_figs, incl_vegtypes_str], f, protocol=-1)
    
incl_vegtypes_str = incl_vegtypes_str[[i for i,c in enumerate(gddaccum_yp_list) if not isinstance(c,type(None))]]

print("Done")

if not h1_ds:
    h1_ds = xr.open_dataset(h1_ds_file)


# %% Get and grid mean GDDs in GGCMI growing season

longname_prefix = "GDD harvest target for "

print('Getting and gridding mean GDDs...')
gdd_maps_ds = gddfn.yp_list_to_ds(gddaccum_yp_list, h1_ds, incl_vegtypes_str, sdates_rx, longname_prefix)
if save_figs:gddharv_maps_ds = gddfn.yp_list_to_ds(gddharv_yp_list, h1_ds, incl_vegtypes_str, sdates_rx, longname_prefix)

# Fill NAs with dummy values
dummy_fill = -1
gdd_fill0_maps_ds = gdd_maps_ds.fillna(0)
gdd_maps_ds = gdd_maps_ds.fillna(dummy_fill)
print('Done getting and gridding means.')

# Add dummy variables for crops not actually simulated
print("Adding dummy variables...")
# Unnecessary?
template_ds = xr.open_dataset(sdate_inFile, decode_times=True)
all_vars = [v.replace("sdate","gdd") for v in template_ds if "sdate" in v]
all_longnames = [template_ds[v].attrs["long_name"].replace("Planting day ", longname_prefix) + " (dummy)" for v in template_ds if "sdate" in v]
dummy_vars = []
dummy_longnames = []
for v, thisVar in enumerate(all_vars):
    if thisVar not in gdd_maps_ds:
        dummy_vars.append(thisVar)
        dummy_longnames.append(all_longnames[v])

def make_dummy(thisCrop_gridded, addend):
    dummy_gridded = thisCrop_gridded
    dummy_gridded.values = dummy_gridded.values*0 + addend
    return dummy_gridded
for v in gdd_maps_ds:
    thisCrop_gridded = gdd_maps_ds[v].copy()
    thisCrop_fill0_gridded = gdd_fill0_maps_ds[v].copy()
    break
dummy_gridded = make_dummy(thisCrop_gridded, -1)
dummy_gridded0 = make_dummy(thisCrop_fill0_gridded, 0)

for v, thisVar in enumerate(dummy_vars):
    if thisVar in gdd_maps_ds:
        raise RuntimeError(f'{thisVar} is already in gdd_maps_ds. Why overwrite it with dummy?')
    dummy_gridded.name = thisVar
    dummy_gridded.attrs["long_name"] = dummy_longnames[v]
    gdd_maps_ds[thisVar] = dummy_gridded
    dummy_gridded0.name = thisVar
    dummy_gridded0.attrs["long_name"] = dummy_longnames[v]
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
if save_figs: gddharv_maps_ds = add_lonlat_attrs(gddharv_maps_ds)

print("Done.")


# %% Save to netCDF
print("Saving...")

# Get output file path
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
    gdd_maps_ds.to_netcdf(outfile, mode="w", format="NETCDF3_CLASSIC")

save_gdds(sdate_inFile, hdate_inFile, outfile, gdd_maps_ds, sdates_rx)
save_gdds(sdate_inFile, hdate_inFile, outfile_fill0, gdd_fill0_maps_ds, sdates_rx)

print("Done saving.")


# %% Save things needed for mapmaking

def add_attrs_to_map_ds(map_ds, incl_vegtypes_str, dummy_fill, outdir_figs, y1, yN):
    return map_ds.assign_attrs({'incl_vegtypes_str': incl_vegtypes_str,
                                'dummy_fill': dummy_fill,
                                'outdir_figs': outdir_figs,
                                'y1': y1,
                                'yN': yN})

if save_figs:
    gdd_maps_ds = add_attrs_to_map_ds(gdd_maps_ds, incl_vegtypes_str, dummy_fill, outdir_figs, y1, yN)
    gddharv_maps_ds = add_attrs_to_map_ds(gddharv_maps_ds, incl_vegtypes_str, dummy_fill, outdir_figs, y1, yN)
    
    gdd_maps_ds.to_netcdf(outdir_figs + "gdd_maps.nc")
    gddharv_maps_ds.to_netcdf(outdir_figs + "gddharv_maps.nc")


# %% Save before/after map and boxplot figures

def make_map(ax, this_map, this_title, vmax, bin_width, fontsize_ticklabels, fontsize_titles): 
    im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
            this_map, shading="auto",
            vmin=0, vmax=vmax)
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines()
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
    plt.axis('off')
    
def get_non_nans(in_da, fillValue):
    in_da = in_da.where(in_da != fillValue)
    return in_da.values[~np.isnan(in_da.values)]

def set_boxplot_props(bp, color):
    linewidth = 3
    plt.setp(bp['boxes'], color=color, linewidth=linewidth)
    plt.setp(bp['whiskers'], color=color, linewidth=linewidth)
    plt.setp(bp['caps'], color=color, linewidth=linewidth)
    plt.setp(bp['medians'], color=color, linewidth=linewidth)
    plt.setp(bp['fliers'], markeredgecolor=color, markersize=6, linewidth=linewidth, markeredgewidth=linewidth/2)

def make_plot(data, offset):
    linewidth = 1.5
    offset = 0.4*offset
    bpl = plt.boxplot(data, positions=np.array(range(len(data)))*2.0+offset, widths=0.6, 
                      boxprops=dict(linewidth=linewidth), whiskerprops=dict(linewidth=linewidth), 
                      capprops=dict(linewidth=linewidth), medianprops=dict(linewidth=linewidth),
                      flierprops=dict(markeredgewidth=0.5))
    return bpl

def make_figures(thisDir=None, gdd_maps_ds=None, gddharv_maps_ds=None):
    if not gdd_maps_ds:
        if not thisDir:
            raise RuntimeError('If not providing gdd_maps_ds, you must provide thisDir (location of gdd_maps.nc)')
        gdd_maps_ds = xr.open_dataset(thisDir + 'gdd_maps.nc')
    if not gddharv_maps_ds:
        if not thisDir:
            raise RuntimeError('If not providing gddharv_maps_ds, you must provide thisDir (location of gddharv_maps.nc)')
        gddharv_maps_ds = xr.open_dataset(thisDir + 'gdd_maps.nc')

    # Get info
    incl_vegtypes_str = gdd_maps_ds.attrs['incl_vegtypes_str']
    dummy_fill = gdd_maps_ds.attrs['dummy_fill']
    outdir_figs = gdd_maps_ds.attrs['outdir_figs']
    y1 = gdd_maps_ds.attrs['y1']
    yN = gdd_maps_ds.attrs['yN']

    # layout = "3x1"
    layout = "2x2"
    bin_width = 15
    lat_bin_edges = np.arange(0, 91, bin_width)

    fontsize_titles = 18
    fontsize_axislabels = 15
    fontsize_ticklabels = 15

    Nbins = len(lat_bin_edges)-1
    bin_names = ["All"]
    for b in np.arange(Nbins):
        lower = lat_bin_edges[b]
        upper = lat_bin_edges[b+1]
        bin_names.append(f"{lower}–{upper}")
        
    color_old = '#beaed4'
    color_new = '#7fc97f'

    # Maps
    ny = 3
    nx = 1
    print("Making before/after maps...")
    for v, vegtype_str in enumerate(incl_vegtypes_str):
        vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
        thisVar = f"gdd1_{vegtype_int}"
        print(f"   {vegtype_str} ({vegtype_int})...")
        
        
        # Maps #####################
        
        gdd_map = gdd_maps_ds[thisVar].isel(time=0, drop=True)
        gdd_map_yx = gdd_map.where(gdd_map != dummy_fill)
        gddharv_map = gddharv_maps_ds[thisVar]
        if "time" in gddharv_map.dims:
            gddharv_map = gddharv_map.isel(time=0, drop=True)
        gddharv_map_yx = gddharv_map.where(gddharv_map != dummy_fill)
                
        vmax = max(np.max(gdd_map_yx), np.max(gddharv_map_yx))
        
        # Set up figure and first subplot
        if layout == "3x1":
            fig = plt.figure(figsize=(7.5,14))
            ax = fig.add_subplot(ny,nx,1,projection=ccrs.PlateCarree())
        elif layout == "2x2":
            fig = plt.figure(figsize=(24,12))
            spec = fig.add_gridspec(nrows=2, ncols=2,
                                    width_ratios=[0.4,0.6])
            ax = fig.add_subplot(spec[0,0],projection=ccrs.PlateCarree())
        else:
            raise RuntimeError(f"layout {layout} not recognized")
        
        thisMin = int(np.round(np.nanmin(gddharv_map_yx)))
        thisMax = int(np.round(np.nanmax(gddharv_map_yx)))
        thisTitle = f"{vegtype_str}: Old (range {thisMin}–{thisMax})"
        make_map(ax, gddharv_map_yx, thisTitle, vmax, bin_width,
                 fontsize_ticklabels, fontsize_titles)
        
        if layout == "3x1":
            ax = fig.add_subplot(ny,nx,2,projection=ccrs.PlateCarree())
        elif layout == "2x2":
            ax = fig.add_subplot(spec[1,0],projection=ccrs.PlateCarree())
        else:
            raise RuntimeError(f"layout {layout} not recognized")
        thisMin = int(np.round(np.nanmin(gdd_map_yx)))
        thisMax = int(np.round(np.nanmax(gdd_map_yx)))
        thisTitle = f"{vegtype_str}: New (range {thisMin}–{thisMax})"
        make_map(ax, gdd_map_yx, thisTitle, vmax, bin_width,
                 fontsize_ticklabels, fontsize_titles)
        
        # Boxplots #####################
        
        gdd_vector = get_non_nans(gdd_map, dummy_fill)
        gddharv_vector = get_non_nans(gddharv_map, dummy_fill)
        
        lat_abs = np.abs(gdd_map.lat.values)
        gdd_bybin_old = [gddharv_vector]
        gdd_bybin_new = [gdd_vector]
        for b in np.arange(Nbins):
            lower = lat_bin_edges[b]
            upper = lat_bin_edges[b+1]
            lat_inds = np.where((lat_abs>=lower) & (lat_abs<upper))[0]
            gdd_vector_thisBin = get_non_nans(gdd_map[lat_inds,:], dummy_fill)
            gddharv_vector_thisBin = get_non_nans(gddharv_map[lat_inds,:], dummy_fill)
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
        plt.savefig(outfile, dpi=300, transparent=False, facecolor='white',
                    bbox_inches='tight')
        plt.close()

    print("Done.")

if save_figs: make_figures(gdd_maps_ds=gdd_maps_ds, gddharv_maps_ds=gdd_maps_ds)



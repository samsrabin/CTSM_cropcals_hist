# %% Setup

# Years of interest (do not include extra year needed for finishing last growing season)
y1 = 1980
yN = 2009

# Minimum harvest threshold allowed in PlantCrop()
gdd_min = 50

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# CLM max growing season length, mxmat, is stored in the following files:
#   * clm5_1: lnd/clm2/paramdata/ctsm51_params.c211112.nc
#   * clm5_0: lnd/clm2/paramdata/clm50_params.c211112.nc
#   * clm4_5: lnd/clm2/paramdata/clm45_params.c211112.nc
paramfile_dir = "/Users/Shared/CESM_inputdata/lnd/clm2/paramdata/"
my_clm_ver = 51
my_clm_subver = "c211112"

# Prescribed sowing and harvest dates
# sdates_rx_file = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"
# hdates_rx_file = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.nc"
# gdds_rx_file = "/Users/Shared/CESM_work/crop_dates/gdds_20220331_144207.nc"
# hdates_rx_file = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37.2000-2000.20220602_230029.nc"
gdds_rx_file = "/Users/Shared/CESM_work/crop_dates/gdds_20220602_231239.nc"
sdates_rx_file = "/Users/Shared/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"
hdates_rx_file = "/Users/Shared/CESM_work/crop_dates/hdates_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17.2000-2000.20220727_164727.nc"
gdds_rx_file = "/Users/Shared/CESM_work/crop_dates/gdds_20220820_163845.nc"

# Directory where model output file(s) can be found (figure files will be saved in subdir here)
indirs = list()
# indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-05-orig/",
#                    used_clm_mxmat = True,
#                    used_rx_sdate = False,
#                    used_rx_harvthresh = False))
# indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-05-gddforced/",
#                    used_clm_mxmat = True,
#                    used_rx_sdate = True,
#                    used_rx_harvthresh = True))
# indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37/2022-04-11-gddforced/",
# indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220601.03.ba902039.gddforced/",
# indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220606.4d9cfa0e.gddforced/",
# indirs.append(dict(path="/Users/Shared/CESM_runs/f10_f10_mg37_20220530/20220610.299035.gddforced/",
#                    used_clm_mxmat = False,
#                    used_rx_sdate = True,
#                    used_rx_harvthresh = True))
indirs.append(dict(path="/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.yield_perharv2.IHistClm50BgcCrop.1958-2014/",
                   used_clm_mxmat = True,
                   used_rx_sdate = False,
                   used_rx_harvthresh = False,
                   landuse_varies = True))
indirs.append(dict(path="/Users/Shared/CESM_runs/cropcals_2deg_v3/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1958-2014.gddforced/",
                   used_clm_mxmat = False,
                   used_rx_sdate = True,
                   used_rx_harvthresh = True,
                   landuse_varies = True))

ggcmi_out_topdir = "/Users/Shared/GGCMI/AgMIP.output"
ggcmi_cropcal_dir = "/Users/Shared/GGCMI/AgMIP.input/phase3/ISIMIP3/crop_calendar"

if len(indirs) != 2:
    raise RuntimeError(f"For now, indirs must have 2 members (found {len(indirs)}")

# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import cropcal_module as cc

import numpy as np
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
import glob
import cftime
import cartopy.crs as ccrs
from matplotlib import cm
import datetime as dt
import nc_time_axis
import re
import importlib

sys.path.append(my_ctsm_python_gallery)
import utils

import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# Directories to save outputs
indir0 = indirs[0]["path"]
outdir_figs = os.path.join(indirs[1]["path"], f"figs_comp_{os.path.basename(os.path.dirname(indir0))}")
if not os.path.exists(outdir_figs):
    os.makedirs(outdir_figs)


fontsize_titles = 8
fontsize_axislabels = 8
fontsize_ticklabels = 7
bin_width = 30
lat_bin_edges = np.arange(0, 91, bin_width)
def make_map(ax, this_map, this_title, ylabel, vmin, vmax, bin_width, fontsize_ticklabels, fontsize_titles, cmap): 
    im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
            this_map, shading="auto",
            vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="white")
    ax.coastlines(linewidth=0.3)
    if this_title:
        ax.set_title(this_title, fontsize=fontsize_titles)
    if ylabel:
        ax.set_ylabel(ylabel)
    # cbar = plt.colorbar(im1, orientation="horizontal", fraction=0.1, pad=0.02)
    # cbar.ax.tick_params(labelsize=fontsize_ticklabels)
    
    # ax.yaxis.set_tick_params(width=0.2)
    # ticks = np.arange(-90, 91, bin_width)
    # ticklabels = [str(x) for x in ticks]
    # for i,x in enumerate(ticks):
    #     if x%2:
    #         ticklabels[i] = ''
    # # ticklabels = []
    # plt.yticks(np.arange(-90,91,bin_width), labels=ticklabels,
    #            fontsize=fontsize_ticklabels,
    #            fontweight=0.1)
    return im1


# %% Import output sowing and harvest dates, etc.

print("Importing CLM output sowing and harvest dates...")

myVars = ["SDATES", "HDATES", "GDDACCUM_PERHARV", "GDDHARV_PERHARV", "HARVEST_REASON_PERHARV", "HUI_PERHARV", "SDATES_PERHARV"]

dates_ds1_orig = utils.import_ds(glob.glob(indirs[1]["path"] + "*h1.*"), \
    myVars=myVars, 
    myVegtypes=utils.define_mgdcrop_list(),
    myVars_missing_ok="SDATES_PERHARV")
dates_ds1_orig = cc.check_and_trim_years(y1, yN, dates_ds1_orig)

dates_ds0_orig = utils.import_ds(glob.glob(indirs[0]["path"] + "*h1.*"), \
    myVars=myVars, 
    myVegtypes=utils.define_mgdcrop_list(),
    myVars_missing_ok="SDATES_PERHARV")
dates_ds0_orig = cc.check_and_trim_years(y1, yN, dates_ds0_orig)

# How many growing seasons can we use? Ignore last season because it can be incomplete for some gridcells.
Ngs = dates_ds1_orig.dims['time'] - 1

# What vegetation types are included?
vegtype_list = [x for x in dates_ds0_orig.vegtype_str.values if x in dates_ds0_orig.patches1d_itype_veg_str.values]

# CLM max growing season length, mxmat, is stored in the following files:
#   * clm5_1: lnd/clm2/paramdata/ctsm51_params.c211112.nc
#   * clm5_0: lnd/clm2/paramdata/clm50_params.c211112.nc
#   * clm4_5: lnd/clm2/paramdata/clm45_params.c211112.nc
pattern = os.path.join(paramfile_dir,f"*{my_clm_ver}_params.{my_clm_subver}.nc")
paramfile = glob.glob(pattern)
if len(paramfile) != 1:
    raise RuntimeError(f"Expected to find 1 match of {pattern}; found {len(paramfile)}")
paramfile_ds = xr.open_dataset(paramfile[0])
# Import max growing season length (stored in netCDF as nanoseconds!)
paramfile_mxmats = paramfile_ds["mxmat"].values / np.timedelta64(1, 'D')
# Import PFT name list
paramfile_pftnames = [x.decode("UTF-8").replace(" ", "") for x in paramfile_ds["pftname"].values]

print("Done.")


# Import GGCMI sowing and harvest dates

sdates_rx_ds = cc.import_rx_dates("sdate", sdates_rx_file, dates_ds0_orig)
hdates_rx_ds = cc.import_rx_dates("hdate", hdates_rx_file, dates_ds0_orig)
gdds_rx_ds = cc.import_rx_dates("gdd", gdds_rx_file, dates_ds0_orig)

gs_len_rx_ds = hdates_rx_ds.copy()
for v in gs_len_rx_ds:
    if v == "time_bounds":
        continue
    gs_len_rx_ds[v] = cc.get_gs_len_da(hdates_rx_ds[v] - sdates_rx_ds[v])


# Align output sowing and harvest dates/etc.
dates_ds0 = cc.convert_axis_time2gs(dates_ds0_orig, myVars=myVars)
dates_ds1 = cc.convert_axis_time2gs(dates_ds1_orig, myVars=myVars)

# Get growing season length
dates_ds0["GSLEN"] = cc.get_gs_len_da(dates_ds0["HDATES"] - dates_ds0["SDATES"])
dates_ds1["GSLEN"] = cc.get_gs_len_da(dates_ds1["HDATES"] - dates_ds1["SDATES"])


#%% Check that some things are constant across years for ds1

constantVars = ["SDATES", "GDDHARV"]
verbose = True

importlib.reload(cc)
cc.check_constant_vars(dates_ds1, constantVars,
                       ignore_nan=indirs[1]['landuse_varies'])


# %% For both datasets, check that GDDACCUM_PERHARV <= HUI_PERHARV

cc.check_v0_le_v1(dates_ds0, ["GDDACCUM", "HUI"], msg_txt=" dates_ds0: ", both_nan_ok=indirs[0]['landuse_varies'])
cc.check_v0_le_v1(dates_ds1, ["GDDACCUM", "HUI"], msg_txt=" dates_ds1: ", both_nan_ok=indirs[0]['landuse_varies'])



# %% Check that prescribed sowing dates were obeyed

if "time" in sdates_rx_ds.dims:
    if sdates_rx_ds.dims["time"] > 1:
        Ntime = sdates_rx_ds.dims["time"]
        raise RuntimeError(f"Expected time dimension length 1; found length {Ntime}")
    sdates_rx_ds = sdates_rx_ds.isel(time=0, drop=True)
    hdates_rx_ds = hdates_rx_ds.isel(time=0, drop=True)
    gdds_rx_ds = gdds_rx_ds.isel(time=0, drop=True)
    

if indirs[0]["used_rx_sdate"]:
    cc.check_rx_obeyed(vegtype_list, sdates_rx_ds, dates_ds0, "dates_ds0", "SDATES")
if indirs[1]["used_rx_sdate"]:
    cc.check_rx_obeyed(vegtype_list, sdates_rx_ds, dates_ds1, "dates_ds1", "SDATES")
if indirs[0]["used_rx_harvthresh"]:
    cc.check_rx_obeyed(vegtype_list, gdds_rx_ds, dates_ds0, "dates_ds0", "GDDHARV", gdd_min=gdd_min)
if indirs[1]["used_rx_harvthresh"]:
    cc.check_rx_obeyed(vegtype_list, gdds_rx_ds, dates_ds1, "dates_ds1", "GDDHARV", gdd_min=gdd_min)
    

# %% Make map of harvest reasons

if "HARVEST_REASON_PERHARV" in "dates_ds0":
    thisVar = "HARVEST_REASON_PERHARV"
else:
    thisVar = "HARVEST_REASON"

reason_list_text_all = [ \
    "???",                 # 0; should never actually be saved
    "Crop mature",         # 1
    "Max gs length",       # 2
    "Bad Dec31 sowing",    # 3
    "Sowing today",        # 4
    "Sowing tomorrow",     # 5
    "Sown a yr ago tmrw.", # 6
    "Sowing tmrw. (Jan 1)" # 7
    ]

reason_list = np.unique(np.concatenate( \
    (np.unique(dates_ds0[thisVar].values), \
    np.unique(dates_ds1[thisVar].values))))
reason_list = [int(x) for x in reason_list if not np.isnan(x)]

reason_list_text = [reason_list_text_all[x] for x in reason_list]

ny = 2
nx = len(reason_list)

figsize = (8, 4)
cbar_adj_bottom = 0.15
cbar_ax_rect = [0.15, 0.05, 0.7, 0.05]
cmap = plt.cm.viridis
wspace = None
hspace = None
if nx == 3:
    figsize = (8, 3)
    cbar_adj_bottom = 0.15
    cbar_ax_rect = [0.15, 0.05, 0.7, 0.05]
    cmap = plt.cm.viridis
    wspace = 0.1
    hspace = 0
elif nx != 2:
    print(f"Since nx = {nx}, you may need to rework some parameters")

for v, vegtype_str in enumerate(vegtype_list):
    if 'winter' in vegtype_str or 'miscanthus' in vegtype_str:
        continue
    print(f"{thisVar}: {vegtype_str}...")
    vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
    
    # Get variations on vegtype string
    vegtype_str_title = cc.get_vegtype_str_for_title(vegtype_str)
    vegtype_str_figfile = cc.get_vegtype_str_figfile(vegtype_str)
    
    # Grid
    thisCrop0_gridded = utils.grid_one_variable(dates_ds0, thisVar, \
        vegtype=vegtype_int).squeeze(drop=True)
    thisCrop1_gridded = utils.grid_one_variable(dates_ds1, thisVar, \
        vegtype=vegtype_int).squeeze(drop=True)
    
    # Set up figure
    fig = plt.figure(figsize=figsize)
    
    # Map each reason's frequency
    for f, reason in enumerate(reason_list):
        reason_text = reason_list_text[f]
        
        ylabel = "CLM5-style" if f==0 else None
        map0_yx = cc.get_reason_freq_map(Ngs, thisCrop0_gridded, reason)
        ax = cc.make_axis(fig, ny, nx, f+1)
        im0 = make_map(ax, map0_yx, f"v0: {reason_text}", ylabel, 0.0, 1.0, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
        
        ylabel = "GGCMI-style" if f==0 else None
        ax = cc.make_axis(fig, ny, nx, f+nx+1)
        map1_yx = cc.get_reason_freq_map(Ngs, thisCrop1_gridded, reason)
        im1 = make_map(ax, map1_yx, f"v1: {reason_text}", ylabel, 0.0, 1.0, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
        
    fig.suptitle(f"Harvest reason: {vegtype_str_title}")
    fig.subplots_adjust(bottom=cbar_adj_bottom)
    cbar_ax = fig.add_axes(cbar_ax_rect)
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal")
    cbar_ax.tick_params(labelsize=fontsize_ticklabels)
    plt.xlabel("Frequency", fontsize=fontsize_titles)
    if wspace != None:
        plt.subplots_adjust(wspace=wspace)
    if hspace != None:
        plt.subplots_adjust(hspace=hspace)
    
    # plt.show()
    # break
    
    # Save
    outfile = os.path.join(outdir_figs, f"harvest_reason_0vs1_{vegtype_str_figfile}.png")
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
            bbox_inches='tight')
    plt.close()
    

# %% Make map of means 

varList = ["GSLEN.onlyMature.useMedian", "GDDHARV_PERHARV", "HUI_PERHARV", "HUI_PERHARV.onlyMature", "GSLEN", "GSLEN.onlyMature", "SDATES", "HDATES.onlyMature"]
# varList = ["GDDHARV_PERHARV"]
# varList = ["HUI_PERHARV"]
# varList = ["GSLEN"]
# varList = ["GSLEN.onlyMature"]
# varList = ["GSLEN", "GSLEN.onlyMature"]
# varList = ["GSLEN.onlyMature.noOutliers"]
# varList = ["GSLEN.onlyMature.useMedian"]
# varList = ["SDATES", "HDATES.onlyMature"]
# varList = ["SDATES"]

varList2 = [v.replace('_PERHARV','') for v in varList if v not in dates_ds0]
varList = varList2

vertical = False

fontsize_titles = 24
fontsize_axislabels = 24
fontsize_ticklabels = 21

for thisVar in varList:
    
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

    ny = 2
    nx = 1
    vmin = 0.0
    cmap = plt.cm.viridis
    if 'GDDHARV' in thisVar:
        title_prefix = "Harv. thresh." + title_prefix
        filename_prefix = "harvest_thresh" + filename_prefix
        ny = 3
        units = "GDD"
    elif 'HUI' in thisVar:
        title_prefix = "HUI @harv." + title_prefix
        filename_prefix = "hui" + filename_prefix
        ny = 3
        units = "GDD"
    elif thisVar == "GSLEN":
        title_prefix = "Seas. length" + title_prefix
        filename_prefix = "seas_length" + filename_prefix
        units = "Days"
        ny = 3
        vmin = None
    elif thisVar == "SDATES":
        title_prefix = "Sowing date" + title_prefix
        filename_prefix = "sdate" + filename_prefix
        units = "Day of year"
        ny = 3
        cmap = plt.cm.twilight
    elif thisVar == "HDATES":
        title_prefix = "Harvest date" + title_prefix
        filename_prefix = "hdate" + filename_prefix
        units = "Day of year"
        ny = 3
        cmap = plt.cm.twilight
    else:
        raise RuntimeError(f"thisVar {thisVar} not recognized")
    
    if not vertical:
        tmp = nx
        nx = ny
        ny = tmp
        figsize = (24, 5.7)
        cbar_adj_bottom = 0
        cbar_ax_rect = [0.3, 0.05, 0.4, 0.07]
        suptitle_y_adj = 0.9
        wspace = 0.1
        hspace = 0
        if ny != 1:
            print(f"Since ny = {ny}, you may need to rework some parameters")
        if nx != 3:
            print(f"Since nx = {nx}, you may need to rework some parameters")
    else:
        figsize = (4, 4)
        cbar_adj_bottom = 0.15
        cbar_ax_rect = [0.15, 0.05, 0.7, 0.05]
        suptitle_y_adj = 1.04
        wspace = None
        hspace = None
        if nx != 1:
            print(f"Since nx = {nx}, you may need to rework some parameters")
        if ny == 3:
            cbar_width = 0.46
            cbar_ax_rect = [(1-cbar_width)/2, 0.05, cbar_width, 0.05]
        elif ny != 2:
            print(f"Since ny = {ny}, you may need to rework some parameters")
    nplots = nx*ny

    for v, vegtype_str in enumerate(vegtype_list):
        
        if "winter" in vegtype_str or "miscanthus" in vegtype_str:
            continue
        
        print(f"{thisVar}: {vegtype_str}...")
        vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
        
        # Get variations on vegtype string
        vegtype_str_paramfile = cc.get_vegtype_str_paramfile(vegtype_str)
        vegtype_str_title = cc.get_vegtype_str_for_title(vegtype_str)
        vegtype_str_figfile = cc.get_vegtype_str_figfile(vegtype_str)
        
        # Grid
        thisCrop0_gridded = utils.grid_one_variable(dates_ds0, thisVar, \
            vegtype=vegtype_int).squeeze(drop=True)
        thisCrop1_gridded = utils.grid_one_variable(dates_ds1, thisVar, \
            vegtype=vegtype_int).squeeze(drop=True)
        
        # If needed, only include seasons where crop reached maturity
        if onlyMature:
            thisCrop0_gridded = cc.mask_immature(dates_ds0, vegtype_int, thisCrop0_gridded)
            thisCrop1_gridded = cc.mask_immature(dates_ds1, vegtype_int, thisCrop1_gridded)
            
        # If needed, remove outliers
        if noOutliers:
            thisCrop0_gridded = cc.remove_outliers(thisCrop0_gridded)
            thisCrop1_gridded = cc.remove_outliers(thisCrop1_gridded)
            
        # Get summary statistic
        if useMedian:
            if thisVar in ["SDATES", "HDATES"]:
                raise RuntimeError(f"Median of {thisVar} not yet supported")
            map0_yx = thisCrop0_gridded.median(axis=0)
            map1_yx = thisCrop1_gridded.median(axis=0)
        else:
            map0_yx = np.mean(thisCrop0_gridded, axis=0)
            map1_yx = np.mean(thisCrop1_gridded, axis=0)
            if thisVar in ["SDATES", "HDATES"]:
                map0_yx.values = stats.circmean(thisCrop0_gridded, high=365, axis=0)
                map1_yx.values = stats.circmean(thisCrop1_gridded, high=365, axis=0)
                
        
        # Set up figure 
        fig = plt.figure(figsize=figsize)
        subplot_title_suffixes = ["", ""]
        
        # Set colorbar etc.
        max0 = int(np.ceil(np.nanmax(map0_yx)))
        max1 = int(np.ceil(np.nanmax(map1_yx)))
        vmax = max(max0, max1)
        if vmin == None:
            vmin = int(np.floor(min(np.nanmin(map0_yx), np.nanmin(map1_yx))))
        if nplots == 3:
            if thisVar == "GSLEN":
                mxmat = int(paramfile_mxmats[paramfile_pftnames.index(vegtype_str_paramfile)])
                units = f"Days (mxmat: {mxmat})"
                if not mxmat > 0:
                    raise RuntimeError(f"Error getting mxmat: {mxmat}")
                
                longest_gs = max(max0, max1)
                subplot_title_suffixes = [f" (max={max0})",
                                        f" (max={max1})"]
                if indirs[0]["used_clm_mxmat"]:
                    if max0 > mxmat:
                        raise RuntimeError(f"v0: mxmat {mxmat} but max simulated {max0}")
                if indirs[1]["used_clm_mxmat"]:
                    if max1 > mxmat:
                        raise RuntimeError(f"v1: mxmat {mxmat} but max simulated {max1}")
                map2_yx = gs_len_rx_ds[f"gs1_{vegtype_int}"].isel(time=0, drop=True)
                map2_yx = map2_yx.where(np.bitwise_not(np.isnan(map1_yx)))
                max2 = int(np.nanmax(map2_yx.values))
                vmax = max(max0, max1, max2)
                
                if vmax > mxmat:
                    Nok = mxmat - vmin + 1
                    Nbad = vmax - mxmat + 1
                    cmap_to_mxmat = plt.cm.viridis(np.linspace(0, 1, num=Nok))
                    cmap_after_mxmat = plt.cm.OrRd(np.linspace(0, 1, num=Nbad))
                    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', np.vstack((cmap_to_mxmat, cmap_after_mxmat)))
            elif thisVar in ["HUI_PERHARV", "GDDHARV_PERHARV", "HUI", "GDDHARV", "SDATES", "HDATES"]:
                thisVar_rx = f"gs1_{vegtype_int}"
                if thisVar in ["HUI_PERHARV", "GDDHARV_PERHARV", "HUI", "GDDHARV"]:
                    this_rx_ds = gdds_rx_ds
                elif thisVar == "SDATES":
                    this_rx_ds = sdates_rx_ds
                elif thisVar == "HDATES":
                    this_rx_ds = hdates_rx_ds
                else:
                    raise RuntimeError(f"thisVar {thisVar} not recognized: Choosing rx DataSet")
                map2_yx = this_rx_ds[thisVar_rx]
                if "time" in map2_yx.dims:
                    map2_yx = map2_yx.isel(time=0, drop=True)
                map2_yx = map2_yx.where(np.bitwise_not(np.isnan(map1_yx)))
                max2 = int(np.nanmax(map2_yx.values))
                if thisVar in ["SDATES", "HDATES"]:
                    vmax = 365
                else:
                    vmax = max(max0, max1, max2)
            else:
                raise RuntimeError(f"thisVar {thisVar} not recognized: Setting up third plot")
        
        ylabel = "CLM5-style"
        ax = cc.make_axis(fig, ny, nx, 1)
        im0 = make_map(ax, map0_yx, f"v0{subplot_title_suffixes[0]}", ylabel, vmin, vmax, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
        
        ylabel = "GGCMI-style"
        ax = cc.make_axis(fig, ny, nx, 2)
        im1 = make_map(ax, map1_yx, f"v1{subplot_title_suffixes[1]}", ylabel, vmin, vmax, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
        
        if nplots == 3:
            ax = cc.make_axis(fig, ny, nx, 3)
            if thisVar == "GSLEN":
                tmp_title = f"GGCMI (max={max2})"
            elif thisVar in ["HUI_PERHARV", "GDDHARV_PERHARV", "HUI", "GDDHARV", "SDATES"]:
                tmp_title = "Prescribed"
            elif thisVar == "HDATES":
                tmp_title = "GGCMI"
            else:
                raise RuntimeError(f"thisVar {thisVar} not recognized: Getting title of third plot")
            im1 = make_map(ax, map2_yx, tmp_title, ylabel, vmin, vmax, bin_width, fontsize_ticklabels, fontsize_titles, cmap)
        
        this_title = f"{title_prefix}:\n{vegtype_str_title}"
        if not vertical:
            this_title = this_title.replace("\n", " ")
        fig.suptitle(this_title, y=suptitle_y_adj)
        fig.subplots_adjust(bottom=cbar_adj_bottom)
        cbar_ax = fig.add_axes(cbar_ax_rect)
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal")
        cbar_ax.tick_params(labelsize=fontsize_ticklabels)
        plt.xlabel(units, fontsize=fontsize_titles)
        if wspace != None:
            plt.subplots_adjust(wspace=wspace)
        if hspace != None:
            plt.subplots_adjust(hspace=hspace)
        
        # plt.show()
        # break
        
        # Save
        outfile = os.path.join(outdir_figs, f"{filename_prefix}_0vs1_{vegtype_str_figfile}.png")
        plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
                bbox_inches='tight')
        plt.close()


# %% Compare mean growing season length (v1 only) to GGCMI models

# varList = ["GSLEN", "GSLEN.onlyMature", "GSLEN.onlyMature.noOutliers", "GSLEN.onlyMature.useMedian"]
# varList = ["GSLEN"]
# varList = ["GSLEN.onlyMature"]
# varList = ["GSLEN.onlyMature.diffExpected"]
# varList = ["GSLEN.onlyMature.diffExpected.noOutliers"]
varList = ["GSLEN.onlyMature.diffExpected.useMedian"]
# varList = ["GSLEN", "GSLEN.onlyMature"]
# varList = ["GSLEN.onlyMature.noOutliers"]
# varList = ["GSLEN.onlyMature.useMedian"]

verbose = False

ggcmi_models_orig = ["ACEA", "CROVER", "CYGMA1p74", "DSSAT-Pythia", "EPIC-IIASA", "ISAM", "LDNDC", "LPJ-GUESS", "LPJmL", "pDSSAT", "PEPIC", "PROMET", "SIMPLACE-LINTUL5"]
Nggcmi_models_orig = len(ggcmi_models_orig)

def get_new_filename(pattern):
    thisFile = glob.glob(pattern)
    if len(thisFile) > 1:
        raise RuntimeError(f"Expected at most 1 match of {pattern}; found {len(thisFile)}")
    return thisFile

def trim_years(y1, yN, Ngs, ds_in):
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
    nx = 4
    if Nggcmi_models_orig > ny*nx + 3:
        raise RuntimeError(f"{Nggcmi_models_orig} GGCMI models + 3 other maps > ny*nx ({ny*nx})")
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

    for v, vegtype_str in enumerate(vegtype_list):
        
        if "corn" in vegtype_str:
            vegtype_str_ggcmi = "mai"
        elif "rice" in vegtype_str:
            vegtype_str_ggcmi = "ri1" # Ignoring ri2, which isn't simulated in CLM yet
        elif "soybean" in vegtype_str:
            vegtype_str_ggcmi = "soy"
        elif "spring_wheat" in vegtype_str:
            vegtype_str_ggcmi = "swh"
        # elif "winter_wheat" in vegtype_str:
        #     vegtype_str_ggcmi = "wwh"
        else:
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
                ggcmiDS = xr.Dataset(coords={"gs": dates_ds1.gs.values,
                                            "lat": thisDS.lat,
                                            "lon": thisDS.lon,
                                            "model": ggcmi_models_orig,
                                            "cft": vegtype_list})
                ggcmiDS_started = True
            
            # Set up DataArray for this crop-irr
            if g==0:
                matyday_da = xr.DataArray(data=np.full((Ngs,
                                                        thisDS.dims["lat"],
                                                        thisDS.dims["lon"],
                                                        Nggcmi_models_orig
                                                    ),
                                                    fill_value=np.nan),
                                                coords=[ggcmiDS.coords[x] for x in ["gs","lat","lon","model"]])
            
            # Get just the seasons you need
            thisDS = trim_years(y1, yN, Ngs, thisDS)
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
                    filterDS = trim_years(y1, yN, Ngs, filterDS)
                    filter_str = f"(after filtering by {filterVar} == 1)"
                    thisDA = thisDA.where(filterDS[ncvar.replace("matyday", filterVar)] == 1)
                else:
                    filterVar = "maturitystatus"
                    thisFile = get_new_filename(pattern.replace("matyday", filterVar))
                    if thisFile:
                        filterDS = xr.open_dataset(thisFile[0], decode_times=False)
                        filterDS = trim_years(y1, yN, Ngs, filterDS)
                        filter_str = f"(after filtering by {filterVar} >= 1)"
                        thisDA = thisDA.where(filterDS[ncvar.replace("matyday", filterVar)] >= 1)
                    else:
                        filterVar = "yield"
                        thisFile = get_new_filename(pattern.replace("matyday", filterVar))
                        if thisFile:
                            filterDS = xr.open_dataset(thisFile[0], decode_times=False)
                            filterDS = trim_years(y1, yN, Ngs, filterDS)
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
                filterDS = trim_years(y1, yN, Ngs, filterDS)
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
        map3_yx = ggcmiExpDS["growing_season_length"] / np.timedelta64(1, 'D')
        
        # Grid
        thisCrop1_gridded = utils.grid_one_variable(dates_ds1, thisVar, \
            vegtype=vegtype_int).squeeze(drop=True)
        
        # If needed, only include seasons where crop reached maturity
        if onlyMature:
            thisCrop1_gridded = cc.mask_immature(dates_ds1, vegtype_int, thisCrop1_gridded)
            
        # If needed, remove outliers
        if noOutliers:
            thisCrop1_gridded = cc.remove_outliers(thisCrop1_gridded)
            ggcmiDA = cc.remove_outliers(ggcmiDA)
            
        # Get summary statistic
        if useMedian:
            map1_yx = thisCrop1_gridded.median(axis=0)
            ggcmiDA_mn = ggcmiDA.median(axis=0)
        else:
            map1_yx = np.mean(thisCrop1_gridded, axis=0)
            ggcmiDA_mn = np.mean(ggcmiDA, axis=0)
        
        # Get "prescribed" growing season length
        map2_yx = gs_len_rx_ds[f"gs1_{vegtype_int}"].isel(time=0, drop=True)
        map2_yx = map2_yx.where(np.bitwise_not(np.isnan(map1_yx)))
        
        # Set up figure 
        fig = plt.figure(figsize=figsize)
        subplot_title_suffixes = ["", ""]
        
        # Set colorbar etc.
        if diffExpected:
            map1_yx = map1_yx - map2_yx
            ggcmiDA_mn = ggcmiDA_mn - map3_yx
            tmp1 = int(np.nanmax(abs(map1_yx)))
            tmpG = int(np.nanmax(abs(ggcmiDA_mn.values)))
            tmp = max(tmp1, tmpG)
            vmin = -tmp
            vmax = tmp
        else:
            min1 = int(np.ceil(np.nanmin(map1_yx)))
            min2 = int(np.ceil(np.nanmin(map2_yx)))
            min3 = int(np.ceil(np.nanmin(map3_yx)))
            vmin = min(min1, min2, min3, np.nanmin(ggcmiDA_mn.values))
            max1 = int(np.ceil(np.nanmax(map1_yx)))
            max2 = int(np.ceil(np.nanmax(map2_yx)))
            max3 = int(np.ceil(np.nanmax(map3_yx)))
            vmax = max(max1, max2, max3, np.nanmax(ggcmiDA_mn.values))
        
        ax = cc.make_axis(fig, ny, nx, 1)
        im1 = make_map(ax, map1_yx, "CLM", "", vmin, vmax, bin_width, fontsize_ticklabels*2, fontsize_titles*2, cmap)
        
        if not diffExpected:
            ax = cc.make_axis(fig, ny, nx, 2)
            im1 = make_map(ax, map2_yx, "CLM expected", "", vmin, vmax, bin_width, fontsize_ticklabels*2, fontsize_titles*2, cmap)
            
            ax = cc.make_axis(fig, ny, nx, 3)
            im1 = make_map(ax, map3_yx, "GGCMI expected", "", vmin, vmax, bin_width, fontsize_ticklabels*2, fontsize_titles*2, cmap)
        
        for g in np.arange(Nggcmi_models):
            ggcmi_yx = ggcmiDA_mn.isel(model=g, drop=True)
            ax = cc.make_axis(fig, ny, nx, 3+g+1)
            im1 = make_map(ax, ggcmi_yx, ggcmi_models[g], "", vmin, vmax, bin_width, fontsize_ticklabels*2, fontsize_titles*2, cmap)
            
        fig.suptitle(f"{title_prefix}: {vegtype_str_title}", y=0.95, fontsize=fontsize_titles*2.2)
        fig.subplots_adjust(bottom=cbar_adj_bottom)
        cbar_ax = fig.add_axes(cbar_ax_rect)
        cbar = fig.colorbar(im1, cax=cbar_ax, orientation="horizontal")
        cbar_ax.tick_params(labelsize=fontsize_ticklabels*2)
        plt.xlabel(units, fontsize=fontsize_titles*2)
        
        plt.subplots_adjust(wspace=0, hspace=0.3)
        
        # plt.show()
        # break
        
        # Save
        outfile = os.path.join(outdir_figs, f"{filename_prefix}_{vegtype_str_figfile}.png")
        plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
                bbox_inches='tight')
        plt.close()


# # %% Grid a variable and save to netCDF

# thisVar = "GRAINC_TO_FOOD_ACCUM_PERHARV"

# outdir_nc = os.path.join(indirs[1]["path"], "netcdfs")
# if not os.path.exists(outdir_nc):
#     os.makedirs(outdir_nc)

# for v, vegtype_str in enumerate(vegtype_list):
#     print(f"{thisVar}: {vegtype_str}...")
#     vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
#     thisCrop_gridded = utils.grid_one_variable(dates_ds1, thisVar, \
#             vegtype=vegtype_int).squeeze(drop=True)
#     thisCrop_gridded = thisCrop_gridded.rename(vegtype_str)
#     thisCrop_gridded = utils.lon_pm2idl(thisCrop_gridded)
    
#     if v==0:
#         out_ds = thisCrop_gridded.to_dataset()
#     else:
#         out_ds[vegtype_str] = thisCrop_gridded

# outFile = os.path.join(outdir_nc, thisVar+".nc")
# out_ds.to_netcdf(outFile)









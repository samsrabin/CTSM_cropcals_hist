import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.collections as mplcol
import cartopy.feature as cfeature
import xarray as xr
import cftime
import sys
from scipy import stats

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

import importlib
importlib.reload(sys.modules['cropcal_figs_module'])
from cropcal_figs_module import *

colormaps = {
    'seq_timeofyear': 'twilight_shifted',
    'seq_other': 'plasma_r', # magma_r? CMRmap_r?
    'div_yieldirr': 'BrBG',
    'div_timeofyear': 'twilight_shifted',
    'div_other_nonnorm': 'PuOr',
    'div_other_norm': 'RdBu_r'
}

def get_non_rx_map(var_info, cases, casename, this_var, thisCrop_main, found_types, plot_y1, plot_yN, ref_casename):
    time_dim = var_info['time_dim']
    case = cases[casename]
    
    # Trim to included years
    try:
        this_ds = case['ds'].sel({time_dim: slice(plot_y1, plot_yN)})
    except:
        # Try converting years to cftime
        plot_y1 = cftime.DatetimeNoLeap(plot_y1, 1, 1)
        plot_yN = cftime.DatetimeNoLeap(plot_yN, 1, 1)
        this_ds = case['ds'].sel({time_dim: slice(plot_y1, plot_yN)})
    
    if this_var not in case['ds']:
        return xr.DataArray(), "continue"
    elif ref_casename and ref_casename!="rx" and cases[ref_casename]['res'] != case['res']:
        # Not bothering with regridding (for now?)
        return xr.DataArray(), "continue"
    this_map = this_ds[this_var]
    
    # Prepare to mask out patch-years with no area
    if "gs" in this_map.dims:
        croparea_ever_positive = this_ds['croparea_positive_wholeseason'].sum(dim="gs")
    elif "time" in this_map.dims or this_var in ["QIRRIG_DEMAND_PATCH_PKMTH"]:
        croparea_ever_positive = this_ds['croparea_positive_sowing'].sum(dim="time")
    else:
        raise RuntimeError(f"Unsure how to mask patch-years with no area for {this_var} with dims {this_map.dims}")
    this_ds['croparea_ever_positive'] = croparea_ever_positive
    
    # Grid the included vegetation types, if needed
    if "lon" not in this_map.dims:
        this_map = utils.grid_one_variable(this_ds, this_var, vegtype=found_types)
        croparea_ever_positive = utils.grid_one_variable(this_ds, 'croparea_ever_positive', vegtype=found_types) > 0
    # If not, select the included vegetation types
    else:
        this_map = this_map.sel(ivt_str=found_types)
        croparea_ever_positive = this_ds['croparea_ever_positive'].sel(ivt_str=found_types) > 0
        
    return this_map, croparea_ever_positive, time_dim


def make_map(ax, this_map, fontsize, bounds=None, cbar=None, cbar_labelpad=4.0, cbar_max=None, cbar_spacing='uniform', cmap=colormaps['seq_other'], extend_bounds='both', extend_nonbounds='both', linewidth=1.0, lonlat_bin_width=None, show_cbar=False, subplot_label=None, this_title=None, ticklabels=None, underlay=None, underlay_color=[0.75, 0.75, 0.75, 1], units=None, vmax=None, vmin=None, vrange=None):
    
    if underlay is not None:
        underlay_cmap = mcolors.ListedColormap(np.array([underlay_color, [1, 1, 1, 1]]))
        ax.pcolormesh(underlay.lon.values, underlay.lat.values,
                      underlay, cmap=underlay_cmap)
    
    if bounds is not None:
        norm = mcolors.BoundaryNorm(bounds, cmap.N, extend=extend_bounds)
        im = ax.pcolormesh(this_map.lon.values, this_map.lat.values,
                           this_map, shading="auto",
                           norm=norm,
                           cmap=cmap)
    else:
        im = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
                           this_map, shading="auto",
                           cmap=cmap,
                           vmin=vmin, vmax=vmax)
        if vrange:
            im.set_clim(vrange[0], vrange[1])
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    
    if subplot_label is not None:
        plt.text(0, 0.95, f"({subplot_label})", transform=ax.transAxes,
             fontsize=fontsize['axislabels'])
    
    # # Country borders
    # ax.add_feature(cfeature.BORDERS, linewidth=linewidth, edgecolor="white")
    # ax.add_feature(cfeature.BORDERS, linewidth=linewidth*0.6)
    
    # Coastlines
    ax.coastlines(linewidth=linewidth, color="white", alpha=0.5)
    ax.coastlines(linewidth=linewidth*0.6, alpha=0.3)
    
    if this_title:
        ax.set_title(this_title, fontsize=fontsize['titles'])
    if show_cbar:
        if cbar:
            cbar.remove()
        
        if bounds is not None:
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', fraction=0.1, pad=0.02, spacing=cbar_spacing)
        else:
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.1, pad=0.02, extend=extend_nonbounds, spacing=cbar_spacing)
        
        deal_with_ticklabels(cbar, cbar_max, ticklabels, units, im)
        cbar.set_label(label=units, fontsize=fontsize['axislabels'], verticalalignment="center", labelpad=cbar_labelpad)
        cbar.ax.tick_params(labelsize=fontsize['ticklabels'])
        if units is not None and "month" in units.lower():
            cbar.ax.tick_params(length=0)
    
    
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

def deal_with_ticklabels(cbar, cbar_max, ticklabels, units, im):
    if ticklabels is not None:
        cbar.set_ticks(ticklabels)
        if units is not None and units.lower() == "month":
            cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            units == "Month"
    if isinstance(im, mplcol.QuadMesh):
        clim_max = im.get_clim()[1]
    else:
        clim_max = im
    if cbar_max is not None and clim_max > cbar_max:
        ticks = cbar.get_ticks()
        if ticks[-2] > cbar_max:
            raise RuntimeError(f"Specified cbar_max is {cbar_max} but highest bin BEGINS at {ticks[-2]}")
        ticklabels = ticks.copy()
        ticklabels[-1] = cbar_max
        for i, x in enumerate(ticklabels):
            if x == int(x):
                ticklabels[i] = str(int(x))
        cbar.set_ticks(ticks) # Calling this before set_xticklabels() avoids "UserWarning: FixedFormatter should only be used together with FixedLocator" (https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator)
        cbar.set_ticklabels(ticklabels)


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
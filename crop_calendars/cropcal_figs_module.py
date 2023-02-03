import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.feature as cfeature
import xarray as xr
import cftime
import sys
from scipy import stats

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

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
    
    # Grid the included vegetation types, if needed
    if "lon" not in this_map.dims:
        this_map = utils.grid_one_variable(this_ds, this_var, vegtype=found_types)
    # If not, select the included vegetation types
    else:
        this_map = this_map.sel(ivt_str=found_types)
    
    return this_map, time_dim


def make_map(ax, this_map, fontsize, lonlat_bin_width=None, units=None, cmap='viridis', vrange=None, linewidth=1.0, this_title=None, show_cbar=False, bounds=None, extend_bounds='both', vmin=None, vmax=None, cbar=None, ticklabels=None, extend_nonbounds='both', subplot_label=None, cbar_max=None): 
    
    if bounds is not None:
        norm = mcolors.BoundaryNorm(bounds, cmap.N, extend=extend_bounds)
        im = ax.pcolormesh(this_map.lon.values, this_map.lat.values,
                           this_map, shading="auto",
                           norm=norm,
                           cmap=cmap,
                           vmin=vmin, vmax=vmax)
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
    
    # ax.add_feature(cfeature.BORDERS, linewidth=linewidth, edgecolor="white")
    # ax.add_feature(cfeature.BORDERS, linewidth=linewidth*0.6)
    ax.coastlines(linewidth=linewidth, color="white")
    ax.coastlines(linewidth=linewidth*0.6)
    
    if this_title:
        ax.set_title(this_title, fontsize=fontsize['titles'])
    if show_cbar:
        if cbar:
            cbar.remove()
        
        if bounds is not None:
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), orientation='horizontal', fraction=0.1, pad=0.02)
        else:    
            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.1, pad=0.02, extend=extend_nonbounds)
        
        if ticklabels is not None:
            cbar.set_ticks(ticklabels)
            if units.lower() == "month":
                cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                units == "Month"
        elif cbar_max is not None and im.get_clim()[1] > cbar_max:
            ticks = cbar.get_ticks()
            if ticks[-2] > cbar_max:
                raise RuntimeError(f"Specified cbar_max is {cbar_max} but highest bin BEGINS at {ticks[-2]}")
            ticklabels = ticks.copy()
            ticklabels[-1] = cbar_max
            ticklabels = [str(int(x)) for x in ticklabels]
            cbar.ax.set_xticks(ticks) # Calling this before set_xticklabels() avoids "UserWarning: FixedFormatter should only be used together with FixedLocator" (https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator)
            cbar.ax.set_xticklabels(ticklabels)
        cbar.set_label(label=units, fontsize=fontsize['axislabels'])
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
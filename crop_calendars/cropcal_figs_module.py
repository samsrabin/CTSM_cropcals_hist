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


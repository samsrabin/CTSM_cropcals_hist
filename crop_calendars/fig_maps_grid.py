import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import xarray as xr
import cropcal_figs_module as ccf
import cartopy.crs as ccrs
import os
import cropcal_module as cc
from cropcal_figs_module import colormaps

# Import general CTSM Python utilities
import sys
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

# Settings

fontsize = {}
fontsize['titles'] = 18
fontsize['axislabels'] = 14
fontsize['ticklabels'] = 14
fontsize['suptitle'] = 22

dpi = 150


# Functions

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_plot_years(cases, plot_y1, plot_yN, timedim):
    for casename, case in cases.items():
        t0 = case['ds'][timedim].values[0]
        if plot_y1 is None:
            plot_y1 = case['ds'][timedim].values[0]
        plot_y1_comp = plot_y1
        if not isinstance(t0, (int, np.int64)):
            plot_y1_comp = type(t0)(plot_y1, 1, 1)
        if case['ds'][timedim].values[0] > plot_y1_comp:
            raise RuntimeError(f"Case {casename} has first date {case['ds'][timedim].values[0]}, after plot_y1 {plot_y1}")
        
        if plot_yN is None:
            plot_yN = case['ds'][timedim].values[-1]
        plot_yN_comp = plot_yN
        if not isinstance(t0, (int, np.int64)):
            plot_yN_comp = type(t0)(plot_yN, 1, 1)
        if case['ds'][timedim].values[-1] < plot_yN_comp:
            raise RuntimeError(f"Case {casename} has last date {case['ds'][timedim].values[-1]}, before plot_yN {plot_yN}")
        
    figname_y1 = plot_y1
    if not isinstance(figname_y1, int):
        figname_y1 = figname_y1.year
    figname_yN = plot_yN
    if not isinstance(figname_yN, int):
        figname_yN = figname_yN.year
        
    return plot_y1, plot_yN, figname_y1, figname_yN


def set_custom_middle_color(cmap, bounds):
    cmap_to_use = cm.get_cmap(cmap, len(bounds)+1)
    
    # Set custom color for zero-centered bin
    Nbins = len(bounds) - 1
    if Nbins % 2 == 0:
        raise RuntimeError(f"Expected odd Nbins, not {Nbins}")
    if isinstance(cmap_to_use, mcolors.LinearSegmentedColormap):
        color_list = [cmap_to_use(x) for x in np.arange(0, 1+1e-9, 1/Nbins)]
        cmap_to_use = mcolors.ListedColormap(color_list)
    new_colors = np.concatenate((cmap_to_use.colors[:int(Nbins/2)+1],
                                 np.array([colormaps['underlay']]),
                                 cmap_to_use.colors[int(Nbins/2)+1:]),
                                axis=0)
    cmap_to_use = mcolors.ListedColormap(new_colors)
    
    return cmap_to_use


def maps_gridlevel_vars(cases, varList, dpi=150, outDir_figs=None, y1=None, yN=None, nx=1, custom_figname=None):
    
    cbar_labelpad = 13
    subplot_str = None
    title_y = 1.1
    if nx == 1:
        figsize = (14, 16)
    elif nx == 2:
        figsize = (14, 8)
    else:
        raise RuntimeError(f"figsize unknown for nx={nx}")
    
    if nx > 1:
        if custom_figname is None:
            raise RuntimeError("You must provide a custom figname when nx > 1")
        elif len(varList) != nx:
            raise RuntimeError(f"nx {nx} does not match number of variables to plot {len(varList)}")
    
    # Get derived variables
    caselist = [x for x in cases.keys()]
    
    # Get list of original variables in each case's Dataset
    orig_vars = [[v for v in case['ds']] for _, case in cases.items()]
    
    # Make figures
    subplot_num = -1
    for (this_var, var_info) in varList.items():
        print(this_var)
        
        ds0 = None
        cmap = None
        vrange = None
        cbar_max = None
        extend = "neither"
        ticklabels_to_use = None
        cbar_units = None
        cmap_to_use = None
        bounds = None
        added_vars = []
        if "_PKMTH" in this_var:
            cmap = ccf.colormaps['seq_timeofyear']
            this_var2add = this_var.replace("_DIFF", "")
            added_vars.append(this_var2add)
            cases = cc.get_peakmonth(cases, this_var2add, y1=y1, yN=yN)
        if "IRRIG_WITHDRAWAL_FRAC_SUPPLY" in this_var:
            # vrange = [0, 0.9]
            bounds = [0, 0.01] + list(np.arange(0.1, 0.95, 0.1))
            cmap = truncate_colormap(cm.get_cmap(ccf.colormaps['seq_other']), minval=0.05, maxval=1.0, n=11)
            cmap_to_use = cm.get_cmap(cmap, len(bounds))
            extend = "max"
            cbar_units = "Fraction"
            cases = cc.get_irrigation_use_relative_to_supply(cases)
            this_var2add = this_var.replace("_DIFF", "")
            added_vars.append(this_var2add)
            
        # Difference plots
        if "DIFF" in this_var:
            if "IRRIG_WITHDRAWAL_FRAC_SUPPLY" in this_var:
                # vrange = [-max(vrange), max(vrange)]
                vrange = None
                extend = "both"
                bounds = np.arange(-0.9, 1, 0.2) * var_info['multiplier']
                
                # Get colormap, setting custom color for zero-centered bin
                cmap_to_use = set_custom_middle_color(ccf.colormaps['div_yieldirr'], bounds)
                
                if var_info['units'] is None:
                    cbar_units = "Change in fraction"
                else:
                    cbar_units = var_info['units']
            this_var = this_var.replace("_DIFF", "")
            case0 = caselist[0]
            case1 = caselist[1]
            ds0 = cases[case0]['ds'].copy()
            ds1 = cases[case1]['ds'].copy()
            da0 = ds0[this_var]
            da1 = ds1[this_var]
            if "time_mth" in da0.dims:
                raise RuntimeError(f"Can't handle dim time_mth")
            elif "time" in da0.dims:
                raise RuntimeError("Code this")
            da = da1 - da0
            da *= var_info['multiplier']
            if var_info['units'].lower() == "months":
                
                # Difference is modulo 6 months
                da_vals = da.copy().values
                da_vals[da_vals > 6] -= 12
                da_vals[da_vals < -6] += 12
                da = xr.DataArray(data=da_vals,
                                  coords=da.coords,
                                  attrs=da.attrs)
                da.load()
                
                # Get plot info
                bounds = np.arange(-6.5, 7.5, 1)
                extend = "neither"
                ticklabels_to_use = np.arange(-6,7,1)
                if "suppress_difftext" in var_info and var_info['suppress_difftext']:
                    cbar_units = var_info['units']
                else:
                    cbar_units = f"{case1} minus {case0} ({var_info['units']})"
                    
                 # Get colormap, setting custom color for zero-centered bin
                cmap_to_use = set_custom_middle_color(cmap, bounds) 
            
            if "time_mth" in da.dims:
                raise RuntimeError(f"Can't handle dim time_mth")
            elif "time" in da.dims:
                plot_y1, plot_yN, figname_y1, figname_yN = get_plot_years(cases, y1, yN, "time")
                timeslice = slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31")
                da = da.sel(time=timeslice).mean(dim="time")
            elif "year" in da.dims:
                plot_y1, plot_yN, figname_y1, figname_yN = get_plot_years(cases, y1, yN, "year")
                timeslice = slice(plot_y1, plot_yN)
                da = da.sel(year=timeslice).mean(dim="year")
            
            if any(d != "gridcell" for d in da.dims):
                raise RuntimeError(f"Expected one dimension ('gridcell'); got {da.dims}")
            
            if cbar_units is None:
                cbar_units = var_info['units']
                
            ds0['tmp'] = da
            this_map = utils.grid_one_variable(ds0, 'tmp')
            
            subplot_num += 1
            if (subplot_num) % nx == 0:
                f = plt.figure(figsize=figsize, facecolor="white")
            if nx > 1:
                subplot_str = chr(ord('`') + subplot_num+1) # or ord('@') for capital
            ax = f.add_subplot(1, nx, subplot_num%nx + 1, projection=ccrs.PlateCarree())
            im, cb = ccf.make_map(ax, this_map, fontsize, units=cbar_units, cmap=cmap_to_use, vrange=vrange, show_cbar=True, ticklabels=ticklabels_to_use, extend_nonbounds=extend, bounds=bounds, extend_bounds=extend, cbar_max=cbar_max, subplot_label=subplot_str, cbar_labelpad=cbar_labelpad)
            
            if ds0 is not None and "tmp" in ds0:
                ds0 = ds0.drop_vars("tmp")

            if "_PKMTH" in this_var:
                cb.ax.tick_params(length=0)
            
            minus = f": {case1} minus {case0}"
            file_basename = f"{var_info['suptitle']}{minus}"
            this_title = file_basename
            if "suppress_difftext" in var_info and var_info['suppress_difftext']:
                this_title = this_title.replace(minus, "")
            ax.set_title(this_title.replace(": ", "\n"), fontsize=fontsize['titles'], y=title_y)
            
            if outDir_figs is not None:
                if (subplot_num%nx - 1) == 0:
                    if custom_figname is not None:
                        file_basename = custom_figname
                    else:
                        file_basename = file_basename.replace(": ", "—").replace("\n", "")
                    fig_outfile = os.path.join(outDir_figs, file_basename + ".png")
                    f.savefig(fig_outfile,
                            bbox_inches='tight', facecolor='white', dpi=dpi)
                    plt.close()
            else:
                plt.show()
        
        
        # Individual cases
        
        else:
            if "_PKMTH" in this_var:
                bounds = np.arange(0.5, 13.5, 1)
                cmap_to_use = cm.get_cmap(ccf.colormaps['seq_timeofyear'])
                extend = "neither"
                ticklabels_to_use = np.arange(1,13,1)
                cbar_units = "Month"
                
            if cbar_units is None:
                cbar_units = var_info['units']
            
            plot_y1 = None
            plot_yN = None
            for i, (casename, case) in enumerate(cases.items()):
                
                ds = case['ds'].copy()
                
                if "time_mth" in ds[this_var].dims:
                    raise RuntimeError(f"Can't handle dim time_mth")
                elif "time" in ds[this_var].dims:
                    plot_y1, plot_yN, figname_y1, figname_yN = get_plot_years(cases, y1, yN, "time")
                    timeslice = slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31")
                    ds['tmp'] = ds[this_var].sel(time=timeslice).mean(dim="time")
                elif "year" in ds[this_var].dims:
                    plot_y1, plot_yN, figname_y1, figname_yN = get_plot_years(cases, y1, yN, "year")
                    timeslice = slice(plot_y1, plot_yN)
                    ds['tmp'] = ds[this_var].sel(year=timeslice).mean(dim="year")
                else:
                    ds['tmp'] = ds[this_var]
                
                if any(d != "gridcell" for d in ds['tmp'].dims):
                    raise RuntimeError(f"Expected one dimension ('gridcell'); got {ds['tmp'].dims}")
                
                this_map = utils.grid_one_variable(ds, 'tmp')
                this_map *= var_info['multiplier']
                
                subplot_num += 1
                if subplot_num % nx == 0:
                    f = plt.figure(figsize=figsize, facecolor="white")
                if nx > 1:
                    subplot_str = chr(ord('`') + subplot_num+1) # or ord('@') for capital
                ax = f.add_subplot(1, nx, subplot_num%nx + 1, projection=ccrs.PlateCarree())
                im, cb = ccf.make_map(ax, this_map, fontsize, units=cbar_units, cmap=cmap_to_use, vrange=vrange, show_cbar=True, ticklabels=ticklabels_to_use, extend_nonbounds=extend, bounds=bounds, extend_bounds=extend, cbar_max=cbar_max, subplot_label=subplot_str, cbar_labelpad=cbar_labelpad)
                
                if "_PKMTH" in this_var:
                    cb.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                    cb.ax.tick_params(length=0)
                elif "IRRIG_WITHDRAWAL_FRAC_SUPPLY" in this_var:
                    new_ticks = np.sort(list(cb.get_ticks()) + [0.01])
                    cb.set_ticks(new_ticks)
                    new_ticklabels = np.round(new_ticks, 2)
                    new_ticklabels = [str(x) for x in new_ticklabels]
                    new_ticklabels[0] = "0"
                    cb.set_ticklabels(new_ticklabels)
                
                this_title = f"{var_info['suptitle']}: {casename}"
                if plot_y1 is not None:
                    this_title += f", {figname_y1}-{figname_yN}"
                ax.set_title(this_title.replace(": ", "\n"), fontsize=fontsize['titles'], y=title_y)
                
                if outDir_figs is not None:
                    if nx==1 or (subplot_num%nx - 1) == 0:
                        print("saving")
                        if custom_figname is not None:
                            file_basename = custom_figname
                        else:
                            file_basename = this_title.replace(": ", "—").replace(",", "").replace("\n", " ")
                        fig_outfile = os.path.join(outDir_figs, file_basename + ".png")
                        f.savefig(fig_outfile,
                                bbox_inches='tight', facecolor='white', dpi=dpi)
                        plt.close()
                else:
                    plt.show()
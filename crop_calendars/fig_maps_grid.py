import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import xarray as xr
import cropcal_figs_module as ccf
import cartopy.crs as ccrs
import os
import cropcal_module as cc

# Import general CTSM Python utilities
import sys
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

# Settings

fontsize = {}
fontsize['titles'] = 18
fontsize['axislabels'] = 16
fontsize['ticklabels'] = 16
fontsize['suptitle'] = 20

figsize = (10, 7)
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


def maps_gridlevel_vars(cases, varList, dpi=150, outDir_figs=None, y1=None, yN=None):
    
    # Get derived variables
    caselist = [x for x in cases.keys()]
    
    # Get list of original variables in each case's Dataset
    orig_vars = [[v for v in case['ds']] for _, case in cases.items()]
    
    # Make figures
    for (this_var, var_info) in varList.items():
        ds0 = None
        cmap = None
        vrange = None
        cbar_max = None
        extend = "neither"
        ticklabels_to_use = None
        cbar_units = None
        cmap_to_use = None
        bounds = None
        spacing = 'uniform'
        added_vars = []
        if "_PKMTH" in this_var:
            cmap = "twilight_shifted"
            this_var2add = this_var.replace("_DIFF", "")
            added_vars.append(this_var2add)
            cases = cc.get_peakmonth(cases, this_var2add, y1=y1, yN=yN)
        if "IRRIG_WITHDRAWAL_FRAC_SUPPLY" in this_var:
            # vrange = [0, 0.9]
            bounds = [0, 0.01] + list(np.arange(0.1, 0.95, 0.1))
            cmap = truncate_colormap(cm.get_cmap("CMRmap_r"), minval=0.05, maxval=1.0, n=11)
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
                bounds = np.arange(-0.9, 1, 0.2)
                cmap_to_use = cm.get_cmap("PiYG", len(bounds)+1)
                cbar_units = "Change in fraction"
            this_var = this_var.replace("_DIFF", "")
            case0 = caselist[0]
            case1 = caselist[1]
            ds0 = cases[case0]['ds']
            ds1 = cases[case1]['ds']
            da0 = ds0[this_var]
            da1 = ds1[this_var]
            if "time_mth" in da0.dims:
                raise RuntimeError(f"Can't handle dim time_mth")
            elif "time" in da0.dims:
                raise RuntimeError("Code this")
            da = da1 - da0
            if var_info['units'] == "months":
                
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
                cmap_to_use = cm.get_cmap(cmap)
                extend = "neither"
                ticklabels_to_use = np.arange(-6,7,1)
                cbar_units=f"{case1} minus {case0} ({var_info['units']})"
            
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
            
            f = plt.figure(figsize=figsize, facecolor="white")
            ax = f.add_axes([0,0,1,1], projection=ccrs.PlateCarree())
            im, cb = ccf.make_map(ax, this_map, fontsize, units=cbar_units, cmap=cmap_to_use, vrange=vrange, linewidth=0.5, show_cbar=True, ticklabels=ticklabels_to_use, extend_nonbounds=extend, bounds=bounds, extend_bounds=extend, cbar_max=cbar_max)
            
            if ds0 is not None and "tmp" in ds0:
                ds0 = ds0.drop_vars("tmp")

            if "_PKMTH" in this_var:
                cb.ax.tick_params(length=0)
            
            this_title = f"{var_info['suptitle']}: {case1} minus {case0}"
            ax.set_title(this_title.replace(": ", "\n"), fontsize=20)
            
            if outDir_figs is not None:
                fig_outfile = os.path.join(outDir_figs, this_title.replace(": ", "—") + ".png")
                f.savefig(fig_outfile,
                        bbox_inches='tight', facecolor='white', dpi=dpi)
                plt.close()
            else:
                plt.show()
        
        
        # Individual cases
        
        else:
            if "_PKMTH" in this_var:
                bounds = np.arange(0.5, 13.5, 1)
                cmap_to_use = cm.get_cmap('twilight_shifted')
                extend = "neither"
                ticklabels_to_use = np.arange(1,13,1)
                cbar_units = "Month"
                
            if cbar_units is None:
                cbar_units = var_info['units']
            
            plot_y1 = None
            plot_yN = None
            for i, (casename, case) in enumerate(cases.items()):
                
                if "time_mth" in case['ds'][this_var].dims:
                    raise RuntimeError(f"Can't handle dim time_mth")
                elif "time" in case['ds'][this_var].dims:
                    plot_y1, plot_yN, figname_y1, figname_yN = get_plot_years(cases, y1, yN, "time")
                    timeslice = slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31")
                    case['ds']['tmp'] = case['ds'][this_var].sel(time=timeslice).mean(dim="time")
                elif "year" in case['ds'][this_var].dims:
                    plot_y1, plot_yN, figname_y1, figname_yN = get_plot_years(cases, y1, yN, "year")
                    timeslice = slice(plot_y1, plot_yN)
                    case['ds']['tmp'] = case['ds'][this_var].sel(year=timeslice).mean(dim="year")
                else:
                    case['ds']['tmp'] = case['ds'][this_var]
                
                if any(d != "gridcell" for d in case['ds']['tmp'].dims):
                    raise RuntimeError(f"Expected one dimension ('gridcell'); got {case['ds']['tmp'].dims}")
                
                this_map = utils.grid_one_variable(case['ds'], 'tmp')
                
                f = plt.figure(figsize=figsize, facecolor="white")
                ax = f.add_axes([0,0,1,1], projection=ccrs.PlateCarree())
                im, cb = ccf.make_map(ax, this_map, fontsize, units=cbar_units, cmap=cmap_to_use, vrange=vrange, linewidth=0.5, show_cbar=True, ticklabels=ticklabels_to_use, extend_nonbounds=extend, bounds=bounds, extend_bounds=extend, cbar_max=cbar_max)
                
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
                ax.set_title(this_title.replace(": ", "\n"), fontsize=20)
                
                if outDir_figs is not None:
                    fig_outfile = os.path.join(outDir_figs, this_title.replace(": ", "—").replace(",", "") + ".png")
                    f.savefig(fig_outfile,
                              bbox_inches='tight', facecolor='white', dpi=dpi)
                    plt.close()
                else:
                    plt.show()
        
    # Drop any variables we added while making figures
    for c, (_, case) in enumerate(cases.items()):
        case['ds'] = case['ds'][orig_vars[c]]
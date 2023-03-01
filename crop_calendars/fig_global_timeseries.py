# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import cropcal_module as cc
import cropcal_figs_module as ccf

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

import numpy as np
import xarray as xr
from scipy import signal, stats
import matplotlib.pyplot as plt
from matplotlib import collections as mplcollections
from matplotlib import cm
from matplotlib.ticker import MultipleLocator

fontsize = {'axis_label': 28,
            'legend': 28,
            'suptitle': 36,
            'title': 30,
            'tick_label': 24}
max_yticks = 8 # Should depend on fontsize['tick_label']
ticklength_major = 10
ticklength_minor = ticklength_major / 2


def reduce_yticks(ax):
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()
    yticks_shown = yticks
    lo_not_shown = ylim[0] > yticks[0]
    if lo_not_shown:
        yticks_shown = yticks_shown[1:]
    hi_not_shown = ylim[-1] < yticks[-1]
    if hi_not_shown:
        yticks_shown = yticks_shown[:-1]
    Nticks_shown = len(yticks_shown)
    if Nticks_shown > max_yticks:
        if Nticks_shown % 2:
            ax.set_yticks([yticks_shown[y] for y in np.arange(0,9,2)])
        else:
            raise RuntimeWarning("Not sure how to trim an odd number of yticks")
        

def get_figs_axes(ny, nx, figsize, sharex=False):
    f_list, axes_list = plt.subplots(ny, nx, figsize=figsize, sharex=sharex)
    axes_list = axes_list.flatten()
    return f_list, axes_list


def get_ydata_gridcells(thisVar, plot_y1, plot_yN, cases, fig_caselist):
    first_case = True
    for casename, case in cases.items():
        if casename not in fig_caselist:
            continue
        ts_thisVar_y = case['ds'][thisVar].sum(dim="gridcell") * 1e-9 # m3 to km3
        ts_thisVar_y = ts_thisVar_y.sel(time=slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31"))
        
        # Save
        if first_case:
            ydata_thisVar = np.expand_dims(ts_thisVar_y.values, axis=0)
            first_case = False
        else:
            ydata_thisVar = np.concatenate((ydata_thisVar,
                                            np.expand_dims(ts_thisVar_y.values, axis=0)),
                                           axis=0)
    return ydata_thisVar


def make_1plot_lines(ax_this, ydata_this, caselist, thisCrop_clm, ylabel, xlabel, y1, yN, stats2=None, stats_round=None, shift_symbols=None, subplot_label=None, show_legend=False, ylabel_xcoord=-0.125, ny=None):
    da = xr.DataArray(data = ydata_this,
                            coords = {'Case': caselist,
                                      'x_name': np.arange(y1,yN+1)})
    
    light_gray = [x/255 for x in [148, 148, 148]]
    dark_gray = [x/255 for x in [64, 64, 74]]
    grays = [dark_gray, light_gray]
    
    for i, casename in enumerate(caselist):
        if casename.lower() in ccf.cropcal_colors:
            color = ccf.cropcal_colors[casename.lower()]
        elif i <= 1:
            color = grays[i]
        else:
            color = cm.Dark2(i-2)
        if (casename == "CLM Default" and "Original baseline" in caselist) or (casename in ["Prescribed Sowing", "Prescribed Maturity"]) or "1)" in casename:
            linestyle = ":"
        elif "0)" in casename:
            linestyle = "--"
        else:
            linestyle = "-"
        da.isel(Case=i).plot.line(x='x_name', ax=ax_this, 
                                  color=color, linestyle=linestyle,
                                  linewidth=3)
    
    if subplot_label is not None:
        ypos = 0.93 # From before, with ny=4
        if ny == 3:
            ypos = 0.9
        ax_this.text(0.02, ypos, f"({subplot_label})", transform=ax_this.transAxes, fontsize=fontsize['axis_label'])
    
    thisTitle = r"$\bf{" + thisCrop_clm.replace(" ", "\ ") + "}$"
    if stats2 is not None:
        skip_this = False
        if len(stats2) != 2:
            if all(x in caselist for x in ["CLM Default", "Prescribed Calendars"]):
                stats2 = [stats2[caselist.index(y)-2] for y in ["CLM Default", "Prescribed Calendars"]]
            else:
                skip_this = True
                print(f"Not adding stats to title because Nstats {len(stats2)} != 2")
        if not skip_this:
            if stats_round is not None:
                stats2 = np.round(stats2, stats_round)
            if shift_symbols is None:
                thisTitle += f" ({stats2[0]} → {stats2[1]})"
            else:
                thisTitle += f" ({shift_symbols[0]}{stats2[0]} → {shift_symbols[1]}{stats2[1]})"
    
    ax_this.title.set_text(thisTitle)
    ax_this.title.set_size(fontsize['title'])
    ax_this.tick_params(axis='x', which='major', labelsize=fontsize['tick_label'], length=ticklength_major)
    ax_this.tick_params(axis='y', which='major', labelsize=fontsize['tick_label'], length=ticklength_minor)
    ax_this.xaxis.set_minor_locator(MultipleLocator(5))
    ax_this.tick_params(axis='x', which='minor', length=ticklength_minor)
    reduce_yticks(ax_this)
    if xlabel is not None:
        ax_this.set_xlabel(xlabel, fontsize=fontsize['axis_label'])
        ypos = -0.1 # From before, with ny=4
        if ny == 3:
            ypos=-0.15
        ax_this.xaxis.set_label_coords(0.5, ypos)
    else:
        ax_this.set_xlabel("")
        ax_this.set_xticklabels("")
    
    ax_this.set_ylabel(ylabel, fontsize=fontsize['axis_label'])
    ax_this.yaxis.set_label_coords(ylabel_xcoord, 0.5)
    if not show_legend and ax_this.get_legend():
        ax_this.get_legend().remove()
    elif show_legend:
        ax_this.legend(handles = ax_this.lines,
                    labels = caselist,
                    loc = None,
                    fontsize=fontsize['legend'])


def make_1plot_scatter(ax_this, xdata, ydata_this, caselist, thisCrop_clm, xlabel, ylabel, equalize_scatter_axes, bottommiddle_plot, stats2=None, stats_round=None, p=None, shift_symbols=None, subplot_label=None, ny=2):
    
    p_symbols = []
    for i, casename in enumerate(caselist):
        if casename.lower() in ccf.cropcal_colors:
            color = ccf.cropcal_colors[casename.lower()]
        else:
            color = cm.Dark2(i)
        if casename in ["Prescribed Sowing", "Prescribed Maturity"]:
            facecolors = 'none'
        else:
            facecolors = color
        plt.sca(ax_this)
        plt.scatter(xdata, ydata_this[i,:], color=color, s=100, alpha=0.8, facecolors=facecolors)
        m, b = np.polyfit(xdata, ydata_this[i,:], 1)
        plt.plot(xdata, m*xdata+b, color=color, linewidth=3)
        
        # Process stat significance
        p_symbols.append("")
        if p is not None:
            # Levels and symbols after Müller et al. (2017) GGCMI evaluation
            if p[i] < 0.001:
                p_symbols[i] = "$^{***}$"
            elif p[i] < 0.05:
                p_symbols[i] = "$^{**}$"
            elif p[i] < 0.01:
                p_symbols[i] = "$^*$"
    
    if subplot_label is not None:
        ypos = 0.93 # From before, with ny=4
        if ny == 3:
            ypos = 0.9
        ax_this.text(0.02, ypos, f"({subplot_label})", transform=ax_this.transAxes, fontsize=fontsize['axis_label'])
    
    if equalize_scatter_axes:
        xlim = ax_this.get_xlim()
        ylim = ax_this.get_ylim()
        newlim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        ax_this.set_xlim(newlim)
        ax_this.set_ylim(newlim)
    
    # Add 1:1 line
    xlim = ax_this.get_xlim()
    ylim = ax_this.get_ylim()
    plt.plot([-100,100], [-100,100], '--', color='gray', alpha=0.5)
    ax_this.set_xlim(xlim)
    ax_this.set_ylim(ylim)
    
    overlap = max(xlim) > min(ylim)
    
    
    thisTitle = r"$\bf{" + thisCrop_clm.replace(" ", "\ ") + "}$"
    if stats2 is not None:
        skip_this = False
        if len(stats2) != 2:
            if all(x in caselist for x in ["CLM Default", "Prescribed Calendars"]):
                stats2 = [stats2[caselist.index(y)-2] for y in ["CLM Default", "Prescribed Calendars"]]
            else:
                skip_this = True
                print(f"Not adding stats to title because Nstats {len(stats2)} != 2")
        if not skip_this:
            
            # Fisher's z test for correlations
            # Fisher (1915, doi:10.2307/2331838)
            # https://garstats.wordpress.com/2019/06/17/compindcorr/
            z0 = np.arctanh(stats2[0])
            z1 = np.arctanh(stats2[1])
            n = len(xdata)
            z = (z0 - z1) / np.sqrt(2/(n-3))
            corr_diff_p = 2*stats.norm().cdf(-np.abs(z))
            if corr_diff_p < 0.1:
                if z > 0:
                    arrow = "↗"
                else:
                    arrow = "↘"
            else:
                arrow = "→"
            
            if stats_round is not None:
                stats2 = np.round(stats2, stats_round)
            if shift_symbols is None:
                thisTitle += f" ({stats2[0]}{p_symbols[0]} {arrow} {stats2[1]}{p_symbols[1]})"
            else:
                thisTitle += f" ({shift_symbols[0]}{stats2[0]}{p_symbols[0]} {arrow} {shift_symbols[1]}{stats2[1]}{p_symbols[1]})"
    
    ax_this.title.set_text(thisTitle)
    ax_this.title.set_size(fontsize['title'])
    ax_this.tick_params(axis='both', which='major', labelsize=fontsize['tick_label'], length=ticklength_minor)
    
    if bottommiddle_plot:
        ax_this.set_xlabel(xlabel, fontsize=fontsize['suptitle'])
        ax_this.xaxis.set_label_coords(0.5, -0.25)
    else:
        ax_this.set_xlabel(xlabel, fontsize=fontsize['axis_label'])
    
    ax_this.set_ylabel(ylabel, fontsize=fontsize['suptitle'])
    if ylabel is not None:
        ax_this.yaxis.set_label_coords(-0.25, 0.5)
    
    reduce_yticks(ax_this)
    if ax_this.get_legend():
        ax_this.get_legend().remove()


def finishup_lines(c, ny, nx, axes_this, f_this, suptitle, outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=None, hide_suptitle=True, filename=None):
    # Delete unused axes, if any
    if isinstance(axes_this, np.ndarray):
        for a in np.arange(c+1, ny*nx):
            f_this.delaxes(axes_this[a])
    elif not isinstance(axes_this, np.ndarray):
        axes_this = np.array([axes_this])
        
    if mxmat_limited:
        suptitle += " (limited season length)"
    
    if not hide_suptitle:
        f_this.suptitle(suptitle,
                        x = 0.1, horizontalalignment = 'left',
                        fontsize=fontsize['suptitle'])
    
    if fig_caselist is not None:
        f_this.legend(handles = axes_this[0].lines,
                    labels = fig_caselist,
                    loc = "upper center",
                    ncol = int(np.ceil(len(fig_caselist)/2)),
                    fontsize=fontsize['legend'])
    
    min_viable_hui_str = ""
    if min_viable_hui_list is not None and min_viable_hui_list != ["ggcmi3"]:
        min_viable_hui_str = " minviableHUI"
        for x in min_viable_hui_list:
            min_viable_hui_str += "_" + str(x)

    if filename is None:
        filename = "Timeseries " + suptitle + " by crop" + min_viable_hui_str + ".pdf"
    f_this.savefig(os.path.join(outDir_figs, filename),
                   bbox_inches='tight')
    plt.close(f_this)


def finishup_scatter(c, ny, nx, axes_this, f_this, suptitle, outDir_figs, mxmat_limited, fig_caselist, inds_sim):
    # Delete unused axes, if any
    legend_in_empty_spot = False
    for a in np.arange(c+1, ny*nx):
        if a == ny*nx-1:
            legend_in_empty_spot = True
            break
        f_this.delaxes(axes_this[a])
    
    if ny==3 and nx==3:
        f_this.subplots_adjust(hspace=0.3)
        
    if mxmat_limited:
        suptitle += " (limited season length)"
        
    # Are we putting the legend in an empty spot?
    if legend_in_empty_spot:
        legend_ncol = 1
        legend_loc = "center"
        legend_owner_handle = axes_this[a]
        legend_bbox_to_anchor = None
        legend_bbox_transform = None
        suptitle_y = 0.96
    else:
        legend_ncol = int(np.ceil(len(fig_caselist)/2))
        legend_loc = "upper center"
        legend_owner_handle = f_this
        legend_bbox_to_anchor = (0.5, 0.972)
        legend_bbox_transform = f_this.transFigure
        suptitle_y = 1
        
    f_this.suptitle(suptitle,
                    x = 0.5, y=suptitle_y, horizontalalignment = 'center',
                    fontsize=fontsize['suptitle'], fontweight='bold')
    
    # Get the handles of just the points (i.e., not including regression lines)    
    legend_handles = [x for x in axes_this[0].get_children() if isinstance(x, mplcollections.PathCollection)]
    
    legend_owner_handle.legend(handles = legend_handles,
                  labels = [fig_caselist[x] for x in inds_sim],
                  loc = legend_loc,
                  bbox_to_anchor = legend_bbox_to_anchor,
                  bbox_transform = legend_bbox_transform,
                  ncol = legend_ncol,
                  fontsize=fontsize['legend'])
    if legend_in_empty_spot:
        legend_owner_handle.axis('off')

    f_this.savefig(outDir_figs + "Scatter " + suptitle + " by crop.pdf",
                   bbox_inches='tight')
    plt.close(f_this)

def get_incl_crops(thisCrop_clm, patches1d_itype_veg_str, cropList_combined_clm, only_irrigated=False):
    if isinstance(patches1d_itype_veg_str, xr.DataArray):
        patches1d_itype_veg_str = patches1d_itype_veg_str.values
        
    if "Total" in thisCrop_clm:
        incl_crops = [x for x in patches1d_itype_veg_str if x.replace('irrigated_','').replace('temperate_','').replace('tropical_','').replace('spring_','') in [y.lower() for y in cropList_combined_clm]]
        if thisCrop_clm == "Total (no sgc)":
            incl_crops = [x for x in incl_crops if "sugarcane" not in x]
        elif thisCrop_clm == "Total (grains)":
            incl_crops = [x for x in incl_crops if "sugarcane" not in x and "cotton" not in x]
        elif thisCrop_clm == "Total":
            pass
        else:
            raise RuntimeError("???")
    else:
        incl_crops = [x for x in patches1d_itype_veg_str if thisCrop_clm.lower() in x]
    
    if only_irrigated:
        incl_crops = [x for x in incl_crops if "irrigated" in x]
    
    return incl_crops


def get_sum_over_patches(da, incl_crops=None, patches1d_itype_veg=None):

    if incl_crops is not None:
        if patches1d_itype_veg is None:
            raise RuntimeError("If specifying incl_crops, you must also provide patches1d_itype_veg")
        elif isinstance(patches1d_itype_veg, xr.DataArray):
            patches1d_itype_veg = patches1d_itype_veg.values
        incl_crops_int = [utils.ivt_str2int(x) for x in incl_crops]
        isel_list = [i for i, x in enumerate(patches1d_itype_veg) if x in incl_crops_int]
        da = da.isel(patch=isel_list)
        
    da = da.sum(dim='patch')
    return da


def get_CLM_ts_area_y(case, lu_ds, thisCrop_clm, cropList_combined_clm, incl_crops=None, only_irrigated=None):
        
    if incl_crops is None:
        incl_crops = get_incl_crops(thisCrop_clm, case['ds'].vegtype_str, cropList_combined_clm, only_irrigated=only_irrigated)
    elif only_irrigated is not None:
        raise RuntimeError("get_CLM_ts_area_y(): Do not specify both incl_crops and only_irrigated")
    
    ts_area_y = get_sum_over_patches(lu_ds.AREA_CFT * 1e-4, # m2 to ha
                                     incl_crops=incl_crops,
                                     patches1d_itype_veg=lu_ds.patches1d_itype_veg)
    return ts_area_y


def get_CLM_ts_prod_y(case, lu_ds, use_annual_yields, min_viable_hui, mxmats, thisCrop_clm, cropList_combined_clm):
    if use_annual_yields:
        case['ds'] = cc.get_yield_ann(case['ds'], min_viable_hui=min_viable_hui, mxmats=mxmats)
        yieldVar = "YIELD_ANN"
    else:
        case['ds'] = cc.zero_immatures(case['ds'], min_viable_hui=min_viable_hui, mxmats=mxmats)
        yieldVar = "YIELD"
    case['ds']['ts_prod_yc'] = cc.get_ts_prod_clm_yc_da2(case['ds'], lu_ds, yieldVar, cropList_combined_clm, quiet=True)
    if thisCrop_clm == "Total (no sgc)":
        ts_prod_y = case['ds'].drop_sel(Crop=['Sugarcane', 'Total'])['ts_prod_yc'].sum(dim="Crop").copy()
    elif thisCrop_clm == "Total (grains)":
        ts_prod_y = case['ds'].drop_sel(Crop=['Sugarcane', 'Cotton', 'Total'])['ts_prod_yc'].sum(dim="Crop").copy()
    else:
        ts_prod_y = case['ds']['ts_prod_yc'].sel(Crop=thisCrop_clm).copy()
    
    return ts_prod_y



def global_timeseries_irrig_inclcrops(thisVar, cases, reses, cropList_combined_clm, outDir_figs, extra="Total (all land)", figsize=(35, 18), noFigs=False, ny=2, nx=4, plot_y1=1980, plot_yN=2010):
    
    if not noFigs:
        # f_lines_area, axes_lines_area = get_figs_axes(ny, nx, figsize)
        f_lines_irrig, axes_lines_irrig = get_figs_axes(ny, nx, figsize)
        
    # Get included cases
    fig_caselist = [x for x in cases]
    for casename in fig_caselist:
        if thisVar not in cases[casename]['ds']:
            fig_caselist.remove(casename)
    
    for c, thisCrop_clm in enumerate(cropList_combined_clm + [extra]):
        print(f"{thisCrop_clm}...")
        is_obs = []
        
        if not noFigs:
            # ax_lines_area = axes_lines_area[c]
            ax_lines_irrig = axes_lines_irrig[c]
                
        # CLM outputs
        first_case = True
        for i, (casename, case) in enumerate(cases.items()):
            if casename not in fig_caselist:
                continue
            is_obs.append(False)
            
            if thisCrop_clm == "Total (all land)":
                ydata_irrig = get_ydata_gridcells("IRRIG_FROM_SURFACE_GRID_ANN", plot_y1, plot_yN, cases, fig_caselist)
            else:

                # Irrigation
                incl_crops = get_incl_crops(thisCrop_clm, case['ds'].vegtype_str, cropList_combined_clm, only_irrigated=True)
                ts_irrig_y = get_sum_over_patches(case['ds'][thisVar] * 1e-9, # m3 to km3
                                                incl_crops=incl_crops,
                                                patches1d_itype_veg=case['ds']['patches1d_itype_veg'])
                ts_irrig_y = ts_irrig_y.sel(time=slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31"))
                
                # Save
                if first_case:
                    ydata_irrig = np.expand_dims(ts_irrig_y.values, axis=0)
                    first_case = False
                else:
                    ydata_irrig = np.concatenate((ydata_irrig,
                                                np.expand_dims(ts_irrig_y.values, axis=0)),
                                                axis=0)
                        
        # Make plots for this crop
        if not noFigs:
            no_xlabel = c < nx*(ny - 1) and not (nx==3 and c==5 and len(cropList_combined_clm + [extra])==8)
            if no_xlabel:
                xlabel = None
            else:
                xlabel = "Year"
            no_ylabel = c%nx
            if no_ylabel:
                ylabel = None
            else:
                ylabel = "km$^3$"
            subplot_str = chr(ord('`') + c+1) # or ord('@') for capital
            make_1plot_lines(ax_lines_irrig, ydata_irrig, fig_caselist, thisCrop_clm, ylabel, xlabel, plot_y1, plot_yN, subplot_label=subplot_str, ny=ny)
            
    # Finish up and save
    if not noFigs:
        print("Finishing and saving...")
        # finishup_lines(c, ny, nx, axes_lines_area, f_lines_area, "Global crop area", outDir_figs, fig_caselist)
        finishup_lines(c, ny, nx, axes_lines_irrig, f_lines_irrig, f"Global irrigation {thisVar.split('_')[1].lower()}", outDir_figs, None, fig_caselist)

    print("Done.")
    

def global_timeseries_irrig_allcrops(thisVar, cases, outDir_figs, figsize=(35, 18), noFigs=False, plot_y1=1980, plot_yN=2010):
    
    if not noFigs:
        f = plt.figure(figsize=figsize, facecolor="white")
        ax = f.add_axes([0,0,1,1])
        
    # Get included cases
    fig_caselist = [x for x in cases]
    for casename in fig_caselist:
        if thisVar not in cases[casename]['ds']:
            fig_caselist.remove(casename)
    
    # CLM outputs
    ydata_thisVar = get_ydata_gridcells(thisVar, plot_y1, plot_yN, cases, fig_caselist)
    
    # Make figure
    if not noFigs:
        xlabel = "Year"
        figtitle = "Global irrigation withdrawals"
        make_1plot_lines(ax, ydata_thisVar, fig_caselist, figtitle, "km$^3$", xlabel, plot_y1, plot_yN, show_legend=True, ylabel_xcoord=-0.075)
        finishup_lines(None, 1, 1, ax, f, "", outDir_figs, False, None, filename=figtitle+".pdf")


def global_timeseries_yieldetc(cases, cropList_combined_clm, earthstats_gd, fao_area, fao_area_nosgc, fao_prod, fao_prod_nosgc, outDir_figs, reses, yearList, \
    equalize_scatter_axes=False, extra="Total (grains)", figsize=(35, 18), include_scatter=True, include_shiftsens=True, min_viable_hui_list="ggcmi3", mxmats=None, noFigs=False, ny=2, nx=4, obs_for_fig="FAOSTAT", plot_y1=1980, plot_yN=2010, remove_scatter_bias=False, bias_round=1, corrcoef_round=3, use_annual_yields=False, verbose=False, w=5):
    
    if not isinstance(min_viable_hui_list, list):
        min_viable_hui_list = [min_viable_hui_list]
    
    # Allow multiple min_viable_hui values only for a certain case list
    if len(min_viable_hui_list) > 1 and [x.replace(' 22', '') for x in cases.keys()] != ['CLM Default', 'Prescribed Calendars']:
        raise RuntimeError("Multiple min_viable_hui values allowed only for case list ['CLM Default', 'Prescribed Calendars']")

    if not noFigs:
        f_lines_area, axes_lines_area = get_figs_axes(ny, nx, figsize)
        f_lines_prod, axes_lines_prod = get_figs_axes(ny, nx, figsize)
        f_lines_yield, axes_lines_yield = get_figs_axes(ny, nx, figsize)
        f_lines_yield_dt, axes_lines_yield_dt = get_figs_axes(ny, nx, figsize)
        if include_shiftsens:
            f_lines_area_orig, axes_lines_area_orig = get_figs_axes(ny, nx, figsize)
            f_lines_area_shiftL, axes_lines_area_shiftL = get_figs_axes(ny, nx, figsize)
            f_lines_area_shiftR, axes_lines_area_shiftR = get_figs_axes(ny, nx, figsize)
            f_lines_prod_orig, axes_lines_prod_orig = get_figs_axes(ny, nx, figsize)
            f_lines_prod_shiftL, axes_lines_prod_shiftL = get_figs_axes(ny, nx, figsize)
            f_lines_prod_shiftR, axes_lines_prod_shiftR = get_figs_axes(ny, nx, figsize)
            f_lines_yield_orig, axes_lines_yield_orig = get_figs_axes(ny, nx, figsize)
            f_lines_yield_shiftL, axes_lines_yield_shiftL = get_figs_axes(ny, nx, figsize)
            f_lines_yield_shiftR, axes_lines_yield_shiftR = get_figs_axes(ny, nx, figsize)
            f_lines_yield_dt_orig, axes_lines_yield_dt_orig = get_figs_axes(ny, nx, figsize)
            f_lines_yield_dt_shiftL, axes_lines_yield_dt_shiftL = get_figs_axes(ny, nx, figsize)
            f_lines_yield_dt_shiftR, axes_lines_yield_dt_shiftR = get_figs_axes(ny, nx, figsize)
        if include_scatter:
            figsize_scatter = figsize
            if ny == 3:
                figsize_scatter = list(figsize_scatter)
                figsize_scatter[1] *= 1
                figsize_scatter = tuple(figsize_scatter)
            f_scatter_yield_dt, axes_scatter_yield_dt = get_figs_axes(ny, nx, figsize_scatter, sharex=False)
            if include_shiftsens:
                f_scatter_yield_dt_orig, axes_scatter_yield_dt_orig = get_figs_axes(ny, nx, figsize_scatter, sharex=False)
                f_scatter_yield_dt_shiftL, axes_scatter_yield_dt_shiftL = get_figs_axes(ny, nx, figsize_scatter, sharex=False)
                f_scatter_yield_dt_shiftR, axes_scatter_yield_dt_shiftR = get_figs_axes(ny, nx, figsize_scatter, sharex=False)

    # Get list 
    fig_caselist = ["FAOSTAT"]
    this_earthstat_res = "f09_g17"
    # fig_caselist += [f"EarthStat ({this_earthstat_res})"]
    fig_caselist += ["EarthStat"]
    for min_viable_hui in min_viable_hui_list:
        for (casename, case) in cases.items():
            fig_casename = casename
            if min_viable_hui != "ggcmi3":
                if nx==3:
                    fig_casename += f" ({min_viable_hui})"
                else:
                    fig_casename += f" (min. HUI {min_viable_hui})"
            fig_caselist.append(fig_casename)

    mxmat_limited = mxmats is not None
    
    for c, thisCrop_clm in enumerate(cropList_combined_clm + [extra]):
        print(f"{thisCrop_clm}...")
        is_obs = []
        
        if not noFigs:
            ax_lines_area = axes_lines_area[c]
            ax_lines_prod = axes_lines_prod[c]
            ax_lines_yield = axes_lines_yield[c]
            ax_lines_yield_dt = axes_lines_yield_dt[c]
            if include_shiftsens:
                ax_lines_area_orig = axes_lines_area_orig[c]
                ax_lines_area_shiftL = axes_lines_area_shiftL[c]
                ax_lines_area_shiftR = axes_lines_area_shiftR[c]
                ax_lines_prod_orig = axes_lines_prod_orig[c]
                ax_lines_prod_shiftL = axes_lines_prod_shiftL[c]
                ax_lines_prod_shiftR = axes_lines_prod_shiftR[c]
                ax_lines_yield_orig = axes_lines_yield_orig[c]
                ax_lines_yield_shiftL = axes_lines_yield_shiftL[c]
                ax_lines_yield_shiftR = axes_lines_yield_shiftR[c]
                ax_lines_yield_dt_orig = axes_lines_yield_dt_orig[c]
                ax_lines_yield_dt_shiftL = axes_lines_yield_dt_shiftL[c]
                ax_lines_yield_dt_shiftR = axes_lines_yield_dt_shiftR[c]
            if include_scatter:
                ax_scatter_yield_dt = axes_scatter_yield_dt[c]
                if include_shiftsens:
                    ax_scatter_yield_dt_orig = axes_scatter_yield_dt_orig[c]
                    ax_scatter_yield_dt_shiftL = axes_scatter_yield_dt_shiftL[c]
                    ax_scatter_yield_dt_shiftR = axes_scatter_yield_dt_shiftR[c]
        
        # FAOSTAT
        is_obs.append(True)
        if thisCrop_clm not in cropList_combined_clm:
            thisCrop_fao = fao_area_nosgc.columns[-1]
            ydata_area = np.array(fao_area_nosgc[thisCrop_fao])
            ydata_prod = np.array(fao_prod_nosgc[thisCrop_fao])
        else:
            thisCrop_fao = fao_area.columns[c]
            ydata_area = np.array(fao_area[thisCrop_fao])
            ydata_prod = np.array(fao_prod[thisCrop_fao])
        
        # FAO EarthStat
        is_obs.append(True)
        if thisCrop_clm == "Total":
            area_tyx = earthstats_gd[this_earthstat_res].HarvestArea.sum(dim="crop").copy()
            prod_tyx = earthstats_gd[this_earthstat_res].Production.sum(dim="crop").copy()
        elif thisCrop_clm == "Total (no sgc)":
            area_tyx = earthstats_gd[this_earthstat_res].drop_sel(crop=['Sugarcane']).HarvestArea.sum(dim="crop").copy()
            prod_tyx = earthstats_gd[this_earthstat_res].drop_sel(crop=['Sugarcane']).Production.sum(dim="crop").copy()
        elif thisCrop_clm == "Total (grains)":
            area_tyx = earthstats_gd[this_earthstat_res].drop_sel(crop=['Sugarcane', 'Cotton']).HarvestArea.sum(dim="crop").copy()
            prod_tyx = earthstats_gd[this_earthstat_res].drop_sel(crop=['Sugarcane', 'Cotton']).Production.sum(dim="crop").copy()
        else:
            area_tyx = earthstats_gd[this_earthstat_res].HarvestArea.sel(crop=thisCrop_clm).copy()
            prod_tyx = earthstats_gd[this_earthstat_res].Production.sel(crop=thisCrop_clm).copy()
        ts_area_y = area_tyx.sum(dim=["lat","lon"]).values
        ydata_area = np.stack((ydata_area,
                            ts_area_y))
        ts_prod_y = 1e-6*prod_tyx.sum(dim=["lat","lon"]).values
        ydata_prod = np.stack((ydata_prod,
                            ts_prod_y))
        
        # CLM outputs
        for min_viable_hui in min_viable_hui_list:
            
            for i, (casename, case) in enumerate(cases.items()):
                lu_ds = reses[case['res']]['ds']
                is_obs.append(False)
            
                # Area
                ts_area_y = get_CLM_ts_area_y(case, lu_ds, thisCrop_clm, cropList_combined_clm)
                ydata_area = np.concatenate((ydata_area,
                                            np.expand_dims(ts_area_y.values, axis=0)),
                                            axis=0)
                
                # Production
                ts_prod_y = get_CLM_ts_prod_y(case, lu_ds, use_annual_yields, min_viable_hui, mxmats, thisCrop_clm, cropList_combined_clm)
                ydata_prod = np.concatenate((ydata_prod,
                                            np.expand_dims(ts_prod_y.values, axis=0)),
                                            axis=0)
                
                # Get yield time dimension name
                non_Crop_dims = [x for x in case['ds']['ts_prod_yc'].dims if x != "Crop"]
                if len(non_Crop_dims) != 1:
                    raise RuntimeError(f"Expected one non-Crop dimension of case['ds']['ts_prod_yc']; found {len(non_Crop_dims)}: {non_Crop_dims}")
                yield_time_dim = non_Crop_dims[0]

        # Convert ha to Mha
        ydata_area *= 1e-6
            
        # Calculate FAO* yields
        ydata_yield = ydata_prod / ydata_area
            
        # Get stats
        
        # Get shifted data
        inds_obs = [i for i,x in enumerate(is_obs) if x]
        inds_sim = [i for i,x in enumerate(is_obs) if not x]
        ydata_area_shiftL = cc.shift_sim_timeseries(ydata_area, "L", inds_obs, inds_sim)
        ydata_area_shiftR = cc.shift_sim_timeseries(ydata_area, "R", inds_obs, inds_sim)
        ydata_prod_shiftL = cc.shift_sim_timeseries(ydata_prod, "L", inds_obs, inds_sim)
        ydata_prod_shiftR = cc.shift_sim_timeseries(ydata_prod, "R", inds_obs, inds_sim)
        ydata_yield_shiftL = cc.shift_sim_timeseries(ydata_yield, "L", inds_obs, inds_sim)
        ydata_yield_shiftR = cc.shift_sim_timeseries(ydata_yield, "R", inds_obs, inds_sim)
        ydata_yield_biased = ydata_yield
        ydata_yield_biased_shiftL = ydata_yield_shiftL
        ydata_yield_biased_shiftR = ydata_yield_shiftR
        
        # Now ignore the outer timesteps to ensure the same years are being considered
        yearList_shifted = yearList[1:-1]
        ydata_area = ydata_area[:,1:-1]
        ydata_prod = ydata_prod[:,1:-1]
        ydata_yield = ydata_yield[:,1:-1]

        # Get detrended data
        if w == 0:
            r = 1
            ydata_yield_debiased_dt = signal.detrend(ydata_yield, axis=1)
            ydata_yield_debiased_shiftL_dt = signal.detrend(ydata_yield_shiftL, axis=1)
            ydata_yield_debiased_shiftR_dt = signal.detrend(ydata_yield_shiftR, axis=1)
            ydata_yield_bias = np.mean(ydata_yield, axis=1, keepdims=True)
            ydata_yield_biased_dt = ydata_yield_debiased_dt + ydata_yield_bias
            ydata_yield_biased_shiftL_dt = ydata_yield_debiased_shiftL_dt + ydata_yield_bias
            ydata_yield_biased_shiftR_dt = ydata_yield_debiased_shiftR_dt + ydata_yield_bias
        elif w==-1:
            raise RuntimeError("Specify w ≥ 0")
        else:
            r = cc.get_window_radius(w)
            center0 = True
            ydata_yield_debiased_dt = cc.christoph_detrend(ydata_yield, w, center0=center0)
            ydata_yield_debiased_shiftL_dt = cc.christoph_detrend(ydata_yield_shiftL, w, center0=center0)
            ydata_yield_debiased_shiftR_dt = cc.christoph_detrend(ydata_yield_shiftR, w, center0=center0)
            center0 = False
            ydata_yield_biased_dt = cc.christoph_detrend(ydata_yield, w, center0=center0)
            ydata_yield_biased_shiftL_dt = cc.christoph_detrend(ydata_yield_shiftL, w, center0=center0)
            ydata_yield_biased_shiftR_dt = cc.christoph_detrend(ydata_yield_shiftR, w, center0=center0)
        if remove_scatter_bias:
            ydata_yield_dt = ydata_yield_debiased_dt
            ydata_yield_shiftL_dt = ydata_yield_debiased_shiftL_dt
            ydata_yield_shiftR_dt = ydata_yield_debiased_shiftR_dt
        else:
            ydata_yield_dt = ydata_yield_biased_dt
            ydata_yield_shiftL_dt = ydata_yield_biased_shiftL_dt
            ydata_yield_shiftR_dt = ydata_yield_biased_shiftR_dt
        yearList_shifted_dt = yearList_shifted[r:-r]
        ydata_yield_sdt = signal.detrend(ydata_yield, axis=1) + np.mean(ydata_yield, axis=1, keepdims=True)
        
        # Restrict non-detrended data to years of interest
        if plot_y1 < yearList_shifted_dt[0]:
            raise RuntimeError(f"plot_y1 {plot_y1} is before the beginning of yearList_shifted_dt {yearList_shifted_dt[0]}")
        if plot_yN > yearList_shifted_dt[-1]:
            raise RuntimeError(f"plot_yN {plot_yN} is after the end of yearList_shifted_dt {yearList_shifted_dt[-1]}")
        yearList_shifted_ok = np.where((yearList_shifted >= plot_y1) & (yearList_shifted <= plot_yN))[0]
        ydata_area = ydata_area[:,yearList_shifted_ok]
        ydata_area_shiftL = ydata_area_shiftL[:,yearList_shifted_ok]
        ydata_area_shiftR = ydata_area_shiftR[:,yearList_shifted_ok]
        ydata_prod = ydata_prod[:,yearList_shifted_ok]
        ydata_prod_shiftL = ydata_prod_shiftL[:,yearList_shifted_ok]
        ydata_prod_shiftR = ydata_prod_shiftR[:,yearList_shifted_ok]
        ydata_yield = ydata_yield[:,yearList_shifted_ok]
        ydata_yield_shiftL = ydata_yield_shiftL[:,yearList_shifted_ok]
        ydata_yield_shiftR = ydata_yield_shiftR[:,yearList_shifted_ok]
        yearList_shifted_dt_ok = np.where((yearList_shifted_dt >= plot_y1) & (yearList_shifted_dt <= plot_yN))[0]
        ydata_yield_dt = ydata_yield_dt[:,yearList_shifted_dt_ok]
        ydata_yield_shiftL_dt = ydata_yield_shiftL_dt[:,yearList_shifted_dt_ok]
        ydata_yield_shiftR_dt = ydata_yield_shiftR_dt[:,yearList_shifted_dt_ok]
        ydata_yield_biased = ydata_yield_biased[:,yearList_shifted_ok]
        ydata_yield_biased_shiftL = ydata_yield_biased_shiftL[:,yearList_shifted_ok]
        ydata_yield_biased_shiftR = ydata_yield_biased_shiftR[:,yearList_shifted_ok]
        ydata_yield_biased_dt = ydata_yield_biased_dt[:,yearList_shifted_dt_ok]
        ydata_yield_biased_shiftL_dt = ydata_yield_biased_shiftL_dt[:,yearList_shifted_dt_ok]
        ydata_yield_biased_shiftR_dt = ydata_yield_biased_shiftR_dt[:,yearList_shifted_dt_ok]
        
        bias0 = None
        for o in inds_obs:
            this_obs = fig_caselist[o]
            if this_obs != obs_for_fig:
                continue
            if verbose:
                print(f"Comparing to {this_obs}:")
            
            bias = cc.get_timeseries_bias(ydata_yield_dt[inds_sim,:], ydata_yield_dt[o,:], fig_caselist, weights=ydata_prod[o,:])
            if this_obs == obs_for_fig:
                bias0 = bias
            if verbose:
                for i, casename in enumerate(fig_caselist[2:]):
                    print(f"   {thisCrop_clm} bias MM window={w} weighted {casename}: {np.round(bias[i], bias_round)}")
            
            # corrcoeff = [stats.linregress(ydata_yield[o,:], ydata_yield[x,:]) for x in np.arange(2, ydata_yield.shape[0])]
            # for i, casename in enumerate(fig_caselist[2:]):
            #     print(f"   {thisCrop_clm} r orig unweighted {casename}: {np.round(corrcoeff[i].rvalue, corrcoef_round)}")
                
            # corrcoeff = [stats.linregress(ydata_yield_sdt[o,r:-r], ydata_yield_sdt[x,r:-r]) for x in inds_sim]
            # for i, casename in enumerate(fig_caselist[2:]):
            #     print(f"   {thisCrop_clm} r signal.detrend() unweighted {casename}: {np.round(corrcoeff[i].rvalue, corrcoef_round)}")
            
            # Weights should never be shifted, as they are observed values.
            weights = ydata_prod[o,:]
            # Actually, no: I don't think Müller et al. (2017) weighted the global numbers.
            weights = np.full_like(weights, fill_value=1.0)
            
            corrcoeff = []
            p = []
            for x in inds_sim:
                rval, pval = cc.weighted_pearsons_r(ydata_yield_dt[o,:], ydata_yield_dt[x,:], weights)
                corrcoeff.append(rval)
                p.append(pval)
            if verbose:
                for i, casename in enumerate(fig_caselist[2:]):
                    print(f"   {thisCrop_clm} r MM window={w} weighted {casename}: {np.round(corrcoeff[i], corrcoef_round)}, p = {p[i]}")
            if this_obs == obs_for_fig:
                corrcoef_ref = corrcoeff
                p_ref = p
                
            corrcoeffL = []
            pL = []
            for x in inds_sim:
                rval, pval = cc.weighted_pearsons_r(ydata_yield_shiftL_dt[o,:], ydata_yield_shiftL_dt[x,:], weights)
                corrcoeffL.append(rval)
                pL.append(pval)
            if verbose:
                for i, casename in enumerate(fig_caselist[2:]):
                    print(f"   {thisCrop_clm} r MM window={w} unweighted shift LEFT {casename}: {np.round(corrcoeffL[i], corrcoef_round)}, p = {pL[i]}")
            if this_obs == obs_for_fig:
                corrcoefL_ref = corrcoeffL
                pL_ref = pL
            
            corrcoeffR = []
            pR = []
            for x in inds_sim:
                rval, pval = cc.weighted_pearsons_r(ydata_yield_shiftR_dt[o,:], ydata_yield_shiftR_dt[x,:], weights)
                corrcoeffR.append(rval)
                pR.append(pval)
            if verbose:
                for i, casename in enumerate(fig_caselist[2:]):
                    print(f"   {thisCrop_clm} r MM window={w} unweighted shift RIGHT {casename}: {np.round(corrcoeffR[i], corrcoef_round)}, p = {pR[i]}")
            if this_obs == obs_for_fig:
                corrcoefR_ref = corrcoeffR
                pR_ref = pR
        
        ydata_area_touse = ydata_area.copy()
        ydata_prod_touse = ydata_prod.copy()
        ydata_yield_touse = ydata_yield.copy()
        ydata_yield_dt_touse = ydata_yield_dt.copy()
        ydata_yield_biased_touse = ydata_yield.copy()
        ydata_yield_biased_dt_touse = ydata_yield_biased_dt.copy()
        corrcoef_ref_touse = corrcoef_ref.copy()
        p_ref_touse = p_ref.copy()
        shift_symbols = []
        for i,s in enumerate(inds_sim):
            Lshift_better = corrcoefL_ref[i] >= 0.3 + corrcoef_ref[i]
            Rshift_better = corrcoefR_ref[i] >= 0.3 + corrcoef_ref[i]
            if Lshift_better and Rshift_better:
                if corrcoefL_ref[i] > corrcoefR_ref[i]:
                    Rshift_better = False
                else:
                    Lshift_better = False
            if Lshift_better:
                if verbose:
                    print(f"Shifting {fig_caselist[s]} sim yield 1 year left")
                ydata_area_touse[s,:] = ydata_area_shiftL[s,:]
                ydata_prod_touse[s,:] = ydata_prod_shiftL[s,:]
                ydata_yield_touse[s,:] = ydata_yield_shiftL[s,:]
                ydata_yield_dt_touse[s,:] = ydata_yield_shiftL_dt[s,:]
                ydata_yield_biased_touse[s,:] = ydata_yield_biased_shiftL[s,:]
                ydata_yield_biased_dt_touse[s,:] = ydata_yield_biased_shiftL_dt[s,:]
                corrcoef_ref_touse[i] = corrcoefL_ref[i]
                p_ref_touse[i] = pL_ref[i]
                shift_symbols.append("$^L$")
            elif Rshift_better:
                if verbose:
                    print(f"Shifting {fig_caselist[s]} sim yield 1 year right")
                ydata_area_touse[s,:] = ydata_area_shiftR[s,:]
                ydata_prod_touse[s,:] = ydata_prod_shiftR[s,:]
                ydata_yield_touse[s,:] = ydata_yield_shiftR[s,:]
                ydata_yield_dt_touse[s,:] = ydata_yield_shiftR_dt[s,:]
                ydata_yield_biased_touse[s,:] = ydata_yield_biased_shiftR[s,:]
                ydata_yield_biased_dt_touse[s,:] = ydata_yield_biased_shiftR_dt[s,:]
                corrcoef_ref_touse[i] = corrcoefR_ref[i]
                p_ref_touse[i] = pR_ref[i]
                shift_symbols.append("$^R$")
            else:
                shift_symbols.append("")

        # Get shifted bias
        o = fig_caselist.index(obs_for_fig)
        bias_shifted = cc.get_timeseries_bias(ydata_yield_biased_dt_touse[inds_sim,:], ydata_yield_biased_dt_touse[o,:], fig_caselist, weights=ydata_prod[o,:])
        bias_shiftL = cc.get_timeseries_bias(ydata_yield_biased_shiftL_dt[inds_sim,:], ydata_yield_biased_dt_touse[o,:], fig_caselist, weights=ydata_prod[o,:])
        bias_shiftR = cc.get_timeseries_bias(ydata_yield_biased_shiftR_dt[inds_sim,:], ydata_yield_biased_dt_touse[o,:], fig_caselist, weights=ydata_prod[o,:])
        
        # Make plots for this crop
        if not noFigs:
            subplot_str = chr(ord('`') + c+1) # or ord('@') for capital
            no_xlabel = c < nx*(ny - 1) and not (nx==3 and c==5 and len(cropList_combined_clm + [extra])==8)
            if no_xlabel:
                xlabel = None
            else:
                xlabel = "Year"
            no_ylabel = c%nx
            label_yield = "Global yield (t/ha)"
            if no_ylabel:
                ylabel_area = None
                ylabel_prod = None
                ylabel_yield = None
            else:
                ylabel_area = "Global area (Mha)"
                ylabel_prod = "Global production (Mt)"
                ylabel_yield = label_yield
            make_1plot_lines(ax_lines_area, ydata_area_touse, fig_caselist, thisCrop_clm, ylabel_area, xlabel, plot_y1, plot_yN, subplot_label=subplot_str, ny=ny)
            if include_shiftsens:
                make_1plot_lines(ax_lines_area_orig, ydata_area, fig_caselist, thisCrop_clm, ylabel_area, xlabel, plot_y1, plot_yN, subplot_label=subplot_str, ny=ny)
                make_1plot_lines(ax_lines_area_shiftL, ydata_area_shiftL, fig_caselist, thisCrop_clm, ylabel_area, xlabel, plot_y1, plot_yN, subplot_label=subplot_str, ny=ny)
                make_1plot_lines(ax_lines_area_shiftR, ydata_area_shiftR, fig_caselist, thisCrop_clm, ylabel_area, xlabel, plot_y1, plot_yN, subplot_label=subplot_str, ny=ny)
            if no_xlabel:
                xlabel = None
            elif yield_time_dim == "Year":
                xlabel = "Year"
            else:
                xlabel = f"Obs year, sim {yield_time_dim.lower()}"
            noshift_symbols = ["", ""]
            shiftL_symbols = ["$^L$", "$^L$"]
            shiftR_symbols = ["$^R$", "$^R$"]
            make_1plot_lines(ax_lines_prod, ydata_prod_touse, fig_caselist, thisCrop_clm, ylabel_prod, xlabel, plot_y1, plot_yN, shift_symbols=shift_symbols, subplot_label=subplot_str, ny=ny)
            if len(min_viable_hui_list) > 1:
                bias_shifted = None
            make_1plot_lines(ax_lines_yield, ydata_yield_touse, fig_caselist, thisCrop_clm, ylabel_yield, xlabel, plot_y1, plot_yN, stats2=bias_shifted, stats_round=bias_round, shift_symbols=shift_symbols, subplot_label=subplot_str, ny=ny)
            make_1plot_lines(ax_lines_yield_dt, ydata_yield_dt_touse, fig_caselist, thisCrop_clm, ylabel_yield, xlabel, plot_y1, plot_yN, stats2=bias_shifted, stats_round=bias_round, shift_symbols=shift_symbols, subplot_label=subplot_str, ny=ny)
            
            if include_shiftsens:
                make_1plot_lines(ax_lines_prod_orig, ydata_prod, fig_caselist, thisCrop_clm, ylabel_prod, xlabel, plot_y1, plot_yN, shift_symbols=noshift_symbols, subplot_label=subplot_str, ny=ny)
                make_1plot_lines(ax_lines_prod_shiftL, ydata_prod_shiftL, fig_caselist, thisCrop_clm, ylabel_prod, xlabel, plot_y1, plot_yN, shift_symbols=shiftL_symbols, subplot_label=subplot_str, ny=ny)
                make_1plot_lines(ax_lines_prod_shiftR, ydata_prod_shiftR, fig_caselist, thisCrop_clm, ylabel_prod, xlabel, plot_y1, plot_yN, shift_symbols=shiftR_symbols, subplot_label=subplot_str, ny=ny)
                if len(min_viable_hui_list) > 1:
                    bias0 = None
                    bias_shiftL = None
                    bias_shiftR = None
                make_1plot_lines(ax_lines_yield_orig, ydata_yield, fig_caselist, thisCrop_clm, ylabel_yield, xlabel, plot_y1, plot_yN, stats2=bias0, stats_round=bias_round, shift_symbols=noshift_symbols, subplot_label=subplot_str, ny=ny)
                make_1plot_lines(ax_lines_yield_shiftL, ydata_yield_shiftL, fig_caselist, thisCrop_clm, ylabel_yield, xlabel, plot_y1, plot_yN, stats2=bias_shiftL, stats_round=bias_round, shift_symbols=shiftL_symbols, subplot_label=subplot_str, ny=ny)
                make_1plot_lines(ax_lines_yield_shiftR, ydata_yield_shiftR, fig_caselist, thisCrop_clm, ylabel_yield, xlabel, plot_y1, plot_yN, stats2=bias_shiftR, stats_round=bias_round, shift_symbols=shiftR_symbols, subplot_label=subplot_str, ny=ny)
                make_1plot_lines(ax_lines_yield_dt_orig, ydata_yield_dt, fig_caselist, thisCrop_clm, ylabel_yield, xlabel, plot_y1, plot_yN, stats2=bias0, stats_round=bias_round, shift_symbols=noshift_symbols, subplot_label=subplot_str, ny=ny)
                make_1plot_lines(ax_lines_yield_dt_shiftL, ydata_yield_shiftL_dt, fig_caselist, thisCrop_clm, ylabel_yield, xlabel, plot_y1, plot_yN, stats2=bias_shiftL, stats_round=bias_round, shift_symbols=shiftL_symbols, subplot_label=subplot_str, ny=ny)
                make_1plot_lines(ax_lines_yield_dt_shiftR, ydata_yield_shiftR_dt, fig_caselist, thisCrop_clm, ylabel_yield, xlabel, plot_y1, plot_yN, stats2=bias_shiftR, stats_round=bias_round, shift_symbols=shiftR_symbols, subplot_label=subplot_str, ny=ny)
            
            # Scatter plots
            if include_scatter:
                xlabel_yield_scatter = None
                ylabel_yield_scatter = None
                if ylabel_yield is not None and not (ny==3 and c!=3):
                    ylabel_yield_scatter = "Simulated " + ylabel_yield.lower()
                even_nx = nx%2==0
                bottommiddle_plot = not even_nx and c - (ny-1)*nx == 1
                if xlabel is not None and (even_nx or bottommiddle_plot):
                    xlabel_yield_scatter = "Observed " + label_yield.lower()
                make_1plot_scatter(ax_scatter_yield_dt, ydata_yield_dt_touse[o,:], ydata_yield_dt_touse[inds_sim,:], [fig_caselist[x] for x in inds_sim], thisCrop_clm, xlabel_yield_scatter, ylabel_yield_scatter, equalize_scatter_axes, bottommiddle_plot, stats2=corrcoef_ref_touse, stats_round=corrcoef_round, p=p_ref_touse, shift_symbols=shift_symbols, subplot_label=subplot_str, ny=ny)
                if include_shiftsens:
                    make_1plot_scatter(ax_scatter_yield_dt_orig, ydata_yield_dt[o,:], ydata_yield_dt[inds_sim,:], [fig_caselist[x] for x in inds_sim], thisCrop_clm, xlabel_yield_scatter, ylabel_yield_scatter, equalize_scatter_axes, bottommiddle_plot, stats2=corrcoef_ref, stats_round=corrcoef_round, p=p_ref_touse, shift_symbols=noshift_symbols, subplot_label=subplot_str, ny=ny)
                    make_1plot_scatter(ax_scatter_yield_dt_shiftL, ydata_yield_shiftL_dt[o,:], ydata_yield_shiftL_dt[inds_sim,:], [fig_caselist[x] for x in inds_sim], thisCrop_clm,xlabel_yield_scatter,  ylabel_yield_scatter, equalize_scatter_axes, bottommiddle_plot, stats2=corrcoeffL, stats_round=corrcoef_round, p=p_ref_touse, shift_symbols=shiftL_symbols, subplot_label=subplot_str, ny=ny)
                    make_1plot_scatter(ax_scatter_yield_dt_shiftR, ydata_yield_shiftR_dt[o,:], ydata_yield_shiftR_dt[inds_sim,:], [fig_caselist[x] for x in inds_sim], thisCrop_clm, xlabel_yield_scatter, ylabel_yield_scatter, equalize_scatter_axes, bottommiddle_plot, stats2=corrcoeffR, stats_round=corrcoef_round, p=p_ref_touse, shift_symbols=shiftR_symbols, subplot_label=subplot_str, ny=ny)
            
    # Finish up and save
    scatter_title_paren = "detrended"
    if remove_scatter_bias:
        scatter_title_paren += ", bias removed"
    if not noFigs:
        print("Finishing and saving...")
        finishup_lines(c, ny, nx, axes_lines_area, f_lines_area, "Global crop area", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
        finishup_lines(c, ny, nx, axes_lines_prod, f_lines_prod, "Global crop production", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
        finishup_lines(c, ny, nx, axes_lines_yield, f_lines_yield, "Global crop yield", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
        finishup_lines(c, ny, nx, axes_lines_yield_dt, f_lines_yield_dt, "Global crop yield (detrended)", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
        if include_shiftsens:
            finishup_lines(c, ny, nx, axes_lines_area_orig, f_lines_area_orig, "Global crop area no-shift", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_area_shiftL, f_lines_area_shiftL, "Global crop area shiftL", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_area_shiftR, f_lines_area_shiftR, "Global crop area shiftR", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_prod_orig, f_lines_prod_orig, "Global crop production no-shift", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_prod_shiftL, f_lines_prod_shiftL, "Global crop production shiftL", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_prod_shiftR, f_lines_prod_shiftR, "Global crop production shiftR", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_yield_orig, f_lines_yield_orig, "Global crop yield no-shift", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_yield_shiftL, f_lines_yield_shiftL, "Global crop yield shiftL", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_yield_shiftR, f_lines_yield_shiftR, "Global crop yield shiftR", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_yield_dt_orig, f_lines_yield_dt_orig, "Global crop yield (detrended) no-shift", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_yield_dt_shiftL, f_lines_yield_dt_shiftL, "Global crop yield (detrended) shiftL", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
            finishup_lines(c, ny, nx, axes_lines_yield_dt_shiftR, f_lines_yield_dt_shiftR, "Global crop yield (detrended) shiftR", outDir_figs, mxmat_limited, fig_caselist, min_viable_hui_list=min_viable_hui_list)
        if include_scatter:
            finishup_scatter(c, ny, nx, axes_scatter_yield_dt, f_scatter_yield_dt, f"Global crop yield ({scatter_title_paren})", outDir_figs, mxmat_limited, fig_caselist, inds_sim)
            if include_shiftsens:
                finishup_scatter(c, ny, nx, axes_scatter_yield_dt_orig, f_scatter_yield_dt_orig, f"Global crop yield ({scatter_title_paren}) no-shift", outDir_figs, mxmat_limited, fig_caselist, inds_sim)
                finishup_scatter(c, ny, nx, axes_scatter_yield_dt_shiftL, f_scatter_yield_dt_shiftL, f"Global crop yield ({scatter_title_paren}) shiftL", outDir_figs, mxmat_limited, fig_caselist, inds_sim)
                finishup_scatter(c, ny, nx, axes_scatter_yield_dt_shiftR, f_scatter_yield_dt_shiftR, f"Global crop yield ({scatter_title_paren}) shiftR", outDir_figs, mxmat_limited, fig_caselist, inds_sim)

    print("Done.")
    return cases

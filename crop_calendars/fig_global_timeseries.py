# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import cropcal_module as cc

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

import numpy as np
import xarray as xr
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import collections as mplcollections
from matplotlib import cm

def get_figs_axes(ny, nx, figsize):
    f_list, axes_list = plt.subplots(ny, nx, figsize=figsize)
    axes_list = axes_list.flatten()
    return f_list, axes_list


def make_1crop_lines(ax_this, ydata_this, caselist, thisCrop_clm, units, xlabel, y1, yN, stats2=None, stats_round=None, shift_symbols=None):
    da = xr.DataArray(data = ydata_this,
                            coords = {'Case': caselist,
                                      xlabel: np.arange(y1,yN+1)})
    
    light_gray = [x/255 for x in [148, 148, 148]]
    dark_gray = [x/255 for x in [64, 64, 74]]
    grays = [dark_gray, light_gray]
    
    for i, casename in enumerate(caselist):
        if i <= 1:
            color = grays[i]
        elif "CLM Default" in casename:
            color = [x/255 for x in [92, 219, 219]]
        elif "Prescribed Calendars" in casename:
            color = [x/255 for x in [250, 102, 240]]
        elif "Prescribed Sowing" in casename:
            color = [x/255 for x in [150, 110, 0]]
        elif "Prescribed Maturity" in casename:
            color = [x/255 for x in [128,0,0]]
        elif i==4:
            # Purple more distinct from my light gray
            color = [x/255 for x in [133, 92, 255]]
        else:
            color = cm.Dark2(i-2)
        if (casename == "CLM Default" and "Original baseline" in caselist) or (casename in ["Prescribed Sowing", "Prescribed Maturity"]) or "min. HUI 1" in casename:
            linestyle = ":"
        elif "min. HUI 0" in casename:
            linestyle = "--"
        else:
            linestyle = "-"
        da.isel(Case=i).plot.line(x=xlabel, ax=ax_this, 
                                  color=color, linestyle=linestyle,
                                  linewidth=3)
    
    thisTitle = thisCrop_clm
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
    ax_this.title.set_size(32)
    ax_this.tick_params(axis='both', which='major', labelsize=20)
    ax_this.set_xlabel(xlabel, fontsize=24)
    ax_this.set_ylabel(units, fontsize=24)
    if ax_this.get_legend():
        ax_this.get_legend().remove()


def make_1crop_scatter(ax_this, xdata, ydata_this, caselist, thisCrop_clm, axlabel, equalize_scatter_axes, stats2=None, stats_round=None, shift_symbols=None):
    
    for i, casename in enumerate(caselist):
        if casename == "CLM Default":
            color = [x/255 for x in [92, 219, 219]]
        elif casename == "Prescribed Calendars":
            color = [x/255 for x in [250, 102, 240]]
        elif casename == "Prescribed Sowing":
            color = [x/255 for x in [150, 110, 0]]
        elif casename == "Prescribed Maturity":
            color = [x/255 for x in [128,0,0]]
        elif i==2:
            # Purple more distinct from my light gray
            color = [x/255 for x in [133, 92, 255]]
        else:
            color = cm.Dark2(i)
        if casename in ["Prescribed Sowing", "Prescribed Maturity"]:
            facecolors = 'none'
        else:
            facecolors = color
        plt.sca(ax_this)
        plt.scatter(xdata, ydata_this[i,:], color=color, s=100, alpha=0.8, facecolors=facecolors)
        m, b = np.polyfit(xdata, ydata_this[i,:], 1)
        plt.plot(xdata, m*xdata+b, color=color)
    
    if equalize_scatter_axes:
        xlim = ax_this.get_xlim()
        ylim = ax_this.get_ylim()
        newlim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        ax_this.set_xlim(newlim)
        ax_this.set_ylim(newlim)
    
    thisTitle = thisCrop_clm
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
    ax_this.title.set_size(32)
    ax_this.tick_params(axis='both', which='major', labelsize=20)
    ax_this.set_xlabel(axlabel, fontsize=24)
    ax_this.set_ylabel(axlabel, fontsize=24)
    if ax_this.get_legend():
        ax_this.get_legend().remove()


def finishup_allcrops_lines(c, ny, nx, axes_this, f_this, suptitle, outDir_figs, mxmat_limited, fig_caselist):
    # Delete unused axes, if any
    for a in np.arange(c+1, ny*nx):
        f_this.delaxes(axes_this[a])
        
    if mxmat_limited:
        suptitle += " (limited season length)"
        
    f_this.suptitle(suptitle,
                    x = 0.1, horizontalalignment = 'left',
                    fontsize=24)
    
    f_this.legend(handles = axes_this[0].lines,
                  labels = fig_caselist,
                  loc = "upper center",
                  ncol = int(np.ceil(len(fig_caselist)/2)),
                  fontsize=28)

    f_this.savefig(outDir_figs + "Timeseries " + suptitle + " by crop.pdf",
                   bbox_inches='tight')
    plt.close(f_this)


def finishup_allcrops_scatter(c, ny, nx, axes_this, f_this, suptitle, outDir_figs, mxmat_limited, fig_caselist, inds_sim):
    # Delete unused axes, if any
    for a in np.arange(c+1, ny*nx):
        f_this.delaxes(axes_this[a])
        
    if mxmat_limited:
        suptitle += " (limited season length)"
        
    f_this.suptitle(suptitle,
                    x = 0.1, horizontalalignment = 'left',
                    fontsize=24)
    
    # Get the handles of just the points (i.e., not including regression lines)    
    legend_handles = [x for x in axes_this[0].get_children() if isinstance(x, mplcollections.PathCollection)]
    
    f_this.legend(handles = legend_handles,
                  labels = [fig_caselist[x] for x in inds_sim],
                  loc = "upper center",
                  ncol = int(np.ceil(len(fig_caselist)/2)),
                  fontsize=28)

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
        elif thisCrop_clm == "Total (grains only)":
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
    elif thisCrop_clm == "Total (grains only)":
        ts_prod_y = case['ds'].drop_sel(Crop=['Sugarcane', 'Cotton', 'Total'])['ts_prod_yc'].sum(dim="Crop").copy()
    else:
        ts_prod_y = case['ds']['ts_prod_yc'].sel(Crop=thisCrop_clm).copy()
    
    return ts_prod_y



def global_timeseries(cases, cropList_combined_clm, earthstats_gd, fao_area, fao_area_nosgc, fao_prod, fao_prod_nosgc, outDir_figs, reses, yearList, \
    equalize_scatter_axes=False, extra="Total (grains only)", figsize=(35, 18), include_scatter=True, include_shiftsens=True, min_viable_hui_list="ggcmi3", mxmats=None, noFigs=False, ny=2, nx=4, obs_for_fig="FAOSTAT", plot_y1=1980, plot_yN=2010, stats_round=3, use_annual_yields=False, verbose=False, w=5):
    
    if not isinstance(min_viable_hui_list, list):
        min_viable_hui_list = [min_viable_hui_list]
    
    # Allow multiple min_viable_hui values only if this is the case list
    if len(min_viable_hui_list) > 1 and [x for x in cases.keys()] != ['CLM Default', 'Prescribed Calendars']:
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
            f_scatter_yield_dt, axes_scatter_yield_dt = get_figs_axes(ny, nx, figsize)
            if include_shiftsens:
                f_scatter_yield_dt_orig, axes_scatter_yield_dt_orig = get_figs_axes(ny, nx, figsize)
                f_scatter_yield_dt_shiftL, axes_scatter_yield_dt_shiftL = get_figs_axes(ny, nx, figsize)
                f_scatter_yield_dt_shiftR, axes_scatter_yield_dt_shiftR = get_figs_axes(ny, nx, figsize)

    # Get list 
    fig_caselist = ["FAOSTAT"]
    this_earthstat_res = "f09_g17"
    fig_caselist += [f"EarthStat ({this_earthstat_res})"]
    for min_viable_hui in min_viable_hui_list:
        for (casename, case) in cases.items():
            fig_casename = casename
            if min_viable_hui != "ggcmi3":
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
        elif thisCrop_clm == "Total (grains only)":
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
        
        # Now ignore the outer timesteps to ensure the same years are being considered
        yearList_shifted = yearList[1:-1]
        ydata_area = ydata_area[:,1:-1]
        ydata_prod = ydata_prod[:,1:-1]
        ydata_yield = ydata_yield[:,1:-1]

        # Get detrended data
        if w == 0:
            r = 1
            ydata_yield_dt = signal.detrend(ydata_yield, axis=1) + np.mean(ydata_yield, axis=1, keepdims=True)
            ydata_yield_shiftL_dt = signal.detrend(ydata_yield_shiftL, axis=1) + np.mean(ydata_yield, axis=1, keepdims=True)
            ydata_yield_shiftR_dt = signal.detrend(ydata_yield_shiftR, axis=1) + np.mean(ydata_yield, axis=1, keepdims=True)
        elif w==-1:
            raise RuntimeError("Specify w ≥ 0")
        else:
            r = cc.get_window_radius(w)
            ydata_yield_dt = cc.christoph_detrend(ydata_yield, w)
            ydata_yield_shiftL_dt = cc.christoph_detrend(ydata_yield_shiftL, w)
            ydata_yield_shiftR_dt = cc.christoph_detrend(ydata_yield_shiftR, w)
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
                    print(f"   {thisCrop_clm} bias MM window={w} weighted {casename}: {np.round(bias[i], stats_round)}")
            
            # corrcoeff = [stats.linregress(ydata_yield[o,:], ydata_yield[x,:]) for x in np.arange(2, ydata_yield.shape[0])]
            # for i, casename in enumerate(fig_caselist[2:]):
            #     print(f"   {thisCrop_clm} r orig unweighted {casename}: {np.round(corrcoeff[i].rvalue, stats_round)}")
                
            # corrcoeff = [stats.linregress(ydata_yield_sdt[o,r:-r], ydata_yield_sdt[x,r:-r]) for x in inds_sim]
            # for i, casename in enumerate(fig_caselist[2:]):
            #     print(f"   {thisCrop_clm} r signal.detrend() unweighted {casename}: {np.round(corrcoeff[i].rvalue, stats_round)}")
            
            # Weights should never be shifted, as they are observed values.
            weights = ydata_prod[o,:]
            
            corrcoeff = [cc.weighted_pearsons_r(ydata_yield_dt[o,:], ydata_yield_dt[x,:], weights) for x in inds_sim]
            if verbose:
                for i, casename in enumerate(fig_caselist[2:]):
                    print(f"   {thisCrop_clm} r MM window={w} weighted {casename}: {np.round(corrcoeff[i], stats_round)}")
            if this_obs == obs_for_fig:
                corrcoef_ref = corrcoeff
                
            corrcoeffL = [cc.weighted_pearsons_r(ydata_yield_shiftL_dt[o,:], ydata_yield_shiftL_dt[x,:], weights) for x in inds_sim]
            if verbose:
                for i, casename in enumerate(fig_caselist[2:]):
                    print(f"   {thisCrop_clm} r MM window={w} unweighted shift LEFT {casename}: {np.round(corrcoeffL[i], stats_round)}")
            if this_obs == obs_for_fig:
                corrcoefL_ref = corrcoeffL
            
            corrcoeffR = [cc.weighted_pearsons_r(ydata_yield_shiftR_dt[o,:], ydata_yield_shiftR_dt[x,:], weights) for x in inds_sim]
            if verbose:
                for i, casename in enumerate(fig_caselist[2:]):
                    print(f"   {thisCrop_clm} r MM window={w} unweighted shift RIGHT {casename}: {np.round(corrcoeffR[i], stats_round)}")
            if this_obs == obs_for_fig:
                corrcoefR_ref = corrcoeffR
        
        ydata_area_touse = ydata_area.copy()
        ydata_prod_touse = ydata_prod.copy()
        ydata_yield_touse = ydata_yield.copy()
        ydata_yield_dt_touse = ydata_yield_dt.copy()
        corrcoef_ref_touse = corrcoef_ref.copy()
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
                corrcoef_ref_touse[i] = corrcoefL_ref[i]
                shift_symbols.append("$^L$")
            elif Rshift_better:
                if verbose:
                    print(f"Shifting {fig_caselist[s]} sim yield 1 year right")
                ydata_area_touse[s,:] = ydata_area_shiftR[s,:]
                ydata_prod_touse[s,:] = ydata_prod_shiftR[s,:]
                ydata_yield_touse[s,:] = ydata_yield_shiftR[s,:]
                ydata_yield_dt_touse[s,:] = ydata_yield_shiftR_dt[s,:]
                corrcoef_ref_touse[i] = corrcoefR_ref[i]
                shift_symbols.append("$^R$")
            else:
                shift_symbols.append("")

        # Get shifted bias
        o = fig_caselist.index(obs_for_fig)
        bias_shifted = cc.get_timeseries_bias(ydata_yield_dt_touse[inds_sim,:], ydata_yield_dt_touse[o,:], fig_caselist, weights=ydata_prod[o,:])
        bias_shiftL = cc.get_timeseries_bias(ydata_yield_shiftL_dt[inds_sim,:], ydata_yield_dt_touse[o,:], fig_caselist, weights=ydata_prod[o,:])
        bias_shiftR = cc.get_timeseries_bias(ydata_yield_shiftR_dt[inds_sim,:], ydata_yield_dt_touse[o,:], fig_caselist, weights=ydata_prod[o,:])
        
        # Make plots for this crop
        if not noFigs:
            xlabel = "Year"
            make_1crop_lines(ax_lines_area, ydata_area_touse, fig_caselist, thisCrop_clm, "Mha", xlabel, plot_y1, plot_yN)
            if include_shiftsens:
                make_1crop_lines(ax_lines_area_orig, ydata_area, fig_caselist, thisCrop_clm, "Mha", xlabel, plot_y1, plot_yN)
                make_1crop_lines(ax_lines_area_shiftL, ydata_area_shiftL, fig_caselist, thisCrop_clm, "Mha", xlabel, plot_y1, plot_yN)
                make_1crop_lines(ax_lines_area_shiftR, ydata_area_shiftR, fig_caselist, thisCrop_clm, "Mha", xlabel, plot_y1, plot_yN)
            if yield_time_dim == "Year":
                xlabel = "Year"
            else:
                xlabel = f"Obs year, sim {yield_time_dim.lower()}"
            noshift_symbols = ["", ""]
            shiftL_symbols = ["$^L$", "$^L$"]
            shiftR_symbols = ["$^R$", "$^R$"]
            make_1crop_lines(ax_lines_prod, ydata_prod_touse, fig_caselist, thisCrop_clm, "Mt", xlabel, plot_y1, plot_yN, shift_symbols=shift_symbols)
            make_1crop_lines(ax_lines_yield, ydata_yield_touse, fig_caselist, thisCrop_clm, "t/ha", xlabel, plot_y1, plot_yN, stats2=bias0, stats_round=stats_round, shift_symbols=shift_symbols)
            make_1crop_lines(ax_lines_yield_dt, ydata_yield_dt_touse, fig_caselist, thisCrop_clm, "t/ha", xlabel, plot_y1, plot_yN, stats2=bias0, stats_round=stats_round, shift_symbols=shift_symbols)
            
            if include_shiftsens:
                make_1crop_lines(ax_lines_prod_orig, ydata_prod, fig_caselist, thisCrop_clm, "Mt", xlabel, plot_y1, plot_yN, shift_symbols=noshift_symbols)
                make_1crop_lines(ax_lines_prod_shiftL, ydata_prod_shiftL, fig_caselist, thisCrop_clm, "Mt", xlabel, plot_y1, plot_yN, shift_symbols=shiftL_symbols)
                make_1crop_lines(ax_lines_prod_shiftR, ydata_prod_shiftR, fig_caselist, thisCrop_clm, "Mt", xlabel, plot_y1, plot_yN, shift_symbols=shiftR_symbols)
                make_1crop_lines(ax_lines_yield_orig, ydata_yield, fig_caselist, thisCrop_clm, "t/ha", xlabel, plot_y1, plot_yN, stats2=bias_shifted, stats_round=stats_round, shift_symbols=noshift_symbols)
                make_1crop_lines(ax_lines_yield_shiftL, ydata_yield_shiftL, fig_caselist, thisCrop_clm, "t/ha", xlabel, plot_y1, plot_yN, stats2=bias_shiftL, stats_round=stats_round, shift_symbols=shiftL_symbols)
                make_1crop_lines(ax_lines_yield_shiftR, ydata_yield_shiftR, fig_caselist, thisCrop_clm, "t/ha", xlabel, plot_y1, plot_yN, stats2=bias_shiftR, stats_round=stats_round, shift_symbols=shiftR_symbols)
                make_1crop_lines(ax_lines_yield_dt_orig, ydata_yield_dt, fig_caselist, thisCrop_clm, "t/ha", xlabel, plot_y1, plot_yN, stats2=bias_shifted, stats_round=stats_round, shift_symbols=noshift_symbols)
                make_1crop_lines(ax_lines_yield_dt_shiftL, ydata_yield_shiftL_dt, fig_caselist, thisCrop_clm, "t/ha", xlabel, plot_y1, plot_yN, stats2=bias_shiftL, stats_round=stats_round, shift_symbols=shiftL_symbols)
                make_1crop_lines(ax_lines_yield_dt_shiftR, ydata_yield_shiftR_dt, fig_caselist, thisCrop_clm, "t/ha", xlabel, plot_y1, plot_yN, stats2=bias_shiftR, stats_round=stats_round, shift_symbols=shiftR_symbols)
            
            # Scatter plots
            if include_scatter:
                make_1crop_scatter(ax_scatter_yield_dt, ydata_yield_dt_touse[o,:], ydata_yield_dt_touse[inds_sim,:], [fig_caselist[x] for x in inds_sim], thisCrop_clm, "t/ha", equalize_scatter_axes, stats2=corrcoef_ref_touse, stats_round=stats_round, shift_symbols=shift_symbols)
                if include_shiftsens:
                    make_1crop_scatter(ax_scatter_yield_dt_orig, ydata_yield_dt[o,:], ydata_yield_dt[inds_sim,:], [fig_caselist[x] for x in inds_sim], thisCrop_clm, "t/ha", equalize_scatter_axes, stats2=corrcoef_ref, stats_round=stats_round, shift_symbols=noshift_symbols)
                    make_1crop_scatter(ax_scatter_yield_dt_shiftL, ydata_yield_shiftL_dt[o,:], ydata_yield_shiftL_dt[inds_sim,:], [fig_caselist[x] for x in inds_sim], thisCrop_clm, "t/ha", equalize_scatter_axes, stats2=corrcoeffL, stats_round=stats_round, shift_symbols=shiftL_symbols)
                    make_1crop_scatter(ax_scatter_yield_dt_shiftR, ydata_yield_shiftR_dt[o,:], ydata_yield_shiftR_dt[inds_sim,:], [fig_caselist[x] for x in inds_sim], thisCrop_clm, "t/ha", equalize_scatter_axes, stats2=corrcoeffR, stats_round=stats_round, shift_symbols=shiftR_symbols)
            
    # Finish up and save
    if not noFigs:
        print("Finishing and saving...")
        finishup_allcrops_lines(c, ny, nx, axes_lines_area, f_lines_area, "Global crop area", outDir_figs, mxmat_limited, fig_caselist)
        finishup_allcrops_lines(c, ny, nx, axes_lines_prod, f_lines_prod, "Global crop production", outDir_figs, mxmat_limited, fig_caselist)
        finishup_allcrops_lines(c, ny, nx, axes_lines_yield, f_lines_yield, "Global crop yield", outDir_figs, mxmat_limited, fig_caselist)
        finishup_allcrops_lines(c, ny, nx, axes_lines_yield_dt, f_lines_yield_dt, "Global crop yield (detrended)", outDir_figs, mxmat_limited, fig_caselist)
        if include_shiftsens:
            finishup_allcrops_lines(c, ny, nx, axes_lines_area_orig, f_lines_area_orig, "Global crop area no-shift", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_area_shiftL, f_lines_area_shiftL, "Global crop area shiftL", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_area_shiftR, f_lines_area_shiftR, "Global crop area shiftR", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_prod_orig, f_lines_prod_orig, "Global crop production no-shift", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_prod_shiftL, f_lines_prod_shiftL, "Global crop production shiftL", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_prod_shiftR, f_lines_prod_shiftR, "Global crop production shiftR", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_yield_orig, f_lines_yield_orig, "Global crop yield no-shift", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_yield_shiftL, f_lines_yield_shiftL, "Global crop yield shiftL", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_yield_shiftR, f_lines_yield_shiftR, "Global crop yield shiftR", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_yield_dt_orig, f_lines_yield_dt_orig, "Global crop yield (detrended) no-shift", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_yield_dt_shiftL, f_lines_yield_dt_shiftL, "Global crop yield (detrended) shiftL", outDir_figs, mxmat_limited, fig_caselist)
            finishup_allcrops_lines(c, ny, nx, axes_lines_yield_dt_shiftR, f_lines_yield_dt_shiftR, "Global crop yield (detrended) shiftR", outDir_figs, mxmat_limited, fig_caselist)
        if include_scatter:
            finishup_allcrops_scatter(c, ny, nx, axes_scatter_yield_dt, f_scatter_yield_dt, "Global crop yield (detrended)", outDir_figs, mxmat_limited, fig_caselist, inds_sim)
            if include_shiftsens:
                finishup_allcrops_scatter(c, ny, nx, axes_scatter_yield_dt_orig, f_scatter_yield_dt_orig, "Global crop yield (detrended) no-shift", outDir_figs, mxmat_limited, fig_caselist, inds_sim)
                finishup_allcrops_scatter(c, ny, nx, axes_scatter_yield_dt_shiftL, f_scatter_yield_dt_shiftL, "Global crop yield (detrended) shiftL", outDir_figs, mxmat_limited, fig_caselist, inds_sim)
                finishup_allcrops_scatter(c, ny, nx, axes_scatter_yield_dt_shiftR, f_scatter_yield_dt_shiftR, "Global crop yield (detrended) shiftR", outDir_figs, mxmat_limited, fig_caselist, inds_sim)

    print("Done.")
    return cases

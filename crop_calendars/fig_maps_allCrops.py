# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import cropcal_module as cc
from cropcal_figs_module import *

import importlib
importlib.reload(sys.modules['cropcal_figs_module'])
from cropcal_figs_module import *
importlib.reload(cc)

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils
importlib.reload(utils)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import stats

fontsize = {}
fontsize['titles'] = 18
fontsize['axislabels'] = 14
fontsize['ticklabels'] = 14
fontsize['suptitle'] = 22


def get_underlay(this_ds, area_map_sum):
    dummy_var = 'gridcell'
    dummy_data = np.full_like(this_ds[dummy_var].values, 1.0)
    this_ds['is_land'] = xr.DataArray(data=dummy_data,
                                              coords=this_ds[dummy_var].coords)
    is_land = utils.grid_one_variable(this_ds, 'is_land')
    underlay = is_land.where(area_map_sum > 0)
    return underlay


def make_fig(thisVar, varInfo, cropList_combined_clm_nototal, ny, nx, ds_in, this_suptitle, is_diff, is_diffdiff, low_area_threshold_m2, croptitle_side, v, Nvars, fig, earthstats, posNeg, take_subcrop_sum, take_subcrop_wtdmean, rx2_da, rx_parent_casename):
    
    if rx2_da is not None and earthstats is not None:
        raise RuntimeError("How are both rx and earthstats Datasets specified?")
    
    time_dim = "time"
    if "time_dim" in varInfo:
        time_dim = varInfo['time_dim']
    
    if is_diff:
        if is_diffdiff:
            cmap = cropcal_colors['div_other_norm']
        elif np.any([x in thisVar for x in ["PROD", "YIELD", "IRRIG"]]):
            cmap = cropcal_colors['div_yieldirr']
        else:
            cmap = cropcal_colors['div_other_nonnorm']
    else:
        cmap = cropcal_colors['seq_other']
    
    if posNeg:
        if Nvars > 1:
            raise RuntimeError("posNeg not tested with Nvars > 1")
        cropList = ["Crops decreasing", "Crops increasing"]
    else:
        cropList = cropList_combined_clm_nototal

    for c, crop in enumerate(cropList):
        thisPlot = c*Nvars + v + 1
        ax = fig.add_subplot(ny, nx*Nvars, thisPlot, projection=ccrs.PlateCarree())
        
        # Which CFTs comprise this crop?
        if crop in cropList_combined_clm_nototal:
            where_thisCrop = np.where(ds_in['patches1d_itype_combinedCropCLM_str'] == crop)[0]
            theseCrops = np.unique(ds_in['patches1d_itype_veg_str'].isel(patch=where_thisCrop))
            this_ds = ds_in.isel(patch=where_thisCrop)
            if rx2_da is not None:
                this_rx2_da = rx2_da.sel(ivt_str=theseCrops)
        else:
            # Include all of the 6 explicitly-simulated crops
            if not posNeg:
                raise RuntimeError(f"{crop} is not in cropList_combined_clm_nototal but this isn't a posNeg figure")
            this_ds = ds_in
            theseCrops = list()
            for porc in cropList_combined_clm_nototal:
                tmp = list(np.unique(np.unique(ds_in['patches1d_itype_veg_str'].isel(patch=np.where(ds_in['patches1d_itype_combinedCropCLM_str'] == porc)[0]))))
                for tmpx in tmp:
                    if tmpx not in theseCrops:
                        theseCrops.append(tmpx)

        # Get area of these CFTs
        area_map = utils.grid_one_variable(this_ds.mean(dim="time"), "AREA_CFT", vegtype=list(theseCrops))
        if not posNeg:
            area_map_sum = area_map.sum(dim="ivt_str")

        # Grid data for those CFTs, getting their sum or weighted mean
        if rx2_da is None:
            this_map = utils.grid_one_variable(this_ds, thisVar, vegtype=list(theseCrops))
        else:
            this_map = utils.grid_one_variable(this_ds, thisVar.replace("ABSBIAS", "BIAS").replace("_BIAS", ""), vegtype=list(theseCrops))
        this_map_attrs = this_map.attrs
        if not posNeg:
            if take_subcrop_sum:
                this_map = this_map.sum(dim="ivt_str")
                if rx2_da is not None:
                    this_map -= this_rx2_da.sum(dim="ivt_str")
            elif take_subcrop_wtdmean:
                this_map = this_map.weighted(area_map.fillna(0)).mean(dim="ivt_str")
                if rx2_da is not None:
                    this_map -= this_rx2_da.weighted(area_map.fillna(0)).mean(dim="ivt_str")
            else:
                raise RuntimeError("Neither take_subcrop_sum nor take_subcrop_wtdmean???")
        this_map.attrs = this_map_attrs
        this_map = cc.convert_units(this_map, varInfo['units'][v])
        
        # Weight based on EarthStats data, if needed
        if False: #"BIAS" in thisVar and earthstats is not None:
            if posNeg:
                raise RuntimeError("EarthStats weighting not tested for posNeg")
            elif "PROD" not in thisVar:
                raise RuntimeError(f"EarthStat weighting not set up for {thisVar}")
            print(f" * Weighting by Earthstat production...")
            earthstats_thisCrop = earthstats.isel(patch=np.where(earthstats['patches1d_itype_combinedCropCLM_str']==crop)[0])
            earthstats_weights_da = utils.grid_one_variable(earthstats_thisCrop, "Production")
                
            if not (np.array_equal(this_map.lon, earthstats_weights_da.lon) and np.array_equal(this_map.lat, earthstats_weights_da.lat) and np.array_equal(this_map.time, earthstats_weights_da.time)):
                raise RuntimeError("Unexpected dim mismatch between this_map and EarthStats weights")
            this_map_weighted = this_map.weighted(earthstats_weights_da.fillna(0))
        else:
            this_map_weighted = this_map
        
        # Get mean over time
        this_map_timemean = this_map_weighted.mean(dim=time_dim)
        
        # If doing just positive-negative, select and sum the matching cells
        if posNeg:
            if crop == "Crops increasing":
                this_map_timemean = this_map_timemean.where(this_map_timemean > 0)
                area_map = area_map.where(this_map_timemean > 0)
            elif crop == "Crops decreasing":
                this_map_timemean = this_map_timemean.where(this_map_timemean < 0)
                area_map = area_map.where(this_map_timemean < 0)
            else:
                RuntimeError(f"posNeg: crop {crop} not recognized")
            this_map_timemean = this_map_timemean.sum(dim="ivt_str")
            area_map_sum = area_map.sum(dim="ivt_str")
            weights_map = area_map / area_map_sum
        
        # Mask where not much crop area?
        if low_area_threshold_m2 is not None:
            this_map_timemean = this_map_timemean.where(area_map_sum>low_area_threshold_m2)
        
        # Set up for masking
        underlay = None
        any_masked = False
        sumdiff_beforemask = np.nansum(np.abs(this_map_timemean.values))
        max_absdiff_beforemask = np.max(np.abs(this_map_timemean))
        pct_absdiffs_masked_before = 0
        frac_to_include = 0.95
        
        # If doing so, mask out cells not significantly different from zero
        if 'mask_sig_diff_from_0' in varInfo and varInfo['mask_sig_diff_from_0'][v]:
            any_masked = True
            
            # Show gray "underlay" map where crop is grown but masked
            if underlay is None:
                if posNeg:
                    raise RuntimeError("Underlay map not tested for posNeg")
                underlay = get_underlay(this_ds, area_map_sum)
            
            # Get p-values for one-sample t-test
            ttest = stats.ttest_1samp(a=this_map.values, popmean=0, axis=list(this_map.dims).index(time_dim))
            pvalues = xr.DataArray(data=ttest.pvalue,
                                   coords=this_map.isel({time_dim: 0}).coords)
            
            # Get critical p-value (alpha)
            alpha = 0.05
            sidak_correction = False
            if sidak_correction:
                Ntests = np.nansum(underlay.values)
                alpha = 1-(1-alpha)**(1/Ntests)
            
            # Mask map where no significant difference
            this_map_timemean = this_map_timemean.where(pvalues < alpha)
            
            # Diagnostics
            pct_absdiffs_masked_before = get_amount_masked(crop, this_map_timemean, sumdiff_beforemask, pct_absdiffs_masked_before, "difference not significant")
            
        # If doing so, mask out negligible cells
        if 'mask_negligible' in varInfo and varInfo['mask_negligible'][v]:
            any_masked = True
            
            # Show gray "underlay" map where crop is grown but masked
            if underlay is None:
                if posNeg:
                    raise RuntimeError("Underlay map not tested for posNeg")
                underlay = get_underlay(this_ds, area_map_sum)
            
            # Mask map where negligible (< 0.1% of max difference)
            this_map_timemean = this_map_timemean.where(np.abs(this_map_timemean) >= 0.001*max_absdiff_beforemask)
            
            # Diagnostics
            pct_absdiffs_masked_before = get_amount_masked(crop, this_map_timemean, sumdiff_beforemask, pct_absdiffs_masked_before, "difference negligible")
            
        # If doing so, SET UP TO mask out all but cells comprising top 95% of absolute differences.
        # This masking actually happens later, via chunk_colorbar().
        if 'maskcolorbar_near0' in varInfo and varInfo['maskcolorbar_near0'][v]:
            if not take_subcrop_sum:
                raise RuntimeError("Do not maskcolorbar_near0 for variables where summing across different subcrops/gridcells doesn't make sense")
            any_masked = True
            
        # Get color bar info
        if is_diff:
            vmax = np.nanmax(np.abs(this_map_timemean.values))
            vmin = -vmax
        else:
            vmin = None
            vmax = None
        
        # Chunk colormap ONLY when masking
        this_cmap = cmap
        bounds = None
        ticks_orig = None
        cbar_spacing = "uniform"
        if any_masked:
            if is_diff:
                bounds, cbar_spacing, pct_absdiffs_masked_before, this_cmap, ticks_orig, vmin, vmax = chunk_colorbar(this_map_timemean, cbar_spacing, cmap, crop, fontsize, pct_absdiffs_masked_before, sumdiff_beforemask, varInfo, vmin, vmax, posNeg=posNeg, underlay=underlay, v=v)
            else:
                raise RuntimeError("How do you have an underlay without a difference map")
        
        # Mask where not much crop area?
        if underlay is not None and low_area_threshold_m2 is not None:
            underlay = underlay.where(area_map_sum>low_area_threshold_m2)
        
        # Plot map
        subplot_str = chr(ord('`') + thisPlot) # or ord('@') for capital
        im, cb = make_map(ax, this_map_timemean, fontsize, show_cbar=True, vmin=vmin, vmax=vmax, cmap=this_cmap, extend_nonbounds=None, underlay=underlay, underlay_color=cropcal_colors['underlay'], bounds=bounds, extend_bounds="neither", ticklabels=ticks_orig, cbar_spacing=cbar_spacing, subplot_label=subplot_str)
        
        show_cbar_label = True
        cbar_label = varInfo['units'][v]
        cbar_label_x = 0.5
        cbar_labelpad = 0.4
        if ny==3 and nx==2:
            cbar_labelpad = -33
            cbar_label_x = 1.098
            if c % 2:
                show_cbar_label = False
        elif ny==3 and nx==1:
            cbar_labelpad = 13
            cbar_label = cbar_label.replace('\n', ' ')
        elif ny==1 and nx==2:
            cbar_labelpad = 13
            cbar_label = cbar_label.replace('\n', ' ')
        if show_cbar_label:
            cb.set_label(label=cbar_label, fontsize=fontsize['axislabels'], x=cbar_label_x, labelpad=cbar_labelpad)
        
        if croptitle_side == "top":
            ax.set_title(crop, fontsize=fontsize['titles'])
        elif croptitle_side == "left":
            if v==0:
                ax.set_ylabel(crop, fontsize=fontsize['titles'])
        else:
            raise RuntimeError(f"Unexpected value for croptitle_side: {croptitle_side}")
        
        if Nvars > 1 and c==0:
            ax.set_title(this_suptitle, fontsize=fontsize['titles'])
        
        # plt.show()
        # ererqrqwew
    
    hspace = None
    suptitle_y = 0.98
    if ny==3:
        hspace = -0.4
        this_suptitle = this_suptitle.replace(": ", ":\n")
        suptitle_y = 0.79
    elif ny==2 and nx==3:
        hspace = 0
        suptitle_y = 0.82
    
    if Nvars == 1:
        fig.suptitle(this_suptitle,
                    fontsize=fontsize['suptitle'],
                    y=suptitle_y)
    
    plt.subplots_adjust(hspace=hspace)


def maps_allCrops(cases, these_cases, reses, thisVar, varInfo, outDir_figs, cropList_combined_clm_nototal, figsize, crop_subset=None, croptitle_side="top", dpi=150, earthstats=None, low_area_threshold_m2=1e4, min_viable_hui="ggcmi3", mxmats=None, ny=2, nx=3, plot_y1=1980, plot_yN=2009):
    
    if crop_subset==cropList_combined_clm_nototal:
        crop_subset = None
    filename_suffix = ""
    if crop_subset is not None:
        filename_suffix = " "
        for crop in crop_subset:
            filename_suffix += crop
        cropList_combined_clm_nototal = crop_subset
    
    # Set up for one variable per column
    multiCol = "." in thisVar
    if multiCol:
        if nx != 1:
            raise RuntimeError("When doing one variable per column, nx must be 1")
        varList = thisVar.split(".")
        minus = ""
        if "DIFF" in thisVar:
            minus = f"{these_cases[1]} minus {these_cases[0]} "
        fig_outfile = os.path.join(outDir_figs, f"{thisVar} {minus}{plot_y1}-{plot_yN}{filename_suffix}.png").replace('Mean annual ', '').replace(':', '')
        fig = plt.figure(figsize=figsize)
    else:
        varList = [thisVar]
        for key in varInfo.keys():
            varInfo[key] = [varInfo[key]]
    Nvars = len(varList)
        
    for v, thisVar in enumerate(varList):
        rx_parent_casename = None
        rx2_da = None
        earthstats_ds = None
        rx_ds = None
    
        # Process variable info
        posNeg = "POSNEG" in thisVar
        is_diff = thisVar.endswith("_DIFF") or thisVar.endswith("_DIFFPOSNEG")
        if is_diff:
            if len(these_cases) != 2:
                raise RuntimeError(f"You must provide exactly 2 cases in these_cases for DIFF variables")
        is_diffdiff = is_diff and thisVar.replace("BIAS", "DIFF").count("DIFF") == 2
        time_dim = "time"
        if "time_dim" in varInfo:
            time_dim = varInfo['time_dim']
            if isinstance(time_dim, list):
                if len(time_dim) > 1:
                    raise RuntimeError("Handle time_dim list with multiple members")
                time_dim = time_dim[0]
        
        # Does this variable get summed across subcrops, or weighted-meaned?
        take_subcrop_sum = False
        theseUnits = varInfo['units'][v]
        for units in ["Mt", "km$^3$", "km$^3$ yr$^{-1}$"]:
            if units in theseUnits:
                take_subcrop_sum = True
                break
        take_subcrop_wtdmean = False
        for units in ["t/ha", "days"]:
            if units in theseUnits:
                take_subcrop_wtdmean = True
                break
        if take_subcrop_sum and take_subcrop_wtdmean:
            raise RuntimeError(f"Units '{theseUnits}' have both take_subcrop_sum and take_subcrop_wtdmean specified somehow")
        if not (take_subcrop_sum or take_subcrop_wtdmean):
            raise RuntimeError(f"Define across-subcrop operation for units '{theseUnits}'")
        
        # Special setup for posNeg
        if posNeg:
            ny = 1
            nx = 2
            figsize = (14, 3.75)

        if is_diff:
            if not multiCol:
                fig = plt.figure(figsize=figsize)
            this_suptitle = f"{varInfo['suptitle'][v]}"
            minus = f": {these_cases[1]} minus {these_cases[0]}"
            this_suptitle += minus
            print(this_suptitle)
            if not multiCol:
                fig_outfile = os.path.join(outDir_figs, f"Map diff {this_suptitle} {plot_y1}-{plot_yN}{filename_suffix}.png").replace('Mean annual ', '').replace(':', '')
            if 'suppress_difftext' in varInfo and varInfo['suppress_difftext']:
                this_suptitle = this_suptitle.replace(minus, "")
            
            case0 = cases[these_cases[0]]
            case1 = cases[these_cases[1]]
            ds0 = case0['ds']
            ds1 = case1['ds']
            lu_ds = reses[case0['res']]['ds']
            
            # Get mean production over period of interest
            if "PROD" in thisVar or "YIELD" in thisVar:
                ds0 = cc.get_yield_ann(ds0, min_viable_hui=min_viable_hui, mxmats=mxmats, lu_ds=lu_ds)
                ds1 = cc.get_yield_ann(ds1, min_viable_hui=min_viable_hui, mxmats=mxmats, lu_ds=lu_ds)
                
            earthstats_ds = None
            if "BIAS" in thisVar:
                if np.any([x in thisVar for x in ["GSLEN", "HDATE", "SDATE"]]):
                    raise RuntimeError(f"Bias-difference maps not yet supported for rx variables ({thisVar})")
                else:
                    if earthstats is None:
                        raise RuntimeError("Pass earthstats to maps_allCrops() if you want to calculate bias.")
                    earthstats_ds = earthstats[cc.get_gridspec(case0)].sel({time_dim: slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31")})
                    ds0 = cc.get_diff_earthstat(ds0, case0, earthstats_ds, reses, use_absolute_bias="ABSBIAS" in thisVar)
                    ds1 = cc.get_diff_earthstat(ds1, case1, earthstats_ds, reses, use_absolute_bias="ABSBIAS" in thisVar)

            this_ds = ds1.copy()
            if is_diffdiff:
                if posNeg:
                    raise RuntimeError("POSNEG not tested for difference-difference maps")
                diff_yield_var = thisVar.replace("PROD", "YIELD")
                undiff_yield_var = "".join(diff_yield_var.rsplit("_DIFF", 1)) # Delete last occurrence of "_DIFF"
                diff_prod_var = diff_yield_var.replace("YIELD", "PROD")
                undiff_prod_var = undiff_yield_var.replace("YIELD", "PROD")
                if "BIAS" in thisVar:
                    undiff_yield_var = undiff_yield_var.replace("ABSBIAS", "BIAS").replace("BIAS", "DIFF")
                    undiff_prod_var = undiff_prod_var.replace("ABSBIAS", "BIAS").replace("BIAS", "DIFF")
                this_ds[diff_yield_var] = ds1[undiff_yield_var] - ds0[undiff_yield_var]
                this_ds[diff_yield_var].attrs['units'] = ds1[undiff_yield_var].attrs['units']
                if undiff_prod_var in this_ds:
                    this_ds[diff_prod_var] = ds1[undiff_prod_var] - ds0[undiff_prod_var]
                    this_ds[diff_prod_var].attrs['units'] = ds1[undiff_prod_var].attrs['units']
            else:
                if "DIFFPOSNEG" in thisVar:
                    thisVar_base = thisVar.replace("_DIFFPOSNEG", "")
                else:
                    thisVar_base = thisVar.replace("_DIFF", "")
                this_ds[thisVar] = ds1[thisVar_base] - ds0[thisVar_base]
                this_ds[thisVar].attrs['units'] = ds1[thisVar_base].attrs['units']
            this_ds = this_ds\
                    .sel({time_dim : slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31")})
            
            make_fig(thisVar, varInfo, cropList_combined_clm_nototal, ny, nx, this_ds, this_suptitle, is_diff, is_diffdiff, low_area_threshold_m2, croptitle_side, v, Nvars, fig, earthstats_ds, posNeg, take_subcrop_sum, take_subcrop_wtdmean, rx_ds, rx_parent_casename)
            
            if not multiCol:
                # plt.show()
                fig.savefig(fig_outfile, bbox_inches='tight', facecolor='white', dpi=dpi)
                plt.close()
        
        
        else:
            for this_case in these_cases:
                if not multiCol:
                    fig = plt.figure(figsize=figsize)
                case = cases[this_case]
                lu_ds = reses[case['res']]['ds']
                this_ds = case['ds']
                if "PROD" in thisVar or "YIELD" in thisVar:
                    this_ds = cc.get_yield_ann(this_ds, min_viable_hui=min_viable_hui, mxmats=None, lu_ds=lu_ds)
                this_ds = this_ds.copy()\
                    .sel({time_dim: slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31")})
                    
                this_suptitle = f"{varInfo['suptitle'][v]}: {this_case}"
                if not multiCol:
                    fig_outfile = os.path.join(outDir_figs, f"Map {this_suptitle} {plot_y1}-{plot_yN}.png").replace('Mean annual ', '').replace(':', '')
                print(this_suptitle)
                
                if "BIAS" in thisVar:
                    if np.any([x in thisVar for x in ["GSLEN", "HDATE", "SDATE"]]):
                        rx_parent_casename, rx_ds, _, _ = cc.get_rx_case(cases, list(cases.keys()), ny, thisVar.replace("ABSBIAS", "BIAS").replace("_BIAS", ""))
                        
                        # Get the name of a variable of interest in the rx Dataset
                        rx_varPrefix_list = []
                        for x in rx_ds:
                            if "lon" in rx_ds[x].dims:
                                rx_varParts = x.split("_")
                                if rx_varParts[0] not in rx_varPrefix_list:
                                    rx_varPrefix_list.append(rx_varParts[0])
                        if len(rx_varPrefix_list) == 0:
                            raise RuntimeError("No matching variables found in rx_ds")
                        elif len(rx_varPrefix_list) > 1:
                            raise RuntimeError(f"Too many matching variables found in rx_ds; prefixes: {rx_varPrefix_list}")
                        rx_varPrefix = rx_varPrefix_list[0]
                        
                        # Get new DataArray with included crops on dimension ivt_str
                        itype_veg_list = np.unique(this_ds['patches1d_itype_veg'])
                        itype_veg_str_list = [utils.ivt_int2str(x) for x in itype_veg_list]
                        rx_varList = [f"{rx_varPrefix}_{x}" for x in itype_veg_list]
                        rx2_da = xr.concat(objs=[rx_ds[x] for x in rx_varList],
                                           dim="ivt_str")
                        rx2_da = rx2_da.assign_coords({'ivt_str': itype_veg_str_list})
                        if "time" in rx_ds and rx_ds.dims["time"]==1:
                            rx2_da = rx2_da.isel(time=0, drop=True)
                        
                    else:
                        if earthstats is None:
                            raise RuntimeError("Pass earthstats to maps_allCrops() if you want to calculate bias.")
                        earthstats_ds = earthstats[cc.get_gridspec(case)].sel({time_dim: slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31")})
                        this_ds = cc.get_diff_earthstat(this_ds, case, earthstats_ds, reses, use_absolute_bias="ABSBIAS" in thisVar)
                        this_ds[thisVar] = this_ds[thisVar.replace("ABSBIAS", "BIAS").replace("_BIAS", "_DIFF")]
                
                make_fig(thisVar, varInfo, cropList_combined_clm_nototal, ny, nx, this_ds, this_suptitle, "BIAS" in thisVar, False, low_area_threshold_m2, croptitle_side, v, Nvars, fig, earthstats_ds, posNeg, take_subcrop_sum, take_subcrop_wtdmean, rx2_da, rx_parent_casename)
                
                if not multiCol:
                    # plt.show()
                    fig.savefig(fig_outfile, bbox_inches='tight', facecolor='white', dpi=dpi)
                    plt.close()
    
    if multiCol:
        # plt.show()
        fig.savefig(fig_outfile, bbox_inches='tight', facecolor='white', dpi=dpi)
        plt.close()
    
    
    return cases

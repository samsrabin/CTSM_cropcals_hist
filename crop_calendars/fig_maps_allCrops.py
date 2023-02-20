# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import cropcal_module as cc
from cropcal_figs_module import *

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import stats

figsize = (20, 10)
fontsize = {}
fontsize['titles'] = 18
fontsize['axislabels'] = 14
fontsize['ticklabels'] = 14
fontsize['suptitle'] = 22


def get_amount_masked(c, crop, this_map_timemean, sumdiff_beforemask, pct_absdiffs_masked_before, reason):
    sumdiff_aftermask = np.nansum(np.abs(this_map_timemean.values))
    pct_absdiffs_masked = 100 * (1 - sumdiff_aftermask / sumdiff_beforemask)
    pct_absdiffs_masked_here = pct_absdiffs_masked - pct_absdiffs_masked_before
    print(f"   Masked {crop} ({reason}): {round(pct_absdiffs_masked_here, 1)}%")
    pct_absdiffs_masked_before = pct_absdiffs_masked
    return pct_absdiffs_masked_before


def get_lowest_threshold(this_map_timemean, frac_to_include):
    flattened = np.abs(this_map_timemean.values).flatten()
    flattened_is_ok = np.where(~np.isnan(flattened))
    okflattened = flattened[flattened_is_ok]
    oksorted = np.flip(np.sort(okflattened))
    okcumsum = np.cumsum(oksorted)
    okcumprop = okcumsum / np.sum(oksorted)
    for i, x in enumerate(okcumprop):
        if x >= frac_to_include:
            break
    lowest_threshold = oksorted[i]
    return lowest_threshold


def get_underlay(this_ds, area_map_sum):
    dummy_var = 'gridcell'
    dummy_data = np.full_like(this_ds[dummy_var].values, 1.0)
    this_ds['is_land'] = xr.DataArray(data=dummy_data,
                                              coords=this_ds[dummy_var].coords)
    is_land = utils.grid_one_variable(this_ds, 'is_land')
    underlay = is_land.where(area_map_sum > 0)
    return underlay


def make_fig(thisVar, varInfo, cropList_combined_clm_nototal, dpi, figsize, ny, nx, plot_y1, plot_yN, ds_in, this_suptitle, fig_outfile, is_diff, is_diffdiff, low_area_threshold_m2, croptitle_side):
    fig = plt.figure(figsize=figsize)
    
    if is_diff:
        if is_diffdiff:
            cmap = "RdBu_r"
        else:
            cmap = "BrBG"
    else:
        cmap = "viridis"

    for c, crop in enumerate(cropList_combined_clm_nototal):
        ax = fig.add_subplot(ny,nx,c+1,projection=ccrs.PlateCarree())
        
        # Which CFTs comprise this crop?
        where_thisCrop = np.where(ds_in['patches1d_itype_combinedCropCLM_str'] == crop)[0]
        theseCrops = np.unique(ds_in['patches1d_itype_veg_str'].isel(patch=where_thisCrop))
        this_ds = ds_in.isel(patch=where_thisCrop)
        
        # Get area of these CFTs
        area_map = utils.grid_one_variable(this_ds.mean(dim="time"), "AREA_CFT", vegtype=list(theseCrops))
        area_map_sum = area_map.sum(dim="ivt_str")
        weights_map = area_map / area_map_sum

        # Grid data for those CFTs, getting their sum
        this_map = utils.grid_one_variable(this_ds, thisVar, vegtype=list(theseCrops))
        if "YIELD" in thisVar:
            this_map = this_map * weights_map
        this_map = this_map.sum(dim="ivt_str")
        this_map *= varInfo['multiplier']
        this_map_timemean = this_map.mean(dim="time")
        
        # Mask where not much crop area?
        if low_area_threshold_m2 is not None:
            this_map_timemean = this_map_timemean.where(area_map_sum>low_area_threshold_m2)
        
        # Set up for masking
        underlay = None
        underlay_color = [0.75, 0.75, 0.75, 1]
        any_masked = False
        sumdiff_beforemask = np.nansum(np.abs(this_map_timemean.values))
        max_absdiff_beforemask = np.max(np.abs(this_map_timemean))
        pct_absdiffs_masked_before = 0
        frac_to_include = 0.95
        lowest_threshold = get_lowest_threshold(this_map_timemean, frac_to_include)
        
        # If doing so, mask out cells not significantly different from zero
        if 'mask_sig_diff_from_0' in varInfo and varInfo['mask_sig_diff_from_0']:
            any_masked = True
            
            # Show gray "underlay" map where crop is grown but masked
            if underlay is None:
                underlay = get_underlay(this_ds, area_map_sum)
            
            # Get p-values for one-sample t-test
            ttest = stats.ttest_1samp(a=this_map.values, popmean=0, axis=list(this_map.dims).index("time"))
            pvalues = xr.DataArray(data=ttest.pvalue,
                                   coords=this_map.isel(time=0).coords)
            
            # Get critical p-value (alpha)
            alpha = 0.05
            sidak_correction = False
            if sidak_correction:
                Ntests = np.nansum(underlay.values)
                alpha = 1-(1-alpha)**(1/Ntests)
            
            # Mask map where no significant difference
            this_map_timemean = this_map_timemean.where(pvalues < alpha)
            
            # Diagnostics
            pct_absdiffs_masked_before = get_amount_masked(c, crop, this_map_timemean, sumdiff_beforemask, pct_absdiffs_masked_before, "difference not significant")
            
        # If doing so, mask out negligible cells
        if 'mask_negligible' in varInfo and varInfo['mask_negligible']:
            any_masked = True
            
            # Show gray "underlay" map where crop is grown but masked
            if underlay is None:
                underlay = get_underlay(this_ds, area_map_sum)
            
            # Mask map where negligible (< 0.1% of max difference)
            this_map_timemean = this_map_timemean.where(np.abs(this_map_timemean) >= 0.001*max_absdiff_beforemask)
            
            # Diagnostics
            pct_absdiffs_masked_before = get_amount_masked(c, crop, this_map_timemean, sumdiff_beforemask, pct_absdiffs_masked_before, "difference negligible")
            
        # If doing so, mask out all but cells comprising top 95% of absolute differences.
        # NOTE that we're not actually masking here. We are instead setting a special color for such cells.
        if 'mask_lowest' in varInfo and varInfo['mask_lowest']:
            any_masked = True
            
            # FAKE-mask all but top cells
            this_map_timemean_fake = this_map_timemean.where(np.abs(this_map_timemean) >= lowest_threshold)
            
            # Diagnostics
            pct_absdiffs_masked_before = get_amount_masked(c, crop, this_map_timemean_fake, sumdiff_beforemask, pct_absdiffs_masked_before, f"lowest, threshold {lowest_threshold}")
            
            # Because this is just a fake mask we must not do any additional masking, real or fake, after this. If we do, then our diagnostics for that will be messed up. To ensure we don't try to do any subsequent masking, set this to None, which should throw an error in get_amount_masked().
            pct_absdiffs_masked_before = None
            
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
                
                # Make a temporary plot with the same color axis and colorbar settings we would use in make_map().
                plt.pcolormesh(this_map_timemean, vmin=vmin, vmax=vmax)
                cb0 = plt.colorbar(location="bottom")
                cb0.ax.tick_params(labelsize=fontsize['ticklabels'])
                
                # Where did plt.colorbar() draw bin boundaries? These are referred to as "tick marks," but note that the extreme values might lie outside [vmin, vmax].
                ticks_orig = cb0.get_ticks()
                bounds = ticks_orig
                
                # In our plot, we will move vmin left and vmax right to ensure that the tick marks are the color bin boundaries.
                if cb0.vmin < bounds[0]:
                    raise RuntimeError("Handle vmin < bounds[0]")
                elif cb0.vmax > bounds[-1]:
                    raise RuntimeError("Handle vmax > bounds[-1]")
                elif 0 not in bounds:
                    raise RuntimeError("Handle 0 not in bounds")
                vmin = bounds[0]
                vmax = bounds[-1]
                
                # Get number of color bins
                Nbins = len(bounds) - 1
                bottom_of_topbin = bounds[-2]
                bottom_of_2ndbin = bounds[-3]
                binwidth = bounds[-1] - bounds[-2]
                if Nbins < 8:
                    Nbins *= 2
                    bottom_of_2ndbin = bottom_of_topbin
                    binwidth /= 2
                    bottom_of_topbin += binwidth
                
                # Ensure that most extreme bin (on at least one side of 0) has at least one gridcell included. If not, remove the most extreme bins and check again.
                maxinmap = np.nanmax(np.abs(this_map_timemean.values))
                if maxinmap < bottom_of_topbin:
                    if maxinmap < bottom_of_2ndbin:
                        raise RuntimeError("How is maxinmap less than the bottom of the SECOND bin??")
                    vmax -= binwidth
                    vmin += binwidth
                    Nbins-=2
                    if ticks_orig[0] < vmin:
                        ticks_orig = ticks_orig[1:]
                    if ticks_orig[-1] > vmax:
                        ticks_orig = ticks_orig[:-1]
                    
                # Get new colormap with the right number of bins.
                this_cmap = cm.get_cmap(cmap, Nbins)
                if Nbins % 2:
                    raise RuntimeError(f"Expected even number of color bins; got {Nbins}")
                
                # Special color for small-masked cells
                if 'mask_lowest' in varInfo and varInfo['mask_lowest']:
                    
                    # Add near-zero bin
                    bounds = np.concatenate((np.arange(vmin, -binwidth+1e-9, binwidth),
                                             np.array([-lowest_threshold, lowest_threshold]),
                                             np.arange(binwidth, vmax+1e-9, binwidth)))
                    cbar_spacing = "proportional"        
                    
                    # Add color for that bin
                    if underlay is not None:
                        raise RuntimeError("You need a different color to distinguish mask_lowest cells from other-masked cells")
                    if isinstance(this_cmap, mcolors.LinearSegmentedColormap):
                        color_list = [this_cmap(x) for x in np.arange(0, 1+1e-9, 1/Nbins)]
                        this_cmap = mcolors.ListedColormap(color_list)
                    elif not isinstance(this_cmap, mcolors.ListedColormap):
                        raise RuntimeError(f"Not sure how to get list of colors from {type(this_cmap)}")
                    new_colors = np.concatenate((this_cmap.colors[:int(Nbins/2)],
                                                 np.array([underlay_color]),
                                                 this_cmap.colors[int(Nbins/2):]),
                                                axis=0)
                    this_cmap = mcolors.ListedColormap(new_colors)            
                
                # Remove our temporary plot and its colorbar.
                plt.cla()
                cb0.remove()
            else:
                raise RuntimeError("How do you have an underlay without a difference map")
        
        # Mask where not much crop area?
        if underlay is not None and low_area_threshold_m2 is not None:
            underlay = underlay.where(area_map_sum>low_area_threshold_m2)
        
        # Plot map
        im, cb = make_map(ax, this_map_timemean, fontsize, show_cbar=True, vmin=vmin, vmax=vmax, cmap=this_cmap, extend_nonbounds=None, underlay=underlay, underlay_color=underlay_color, bounds=bounds, extend_bounds="neither", ticklabels=ticks_orig, cbar_spacing=cbar_spacing)
        
        show_cbar_label = True
        cbar_label = varInfo['units']
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
        if show_cbar_label:
            cb.set_label(label=cbar_label, fontsize=fontsize['axislabels'], x=cbar_label_x, labelpad=cbar_labelpad)
        
        if croptitle_side == "top":
            ax.set_title(crop, fontsize=fontsize['titles'])
        elif croptitle_side == "left":
            ax.set_ylabel(crop, fontsize=fontsize['titles'])
        else:
            raise RuntimeError(f"Unexpected value for croptitle_side: {croptitle_side}")
        
        # plt.show()
        # return
    
    hspace = None
    suptitle_y = 0.98
    if ny==3:
        hspace = -0.4
        this_suptitle = this_suptitle.replace(": ", ":\n")
        suptitle_y = 0.79
    elif ny==2 and nx==3:
        hspace = 0
        suptitle_y = 0.82
    
    fig.suptitle(this_suptitle,
                fontsize=fontsize['suptitle'],
                y=suptitle_y)
    plt.subplots_adjust(hspace=hspace)
    
    # plt.show()
    # return

    fig.savefig(fig_outfile, bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close()


def maps_allCrops(cases, these_cases, reses, thisVar, varInfo, outDir_figs, cropList_combined_clm_nototal, croptitle_side="top", dpi=150, figsize=figsize, low_area_threshold_m2=1e4, min_viable_hui="ggcmi3", mxmats=None, ny=2, nx=3, plot_y1=1980, plot_yN=2009):
    
    # Process variable info
    is_diff = thisVar[-5:] == "_DIFF"
    if is_diff:
        if len(these_cases) != 2:
            raise RuntimeError(f"You must provide exactly 2 cases in these_cases for DIFF variables")
    is_diffdiff = is_diff and thisVar.count("DIFF") == 2

    if is_diff:
        this_suptitle = f"{varInfo['suptitle']}: {these_cases[1]} minus {these_cases[0]}"
        print(this_suptitle)
        fig_outfile = os.path.join(outDir_figs, f"Map diff {this_suptitle} {plot_y1}-{plot_yN}.png").replace('Mean annual ', '').replace(':', '')
        
        case0 = cases[these_cases[0]]
        ds0 = cases[these_cases[0]]['ds']
        ds1 = cases[these_cases[1]]['ds']
        lu_ds = reses[case0['res']]['ds']

        # Get mean production over period of interest
        if "PROD" in thisVar or "YIELD" in thisVar:
            ds0 = cc.get_yield_ann(ds0, min_viable_hui=min_viable_hui, mxmats=mxmats, lu_ds=lu_ds)
            ds1 = cc.get_yield_ann(ds1, min_viable_hui=min_viable_hui, mxmats=mxmats, lu_ds=lu_ds)

        this_ds = ds1.copy()
        if is_diffdiff:
            diff_yield_var = thisVar.replace("PROD", "YIELD")
            undiff_yield_var = "".join(diff_yield_var.rsplit("_DIFF", 1)) # Delete last occurrence of "_DIFF"
            diff_prod_var = diff_yield_var.replace("YIELD", "PROD")
            undiff_prod_var = undiff_yield_var.replace("YIELD", "PROD")
            this_ds[diff_yield_var] = np.fabs(ds1[undiff_yield_var]) - np.fabs(ds0[undiff_yield_var])
            if undiff_prod_var in this_ds:
                this_ds[diff_prod_var] = np.fabs(ds1[undiff_prod_var]) - np.fabs(ds0[undiff_prod_var])
        else:
            thisVar_base = thisVar.replace("_DIFF", "")
            this_ds[thisVar] = ds1[thisVar_base] - ds0[thisVar_base]
        this_ds = this_ds\
                .sel(time=slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31"))
        
        make_fig(thisVar, varInfo, cropList_combined_clm_nototal, dpi, figsize, ny, nx, plot_y1, plot_yN, this_ds, this_suptitle, fig_outfile, is_diff, is_diffdiff, low_area_threshold_m2, croptitle_side)
    
    
    else:
        for this_case in these_cases:
            case = cases[this_case]
            lu_ds = reses[case['res']]['ds']
            this_ds = case['ds']
            this_ds = cc.get_yield_ann(this_ds, min_viable_hui=min_viable_hui, mxmats=None, lu_ds=lu_ds)
            this_ds = this_ds.copy()\
                .sel(time=slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31"))
                
            this_suptitle = f"{varInfo['suptitle']}: {this_case}"
            fig_outfile = os.path.join(outDir_figs, f"Map {this_suptitle} {plot_y1}-{plot_yN}.png").replace('Mean annual ', '').replace(':', '')
            print(this_suptitle)
            
            make_fig(thisVar, varInfo, cropList_combined_clm_nototal, dpi, figsize, ny, nx, plot_y1, plot_yN, this_ds, this_suptitle, fig_outfile, "DIFF" in thisVar, False, low_area_threshold_m2, croptitle_side)
    
    return cases

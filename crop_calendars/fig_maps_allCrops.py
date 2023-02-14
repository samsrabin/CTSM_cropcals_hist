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

figsize = (20, 10)
fontsize = {}
fontsize['titles'] = 18
fontsize['axislabels'] = 14
fontsize['ticklabels'] = 14
fontsize['suptitle'] = 22


def make_fig(thisVar, varInfo, cropList_combined_clm_nototal, dpi, figsize, ny, nx, plot_y1, plot_yN, ds_in, this_suptitle, fig_outfile, is_diff, is_diffdiff):
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
        this_map = this_map.mean(dim="time")
        
        # Plot map
        if is_diff:
            vmax = np.nanmax(np.abs(this_map.values))
            vmin = -vmax
        else:
            vmin = None
            vmax = None
        im, cb = make_map(ax, this_map.where(area_map_sum>1e4), fontsize, show_cbar=True, vmin=vmin, vmax=vmax, cmap=cmap, extend_nonbounds=None)
        
        show_cbar_label = True
        cbar_label_x = 0.5
        cbar_labelpad = 0.4
        if ny==3 and nx==2:
            cbar_labelpad = -33
            cbar_label_x = 1.098
            if c % 2:
                show_cbar_label = False
        if show_cbar_label:
            cb.set_label(label=varInfo['units'], fontsize=fontsize['axislabels'], x=cbar_label_x, labelpad=cbar_labelpad)
        
        ax.set_title(crop, fontsize=fontsize['titles'])
    
    hspace = None
    suptitle_y = 0.98
    if ny==3 and nx==2:
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


def maps_allCrops(cases, these_cases, reses, thisVar, varInfo, outDir_figs, cropList_combined_clm_nototal, dpi=150, figsize=figsize, min_viable_hui="ggcmi3", mxmats=None, ny=2, nx=3, plot_y1=1980, plot_yN=2009):
    
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
        
        make_fig(thisVar, varInfo, cropList_combined_clm_nototal, dpi, figsize, ny, nx, plot_y1, plot_yN, this_ds, this_suptitle, fig_outfile, is_diff, is_diffdiff)
    
    
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
            
            make_fig(thisVar, varInfo, cropList_combined_clm_nototal, dpi, figsize, ny, nx, plot_y1, plot_yN, this_ds, this_suptitle, fig_outfile, "DIFF" in thisVar, False)
    
    return cases

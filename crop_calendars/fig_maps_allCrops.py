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


def make_fig(thisVar, varInfo, cropList_combined_clm_nototal, dpi, figsize, ny, nx, plot_y1, plot_yN, lu_ds, this_ds, this_suptitle, fig_outfile, is_diff):
    fig = plt.figure(figsize=figsize)
    
    if is_diff:
        cmap = "BrBG"
    else:
        cmap = "viridis"

    for c, crop in enumerate(cropList_combined_clm_nototal):
        ax = fig.add_subplot(ny,nx,c+1,projection=ccrs.PlateCarree())
        
        # Which CFTs comprise this crop?
        where_thisCrop = np.where(this_ds['patches1d_itype_combinedCropCLM_str'] == crop)[0]
        theseCrops = np.unique(this_ds['patches1d_itype_veg_str'].isel(patch=where_thisCrop))
        
        # Get area of these CFTs
        lu_thiscrop = utils.xr_flexsel(lu_ds.sel(time=slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31")),
                                    vegtype=list(theseCrops))
        area_thiscrop_mean_da = lu_thiscrop['AREA_CFT'].mean(dim="time")
        area_thiscrop_ds = xr.Dataset(\
            data_vars={'area_thiscrop_mean': area_thiscrop_mean_da,
                    'patches1d_ixy': lu_thiscrop.patches1d_ixy,
                    'patches1d_jxy': lu_thiscrop.patches1d_jxy,
                    'patches1d_lon': lu_thiscrop.patches1d_lon,
                    'patches1d_lat': lu_thiscrop.patches1d_lat,
                    'patches1d_itype_veg': lu_thiscrop.patches1d_itype_veg,
                    'patches1d_itype_veg_str': lu_thiscrop.patches1d_itype_veg_str,
                    'vegtype_str': this_ds.vegtype_str},
            coords={'patch': lu_thiscrop.patch,
                    'lon': lu_thiscrop.lon,
                    'lat': lu_thiscrop.lat,
                    'ivt': this_ds.ivt})
        area_map = utils.grid_one_variable(area_thiscrop_ds, "area_thiscrop_mean")
        area_map_sum = area_map.sum(dim="ivt_str")
        weights_map = area_map / area_map_sum

        # Grid data for those CFTs, getting their sum
        this_map = utils.grid_one_variable(this_ds.isel(patch=where_thisCrop), thisVar, vegtype=list(theseCrops))
        if "YIELD" in thisVar:
            this_map = this_map * weights_map
        this_map = this_map.sum(dim="ivt_str")
        this_map *= varInfo['multiplier']
        
        # Plot map
        if is_diff:
            vmax = np.nanmax(np.abs(this_map.values))
            vmin = -vmax
        else:
            vmin = None
            vmax = None
        im, cb = make_map(ax, this_map.where(area_map_sum>1e4), fontsize, show_cbar=True, vmin=vmin, vmax=vmax, cmap=cmap, extend_nonbounds=None)
        cb.set_label(label=varInfo['units'], fontsize=fontsize['axislabels'])
        ax.set_title(crop, fontsize=fontsize['titles'])
        
    fig.suptitle(this_suptitle,
                fontsize=fontsize['suptitle'],
                y=0.82)
    plt.subplots_adjust(hspace=0)

    fig.savefig(fig_outfile, bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close()


def maps_allCrops(cases, these_cases, reses, thisVar, varInfo, outDir_figs, cropList_combined_clm_nototal, dpi=150, figsize=figsize, min_viable_hui="ggcmi3", mxmats=None, ny=2, nx=3, plot_y1=1980, plot_yN=2010, use_annual_yields=False):
    
    # Process variable info
    is_diff = "DIFF" in thisVar
    if is_diff:
        if len(these_cases) != 2:
            raise RuntimeError(f"You must provide exactly 2 cases in these_cases for DIFF variables")

    if is_diff:
        this_suptitle = f"{varInfo['suptitle']}: {these_cases[1]} minus {these_cases[0]}"
        print(this_suptitle)
        fig_outfile = os.path.join(outDir_figs, f"Map diff {this_suptitle} {plot_y1}-{plot_yN}.png").replace('Mean annual ', '').replace(':', '')
        
        case0 = cases[these_cases[0]]
        ds0 = cases[these_cases[0]]['ds']
        ds1 = cases[these_cases[1]]['ds']
        lu_ds = reses[case0['res']]['ds']

        # Get mean production over period of interest
        ds0 = cc.get_yield_ann(ds0, min_viable_hui=min_viable_hui, mxmats=None)
        ds1 = cc.get_yield_ann(ds1, min_viable_hui=min_viable_hui, mxmats=None)

        this_ds = ds1.copy()
        this_ds['YIELD_ANN_DIFF'] = ds1['YIELD_ANN'] - ds0['YIELD_ANN']
        this_ds['PROD_ANN_DIFF'] = this_ds['YIELD_ANN_DIFF'] * lu_ds['AREA_CFT']
        this_ds = this_ds\
                .sel(time=slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31"))\
                .mean(dim="time")

        make_fig(thisVar, varInfo, cropList_combined_clm_nototal, dpi, figsize, ny, nx, plot_y1, plot_yN, lu_ds, this_ds, this_suptitle, fig_outfile, is_diff)
    
    
    else:
        for this_case in these_cases:
            case = cases[this_case]
            lu_ds = reses[case['res']]['ds']
            this_ds = case['ds']
            this_ds = cc.get_yield_ann(this_ds, min_viable_hui=min_viable_hui, mxmats=None)
            this_ds['PROD_ANN'] = this_ds['YIELD_ANN'] * lu_ds['AREA_CFT']
            this_ds = this_ds\
                .sel(time=slice(f"{plot_y1}-01-01", f"{plot_yN}-12-31"))\
                .mean(dim="time")
                
            this_suptitle = f"{varInfo['suptitle']}: {this_case}"
            fig_outfile = os.path.join(outDir_figs, f"Map {this_suptitle} {plot_y1}-{plot_yN}.png").replace('Mean annual ', '').replace(':', '')
            print(this_suptitle)
            
            make_fig(thisVar, varInfo, cropList_combined_clm_nototal, dpi, figsize, ny, nx, plot_y1, plot_yN, lu_ds, this_ds, this_suptitle, fig_outfile, is_diff)
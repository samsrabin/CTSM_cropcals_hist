import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import shared functions
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import cropcal_module as cc
from cropcal_figs_module import *

abs_cmap_default = 'rainbow'
gslen_colorbar_max = 364


def add_rowcol_labels(axes, fig_caselist, fontsize, nx, ny, rx_row_label):
    # Add row labels
    leftmost = np.arange(0, nx*ny, nx)
    for a, ax in enumerate(axes):
        if a not in leftmost:
            nearest_leftmost = np.max(leftmost[leftmost < a])
            axes[a].sharey(axes[nearest_leftmost])
    for i, a in enumerate(leftmost):
        
        if fig_caselist[i] == "rx":
            row_label = rx_row_label
        else:
            row_label = fig_caselist[i]
        
        axes[a].set_ylabel(row_label, fontsize=fontsize['titles'])
        axes[a].yaxis.set_label_coords(-0.05, 0.5)

    # Add column labels
    topmost = np.arange(nx)
    column_labels = ['Rainfed', 'Irrigated']
    for a, ax in enumerate(axes):
        if a not in topmost:
            nearest_topmost = a % nx
            axes[a].sharex(axes[nearest_topmost])
    for i, a in enumerate(topmost):
        axes[a].set_title(f"{column_labels[i]}",
                        fontsize=fontsize['titles'],
                        y=1.1)


def get_colorbar_chunks(im, ax, this_var, cmap_name, is_diff):
    force_diffmap_within_vrange = False
    cb2 = plt.colorbar(mappable=im, ax=ax, location='bottom')
    ticks_orig = cb2.get_ticks()
    cb2.remove()
    plt.draw()
    Ncolors = len(ticks_orig)-1
    if Ncolors <= 5:
        Ncolors *= 2
    if is_diff and Ncolors % 2 == 0:
        Ncolors += 1
    if is_diff:
        if "DATES" in this_var:
            vmin = -165
            vmax = 165
            Ncolors = 11
            cbar_ticklabels = np.arange(vmin, vmax+1, 30)
            force_diffmap_within_vrange = True
        elif this_var == "HUIFRAC" or this_var == "MATURE":
            vmin = -1.1
            vmax = 1.1
            Ncolors = 11
            cbar_ticklabels = np.arange(vmin, vmax+0.01, 0.2)
            force_diffmap_within_vrange = False
        else:
            cbar_width = max(ticks_orig) - min(ticks_orig)
            cbin_width = cbar_width / Ncolors
            vmax = cbin_width*Ncolors/2
            ceil_to_nearest = 10**np.floor(np.log10(vmax)-1)
            cbin_width = np.ceil(cbin_width / ceil_to_nearest)*ceil_to_nearest
            vmin = -cbin_width*Ncolors/2
            vmax = cbin_width*Ncolors/2
            cbar_ticklabels = np.arange(vmin, vmax+1, cbin_width)
        if this_var == "GSLEN" and np.array_equal(cbar_ticklabels, [-315., -225., -135.,  -45.,   45.,  135.,  225.,  315.]):
            vmin = -330
            vmax = 330
            Ncolors = 11
            cbin_width = 60
            cbar_ticklabels = np.arange(vmin, vmax+1, cbin_width)
            if max(np.abs(ticks_orig)) < cbar_ticklabels[-2]:
                vmin = -cbar_ticklabels[-2]
                vmax = cbar_ticklabels[-2]
                Ncolors -= 2
                cbar_ticklabels = np.arange(vmin, vmax+1, cbin_width)
        if this_var == "GSLEN" and np.array_equal(cbar_ticklabels, [-175., -125.,  -75.,  -25.,   25.,   75.,  125.,  175.]):
            vmin = -210
            vmax = 210
            Ncolors = 7
            cbin_width = 60
            cbar_ticklabels = np.arange(vmin, vmax+1, cbin_width)
            if max(np.abs(ticks_orig)) < cbar_ticklabels[-2]:
                vmin = -cbar_ticklabels[-2]
                vmax = cbar_ticklabels[-2]
                Ncolors -= 2
                cbar_ticklabels = np.arange(vmin, vmax+1, cbin_width)
                
    elif "DATES" in this_var:
        vmin = 0
        vmax = 400
        cbar_ticklabels = [1, 50, 100, 150, 200, 250, 300, 350, 365]
    else:
        vmin = min(ticks_orig)
        vmax = max(ticks_orig)
        cbar_ticklabels = None
    if not is_diff and this_var == "GSLEN" and cbar_ticklabels is not None and cbar_ticklabels[-1] > gslen_colorbar_max:
        cbar_ticklabels[-1] == gslen_colorbar_max
    if cmap_name:
        this_cmap = cm.get_cmap(cmap_name, Ncolors)
    else:
        this_cmap = cm.get_cmap("viridis", Ncolors)
    return vmin, vmax, Ncolors, this_cmap, cbar_ticklabels, force_diffmap_within_vrange


def get_cases_with_var(cases, this_var, lu_ds, min_viable_hui, mxmats_tmp, ref_casename):
    ny = 0
    fig_caselist = []
    for i, (casename, case) in enumerate(cases.items()):
        
        if this_var in ["YIELD_ANN", "PROD_ANN"]:
            case['ds'] = cc.get_yield_ann(case['ds'], min_viable_hui=min_viable_hui, mxmats=mxmats_tmp, lu_ds=lu_ds)
                    
        elif this_var == "MATURE":
            case['ds'] = cc.zero_immatures(case['ds'], out_var="MATURE", min_viable_hui=min_viable_hui, mxmats=mxmats_tmp)
            
        elif this_var == "MATURE_ANN":
            raise RuntimeError("Need to add code for maturity on annual axis")
            
        
        if ref_casename and ref_casename != "rx" and cases[ref_casename]['res'] != case['res']:
            # Not bothering with regridding (for now?)
            pass
        elif this_var in case['ds']:
            ny += 1
            fig_caselist += [casename]
        elif casename == ref_casename:
            raise RuntimeError(f'ref_case {ref_casename} is missing {this_var}')
    
    return fig_caselist, ny


def get_rx_case(cases, fig_caselist, ny, this_var):
    if this_var == "GDDHARV":
        rx_ds_key = "rx_gdds_ds"
    elif this_var == "GSLEN":
        rx_ds_key = "rx_gslen_ds"
    elif this_var == "HDATES":
        rx_ds_key = "rx_hdates_ds"
    elif this_var == "SDATES":
        rx_ds_key = "rx_sdates_ds"
    else:
        raise RuntimeError(f"What rx_ds_key should I use for {this_var}?")
    if this_var in ["GSLEN", "HDATES", "SDATES"]:
        rx_row_label = "GGCMI3"
    elif this_var == "GDDHARV":
        rx_row_label = "GGCMI3-derived"
    else:
        raise RuntimeError(f"What row label should be used instead of 'rx' for {this_var}?")
    rx_parent_found = False
    for i, (casename, case) in enumerate(cases.items()):
        # For now, we're just assuming all runs with a given prescribed variable use the same input file
        if rx_ds_key in case:
            rx_parent_casename = casename
            rx_ds = case[rx_ds_key]
            fig_caselist += ["rx"]
            ny += 1
            rx_parent_found = True
            break
    if not rx_parent_found:
        raise RuntimeError(f"No case found with {rx_ds_key}")
    
    return rx_parent_casename, rx_ds, rx_row_label, ny


def get_figure_info(ny, ref_casename):
    hspace = None
    if ny == 1:
        print("WARNING: Check that the layout looks good for ny == 1")
        figsize = (24, 7.5)	  # width, height
        suptitle_ypos = 0.85
    elif ny == 2:
        figsize = (15, 8.5)		# width, height
        if ref_casename:
            suptitle_xpos = 0.515
            suptitle_ypos = 0.95
        else:
            suptitle_xpos = 0.55
            suptitle_ypos = 1
        cbar_pos = [0.17, 0.05, 0.725, 0.025]	# left edge, bottom edge, width, height
        new_sp_bottom = 0.11
        new_sp_left = None
    elif ny == 3:
        figsize = (14, 10)	 # width, height
        if ref_casename:
            suptitle_xpos = 0.5
            suptitle_ypos = 0.96
        else:
            suptitle_xpos = 0.55
            suptitle_ypos = 0.98
        cbar_pos = [0.2, 0.05, 0.725, 0.025]  # left edge, bottom edge, width, height
        new_sp_bottom = 0.11 # default: 0.1
        new_sp_left = 0.125
        hspace = 0.3
    elif ny == 4:
        figsize = (22, 16)	 # width, height
        suptitle_xpos = 0.55
        suptitle_ypos = 1
        cbar_pos = [0.2, 0.05, 0.725, 0.025]  # left edge, bottom edge, width, height
        new_sp_bottom = 0.11 # default: 0.1
        new_sp_left = 0.125
    else:
        raise ValueError(f"Set up for ny = {ny}")
    return cbar_pos, figsize, hspace, new_sp_bottom, new_sp_left, suptitle_xpos, suptitle_ypos


def loop_case_maps(cases, ny, nx, fig_caselist, c, ref_casename, fontsize, this_var, var_info, rx_row_label, rx_parent_casename, rx_ds, thisCrop_main, found_types, fig, ims, axes, cbs, plot_y1, plot_yN, chunk_colorbar, vmin=None, vmax=None, new_axes=True, Ncolors=None, abs_cmap=None, diff_cmap=None, diff_vmin=None, diff_vmax=None, diff_Ncolors=None, diff_ticklabels=None, force_diffmap_within_vrange=False):
    
    if this_var == "GSLEN":
        cbar_max = gslen_colorbar_max
    else:
        cbar_max = None
    
    allmaps_min = np.inf
    allmaps_max = -np.inf
    allmaps_diff_min = np.inf
    allmaps_diff_max = -np.inf
    for i, casename in enumerate(fig_caselist):
        
        plotting_diffs = ref_casename and casename != ref_casename
        cmap_to_use = None
        if plotting_diffs and not new_axes:
            vmin_to_use = diff_vmin
            vmax_to_use = diff_vmax
            Ncolors_to_use = diff_Ncolors
            cmap_to_use = diff_cmap
            ticklabels_to_use = diff_ticklabels
        else:
            vmin_to_use = vmin
            vmax_to_use = vmax
            Ncolors_to_use = Ncolors
            if not new_axes:
                cmap_to_use = abs_cmap
            ticklabels_to_use = None
            
        if casename == "rx":
            time_dim = "time"
            these_rx_vars = ["gs1_" + str(x) for x in utils.vegtype_str2int(found_types)]
            this_map = xr.concat((rx_ds[x].assign_coords({'ivt_str': found_types[i]}) for i, x in enumerate(these_rx_vars)), dim="ivt_str")
            this_map = this_map.squeeze(drop=True)
            if "lon" not in this_map.dims:
                this_ds = xr.Dataset(data_vars={'tmp': this_map})
                this_map = utils.grid_one_variable(this_ds, 'tmp')
                
            # Apply LU mask
            parent_map, parent_time_dim = get_non_rx_map(var_info, cases, rx_parent_casename, this_var, thisCrop_main, found_types, plot_y1, plot_yN, ref_casename)
            if parent_time_dim == "continue":
                raise RuntimeError("What should I do here?")
            this_map = this_map.where(~np.isnan(parent_map.mean(dim=parent_time_dim)))
        else:
            this_map, time_dim = get_non_rx_map(var_info, cases, casename, this_var, thisCrop_main, found_types, plot_y1, plot_yN, ref_casename)
            if time_dim == "continue":
                continue
        c += 1
            
        # Get mean, set colormap
        diverging_map = "PiYG_r"
        units = var_info['units']
        bounds = None
        vrange = None
        manual_colors = False
        extend = None
        if units == "day of year":
            if time_dim in this_map.dims:
                ar = stats.circmean(this_map, high=365, low=1, axis=this_map.dims.index(time_dim), nan_policy='omit')
                dummy_map = this_map.isel({time_dim: 0}, drop=True)
            else:
                ar = this_map.copy()
                dummy_map = this_map.copy()
            this_map = xr.DataArray(data = ar,
                                    coords = dummy_map.coords,
                                    attrs = dummy_map.attrs)
            if plotting_diffs:
                this_map_vals = (this_map - refcase_map).values
                this_map_vals[this_map_vals > 365/2] -= 365
                this_map_vals[this_map_vals < -365/2] += 365
                this_map = xr.DataArray(data = this_map_vals,
                                        coords = this_map.coords,
                                        attrs = this_map.attrs)
                if diff_cmap:
                    cmap = diff_cmap
                else:
                    cmap = diverging_map
                
                units = "days"
                if chunk_colorbar:
                    manual_colors = True
                    bounds = np.arange(-165, 165+1, 30)
                    ticklabels_to_use = bounds
                    cmap_to_use = cm.get_cmap(cmap)
                    extend = "both"
                else:
                    vmax = np.nanmax(np.abs(this_map.values))
                    vrange = [-vmax, vmax]
            else:
                if abs_cmap:
                    cmap = abs_cmap
                else:
                    cmap = 'twilight_shifted'
                
                units = "Day of year"
                if chunk_colorbar:
                    manual_colors = True
                    bounds = np.append(np.append(1, np.arange(50, 350+1, 50)), 365)
                    cmap_to_use = cm.get_cmap(cmap)
                    extend = "neither"
                else:
                    vrange = [1, 365]
        elif units == "month":
            if plotting_diffs:
                this_map_vals = (this_map - refcase_map).values
                this_map_vals[this_map_vals > 6] -= 12
                this_map_vals[this_map_vals < -6] += 12
                this_map = xr.DataArray(data = this_map_vals,
                                        coords = this_map.coords,
                                        attrs = this_map.attrs)
                
                if diff_cmap:
                    cmap = diff_cmap
                else:
                    cmap = diverging_map
                    
                manual_colors = True
                bounds = np.arange(-6.5, 7.5, 1)
                cmap_to_use = cm.get_cmap(cmap)
                extend = "neither"
                ticklabels_to_use = np.arange(-6,7,1)
                units="months"
            else:
                manual_colors = True
                bounds = np.arange(0.5, 13.5, 1)
                cmap_to_use = cm.get_cmap('twilight_shifted')
                extend = "neither"
                ticklabels_to_use = np.arange(1,13,1)
                units = "Month"
                
        else: # other units
            if time_dim in this_map.dims:
                this_map = this_map.mean(dim=time_dim)
            this_map *= var_info['multiplier']
            if plotting_diffs:
                this_map = this_map - refcase_map
                if diff_cmap:
                    cmap = diff_cmap
                else:
                    cmap = diverging_map
                
                if new_axes:
                    vrange = list(np.nanmax(np.abs(this_map.values)) * np.array([-1,1]))
                else:
                    vrange = None
            else:
                if abs_cmap:
                    cmap = abs_cmap
                else:
                    cmap = 'viridis'
                vrange = None
            
        if casename == ref_casename:
            refcase_map = this_map.copy()
            
        if Ncolors and cmap_to_use is None:
            cmap_to_use = cm.get_cmap(cmap, Ncolors_to_use)
                
        cbar_units = units
        if plotting_diffs:
            if ref_casename == "rx":
                cbar_units = f"Diff. from {rx_row_label} ({units})"
            else:
                cbar_units = f"Diff. from {ref_casename} ({units})"
            if not np.any(np.abs(this_map) > 0):
                print(f'		{casename} identical to {ref_casename}!')
                cbar_units += ": None!"
            
        rainfed_types = [x for x in found_types if "irrigated" not in x]
        subplot_num = nx*c+1
        subplot_str = chr(ord('`') + subplot_num) # or ord('@') for capital
        if new_axes:
            ax = fig.add_subplot(ny,nx,subplot_num,projection=ccrs.PlateCarree(), ylabel="mirntnt")
            axes.append(ax)
            cb = None
        else:
            ax = axes[i*2]
            cb = cbs[i*2]
        thisCrop = thisCrop_main
        this_map_sel = this_map.copy().sel(ivt_str=thisCrop)
        if plotting_diffs and not new_axes and force_diffmap_within_vrange:
            this_map_sel_vals = this_map_sel.copy().values
            this_map_sel_vals[np.where(this_map_sel_vals < diff_vmin)] = diff_vmin
            this_map_sel_vals[np.where(this_map_sel_vals > diff_vmax)] = diff_vmax
            this_map_sel = xr.DataArray(data=this_map_sel_vals,
                                        coords=this_map_sel.coords,
                                        attrs=this_map_sel.attrs)
            if extend is None:
                extend = "both"
        elif extend is None:
            extend = "neither"
        if cmap_to_use is None:
            cmap_to_use = cmap
        
        # Check current map's extremes against previous extremes
        if plotting_diffs:
            allmaps_diff_min = min(allmaps_diff_min, np.nanmin(this_map_sel))
            allmaps_diff_max = max(allmaps_diff_max, np.nanmax(this_map_sel))
        else:
            allmaps_min = min(allmaps_min, np.nanmin(this_map_sel))
            allmaps_max = max(allmaps_max, np.nanmax(this_map_sel))
        
        im, cb = make_map(ax, this_map_sel, fontsize, units=cbar_units, cmap=cmap_to_use, vrange=vrange, linewidth=0.5, show_cbar=bool(ref_casename), vmin=vmin_to_use, vmax=vmax_to_use, cbar=cb, ticklabels=ticklabels_to_use, extend_nonbounds=extend, bounds=bounds, extend_bounds=extend, subplot_label=subplot_str, cbar_max=cbar_max)
        if new_axes:
            ims.append(im)
            cbs.append(cb)
        else:
            ims[i*2] = im
            cbs[i*2] = cb

        irrigated_types = [x for x in found_types if "irrigated" in x]
        subplot_num = nx*c+2
        subplot_str = chr(ord('`') + subplot_num) # or ord('@') for capital
        if new_axes:
            ax = fig.add_subplot(ny,nx,subplot_num,projection=ccrs.PlateCarree())
            axes.append(ax)
            cb = None
        else:
            ax = axes[i*2 + 1]
            cb = cbs[i*2 + 1]
        thisCrop = "irrigated_" + thisCrop_main
        this_map_sel = this_map.copy().sel(ivt_str=thisCrop)
        if plotting_diffs and not new_axes and force_diffmap_within_vrange:
            this_map_sel_vals = this_map_sel.copy().values
            this_map_sel_vals[np.where(this_map_sel_vals < diff_vmin)] = diff_vmin
            this_map_sel_vals[np.where(this_map_sel_vals > diff_vmax)] = diff_vmax
            this_map_sel = xr.DataArray(data=this_map_sel_vals,
                                        coords=this_map_sel.coords,
                                        attrs=this_map_sel.attrs)
            if extend is None:
                extend = "both"
        else:
            if extend is None:
                extend = "neither"
        if cmap_to_use is None:
            cmap_to_use = cmap
        
        # Check current map's extremes against previous extremes
        if plotting_diffs:
            allmaps_diff_min = min(allmaps_diff_min, np.nanmin(this_map_sel))
            allmaps_diff_max = max(allmaps_diff_max, np.nanmax(this_map_sel))
        else:
            allmaps_min = min(allmaps_min, np.nanmin(this_map_sel))
            allmaps_max = max(allmaps_max, np.nanmax(this_map_sel))
        
        im, cb = make_map(ax, this_map_sel, fontsize, units=cbar_units, cmap=cmap_to_use, vrange=vrange, linewidth=0.5, show_cbar=bool(ref_casename), vmin=vmin_to_use, vmax=vmax_to_use, cbar=cb, ticklabels=ticklabels_to_use, extend_nonbounds=extend, bounds=bounds, extend_bounds=extend, subplot_label=subplot_str, cbar_max=cbar_max)
        if new_axes:
            ims.append(im)
            cbs.append(cb)
        else:
            ims[i*2 + 1] = im
            cbs[i*2 + 1] = cb
            
    vrange = [allmaps_min, allmaps_max]
    if not np.isinf(allmaps_diff_min):
        vmax_diff = max(np.abs([allmaps_diff_min, allmaps_diff_max]))
        vrange_diff = [-vmax_diff, vmax_diff]
    else:
        vrange_diff = None
    
    return units, vrange, vrange_diff, fig, ims, axes, cbs, manual_colors


def maps_eachCrop(cases, clm_types, clm_types_rfir, dpi, fontsize, lu_ds, min_viable_hui, mxmats_tmp, nx, outDir_figs, overwrite, plot_y1, plot_yN, ref_casename, varList, chunk_colorbar=False):

    for (this_var, var_info) in varList.items():
        
        if var_info['time_dim'] == "time":
            yrange_str = f'{plot_y1}-{plot_yN}'
        else:
            yrange_str = f'{plot_y1}-{plot_yN-1} growing seasons'
        suptitle = var_info['suptitle'] + f' ({yrange_str})'
        
        print(f'Mapping {this_var}...')
        
        # Get colormap
        abs_cmap = abs_cmap_default
        if "DATE" in this_var or "PKMTH" in this_var:
            abs_cmap = "twilight_shifted"
        if ("YIELD" in this_var or "PROD" in this_var) and ref_casename:
            diff_cmap = "BrBG"
        else:
            diff_cmap = "PiYG_r"

        # First, determine how many cases have this variable
        fig_caselist, ny = get_cases_with_var(cases, this_var, lu_ds, min_viable_hui, mxmats_tmp, ref_casename)
        if ny == 0:
            print(f"No cases contain {this_var}; skipping.")
            continue
        
        # Add "prescribed" "case," if relevant
        rx_row_label = None
        rx_parent_casename = None
        rx_ds = None
        if this_var in ["GDDHARV", "GSLEN", "HDATES", "SDATES"]:
            rx_parent_casename, rx_ds, rx_row_label, ny = get_rx_case(cases, fig_caselist, ny, this_var)
        elif ref_casename == "rx":
            print(f"Skipping {this_var} because it has no rx dataset against which to compare simulations")
            continue
        
        # Rearrange caselist for this figure so that reference case is first
        if ref_casename:
            if len(fig_caselist) <= 1:
                raise RuntimeError(f"Only ref case {ref_casename} has {this_var}")
            fig_caselist = [ref_casename] + [x for x in fig_caselist if x != ref_casename]
        
        # Now set some figure parameters based on # cases
        cbar_pos, figsize, hspace, new_sp_bottom, new_sp_left, suptitle_xpos, suptitle_ypos = get_figure_info(ny, ref_casename)

        for thisCrop_main in clm_types:
            this_suptitle = thisCrop_main.capitalize() + ": " + suptitle
                    
            # Get the name we'll use in output text/filenames
            thisCrop_out = thisCrop_main
            if "soybean" in thisCrop_out and "tropical" not in thisCrop_out:
                thisCrop_out = thisCrop_out.replace("soy", "temperate_soy")
            
            # Skip if file exists and we're not overwriting
            diff_txt = ""
            if ref_casename == "rx":
                diff_txt = f" Diff {rx_row_label}"
            elif ref_casename:
                diff_txt = f" Diff {ref_casename}"
            fig_outfile = outDir_figs + "Map " + suptitle + diff_txt + f" {thisCrop_out}.png"
            if any(x in this_var for x in ["YIELD", "PROD"]):
                fig_outfile = fig_outfile.replace(".png", f" {min_viable_hui}-mat.png")
            if os.path.exists(fig_outfile) and not overwrite:
                print(f'    Skipping {thisCrop_out} (file exists).')
                continue
            
            print(f'    {thisCrop_out}...')
            found_types = [x for x in clm_types_rfir if thisCrop_main in x]
            
            c = -1
            fig = plt.figure(figsize=figsize)
            ims = []
            axes = []
            cbs = []
            units, vrange, vrange_diff, fig, ims, axes, cbs, manual_colors = loop_case_maps(cases, ny, nx, fig_caselist, c, ref_casename, fontsize, this_var, var_info, rx_row_label, rx_parent_casename, rx_ds, thisCrop_main, found_types, fig, ims, axes, cbs, plot_y1, plot_yN, chunk_colorbar, abs_cmap=abs_cmap, diff_cmap=diff_cmap)

            fig.suptitle(this_suptitle,
                            x = suptitle_xpos,
                            y = suptitle_ypos,
                            fontsize = fontsize['suptitle'])

            # Add row and column labels
            add_rowcol_labels(axes, fig_caselist, fontsize, nx, ny, rx_row_label)
            
            # Draw all-subplot colorbar
            if not ref_casename:
                cbar_ax = fig.add_axes(cbar_pos)
                fig.tight_layout()
                cb = fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal', label=units)
                if "PKMTH" in this_var:
                    cb.set_ticks(np.arange(1,13))
                    cb.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                    cb.ax.tick_params(length=0)
                cb.ax.tick_params(labelsize=fontsize['ticklabels'])
                cb.set_label(units, fontsize=fontsize['titles'])
            
            if not manual_colors:
                if ref_casename:
                    extend, eq_vrange = cc.equalize_colorbars(ims[:nx], this_var=this_var, vrange=vrange)
                    extend, diff_eq_vrange = cc.equalize_colorbars(ims[nx:], this_var=this_var, vrange=vrange_diff)
                else:
                    extend, eq_vrange = cc.equalize_colorbars(ims, this_var=this_var, vrange=vrange)
                
                # Chunk colorbar
                cbar_ticklabels = None
                if chunk_colorbar:
                    vmin, vmax, Ncolors, this_cmap, cbar_ticklabels, force_diffmap_within_vrange = get_colorbar_chunks(ims[0], axes[0], this_var, abs_cmap, False)
                    if ref_casename:
                        diff_vmin, diff_vmax, diff_Ncolors, diff_this_cmap, diff_cbar_ticklabels, force_diffmap_within_vrange = get_colorbar_chunks(ims[2], axes[2], this_var, diff_cmap, True)
                        while diff_eq_vrange[1] <= diff_cbar_ticklabels[-2]:
                            diff_cbar_ticklabels = diff_cbar_ticklabels[1:-1]
                            diff_vmin = diff_cbar_ticklabels[0]
                            diff_vmax = diff_cbar_ticklabels[-1]
                            diff_Ncolors -= 2
                            if diff_Ncolors <= 0:
                                raise RuntimeError("Infinite loop?")
                            diff_this_cmap = cm.get_cmap(diff_cmap, diff_Ncolors)
                        if diff_eq_vrange[1] <= diff_cbar_ticklabels[-2]:
                            print(f"diff_eq_vrange: {diff_eq_vrange}")
                            print(f"range from get_colorbar_chunks(): [{diff_vmin}, {diff_vmax}]")
                            print(f"diff_Ncolors: {diff_Ncolors}")
                            print(f"diff_cbar_ticklabels: {diff_cbar_ticklabels}")
                            print(f"diff_this_cmap: {diff_this_cmap}")
                            raise RuntimeError("Extra bin(s)!")
                    else:
                        diff_vmin = None
                        diff_vmax = None
                        diff_Ncolors = None
                        diff_this_cmap = None
                        diff_cbar_ticklabels = None
                    units, vrange, vrange_diff, fig, ims, axes, cbs, manual_colors = loop_case_maps(cases, ny, nx, fig_caselist, c, ref_casename, fontsize, this_var, var_info, rx_row_label, rx_parent_casename, rx_ds, thisCrop_main, found_types, fig, ims, axes, cbs, plot_y1, plot_yN, chunk_colorbar, vmin=vmin, vmax=vmax, new_axes=False, Ncolors=Ncolors, abs_cmap=this_cmap, diff_vmin=diff_vmin, diff_vmax=diff_vmax, diff_Ncolors=diff_Ncolors, diff_cmap=diff_this_cmap, diff_ticklabels=diff_cbar_ticklabels, force_diffmap_within_vrange=force_diffmap_within_vrange)
                
                # Redraw all-subplot colorbar 
                if not ref_casename:
                    cbar_ax = fig.add_axes(cbar_pos)
                    fig.tight_layout()
                    cb.remove()
                    cb = fig.colorbar(ims[0], cax=cbar_ax, orientation='horizontal', label=units, extend = extend)
                    cb.ax.tick_params(labelsize=fontsize['ticklabels'])
                    cb.set_label(units, fontsize=fontsize['titles'])
                    if cbar_ticklabels is not None:
                        cb.set_ticks(cb.get_ticks()) # Does nothing except to avoid "FixedFormatter should only be used together with FixedLocator" warning in call of cb.set_ticklabels() below
                        cb.set_ticklabels(cbar_ticklabels)
            
            plt.subplots_adjust(bottom=new_sp_bottom, left=new_sp_left)
            if hspace is not None:
                plt.subplots_adjust(hspace=hspace)
            
            # plt.show()
            # return
            
            fig.savefig(fig_outfile,
                        bbox_inches='tight', facecolor='white', dpi=dpi)
            plt.close()

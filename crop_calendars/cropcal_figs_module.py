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


def loop_case_maps(cases, ny, nx, fig_caselist, c, ref_casename, fontsize, this_var, var_info, rx_row_label, rx_parent_casename, rx_ds, thisCrop_main, found_types, fig, ims, axes, cbs, plot_y1, plot_yN, vmin=None, vmax=None, new_axes=True, Ncolors=None, abs_cmap=None, diff_cmap=None, diff_vmin=None, diff_vmax=None, diff_Ncolors=None, diff_ticklabels=None, force_diffmap_within_vrange=False):
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
                    
                vrange = [-165, 165]
                units = "days"
            else:
                if abs_cmap:
                    cmap = abs_cmap
                else:
                    cmap = 'twilight_shifted'
                vrange = [1, 365]
                if vmin != None:
                    vrange[0] = vmin_to_use
                if vmax != None:
                    vrange[1] = vmax_to_use
        else:
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
        if new_axes:
            ax = fig.add_subplot(ny,nx,nx*c+1,projection=ccrs.PlateCarree(), ylabel="mirntnt")
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
            extend = "both"
        else:
            extend = "neither"
        im, cb = make_map(ax, this_map_sel, fontsize, units=cbar_units, cmap=cmap_to_use, vrange=vrange, linewidth=0.5, show_cbar=bool(ref_casename), vmin=vmin_to_use, vmax=vmax_to_use, cbar=cb, ticklabels=ticklabels_to_use, extend_nonbounds=extend)
        if new_axes:
            ims.append(im)
            cbs.append(cb)
        else:
            ims[i*2] = im
            cbs[i*2] = cb

        irrigated_types = [x for x in found_types if "irrigated" in x]
        if new_axes:
            ax = fig.add_subplot(ny,nx,nx*c+2,projection=ccrs.PlateCarree())
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
            extend = "both"
        else:
            extend = "neither"
        im, cb = make_map(ax, this_map_sel, fontsize, units=cbar_units, cmap=cmap, vrange=vrange, linewidth=0.5, show_cbar=bool(ref_casename), vmin=vmin_to_use, vmax=vmax_to_use, cbar=cb, ticklabels=ticklabels_to_use, extend_nonbounds=extend)
        if new_axes:
            ims.append(im)
            cbs.append(cb)
        else:
            ims[i*2 + 1] = im
            cbs[i*2 + 1] = cb
    return units, vrange, fig, ims, axes, cbs


def make_map(ax, this_map, fontsize, lonlat_bin_width=None, units=None, cmap='viridis', vrange=None, linewidth=1.0, this_title=None, show_cbar=False, bounds=None, extend_bounds='both', vmin=None, vmax=None, cbar=None, ticklabels=None, extend_nonbounds='both'): 
    
    if bounds:
        norm = mcolors.BoundaryNorm(bounds, cmap.N, extend=extend_bounds)
        im = ax.pcolormesh(this_map.lon.values, this_map.lat.values,
                           this_map, shading="auto",
                           norm=norm,
                           cmap=cmap,
                           vmin=vmin, vmax=vmax)
    else:
        im = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
                           this_map, shading="auto",
                           cmap=cmap,
                           vmin=vmin, vmax=vmax)
        if vrange:
            im.set_clim(vrange[0], vrange[1])
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    
    # ax.add_feature(cfeature.BORDERS, linewidth=linewidth, edgecolor="white")
    # ax.add_feature(cfeature.BORDERS, linewidth=linewidth*0.6)
    ax.coastlines(linewidth=linewidth, color="white")
    ax.coastlines(linewidth=linewidth*0.6)
    
    if this_title:
        ax.set_title(this_title, fontsize=fontsize['titles'])
    if show_cbar:
        if cbar:
            cbar.remove()
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.1, pad=0.02, extend=extend_nonbounds)
        if ticklabels is not None:
            cbar.set_ticks(ticklabels)
        cbar.set_label(label=units, fontsize=fontsize['axislabels'])
        cbar.ax.tick_params(labelsize=fontsize['ticklabels'])
     
    def set_ticks(lonlat_bin_width, fontsize, x_or_y):
        
        if x_or_y == "x":
            ticks = np.arange(-180, 181, lonlat_bin_width)
        else:
            ticks = np.arange(-60, 91, lonlat_bin_width)
            
        ticklabels = [str(x) for x in ticks]
        for i,x in enumerate(ticks):
            if x%2:
                ticklabels[i] = ''
        
        if x_or_y == "x":
            plt.xticks(ticks, labels=ticklabels,
                       fontsize=fontsize['ticklabels'])
        else:
            plt.yticks(ticks, labels=ticklabels,
                       fontsize=fontsize['ticklabels'])
    
    if lonlat_bin_width:
        set_ticks(lonlat_bin_width, fontsize, "y")
        # set_ticks(lonlat_bin_width, fontsize, "x")
    else:
        # Need to do this for subplot row labels
        set_ticks(-1, fontsize, "y")
        plt.yticks([])
    for x in ax.spines:
        ax.spines[x].set_visible(False)
    
    if show_cbar:
        return im, cbar
    else:
        return im, None


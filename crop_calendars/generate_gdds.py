# What system is the script running on?
import socket
hostname = socket.gethostname()

# Import the CTSM Python utilities, functions for GDD generation
import sys
if hostname == "Sams-2021-MacBook-Pro.local":
    sys.path.append("/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/")
    import utils
    sys.path.append("/Users/sam/Documents/git_repos/CTSM_cropcals_hist/crop_calendars")
    import generate_gdds_functions as gddfn
else:
    # Only possible because I have export PYTHONPATH=$HOME in my .bash_profile
    from ctsm_python_gallery_myfork.ctsm_py import utils
    from CTSM_cropcals_hist.crop_calendars import generate_gdds_functions as gddfn

# Import everything else
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings
import cartopy.crs as ccrs
import datetime as dt
import pickle
import argparse

plt.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


# Suppress some warnings
import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")


def main(argv):

    help_string = "generate_gdds.py -r <run-dir> -s <sdates-file> -h <hdates-file> -1 <first-season> -N <last-season> [--no-save-figs --args.only_make_figs]"

    ###############################
    ### Process input arguments ###
    ###############################
    
    # Set arguments
    parser = argparse.ArgumentParser(description="ADD DESCRIPTION HERE")
    parser.add_argument("-r", "--run-dir", 
                        help="Directory where run outputs can be found (and where outputs will go)",
                        required=True)
    parser.add_argument("-1", "--first-season", 
                        help="First growing season to include in calculation of mean",
                        required=True)
    parser.add_argument("-n", "-N", "--last-season", 
                        help="Last growing season to include in calculation of mean",
                        required=True)
    parser.add_argument("-sd", "--sdates-file", 
                        help="File of prescribed sowing dates",
                        required=True)
    parser.add_argument("-hd", "--hdates-file", 
                        help="File of prescribed harvest dates",
                        required=True)
    figsgroup = parser.add_mutually_exclusive_group()
    figsgroup.add_argument("--dont-save-figs", 
                           help="Do not save figures or files needed to create them",
                           action="store_true", default=False)
    figsgroup.add_argument("--only-make-figs", 
                           help="Use preprocessed files to make figures only",
                           action="store_true", default=False)
    parser.add_argument("--run1-name", 
                        help="Name of original values to show in figures",
                        default="Old")
    parser.add_argument("--run2-name", 
                        help="Name of new values to show in figures",
                        default="New")
    
    # Get arguments
    args = parser.parse_args()
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    save_figs = not args.dont_save_figs
        
    # Directories to save output files and figures
    outdir = os.path.join(args.run_dir, "generate_gdds")
    outdir_figs = os.path.join(outdir, "figs")
    
    
    ##########################
    ### Import and process ###
    ##########################
    
    if not args.only_make_figs:
    
        # Keep 1 extra year to avoid incomplete final growing season for crops harvested after Dec. 31.
        y1_import_str = f"{args.first_season+1}-01-01"
        yN_import_str = f"{args.last_season+2}-01-01"
        
        print(f"Importing netCDF time steps {y1_import_str} through {yN_import_str} (years are +1 because of CTSM output naming)")
        
        pickle_file = os.path.join(outdir, f'{args.first_season}-{args.last_season}.pickle')
        h1_ds_file = os.path.join(outdir, f'{args.first_season}-{args.last_season}.h1_ds.nc')
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                args.first_season, args.last_season, pickle_year, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices_list, incorrectly_daily, gddharv_in_h3, save_figs, incl_vegtypes_str, incl_patches1d_itype_veg, mxsowings = pickle.load(f)
            print(f'Will resume import at {pickle_year+1}')
            h1_ds = None
        else:
            incorrectly_daily = False
            skip_patches_for_isel_nan_lastyear = np.ndarray([])
            gddharv_in_h3 = False
            pickle_year = -np.inf
            gddaccum_yp_list = []
            gddharv_yp_list = []
            incl_vegtypes_str = None
            lastYear_active_patch_indices_list = None
        sdates_rx = args.sdates_file
        hdates_rx = args.hdates_file
        
        for y, thisYear in enumerate(np.arange(args.first_season+1,args.last_season+3)):
            
            if thisYear <= pickle_year:
                continue
            
            h1_ds, sdates_rx, hdates_rx, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices_list, incorrectly_daily, gddharv_in_h3, incl_vegtypes_str, incl_patches1d_itype_veg, mxsowings = gddfn.import_and_process_1yr(args.first_season, args.last_season, y, thisYear, sdates_rx, hdates_rx, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices_list, incorrectly_daily, gddharv_in_h3, save_figs, args.run_dir, incl_vegtypes_str, h1_ds_file)
            
            print(f'   Saving pickle file ({pickle_file})...')
            with open(pickle_file, 'wb') as f:
                pickle.dump([args.first_season, args.last_season, thisYear, gddaccum_yp_list, gddharv_yp_list, skip_patches_for_isel_nan_lastyear, lastYear_active_patch_indices_list, incorrectly_daily, gddharv_in_h3, save_figs, incl_vegtypes_str, incl_patches1d_itype_veg, mxsowings], f, protocol=-1)
                
        
        if isinstance(incl_vegtypes_str, list):
            incl_vegtypes_str = np.array(incl_vegtypes_str)
        plot_vegtypes_str = incl_vegtypes_str[[i for i,c in enumerate(gddaccum_yp_list) if not isinstance(c,type(None))]]
        
        print("Done")
        
        if not h1_ds:
            h1_ds = xr.open_dataset(h1_ds_file)
    
    
    ######################################################
    ### Get and grid mean GDDs in GGCMI growing season ###
    ######################################################
    
    if not args.only_make_figs:
    
        longname_prefix = "GDD harvest target for "
        
        # Could skip this by saving sdates_rx['time_bounds']
        sdates_rx = gddfn.import_rx_dates("s", sdates_rx, incl_patches1d_itype_veg, mxsowings)
        
        print('Getting and gridding mean GDDs...')
        gdd_maps_ds = gddfn.yp_list_to_ds(gddaccum_yp_list, h1_ds, incl_vegtypes_str, sdates_rx, longname_prefix)
        if save_figs:gddharv_maps_ds = gddfn.yp_list_to_ds(gddharv_yp_list, h1_ds, incl_vegtypes_str, sdates_rx, longname_prefix)
        
        # Fill NAs with dummy values
        dummy_fill = -1
        gdd_fill0_maps_ds = gdd_maps_ds.fillna(0)
        gdd_maps_ds = gdd_maps_ds.fillna(dummy_fill)
        print('Done getting and gridding means.')
        
        # Add dummy variables for crops not actually simulated
        print("Adding dummy variables...")
        # Unnecessary?
        template_ds = xr.open_dataset(args.sdates_file, decode_times=True)
        all_vars = [v.replace("sdate","gdd") for v in template_ds if "sdate" in v]
        all_longnames = [template_ds[v].attrs["long_name"].replace("Planting day ", longname_prefix) + " (dummy)" for v in template_ds if "sdate" in v]
        dummy_vars = []
        dummy_longnames = []
        for v, thisVar in enumerate(all_vars):
            if thisVar not in gdd_maps_ds:
                dummy_vars.append(thisVar)
                dummy_longnames.append(all_longnames[v])
        
        def make_dummy(thisCrop_gridded, addend):
            dummy_gridded = thisCrop_gridded
            dummy_gridded.values = dummy_gridded.values*0 + addend
            return dummy_gridded
        for v in gdd_maps_ds:
            thisCrop_gridded = gdd_maps_ds[v].copy()
            thisCrop_fill0_gridded = gdd_fill0_maps_ds[v].copy()
            break
        dummy_gridded = make_dummy(thisCrop_gridded, -1)
        dummy_gridded0 = make_dummy(thisCrop_fill0_gridded, 0)
        
        for v, thisVar in enumerate(dummy_vars):
            if thisVar in gdd_maps_ds:
                raise RuntimeError(f'{thisVar} is already in gdd_maps_ds. Why overwrite it with dummy?')
            dummy_gridded.name = thisVar
            dummy_gridded.attrs["long_name"] = dummy_longnames[v]
            gdd_maps_ds[thisVar] = dummy_gridded
            dummy_gridded0.name = thisVar
            dummy_gridded0.attrs["long_name"] = dummy_longnames[v]
            gdd_fill0_maps_ds[thisVar] = dummy_gridded0
        
        # Add lon/lat attributes
        def add_lonlat_attrs(ds):
            ds.lon.attrs = {\
                "long_name": "coordinate_longitude",
                "units": "degrees_east"}
            ds.lat.attrs = {\
                "long_name": "coordinate_latitude",
                "units": "degrees_north"}
            return ds
        gdd_maps_ds = add_lonlat_attrs(gdd_maps_ds)
        gdd_fill0_maps_ds = add_lonlat_attrs(gdd_fill0_maps_ds)
        if save_figs: gddharv_maps_ds = add_lonlat_attrs(gddharv_maps_ds)
        
        print("Done.")
    
    
    ######################
    ### Save to netCDF ###
    ######################
    
    if not args.only_make_figs:
        print("Saving...")
        
        # Get output file path
        datestr = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join(outdir, "gdds_" + datestr + ".nc")
        outfile_fill0 = os.path.join(outdir, "gdds_fill0_" + datestr + ".nc")
        
        def save_gdds(args, outfile, gdd_maps_ds, sdates_rx):
            # Set up output file from template (i.e., prescribed sowing dates).
            template_ds = xr.open_dataset(args.sdates_file, decode_times=True)
            for v in template_ds:
                if "sdate" in v:
                    template_ds = template_ds.drop(v)
            template_ds.to_netcdf(path=outfile, format="NETCDF3_CLASSIC")
            template_ds.close()
        
            # Add global attributes
            comment = f"Derived from CLM run plus crop calendar input files {os.path.basename(args.sdates_file) and {os.path.basename(args.hdates_file)}}."
            gdd_maps_ds.attrs = {\
                "author": "Sam Rabin (sam.rabin@gmail.com)",
                "comment": comment,
                "created": dt.datetime.now().astimezone().isoformat()
                }
        
            # Add time_bounds
            gdd_maps_ds["time_bounds"] = sdates_rx.time_bounds
        
            # Save cultivar GDDs
            gdd_maps_ds.to_netcdf(outfile, mode="w", format="NETCDF3_CLASSIC")
        
        save_gdds(args, outfile, gdd_maps_ds, sdates_rx)
        save_gdds(args, outfile_fill0, gdd_fill0_maps_ds, sdates_rx)
        
        print("Done saving.")
    
    
    ########################################
    ### Save things needed for mapmaking ###
    ########################################
    
    def add_attrs_to_map_ds(map_ds, incl_vegtypes_str, dummy_fill, outdir_figs, args):
        return map_ds.assign_attrs({'incl_vegtypes_str': incl_vegtypes_str,
                                    'dummy_fill': dummy_fill,
                                    'outdir_figs': outdir_figs,
                                    'args.first_season': args.first_season,
                                    'args.last_season': args.last_season})
    
    if save_figs and not args.only_make_figs:
        if not os.path.exists(outdir_figs):
            os.makedirs(outdir_figs)

        gdd_maps_ds = add_attrs_to_map_ds(gdd_maps_ds, plot_vegtypes_str, dummy_fill, outdir_figs, args)
        gddharv_maps_ds = add_attrs_to_map_ds(gddharv_maps_ds, plot_vegtypes_str, dummy_fill, outdir_figs, args)
        
        gdd_maps_ds.to_netcdf(os.path.join(outdir_figs, "gdd_maps.nc"))
        gddharv_maps_ds.to_netcdf(os.path.join(outdir_figs, "gddharv_maps.nc"))
    
    
    #################################################
    ### Save before/after map and boxplot figures ###
    #################################################
    
    def get_bounds_ncolors(gdd_spacing, diff_map_yx):
        vmax = np.floor(np.nanmax(diff_map_yx.values)/gdd_spacing)*gdd_spacing
        vmin = -vmax
        epsilon = np.nextafter(0, 1)
        bounds = list(np.arange(vmin, vmax, gdd_spacing)) + [vmax-epsilon]
        if 0 in bounds:
            bounds.remove(0)
            bounds[bounds.index(-gdd_spacing)] /= 2
            bounds[bounds.index(gdd_spacing)] /= 2
        Ncolors = len(bounds) + 1
        return vmax, bounds, Ncolors
    
    def make_map(ax, this_map, this_title, vmax, bin_width, fontsize_ticklabels, fontsize_titles, bounds=None, extend='both', cmap=None, cbar_ticks=None):
        
        if bounds:
            if not cmap:
                raise RuntimeError("Calling make_map() with bounds requires cmap to be specified")
            norm = mcolors.BoundaryNorm(bounds, cmap.N, extend=extend)
            im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values,
                                this_map, shading="auto",
                                norm=norm,
                                cmap=cmap)
        else:
            if np.any(this_map.values < 0):
                gdd_spacing = 500
                vmax = np.floor(np.nanmax(this_map.values)/gdd_spacing)*gdd_spacing
                vmin = -vmax
                Ncolors = vmax/gdd_spacing
                if Ncolors % 2 == 0: Ncolors += 1
                if not cmap:
                    cmap = cm.get_cmap("RdYlBu_r", Ncolors)
                
                if np.any(this_map.values > vmax) and np.any(this_map.values < vmin):
                    extend = 'both'
                elif np.any(this_map.values > vmax):
                    extend = 'max'
                elif np.any(this_map.values < vmin):
                    extend = 'min'
                else:
                    extend = 'neither'
                
            else:
                vmin = 0
                vmax = np.floor(vmax/500)*500
                Ncolors = vmax/500
                if not cmap:
                    cmap=cm.get_cmap("jet", Ncolors)
                extend = 'max'
                
            im1 = ax.pcolormesh(this_map.lon.values, this_map.lat.values, 
                    this_map, shading="auto",
                    vmin=vmin, vmax=vmax,
                    cmap=cmap)
            
        ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.3)
        ax.set_title(this_title, fontsize=fontsize_titles, fontweight="bold", y=0.96)
        cbar = plt.colorbar(im1, orientation="horizontal", fraction=0.1, pad=0.02,
                            aspect=40, extend=extend, spacing='proportional')
        cbar.ax.tick_params(labelsize=fontsize_ticklabels)
        cbar.ax.set_xlabel(this_map.attrs['units'],
                           fontsize=fontsize_ticklabels)
        cbar.ax.xaxis.set_label_coords(x=0.115, y=2.6)
        if cbar_ticks:
            cbar.ax.set_xticks(cbar_ticks)
        
        ticks = np.arange(-60, 91, bin_width)
        ticklabels = [str(x) for x in ticks]
        for i,x in enumerate(ticks):
            if x%2:
                ticklabels[i] = ''
        plt.yticks(np.arange(-60,91,15), labels=ticklabels,
                   fontsize=fontsize_ticklabels)
        plt.axis('off')
        
    def get_non_nans(in_da, fillValue):
        in_da = in_da.where(in_da != fillValue)
        return in_da.values[~np.isnan(in_da.values)]
    
    linewidth = 1.5
    def set_boxplot_props(bp, color, linewidth):
        linewidth = linewidth
        plt.setp(bp['boxes'], color=color, linewidth=linewidth)
        plt.setp(bp['whiskers'], color=color, linewidth=linewidth)
        plt.setp(bp['caps'], color=color, linewidth=linewidth)
        plt.setp(bp['medians'], color=color, linewidth=linewidth)
        plt.setp(bp['fliers'], markeredgecolor=color, markersize=6, linewidth=linewidth, markeredgewidth=linewidth/2)
    
    def make_plot(data, offset, linewidth):
        offset = 0.4*offset
        bpl = plt.boxplot(data, positions=np.array(range(len(data)))*2.0+offset, widths=0.6, 
                          boxprops=dict(linewidth=linewidth), whiskerprops=dict(linewidth=linewidth), 
                          capprops=dict(linewidth=linewidth), medianprops=dict(linewidth=linewidth),
                          flierprops=dict(markeredgewidth=0.5))
        return bpl
    
    def make_figures(args, thisDir=None, gdd_maps_ds=None, gddharv_maps_ds=None, outdir_figs=None, linewidth=1.5):
        if not gdd_maps_ds:
            if not thisDir:
                raise RuntimeError('If not providing gdd_maps_ds, you must provide thisDir (location of gdd_maps.nc)')
            gdd_maps_ds = xr.open_dataset(thisDir + 'gdd_maps.nc')
        if not gddharv_maps_ds:
            if not thisDir:
                raise RuntimeError('If not providing gddharv_maps_ds, you must provide thisDir (location of gddharv_maps.nc)')
            gddharv_maps_ds = xr.open_dataset(thisDir + 'gdd_maps.nc')
    
        # Get info
        incl_vegtypes_str = gdd_maps_ds.attrs['incl_vegtypes_str']
        dummy_fill = gdd_maps_ds.attrs['dummy_fill']
        if not outdir_figs:
            outdir_figs = gdd_maps_ds.attrs['outdir_figs']
        y1 = gdd_maps_ds.attrs['y1']
        yN = gdd_maps_ds.attrs['yN']
    
        # layout = "3x1"
        # layout = "2x2"
        layout = "3x2"
        bin_width = 15
        lat_bin_edges = np.arange(0, 91, bin_width)
    
        fontsize_titles = 12
        fontsize_axislabels = 12
        fontsize_ticklabels = 12
    
        Nbins = len(lat_bin_edges)-1
        bin_names = ["All"]
        for b in np.arange(Nbins):
            lower = lat_bin_edges[b]
            upper = lat_bin_edges[b+1]
            bin_names.append(f"{lower}–{upper}")
            
        color_old = '#beaed4'
        color_new = '#7fc97f'
        gdd_units = 'GDD (°C • day)'
    
        # Maps
        ny = 3
        nx = 1
        print("Making before/after maps...")
        for v, vegtype_str in enumerate(incl_vegtypes_str):
            vegtype_int = utils.vegtype_str2int(vegtype_str)[0]
            thisVar = f"gdd1_{vegtype_int}"
            
            vegtype_str_title = vegtype_str.replace("_", " ")
            if "irrigated" not in vegtype_str:
                vegtype_str_title = "rainfed " + vegtype_str_title
            vegtype_str_title = vegtype_str_title.capitalize()
            
            print(f"   {vegtype_str_title} ({vegtype_int})...")
            
            
            # Maps #####################
            
            gdd_map = gdd_maps_ds[thisVar].isel(time=0, drop=True)
            gdd_map_yx = gdd_map.where(gdd_map != dummy_fill)
            gddharv_map = gddharv_maps_ds[thisVar]
            if "time" in gddharv_map.dims:
                gddharv_map = gddharv_map.isel(time=0, drop=True)
            gddharv_map_yx = gddharv_map.where(gddharv_map != dummy_fill)
            
            gdd_map_yx.attrs['units'] = gdd_units
            gddharv_map_yx.attrs['units'] = gdd_units
                    
            vmax = max(np.max(gdd_map_yx), np.max(gddharv_map_yx)).values
            
            # Set up figure and first subplot
            if layout == "3x1":
                fig = plt.figure(figsize=(7.5,14))
                ax = fig.add_subplot(ny,nx,1,projection=ccrs.PlateCarree())
            elif layout == "2x2":
                fig = plt.figure(figsize=(12,6))
                spec = fig.add_gridspec(nrows=2, ncols=2,
                                        width_ratios=[0.4,0.6])
                ax = fig.add_subplot(spec[0,0],projection=ccrs.PlateCarree())
            elif layout == "3x2":
                fig = plt.figure(figsize=(14,9))
                spec = fig.add_gridspec(nrows=3, ncols=2,
                                        width_ratios=[0.5,0.5],
                                        wspace=0.2)
                ax = fig.add_subplot(spec[0,0],projection=ccrs.PlateCarree())
            else:
                raise RuntimeError(f"layout {layout} not recognized")
            
            thisMin = int(np.round(np.nanmin(gddharv_map_yx)))
            thisMax = int(np.round(np.nanmax(gddharv_map_yx)))
            thisTitle = f"{args.run1_name} (range {thisMin}–{thisMax})"
            make_map(ax, gddharv_map_yx, thisTitle, vmax, bin_width,
                     fontsize_ticklabels, fontsize_titles)
            
            if layout == "3x1":
                ax = fig.add_subplot(ny,nx,2,projection=ccrs.PlateCarree())
            elif layout in ["2x2", "3x2"]:
                ax = fig.add_subplot(spec[1,0],projection=ccrs.PlateCarree())
            else:
                raise RuntimeError(f"layout {layout} not recognized")
            thisMin = int(np.round(np.nanmin(gdd_map_yx)))
            thisMax = int(np.round(np.nanmax(gdd_map_yx)))
            thisTitle = f"{args.run2_name} (range {thisMin}–{thisMax})"
            make_map(ax, gdd_map_yx, thisTitle, vmax, bin_width,
                     fontsize_ticklabels, fontsize_titles)
            
            # Difference
            if layout == "3x2":
                ax = fig.add_subplot(spec[2,0],projection=ccrs.PlateCarree())
                thisMin = int(np.round(np.nanmin(gdd_map_yx)))
                thisMax = int(np.round(np.nanmax(gdd_map_yx)))
                thisTitle = "ISIMIP3 minus CLM"
                diff_map_yx = gdd_map_yx - gddharv_map_yx
                diff_map_yx.attrs['units'] = gdd_units
                
                gdd_spacing = 500
                vmax, bounds, Ncolors = get_bounds_ncolors(gdd_spacing, diff_map_yx)
                if Ncolors < 9:
                    gdd_spacing = 250
                    vmax, bounds, Ncolors = get_bounds_ncolors(gdd_spacing, diff_map_yx)
                
                cmap = cm.get_cmap("RdBu_r", Ncolors)
                cbar_ticks = []
                include_0bin_ticks = Ncolors <= 13
                if vmax <= 3000:
                    tick_spacing = gdd_spacing*2
                elif vmax <= 5000:
                    tick_spacing = 1500
                else:
                    tick_spacing = 2000
                previous = -np.inf
                for x in bounds:
                    if (not include_0bin_ticks) and (x>0) and (previous<0):
                        cbar_ticks.append(0)
                    if x % tick_spacing == 0 or (include_0bin_ticks and abs(x)==gdd_spacing/2):
                        cbar_ticks.append(x)
                    previous = x
                
                make_map(ax, diff_map_yx, thisTitle, vmax, bin_width,
                        fontsize_ticklabels, fontsize_titles, bounds=bounds,
                        extend='both', cmap=cmap, cbar_ticks=cbar_ticks)
            
            # Boxplots #####################
            
            gdd_vector = get_non_nans(gdd_map, dummy_fill)
            gddharv_vector = get_non_nans(gddharv_map, dummy_fill)
            
            lat_abs = np.abs(gdd_map.lat.values)
            gdd_bybin_old = [gddharv_vector]
            gdd_bybin_new = [gdd_vector]
            for b in np.arange(Nbins):
                lower = lat_bin_edges[b]
                upper = lat_bin_edges[b+1]
                lat_inds = np.where((lat_abs>=lower) & (lat_abs<upper))[0]
                gdd_vector_thisBin = get_non_nans(gdd_map[lat_inds,:], dummy_fill)
                gddharv_vector_thisBin = get_non_nans(gddharv_map[lat_inds,:], dummy_fill)
                gdd_bybin_old.append(gddharv_vector_thisBin)
                gdd_bybin_new.append(gdd_vector_thisBin)
                    
            if layout == "3x1":
                ax = fig.add_subplot(ny,nx,3)
            elif layout in ["2x2", "3x2"]:
                ax = fig.add_subplot(spec[:,1])
            else:
                raise RuntimeError(f"layout {layout} not recognized")
    
            bpl = make_plot(gdd_bybin_old, -1, linewidth)
            bpr = make_plot(gdd_bybin_new, 1, linewidth)
            set_boxplot_props(bpl, color_old, linewidth)
            set_boxplot_props(bpr, color_new, linewidth)
            
            # draw temporary lines to create a legend
            plt.plot([], c=color_old, label=args.run1_name, linewidth=linewidth)
            plt.plot([], c=color_new, label=args.run2_name, linewidth=linewidth)
            plt.legend(fontsize=fontsize_titles)
            
            plt.xticks(range(0, len(bin_names) * 2, 2), bin_names,
                       fontsize=fontsize_ticklabels)
            plt.yticks(fontsize=fontsize_ticklabels)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            plt.xlabel("latitude zone (absolute value)", fontsize=fontsize_axislabels)
            plt.ylabel(gdd_units, fontsize=fontsize_axislabels)
            ax.yaxis.set_label_coords(-0.11, 0.5)
            plt.title(f"Zonal changes", fontsize=fontsize_titles, fontweight="bold")
            
            plt.suptitle(f"Maturity requirements: {vegtype_str_title}",
                         fontsize=fontsize_titles*1.2,
                         fontweight="bold",
                         y=0.95)
            
            outfile = os.path.join(outdir_figs, f"{thisVar}_{vegtype_str}_gs{y1}-{yN}.png")
            plt.savefig(outfile, dpi=300, transparent=False, facecolor='white',
                        bbox_inches='tight')
            plt.close()
    
        print("Done.")
    
    if save_figs: 
        if args.only_make_figs:
            gdd_maps_ds = xr.open_dataset(os.path.join(args.run_dir, "generate_gdds", "figs", "gdd_maps.nc"))
            gddharv_maps_ds = xr.open_dataset(os.path.join(args.run_dir, "generate_gdds", "figs", "gddharv_maps.nc"))
        make_figures(args, gdd_maps_ds=gdd_maps_ds, gddharv_maps_ds=gddharv_maps_ds, outdir_figs=outdir_figs, linewidth=linewidth)


if __name__ == "__main__":
   main(sys.argv[1:])

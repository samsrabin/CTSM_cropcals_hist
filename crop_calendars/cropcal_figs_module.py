import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import cartopy.feature as cfeature

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


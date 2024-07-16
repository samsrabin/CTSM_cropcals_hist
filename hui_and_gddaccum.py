# %% Setup

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Define the runs in question
# runDir = "/Users/Shared/CESM_runs/1655-rename-gddplant/2022-03-10"
runDir = "/Users/Shared/CESM_runs/1655-rename-gddplant/2022-03-16"

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import glob
import cftime
import cartopy.crs as ccrs
from matplotlib import cm
import datetime as dt
import os

# Get output directory
outdir = runDir + "/figs"
if not os.path.exists(outdir):
    os.makedirs(outdir)

import sys
sys.path.append(my_ctsm_python_gallery)
import utils

def make_plot(thisPatch, hui, gddaccum, lon, lat, Ntime, newFig=True):
    time_count = np.arange(Ntime) + 1
    if newFig:
        plt.figure(figsize=(12, 6), dpi=80)
        plt.clf()
    plt.plot(time_count, hui.values)
    plt.plot(time_count, gddaccum.values, '--')
    yearStarts = [x for x in time_count if (x % 365)==1]
    for d in yearStarts:
        plt.axvline(x=d, ls='-', color="0.9")
    if lat <= 0.0:
        yearStarts_sh = [x for x in time_count if (x % 365)==182]
        for d in yearStarts_sh:
            plt.axvline(x=d, ls='--', color="0.9")
    plt.gca().set_prop_cycle(None)
    plt.plot(time_count, hui.values)
    plt.plot(time_count, gddaccum.values, '--')
    plt.legend(["HUI", "GDDACCUM"])
    plt.locator_params(axis="x", nbins=20)
    plt.xlabel("Day of simulation")
    plt.ylabel("growing degree-days")
    plt.title(f"{run1_ds['patches1d_itype_veg_str'].sel(patch=thisPatch).values}, lon {lon} lat {lat} (patch {thisPatch})")


# %% Import

import importlib
importlib.reload(utils)

varList = ["CPHASE", "GDDHARV", "HUI", "GDDACCUM", "GPP", "GRAINC_TO_FOOD", "NPP", "TLAI", "TOTVEGC"]

run1_ds = utils.import_ds(glob.glob(runDir + "/*clm2.h1.*"), \
    myVars=varList, 
    myVegtypes=utils.define_mgdcrop_list())
# run1dates_ds = utils.import_ds(glob.glob(runDir + "/*clm2.h2.*"), \
#     myVars=["SDATES", "HDATES"],
#     myVegtypes=utils.define_mgdcrop_list())



# # %% Look at some random patches' info

# Npatch = run1_ds.sizes["patch"]
# for i in np.sort(np.random.choice(np.arange(Npatch), 10, replace=False)):
#     p = run1_ds["patch"].values[i]
#     lon = round(run1_ds['patches1d_lon'].values[i], 3)
#     lat = round(run1_ds['patches1d_lat'].values[i], 3)
#     print(f"patch {p} ({run1_ds['patches1d_itype_veg_str'].values[i]}): "
#           + f"lon {lon} lat {lat}")


# %% Plot all patches where variables differ

max_patches = 1
n_patches = 0

for i, thisPatch in enumerate(run1_ds.patch.values):
    hui = run1_ds.sel(patch=thisPatch)["HUI"]
    gddaccum = run1_ds.sel(patch=thisPatch)["GDDACCUM"]
    lon = round(run1_ds['patches1d_lon'].values[i], 3)
    lat = round(run1_ds['patches1d_lat'].values[i], 3)

    if np.array_equal(hui.values, gddaccum.values, equal_nan=True):
        continue
    
    n_patches = n_patches + 1
    vt_str = run1_ds['patches1d_itype_veg_str'].sel(patch=thisPatch).values
    print(f"HUI and GDDACCUM differ in patch {thisPatch} ({vt_str}, lon {lon} lat {lat})")
    make_plot(thisPatch, hui, gddaccum, lon, lat, run1_ds.sizes["time"])

    outfile = f"{outdir}/patch{thisPatch}-{vt_str}.png"
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
        bbox_inches='tight')
    plt.close()
    
    if n_patches == max_patches:
        break
            
if n_patches == max_patches:
    print(f"Stopping after {max_patches} patches")
else:
    print("All done.")


# %% Make sure GDDACCUM is never > HUI

np.any(run1_ds["GDDACCUM"].values > run1_ds["HUI"].values)

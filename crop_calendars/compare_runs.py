# %% Setup

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Define the runs in question
# run_1 = {"name": "bill_1628",
#          "path": "/Users/Shared/CESM_runs/2022-02-18/bill_1628"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/bill_1628_decStart/40ishdays"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/bill_1628_decStart/1125days"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/bill_1628_decStart_IHistClm51BgcCrop"
#          }
# run_2 = {"name": "sam_1616",
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616/2022-02-18"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616/2022-02-21"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616/2022-02-22"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616_decStart/v01"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616_decStart/v02"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616_decStart/v03"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616_decStart/v04"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616_decStart/v04_1125days"
#         #  "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616/2022-03-01_9abb3d"
#         # "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616/2022-03-01_e22bb35c1"
#         # "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616_decStart/v05"
#         # "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616_decStart_IHistClm51BgcCrop/v1"
#         # "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616_decStart_IHistClm51BgcCrop/v2"
#         # "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616_decStart_IHistClm51BgcCrop/v3"
#         "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616/2022-03-07_ab7baa"
#          }
# run_1 = {"name": "sam_1616_diff",
#          "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616/2022-03-07_fb114ae2f"
#          }
# run_2 = {"name": "sam_1616_pass",
#          "path": "/Users/Shared/CESM_runs/2022-02-18/1537-crop-date-outputs3_ctsm1616/2022-03-07_fea49ff92"
#          }
run_1 = {"name": "bill_1628",
         "path": "/Users/Shared/CESM_runs/1537-smallville/ERS_Ly5_Mmpi-serial.1x1_smallvilleIA.I2000Clm50BgcCropQianRs.izumi_gnu.clm-ciso_monthly/1537-smallville-bill"
         }
run_2 = {"name": "mine_ts",
        #  "path": "/Users/Shared/CESM_runs/1537-smallville/ERS_Ly5_Mmpi-serial.1x1_smallvilleIA.I2000Clm50BgcCropQianRs.izumi_gnu.clm-ciso_monthly/1537-smallville-mine/v01"
        # "path": "/Users/Shared/CESM_runs/1537-smallville/ERS_Ly5_Mmpi-serial.1x1_smallvilleIA.I2000Clm50BgcCropQianRs.izumi_gnu.clm-ciso_monthly/1537-smallville-mine/v02"
        # "path": "/Users/Shared/CESM_runs/1537-smallville/ERS_Ly5_Mmpi-serial.1x1_smallvilleIA.I2000Clm50BgcCropQianRs.izumi_gnu.clm-ciso_monthly/1537-smallville-mine/v03"
        # "path": "/Users/Shared/CESM_runs/1537-smallville/ERS_Ly5_Mmpi-serial.1x1_smallvilleIA.I2000Clm50BgcCropQianRs.izumi_gnu.clm-ciso_monthly/1537-smallville-mine/v04"
        "path": "/Users/Shared/CESM_runs/1537-smallville/ERS_Ly5_Mmpi-serial.1x1_smallvilleIA.I2000Clm50BgcCropQianRs.izumi_gnu.clm-ciso_monthly/1537-smallville-mine/v05"
         }

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
outdir = run_2["path"] + "/figs"
if not os.path.exists(outdir):
    os.makedirs(outdir)

import sys
sys.path.append(my_ctsm_python_gallery)
import utils

def make_plot(thisPatch, patch1_da, patch2_da, lon, lat, Ntime, newFig=True):
    time_count = np.arange(Ntime) + 1
    if newFig:
        plt.figure(figsize=(12, 6), dpi=80)
        plt.clf()
    plt.plot(time_count, patch1_da.values)
    plt.plot(time_count, patch2_da.values, '--')
    yearStarts = [x for x in time_count if (x % 365)==1]
    for d in yearStarts:
        plt.axvline(x=d, ls='-', color="0.9")
    if lat <= 0.0:
        yearStarts_sh = [x for x in time_count if (x % 365)==182]
        for d in yearStarts_sh:
            plt.axvline(x=d, ls='--', color="0.9")
    plt.gca().set_prop_cycle(None)
    plt.plot(time_count, patch1_da.values)
    plt.plot(time_count, patch2_da.values, '--')
    plt.legend([run_1['name'], run_2['name']])
    plt.locator_params(axis="x", nbins=20)
    plt.xlabel("Day of simulation")
    plt.ylabel(thisVar)
    plt.title(f"{run1_ds['patches1d_itype_veg_str'].sel(patch=thisPatch).values}, lon {lon} lat {lat} (patch {thisPatch})")


# %% Import

import importlib
importlib.reload(utils)

varList = ["CPHASE", "GDDHARV", "GDDPLANT", "GPP", "GRAINC_TO_FOOD", "NPP", "TLAI", "TOTVEGC"]
# varList = ["CPHASE", "GDDHARV", "GDDPLANT", "GPP", "GRAINC_TO_FOOD", "NPP", "TLAI", "TOTVEGC", "HUIGRAIN"]

run1_ds = utils.import_ds(glob.glob(run_1["path"] + "/*clm2.h1.*"), \
    myVars=varList, 
    myVegtypes=utils.define_mgdcrop_list())

run2_ds = utils.import_ds(glob.glob(run_2["path"] + "/*clm2.h1.*"), \
    myVars=varList, 
    myVegtypes=utils.define_mgdcrop_list())
# run2dates_ds = utils.import_ds(glob.glob(run_2["path"] + "/*clm2.h2.*"), \
#     myVars=["SDATES", "HDATES"], 
#     myVegtypes=utils.define_mgdcrop_list())

# Only include patches that have non-NaN CPHASE in either run1 or run2
incl_run1 = np.any(np.bitwise_not(np.isnan(run1_ds.CPHASE.values)), axis=0)
incl_run2 = np.any(np.bitwise_not(np.isnan(run2_ds.CPHASE.values)), axis=0)
incl = np.where(np.bitwise_or(incl_run1, incl_run2))[0]
run1_ds = run1_ds.isel(patch=incl)
run2_ds = run2_ds.isel(patch=incl)

# Sanity checks
if not np.array_equal(run1_ds['patches1d_lon'].values, run2_ds['patches1d_lon'].values):
    raise RuntimeError("Longitudes differ")
if not np.array_equal(run1_ds['patches1d_lat'].values, run2_ds['patches1d_lat'].values):
    raise RuntimeError("Latitudes differ")
if not np.array_equal(run1_ds['time'].values, run2_ds['time'].values):
    def intersect_mtlb(a, b):
        a1, ia = np.unique(a, return_index=True)
        b1, ib = np.unique(b, return_index=True)
        aux = np.concatenate((a1, b1))
        aux.sort()
        c = aux[:-1][aux[1:] == aux[:-1]]
        return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]
    c, ia, ib = intersect_mtlb(run1_ds['time'].values, run2_ds['time'].values)
    if not len(c):
        raise RuntimeError("Times differ")
    print(f"Times differ; using overlap ({c[0]} to {c[-1]})")
    run1_ds = run1_ds.isel(time=ia)
    run2_ds = run2_ds.isel(time=ib)


# # %% Look at some random patches' info

# Npatch = run1_ds.sizes["patch"]
# for i in np.sort(np.random.choice(np.arange(Npatch), 10, replace=False)):
#     p = run1_ds["patch"].values[i]
#     lon = round(run1_ds['patches1d_lon'].values[i], 3)
#     lat = round(run1_ds['patches1d_lat'].values[i], 3)
#     print(f"patch {p} ({run1_ds['patches1d_itype_veg_str'].values[i]}): "
#           + f"lon {lon} lat {lat}")


# %% Plot all patches where codebases differ in a given variable (or a list)
# this_varList = ["CPHASE"]
this_varList = varList

for thisVar in this_varList:
    print(thisVar)

    if not np.array_equal(run1_ds[thisVar].values, run2_ds[thisVar].values, equal_nan=True):
        for i, thisPatch in enumerate(run1_ds.patch.values):
            patch1_da = run1_ds.sel(patch=thisPatch)[thisVar]
            patch2_da = run2_ds.sel(patch=thisPatch)[thisVar]
            lon = round(run1_ds['patches1d_lon'].values[i], 3)
            lat = round(run1_ds['patches1d_lat'].values[i], 3)

            if np.array_equal(patch1_da.values, patch2_da.values, equal_nan=True):
                continue
            
            vt_str = run1_ds['patches1d_itype_veg_str'].sel(patch=thisPatch).values
            print(f"{run_1['name']} and {run_2['name']} have different {thisVar} in patch {thisPatch} ({vt_str}, lon {lon} lat {lat})")
            make_plot(thisPatch, patch1_da, patch2_da, lon, lat, run1_ds.sizes["time"])

            outfile = f"{outdir}/patch{thisPatch}-{vt_str}-{thisVar}.png"
            plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
                bbox_inches='tight')
            plt.close()
            
    print('Done.')
print("All done.")


# %% Plot patches with multiple harvests in a year
thisVar = "GRAINC_TO_FOOD"

def n_harvests(data, y):
    d1 = y*365
    dN = d1 + 364
    return np.sum(data[d1:dN] > 0)

for i, thisPatch in enumerate(run1_ds.patch.values):
    patch1_da = run1_ds.sel(patch=thisPatch)[thisVar]
    patch2_da = run2_ds.sel(patch=thisPatch)[thisVar]
    lon = round(run1_ds['patches1d_lon'].values[i], 3)
    lat = round(run1_ds['patches1d_lat'].values[i], 3)

    if not np.array_equal(patch1_da.values, patch2_da.values):
        continue
    multi_harvests = False
    zero_harvests = False
    do_make_plot = False
    for y in np.arange(3):
        n = n_harvests(patch1_da.values, y)
        multi_harvests = multi_harvests or n > 1
        zero_harvests = zero_harvests or n==0
        if multi_harvests and zero_harvests:
            do_make_plot = True
            break
    if not do_make_plot:
        continue
    
    vt_str = run1_ds['patches1d_itype_veg_str'].sel(patch=thisPatch).values
    print(f"{run_1['name']} and {run_2['name']} have different {thisVar} in patch {thisPatch} ({vt_str}, lon {lon} lat {lat})")
    make_plot(thisPatch, patch1_da, patch2_da, lon, lat, run1_ds.sizes["time"])
    break

    outfile = f"{outdir}/patch{thisPatch}-{vt_str}-{thisVar}.png"
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
        bbox_inches='tight')
    plt.close()

print('Done.')


# %% Plot patches with multiple harvests in a year
thisVar = "GRAINC_TO_FOOD"

for i, thisPatch in enumerate(run1_ds.patch.values):
    patch1_da = run1_ds.sel(patch=thisPatch)[thisVar]
    patch2_da = run2_ds.sel(patch=thisPatch)[thisVar]
    lon = round(run1_ds['patches1d_lon'].values[i], 3)
    lat = round(run1_ds['patches1d_lat'].values[i], 3)

    if not np.array_equal(patch1_da.values, patch2_da.values):
        continue
    patch2_hdates = run2dates_ds.sel(patch=thisPatch)["HDATES"].values[1:,:]
    patch2_hcounts = np.sum(patch2_hdates > 0, axis=1)
    multi_harvests = np.any(patch2_hcounts > 1)
    zero_harvests = np.any(patch2_hcounts == 0)
    if not (multi_harvests and zero_harvests):
        continue
    
    vt_str = run1_ds['patches1d_itype_veg_str'].sel(patch=thisPatch).values
    print(f"patch {thisPatch} ({vt_str}, lon {lon} lat {lat})")
    print(patch2_hdates)
    make_plot(thisPatch, patch1_da, patch2_da, lon, lat, run1_ds.sizes["time"])
    plt.show(); continue

    outfile = f"{outdir}/patch{thisPatch}-{vt_str}-{thisVar}.png"
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
        bbox_inches='tight')
    plt.close()

print('Done.')


        
        
# %% Plot all variables for a given patch
    
thisPatch = 547
lon = np.round(run1_ds['patches1d_lon'].sel(patch=thisPatch).values, 3)
lat = np.round(run1_ds['patches1d_lat'].sel(patch=thisPatch).values, 3)

for thisVar in varList:
    patch1_da = run1_ds.sel(patch=thisPatch)[thisVar]
    patch2_da = run2_ds.sel(patch=thisPatch)[thisVar]
    # if np.array_equal(patch1_da.values, patch2_da.values):
    #     continue
    make_plot(thisPatch, patch1_da, patch2_da, lon, lat, run1_ds.sizes["time"])
    
    vt_str = run1_ds['patches1d_itype_veg_str'].sel(patch=thisPatch).values
    outfile = f"{outdir}/patch{thisPatch}-{vt_str}-{thisVar}.png"
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
        bbox_inches='tight')
    plt.close()
    
    
# %% Plot CPHASE and HUI(GRAIN) for a given patch or list of patches

patchList = [547, 1071, 1790, 3923]

this_varList = ["CPHASE", "GDDHARV", "HUIGRAIN"]
lon = np.round(run1_ds['patches1d_lon'].sel(patch=thisPatch).values, 3)
lat = np.round(run1_ds['patches1d_lat'].sel(patch=thisPatch).values, 3)

for thisPatch in patchList:
    fig, axs = plt.subplots(3, 1, figsize=(8,15))
    for v, thisVar in enumerate(this_varList):
        patch1_da = run1_ds.sel(patch=thisPatch)[thisVar]
        patch2_da = run2_ds.sel(patch=thisPatch)[thisVar]

        plt.axes(axs[v])
        make_plot(thisPatch, patch1_da, patch2_da, lon, lat, run1_ds.sizes["time"],
                newFig=False)
    
    vt_str = run1_ds['patches1d_itype_veg_str'].sel(patch=thisPatch).values
    outfile = f"{outdir}/patch{thisPatch}-{vt_str}-CPHASE_GDDHARV_HUIGRAIN.png"
    plt.savefig(outfile, dpi=150, transparent=False, facecolor='white', \
        bbox_inches='tight')
    plt.close()


# %% Hypothesis testing

thesePatches = [1566, 1567, 1594, 1632, 1633, 1664, 1665, 1687, 1688, 1707, 1739, 1740, 1818, 1853, 1854]
thisVar = "SDATES"
for p in thesePatches:
    patch2_da = run2dates_ds.sel(patch=p)[thisVar]
    # 1, not 0, because 0th value is from simulated 1999
    print(f"Patch {p}: Year 1 sown on day {patch2_da.values[4]}")
    
# %%
# %% Hypothesis testing

thesePatches = [1566, 1567, 1594, 1632, 1633, 1664, 1665, 1687, 1688, 1707, 1739, 1740, 1818, 1853, 1854]
thisVar = "IDOP"
for p in thesePatches:
    patch2_da = run2dates_ds.sel(patch=p)[thisVar]
    # 1, not 0, because 0th value is from simulated 1999
    print(f"Patch {p}: Year 1 sown on day {patch2_da.values[4]}")







    
    
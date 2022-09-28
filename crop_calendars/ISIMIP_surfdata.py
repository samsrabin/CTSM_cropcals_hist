# %%

thisDir = "/Users/Shared/CESM_inputdata/lnd/clm2/isimip_surfdata"
infile = os.path.join(thisDir, "surfdata_360x720cru_78pfts_CMIP6_simyr1850_c170824_ssr.nc")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import shutil
import os
import time
import datetime as dt
import cartopy.crs as ccrs
import datetime as dt

import sys
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils


def make_map(ax, this_map): 
    ax.pcolormesh(this_map.lsmlon.values, this_map.lsmlat.values, 
            this_map, shading="auto")
    ax.set_extent([-180,180,-63,90],crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, color="white")
    ax.coastlines(linewidth=0.3)
    
def make_axis(fig, ny, nx, n):
    ax = fig.add_subplot(ny,nx,n,projection=ccrs.PlateCarree())
    ax.spines['geo'].set_visible(False) # Turn off box outline
    return ax

# Define crop dictionary
# As "CLMname: [number, GGCMIname]"
# - CLM names and numbers taken from commit `3dcbc7499a57904750a994672fc36b4221b9def5`
# - Using one global GGCMI value for both temperate and tropical versions of corn and soybean.
# - There is no GGCMI equivalent of CLM's winter barley and rye. Using winter wheat instead.
# - Using GGCMI `pea` for CLM pulses, as suggested by GGCMI phase 3 protocol.
# - Only using GGCMI `ri1` for rice; ignoring `ri2`.
def set_crop_dict(thisnum, thisname):
    return {"clm_num": thisnum, "thiscrop_ggcmi": thisname}
    
crop_dict = {
    "unmanaged_c3_crop": set_crop_dict(15, "swh_rf"),
    "unmanaged_c3_irrigated": set_crop_dict(16, "swh_ir"),
    "temperate_corn": set_crop_dict(17, "mai_rf"),
    "irrigated_temperate_corn": set_crop_dict(18, "mai_ir"),
    "spring_wheat": set_crop_dict(19, "swh_rf"),
    "irrigated_spring_wheat": set_crop_dict(20, "swh_ir"),
    "winter_wheat": set_crop_dict(21, "wwh_rf"),
    "irrigated_winter_wheat": set_crop_dict(22, "wwh_ir"),
    "temperate_soybean": set_crop_dict(23, "soy_rf"),
    "irrigated_temperate_soybean": set_crop_dict(24, "soy_ir"),
    "barley": set_crop_dict(25, "bar_rf"),
    "irrigated_barley": set_crop_dict(26, "bar_ir"),
    "winter_barley": set_crop_dict(27, "wwh_rf"),
    "irrigated_winter_barley": set_crop_dict(28, "wwh_ir"),
    "rye": set_crop_dict(29, "rye_rf"),
    "irrigated_rye": set_crop_dict(30, "rye_ir"),
    "winter_rye": set_crop_dict(31, "wwh_rf"),
    "irrigated_winter_rye": set_crop_dict(32, "wwh_ir"),
    "cassava": set_crop_dict(33, "cas_rf"),
    "irrigated_cassava": set_crop_dict(34, "cas_ir"),
    "citrus": set_crop_dict(35, None),
    "irrigated_citrus": set_crop_dict(36, None),
    "cocoa": set_crop_dict(37, None),
    "irrigated_cocoa": set_crop_dict(38, None),
    "coffee": set_crop_dict(39, None),
    "irrigated_coffee": set_crop_dict(40, None),
    "cotton": set_crop_dict(41, "cot_rf"),
    "irrigated_cotton": set_crop_dict(42, "cot_ir"),
    "datepalm": set_crop_dict(43, None),
    "irrigated_datepalm": set_crop_dict(44, None),
    "foddergrass": set_crop_dict(45, None),
    "irrigated_foddergrass": set_crop_dict(46, None),
    "grapes": set_crop_dict(47, None),
    "irrigated_grapes": set_crop_dict(48, None),
    "groundnuts": set_crop_dict(49, "nut_rf"),
    "irrigated_groundnuts": set_crop_dict(50, "nut_ir"),
    "millet": set_crop_dict(51, "mil_rf"),
    "irrigated_millet": set_crop_dict(52, "mil_ir"),
    "oilpalm": set_crop_dict(53, None),
    "irrigated_oilpalm": set_crop_dict(54, None),
    "potatoes": set_crop_dict(55, "pot_rf"),
    "irrigated_potatoes": set_crop_dict(56, "pot_ir"),
    "pulses": set_crop_dict(57, "pea_rf"),
    "irrigated_pulses": set_crop_dict(58, "pea_ir"),
    "rapeseed": set_crop_dict(59, "rap_rf"),
    "irrigated_rapeseed": set_crop_dict(60, "rap_ir"),
    "rice": set_crop_dict(61, "ri1_rf"),
    "irrigated_rice": set_crop_dict(62, "ri1_ir"),
    "sorghum": set_crop_dict(63, "sor_rf"),
    "irrigated_sorghum": set_crop_dict(64, "sor_ir"),
    "sugarbeet": set_crop_dict(65, "sgb_rf"),
    "irrigated_sugarbeet": set_crop_dict(66, "sgb_ir"),
    "sugarcane": set_crop_dict(67, "sgc_rf"),
    "irrigated_sugarcane": set_crop_dict(68, "sgc_ir"),
    "sunflower": set_crop_dict(69, "sun_rf"),
    "irrigated_sunflower": set_crop_dict(70, "sun_ir"),
    "miscanthus": set_crop_dict(71, None),
    "irrigated_miscanthus": set_crop_dict(72, None),
    "switchgrass": set_crop_dict(73, None),
    "irrigated_switchgrass": set_crop_dict(74, None),
    "tropical_corn": set_crop_dict(75, "mai_rf"),
    "irrigated_tropical_corn": set_crop_dict(76, "mai_ir"),
    "tropical_soybean": set_crop_dict(77, "soy_rf"),
    "irrigated_tropical_soybean": set_crop_dict(78, "soy_ir"),
}


# %% Import

in_ds = xr.open_dataset(infile)

# Import GGCMI mask
ggcmi_mask_ds = xr.open_dataset("/Users/Shared/GGCMI/AgMIP.input/phase3/ISIMIP3/landseamask_v1.1.1/mask_ggcmi_v1.1.1.nc")
ggcmi_mask_da = ggcmi_mask_ds.has_all


# %% Get cells to include

# Rework lon/lat to match GGCMI
lsmlat = in_ds.lsmlat.copy()
lsmlon = in_ds.lsmlon.copy()
in_ds = in_ds.rename_dims(dict(lsmlat="lat",lsmlon="lon"))
in_ds = in_ds.assign_coords(lon = utils.lon_pm2idl(in_ds.lon.values/2+0.25),
                            lat = (in_ds.lat.values-180)/2 + 0.25)

# incl_cells_da = ((infile.PCT_CROP + infile.PCT_NATVEG + np.sum(infile.PCT_URBAN, axis=0)) > 0)
# incl_cells = np.where(incl_cells_da.values)
# incl_cells_3d = np.where(np.tile(incl_cells_da.expand_dims(dim='numurbl', axis=0), (3, 1, 1)))

incl_cells_da = np.logical_and(in_ds.PCT_CROP + in_ds.PCT_NATVEG > 0, ggcmi_mask_da)
incl_cells = np.where(incl_cells_da.values)

# Undo lon/lat rework
in_ds = in_ds.assign_coords(lon = lsmlon.values,
                            lat = lsmlat.values)
in_ds = in_ds.rename_dims(dict(lat="lsmlat",lon="lsmlon"))


# %% Get list of CFTs to include

# Which CFTs have *any* area?
pct_cropxcft_da = in_ds.PCT_CROP * in_ds.PCT_CFT
clm_cft_indices = np.arange(len(crop_dict))
inclm_cft_indices = np.where(np.sum(pct_cropxcft_da, axis=(0,1)))[0]
inclm_vegtype_int = in_ds.cft.values[inclm_cft_indices]

# Which of those CFTs have GGCMI equivalents?
inboth_vegtype_str = []
inboth_vegtype_int = []
inboth_cft_indices = []
for v, vegtype_int in enumerate(inclm_vegtype_int):
    found = False
    for vegtype_str in crop_dict:
        if crop_dict[vegtype_str]["clm_num"] == vegtype_int:
            found = True
            break
    if not found:
        raise RuntimeError(f"Vegtype {vegtype_int} not found in crop_dict")
    vegtype_str_ggcmi = crop_dict[vegtype_str]["thiscrop_ggcmi"]
    if vegtype_str_ggcmi:
        print(f"* {vegtype_int}: {vegtype_str} -> {vegtype_str_ggcmi}")
        inboth_vegtype_str.append(vegtype_str)
        inboth_vegtype_int.append(vegtype_int)
        inboth_cft_indices.append(inclm_cft_indices[v])
    else:
        print(f"  {vegtype_int}: {vegtype_str} EXCLUDED")

notinboth_cft_indices = [x for x in clm_cft_indices if x not in inboth_cft_indices]

Ncft = len(inboth_vegtype_str)


# %% Run 100% cropland, split among all crops

in_ds.PCT_CROP.values[incl_cells] = 100
in_ds.PCT_NATVEG.values[incl_cells] = 0

in_ds.PCT_CFT.values[inboth_cft_indices,:,:] = 100/Ncft
in_ds.PCT_CFT.values[notinboth_cft_indices,:,:] = 0



# %% Save

outfile = infile.replace(".nc", f".{Ncft}crops_everywhere.{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.nc")
in_ds.to_netcdf(outfile, format="NETCDF3_CLASSIC")




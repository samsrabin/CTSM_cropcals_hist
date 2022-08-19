# %% Setup

import numpy as np
import xarray as xr

infile = '/Users/Shared/CESM_work/CropEvalData_ssr/landuse.timeseries_1.9x2.5_hist_78pfts_CMIP6_simyr1850-2015_c170824.nc4'
first_fake_year = 1976
outfile = infile.replace('.nc4', f'.gddgen{first_fake_year}-2015.nc4')

# %% Process

ds_in = xr.open_dataset(infile)
ds_tmp = ds_in.sel(time=slice(first_fake_year,2015)).copy()

# Where is each crop ever active?
ds_tmp['AREA_CROP'] = (ds_tmp.AREA * ds_tmp.LANDFRAC_PFT * ds_tmp.PCT_CROP/100).transpose("time", "lsmlat", "lsmlon")
ds_tmp['AREA_CFT'] = (ds_tmp['AREA_CROP'] * ds_tmp.PCT_CFT/100).transpose("time", "cft", "lsmlat", "lsmlon")
ds_area_max = ds_tmp['AREA_CFT'].max(dim="time")
ds_tmp['ever_active_bycft'] = (ds_area_max > 0).transpose("cft", "lsmlat", "lsmlon")
ds_tmp['ever_active'] = ds_tmp['ever_active_bycft'].any(dim="cft")

# For cells that EVER have cropland, when do they have 0 crop area? Change those 0% values to 1%.
new_pct_crop_ar = ds_tmp['PCT_CROP'].values
new_pct_crop_ar[np.where((ds_tmp['AREA_CROP']==0) & ds_tmp['ever_active'])] = 1.0
ds_tmp['PCT_CROP'] = xr.DataArray(data = new_pct_crop_ar,
                                  coords = ds_tmp['PCT_CROP'].coords,
                                  attrs = ds_tmp['PCT_CROP'].attrs)
ds_tmp['AREA_CROP'] = (ds_tmp.AREA * ds_tmp.LANDFRAC_PFT * ds_tmp.PCT_CROP/100).transpose("time", "lsmlat", "lsmlon")
if np.any((ds_tmp['AREA_CROP']==0) & ds_tmp['ever_active']):
   raise RuntimeError("Failed to fill 0% CROP with 1% where needed.")

# For cells that EVER have each CFT, when do they have 0 area of that CFT? Change those 0% values to something positive.
new_pct_cft_tcyx = ds_tmp['PCT_CFT'].values
new_pct_cft_tcyx[np.where((ds_tmp['AREA_CFT']==0) & ds_tmp['ever_active'])] = 1.0
# Ensure sum to 100
i = 0
while np.any(~np.isclose(np.sum(new_pct_cft_tcyx, axis=1), 100.0)):
   i+=1
   if i > 10:
      raise RuntimeError('too many iterations')
   new_pct_cft_tcyx = 100 * (new_pct_cft_tcyx / np.expand_dims(np.sum(new_pct_cft_tcyx, axis=1), axis=1))
ds_tmp['PCT_CFT'] = xr.DataArray(data = new_pct_cft_tcyx,
                                 coords = ds_tmp['PCT_CFT'].coords,
                                 attrs = ds_tmp['PCT_CFT'].attrs)


# %% Save output netCDF
ds_out = xr.concat((ds_in.copy().sel(time=slice(0, first_fake_year-1)),
                    ds_tmp), dim="time")
ds_out = ds_out.drop([v for v in ds_out if v not in ds_in])
ds_out.to_netcdf(outfile)
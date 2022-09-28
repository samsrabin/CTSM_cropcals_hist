# %% Setup

import numpy as np
import xarray as xr
import cftime
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
import sys
sys.path.append(my_ctsm_python_gallery)
import utils


# %% Import and process

y1 = 1955
yN = 2020
yearList = np.arange(y1, yN+1)
Nyears = len(yearList)

infile = "/Users/Shared/CESM_work/crop_dates/cropcals3.f19-g17.rx_crop_calendars2.IHistClm50BgcCrop.ggcmi.1977-2014.gddgen/gdds_20220904_111456.nc"
ds_in = xr.open_dataset(infile)

ds_out = utils.tile_over_time(ds_in, years=yearList)


# %% Save

outfile = infile
while outfile[-1]!='.':
    outfile = outfile[:-1]
outfile = outfile + f'{y1}-{yN}.nc'

ds_out.to_netcdf(outfile, mode='w', format='NETCDF3_CLASSIC')

# %%


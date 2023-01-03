# %%

import numpy as np
import xarray as xr
import os
import glob

ggcmi3_cropcal_dir = '/Users/Shared/GGCMI/AgMIP.input/phase3/ISIMIP3/crop_calendar'

# %%

def get_ds(ggcmi3_cropcal_dir, i, s):
    pattern = os.path.join(ggcmi3_cropcal_dir, f"ri{s}_{i}*nc4")
    filename = glob.glob(pattern)
    if len(filename) != 1:
        raise RuntimeError(f"Expected 1 match of pattern; found {len(filename)}")
    filename = filename[0]
    ds = xr.open_dataset(filename)
    ds.attrs['filename'] = filename
    return ds

for i in ['rf', 'ir']:
    ri1 = get_ds(ggcmi3_cropcal_dir, i, 1)
    ri2 = get_ds(ggcmi3_cropcal_dir, i, 2)
    
    main_rice = np.full(ri1.fraction_of_harvested_area.shape, 0)
    main_rice[np.where(ri1.fraction_of_harvested_area >= ri2.fraction_of_harvested_area)] = 1.0
    main_rice[np.where(ri1.fraction_of_harvested_area < ri2.fraction_of_harvested_area)] = 2.0
    
    main_rice_da = xr.DataArray(data = main_rice,
                             coords = ri1['fraction_of_harvested_area'].coords)
    
    ric = xr.Dataset(data_vars = {'which_rice': main_rice_da})
    ric.attrs['provenance'] = "Combined ri1 and ri2 based on which is dominant by area in each gridcell. See combine_r1-2_by_fractionharvestedarea.py."
    encoding = {'which_rice': {"zlib": True, "complevel": 9}}
    for v in ["planting_day", "maturity_day", "growing_season_length"]:
        ra = np.full(ri1.fraction_of_harvested_area.shape, np.nan)
        ra[np.where(main_rice == 1.0)] = ri1[v].values[np.where(main_rice == 1.0)]
        ra[np.where(main_rice == 2.0)] = ri2[v].values[np.where(main_rice == 2.0)]
        ric[v] = xr.DataArray(data = ra,
                              coords = ri1[v].coords,
                              attrs = ri1[v].attrs)
        encoding[v] = {"zlib": True, "complevel": 9}
    
    outfile = ri1.attrs['filename'].replace('ri1', 'ric')
    ric.to_netcdf(outfile, format='NETCDF4', encoding=encoding)
    

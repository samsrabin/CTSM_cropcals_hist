import numpy as np
import xarray as xr
import argparse
import sys
import os

def main(argv):
    
    ###############################
    ### Process input arguments ###
    ###############################
    
    # Set arguments
    parser = argparse.ArgumentParser(description="ADD DESCRIPTION HERE")
    parser.add_argument("-l", "--flanduse_timeseries", "--flanduse-timeseries", 
                        help="Land-use timeseries file (flanduse_timeseries) for CLM run",
                        required=True)
    parser.add_argument("-p", "--paramfile", 
                        help="Parameter file (paramfile) for CLM run",
                        required=True)
    parser.add_argument("-s", "--fsurdat", 
                        help="Surface dataset (fsurdat) for CLM run",
                        required=True)
    args = parser.parse_args(argv)


    ##############
    ### Import ###
    ##############

    lu = xr.open_dataset(args.flanduse_timeseries)
    surf = xr.open_dataset(args.fsurdat)
    params = xr.open_dataset(args.paramfile)


    ##########################################
    ### %% Get new PCT_CROP and PCT_NATVEG ###
    ##########################################

    new_pct_crop_da = lu['PCT_CROP'].max(dim="time")
    new_pct_crop_da.attrs = surf['PCT_CROP'].attrs

    surf_pct_crop_plus_natveg = surf['PCT_CROP'] + surf['PCT_NATVEG']
    if np.any(surf_pct_crop_plus_natveg < new_pct_crop_da):
        raise RuntimeError("Max CROP > CROP+NATVEG")

    new_natveg_da = surf_pct_crop_plus_natveg - new_pct_crop_da
    if np.any((new_natveg_da > 0) & (surf['PCT_NATVEG'] == 0)):
        print("You created some NATVEG area. Not necessarily a problem, but unexpected.")
    new_natveg_da.attrs = surf['PCT_NATVEG'].attrs


    ##################################################################
    ### Get new PCT_CFT (percentage of cropland that is each crop) ###
    ##################################################################

    # Sum all crops' max area, merging unrepresented types into their representative type
    cft_list_int = surf['cft'].values[surf['cft'].values>=17]
    max_merged_pct_crop = np.full_like(surf['PCT_CFT'], 0.0)
    for i, c in enumerate(cft_list_int):
        mergetarget = params['mergetoclmpft'].sel(pft=c).values
        m = np.where(cft_list_int==mergetarget)[0]
        max_merged_pct_crop[m,:,:] += np.expand_dims(lu['PCT_CFT'].sel(cft=c).max(dim="time"), axis=0)
    max_merged_pct_crop_da = xr.DataArray(data=max_merged_pct_crop,
                                          dims=surf['PCT_CFT'].dims,
                                          attrs=surf['PCT_CFT'].attrs)
    
    # Ensure no area in merged-away crops
    for i, c in enumerate(cft_list_int):
        if (params['mergetoclmpft'].sel(pft=c) != c) and (max_merged_pct_crop_da.sel(cft=i).max() > 0):
            raise RuntimeError(f"Unexpected max_merged_pct_crop area for pft {c}")

    # Determine how many crops ever have any area
    ever_has_this_crop = np.full_like(max_merged_pct_crop, 0.0)
    ever_has_this_crop[max_merged_pct_crop > 0] = 1
    N_crops_ever_active = ever_has_this_crop.sum(axis=0)

    # Split crop area evenly among ever-included crops
    new_pct_cft = np.full_like(surf['PCT_CFT'].isel(cft=0), 0.0)
    new_pct_cft[N_crops_ever_active > 0] = 1.0 / N_crops_ever_active[N_crops_ever_active > 0]
    new_pct_cft = np.expand_dims(new_pct_cft, axis=0)
    new_pct_cft = np.tile(new_pct_cft, reps=[surf.dims['cft'], 1, 1])
    where_zero = np.where(max_merged_pct_crop_da)
    new_pct_cft[where_zero] = 0.0
    new_pct_cft_da = xr.DataArray(data=new_pct_cft,
                                  dims=surf['PCT_CFT'].dims,
                                  attrs=surf['PCT_CFT'].attrs)
    
    
    #############################
    ### Save to run directory ###
    #############################
    
    # Make new Dataset
    new_surf = surf
    new_surf['PCT_CROP'] = new_pct_crop_da
    new_surf['PCT_CFT'] = new_pct_cft_da
    new_surf['PCT_NATVEG'] = new_natveg_da
    
    # Save to new file
    fsurdat_noext, ext = os.path.splitext(args.fsurdat)
    new_fsurdat = f"{fsurdat_noext}.GDDgen{ext}"
    new_fsurdat = os.path.basename(new_fsurdat)
    new_surf.to_netcdf(new_fsurdat, format="NETCDF3_64BIT")

    print(new_fsurdat)


if __name__ == "__main__":
    main(sys.argv[1:])
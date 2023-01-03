# %% Options

# Years to include (set to None to leave that edge unbounded)
y1 = 2000
yN = 2000

# Global attributes for output files
out_attrs = {
    "title": "GGCMI crop calendar for Phase 3, v1.01",
    "author_thisfile": "Sam Rabin (sam.rabin@gmail.com)",
    "author_original": "Jonas Jägermeyr (jaegermeyr@uchicago.edu)",
    "comment": "Day of year is 1-indexed (i.e., Jan. 1 = 1). Filled using cdo -remapnn,$original -setmisstonn"
}

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"


# %% Imports

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import shutil
import os
import time
import datetime as dt
import cftime

import sys
sys.path.append(my_ctsm_python_gallery)
import utils


# %% Setup

# Where to save output files
outdir = "/Users/Shared/CESM_work/crop_dates_mostrice/"

# Files/directories to use for inputs
### f10_f10_mg37
# templatefile = "/Users/Shared/CESM_runs/f10_f10_mg37/2021-11-23/f10_f10_mg37.clm2.h3.2000-01-01-00000.nc"
# indir = "/Users/Shared/GGCMI/AgMIP.input/phase3/ISIMIP3/crop_calendar-nninterp-f10_f10_mg37/"
# file_specifier = "_ggcmi_crop_calendar_phase3_v1.01_nninterp-f10_f10_mg37" # In name of input and output files
### half degree
# templatefile = "/Users/Shared/CESM_runs/halfdeg_test/2022-04-22/halfdeg_test.clm2.h1.1850-01-01-00000.nc"
# indir = "/Users/Shared/GGCMI/AgMIP.input/phase3/ISIMIP3/crop_calendar/"
# file_specifier = "_ggcmi_crop_calendar_phase3_v1.01" # In name of input and output files
### f19_g17 ("2-degree", actually 2.5° lon x 1.875° lat; i.e., 144x96)
templatefile = "/Users/Shared/CESM_runs/spinup_ctsm5.1.dev092_I1850Clm50BgcCrop_f19-g17_pt2/lnd/hist/spinup_ctsm5.1.dev092_I1850Clm50BgcCrop_f19-g17_pt2.clm2.h0.1801-12.nc"
indir = "/Users/Shared/GGCMI/AgMIP.input/phase3/ISIMIP3/crop_calendar-nninterp-f19_g17/"
file_specifier = "_ggcmi_crop_calendar_phase3_v1.01_nninterp-f19_g17" # In name of input and output files


# Add current date/time to output attributes
out_attrs["created"] = dt.datetime.now().replace(microsecond=0).astimezone().isoformat()

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
    "rice": set_crop_dict(61, "ric_rf"),
    "irrigated_rice": set_crop_dict(62, "ric_ir"),
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
    "c3_crop": set_crop_dict(15, None),
    "c3_irrigated": set_crop_dict(16, None),
}


# %% Define variable dictionary and output files
# As "CLM: [GGCMI, outfile]"

def set_var_dict(name_ggcmi, outfile):
    return {"name_ggcmi": name_ggcmi, "outfile": outfile}

variable_dict = {
    "sdate": set_var_dict("planting_day", ""),
    "hdate": set_var_dict("maturity_day", "")
}

def slice_yr(y):
    if y == None:
        return None
    else:
        return str(y)

# Open and time-slice template dataset
template_ds = xr.open_dataset(templatefile, decode_times=True).sel(time=slice(slice_yr(y1), slice_yr(yN)))
# template_ds = template_ds.sel(time=slice(slice_yr(y1), slice_yr(yN)))
if np.size(template_ds.time) == 0:
    print(f"Time slice {y1}-{yN} not found in template dataset. Faking.")
    if y1 != yN:
        raise RuntimeError("Not sure how to fake time axis when y1 != yN")
    template_ds = xr.open_dataset(templatefile, decode_times=True)
    template_ds = template_ds.isel(time=slice(0,1)) # Get the first timestep. Just using .values[0] causes trouble.
    val0 = template_ds.time.values[0]
    if not isinstance(template_ds.time.values[0], cftime.datetime):
        raise TypeError(f"Template file time axis is type {type(template_ds.time.values[0])}, which isn't a cftime.datetime subclass; not sure that next line (assigning fake value) will work.")
    template_ds.time.values[0] = type(val0)(y1, 1, 1, 0, 0, 0, 0, has_year_zero=val0.has_year_zero)
    template_ds.time_bounds.values[0] = np.array([template_ds.time.values[0,], template_ds.time.values[0]])
    if "mcdate" in template_ds:
        template_ds.mcdate.values = np.array([20000101,]).astype(type(template_ds.mcdate.values[0]))
    if "mcsec" in template_ds:
        template_ds.mcsec.values = np.array([0,]).astype(type(template_ds.mcsec.values[0]))
    if "mdcur" in template_ds:
        template_ds.mdcur.values = np.array([0,]).astype(type(template_ds.mdcur.values[0]))
    if "mscur" in template_ds:
        template_ds.mscur.values = np.array([0,]).astype(type(template_ds.mscur.values[0]))
    if "nstep" in template_ds:
        template_ds.nstep.values = np.array([0,]).astype(type(template_ds.nstep.values[0]))
y1 = template_ds.time.values[0].year
yN = template_ds.time.values[-1].year
template_ds.attrs = out_attrs

#  Remove variable(s) we don't need (hdm, or any all-uppercase variables)
for v in template_ds:
    if v == "hdm" or v.upper()==v:
        template_ds = template_ds.drop(v)

# Create output files
for v in variable_dict:
    outfile = "%s%ss%s.%d-%d.%s.nc" % (outdir, v, file_specifier, y1, yN, dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    variable_dict[v]["outfile"] = outfile
    variable_dict[v]["outfile_fill1"] = outfile.replace(".nc", ".fill1.nc")
    template_ds.to_netcdf(path=variable_dict[v]["outfile"])
    template_ds.to_netcdf(path=variable_dict[v]["outfile_fill1"])

template_ds.close()


# %% Process all crops

verbose = True

for thiscrop_clm in crop_dict:

    # Which crop are we on?
    c = list(crop_dict.keys()).index(thiscrop_clm) + 1

    # Get information about this crop
    this_dict = crop_dict[thiscrop_clm]
    thiscrop_int = this_dict["clm_num"]
    thiscrop_ggcmi = this_dict["thiscrop_ggcmi"]
    
    # If no corresponding GGCMI crop, skip opening dataset.
    # Will use previous cropcal_ds as a template.
    if thiscrop_ggcmi == None:
        if c == 1:
            raise ValueError(f"First crop ({thiscrop_clm}) must have a GGCMI type")
        print("Filling %s with dummy data (%d of %d)..." \
            % (str(thiscrop_clm),
            c, 
            len(crop_dict)))
    
    # Otherwise, import crop calendar file
    else:
        if verbose:
            print("Importing %s -> %s (%d of %d)..." \
                % (str(thiscrop_ggcmi), 
                str(thiscrop_clm),
                c, 
                len(crop_dict)))
        
        file_ggcmi = indir + thiscrop_ggcmi + file_specifier + ".nc4"
        if not os.path.exists(file_ggcmi):
            raise Exception("Input file not found: " + file_ggcmi)
        cropcal_ds = xr.open_dataset(file_ggcmi)
        # Flip latitude to match destination
        cropcal_ds = cropcal_ds.reindex(lat=cropcal_ds.lat[::-1])
        # Rearrange longitude to match destination (does nothing if not needed)
        cropcal_ds = utils.lon_idl2pm(cropcal_ds, fail_silently=True)
    
    for thisvar_clm in variable_dict:
        # Get GGCMI netCDF info
        varname_ggcmi = variable_dict[thisvar_clm]["name_ggcmi"]
        if verbose:
            print("    Processing %s..." % varname_ggcmi)
        
        # Get CLM netCDF info
        varname_clm = thisvar_clm + "1_" + str(thiscrop_int)
        file_clm = variable_dict[thisvar_clm]["outfile"]
        if not os.path.exists(file_clm):
            raise Exception("Output file not found: " + file_clm)
        file_clm_tmp = file_clm + ".tmp"
        file_clm_fill1 = variable_dict[thisvar_clm]["outfile_fill1"]
        if not os.path.exists(file_clm_fill1):
            raise Exception("Output file not found: " + file_clm_fill1)
        file_clm_fill1_tmp = file_clm_fill1 + ".tmp"
        
        # "Read" the file (doesn't actually bring anything into memory yet)
        out_ds = xr.open_dataset(file_clm)
        out_fill1_ds = xr.open_dataset(file_clm_fill1)
        
        # Strip dataset to just this variable
        droplist = []
        for i in list(cropcal_ds.keys()):
            if i != varname_ggcmi:
                droplist.append(i)
        thisvar_ds = cropcal_ds.drop(droplist)
        thisvar_ds = thisvar_ds.load()
        thisvar_fill1_ds = thisvar_ds.copy()

        # Convert to integer
        new_fillvalue = -1
        dummyvalue = -1
        thisvar_ds.variables[varname_ggcmi].encoding["_FillValue"] \
            = new_fillvalue
        thisvar_fill1_ds.variables[varname_ggcmi].encoding["_FillValue"] \
            = new_fillvalue
        if thiscrop_ggcmi == None:
            thisvar_ds.variables[varname_ggcmi].values.fill(dummyvalue)
            thisvar_fill1_ds.variables[varname_ggcmi].values.fill(1)
        else:
            thisvar_ds.variables[varname_ggcmi].values[np.isnan(thisvar_ds.variables[varname_ggcmi].values)] \
                = new_fillvalue
            thisvar_ds.variables[varname_ggcmi].values \
                = thisvar_ds.variables[varname_ggcmi].values.astype("int16")
            thisvar_fill1_ds.variables[varname_ggcmi].values[np.isnan(thisvar_fill1_ds.variables[varname_ggcmi].values)] \
                = 1
            thisvar_fill1_ds.variables[varname_ggcmi].values \
                = thisvar_fill1_ds.variables[varname_ggcmi].values.astype("int16")
        
        # Add time dimension (https://stackoverflow.com/a/62862440)
        # (Repeats original map for every timestep)
        # Probably not necessary to use this method, since I only end up extracting thisvar_ds.values anyway---I could probably use some numpy method instead.
        thisvar_ds = thisvar_ds.expand_dims(time = template_ds.time)
        thisvar_fill1_ds = thisvar_fill1_ds.expand_dims(time = template_ds.time)
        # "True" here shows that the time dimension was created by just repeating the one map.
        # tmp = thisvar_ds[varname_ggcmi]
        # np.all((np.diff(tmp.values, axis=0) == 0.0) | np.isnan(np.diff(tmp.values, axis=0)))

        # Add variable to output dataset
        out_ds[varname_clm]=(thisvar_ds[varname_ggcmi].dims,
                             thisvar_ds[varname_ggcmi].values)
        out_fill1_ds[varname_clm]=(thisvar_fill1_ds[varname_ggcmi].dims,
                             thisvar_fill1_ds[varname_ggcmi].values)

        # Edit/add variable attributes etc.
        longname = thisvar_ds[varname_ggcmi].attrs["long_name"]
        longname = longname.replace("rainfed", thiscrop_clm).replace("irrigated", thiscrop_clm)
        def set_var_attrs(out_ds, varname_clm, longname, thiscrop_clm, thiscrop_ggcmi, varname_ggcmi, new_fillvalue):
            out_ds[varname_clm].attrs["long_name"] = longname
            if thiscrop_ggcmi == None:
                out_ds[varname_clm].attrs["crop_name_clm"] = "none"
                out_ds[varname_clm].attrs["crop_name_ggcmi"] = "none"
            else:
                out_ds[varname_clm].attrs["crop_name_clm"] = thiscrop_clm
                out_ds[varname_clm].attrs["crop_name_ggcmi"] = thiscrop_ggcmi
            out_ds[varname_clm].attrs["short_name_ggcmi"] = varname_ggcmi
            out_ds[varname_clm].attrs["units"] = "day of year"
            out_ds[varname_clm].encoding["_FillValue"] = new_fillvalue
            # scale_factor and add_offset are required by I/O library for short data
            # From https://www.unidata.ucar.edu/software/netcdf/workshops/2010/bestpractices/Packing.html:
            #    unpacked_value = packed_value * scale_factor + add_offset
            out_ds[varname_clm].attrs["scale_factor"] = np.int16(1)
            out_ds[varname_clm].attrs["add_offset"] = np.int16(0)
            return out_ds
        out_ds = set_var_attrs(out_ds, varname_clm, longname, thiscrop_clm, thiscrop_ggcmi, varname_ggcmi, new_fillvalue)
        out_fill1_ds = set_var_attrs(out_fill1_ds, varname_clm, longname, thiscrop_clm, thiscrop_ggcmi, varname_ggcmi, new_fillvalue)

        # Save
        if verbose:
            print("    Saving %s..." % varname_ggcmi)
        # start = time.time()
        # Can't overwrite file_clm while you have it open (as out_ds), so first copy it to a temporary file...
        shutil.copyfile(file_clm, file_clm_tmp)
        shutil.copyfile(file_clm_fill1, file_clm_fill1_tmp)
        # ... then save out_ds to the temporary file...
        out_ds.to_netcdf(file_clm_tmp, format="NETCDF3_CLASSIC")
        out_fill1_ds.to_netcdf(file_clm_fill1_tmp, format="NETCDF3_CLASSIC")
        # ... then close out_ds...
        out_ds.close()
        out_fill1_ds.close()
        # ... and finally replace the original file with the new temporary file (deleting the temporary file in the process)
        os.replace(file_clm_tmp, file_clm)
        os.replace(file_clm_fill1_tmp, file_clm_fill1)
        # end = time.time()
        # print(end - start)
        ### NOTE: This method gets slower and slower as the file gets bigger! (The entire process, but also the out_ds.to_netcdf() step.) Is there a better way?
    
    cropcal_ds.close()

print("Done!")

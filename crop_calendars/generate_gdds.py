# %% Setup

# Your path to ctsm_py directory (i.e., where utils.py lives)
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"

# Directory where input file(s) can be found
# indir = "/Volumes/Reacher/CESM_runs/numa_20211014_rx/normal/"
# generate_gdds = False
indir = "/Volumes/Reacher/CESM_runs/numa_20211014_rx/generate_gdds/"
generate_gdds = True

# Either the name of a file within $indir, or a pattern that will return a list of files.
pattern = "*h1.*-01-01-00000.nc"

# List of variables to import from file(s) in $indir matching $pattern. Additional variables will be imported as necessary if they will be useful in gridding any of these. So, e.g., since CPHASE 
myVars = ["CPHASE"]

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import warnings
import glob
import cftime

import sys
sys.path.append(my_ctsm_python_gallery)
import utils


# %% Import dataset

# Get list of all files in $indir matching $pattern
filelist = glob.glob(indir + pattern)

# Import
this_ds = utils.import_ds(filelist, myVars=myVars, myVegtypes=utils.define_mgdcrop_list())

# Get dates in a format that matplotlib can use
with warnings.catch_warnings():
    # Ignore this warning in this with-block
    warnings.filterwarnings("ignore", message="Converting a CFTimeIndex with dates from a non-standard calendar, 'noleap', to a pandas.DatetimeIndex, which uses dates from the standard calendar.  This may lead to subtle errors in operations that depend on the length of time between dates.")
    datetime_vals = this_ds.indexes["time"].to_datetimeindex()


# %% Plot timeseries

thisVar = "CPHASE"

with utils.get_thisVar_da(thisVar, this_ds) as thisvar_da:
    for p in np.arange(0,np.size(this_ds.patches1d_itype_veg_str.values)):
        this_pft_char = this_ds.patches1d_itype_veg_str.values[p]
        this_pft_char = this_pft_char.replace("_", " ")
        plt.plot(datetime_vals, thisvar_da.values[:,p], label = this_pft_char)
    plt.title(thisVar)
    plt.ylabel(this_ds.variables[thisVar].attrs['units'])
    plt.legend()
    plt.show()


# %% Get simulated sowing and harvest dates

# Get year and day number
def get_jday(cftime_datetime_object):
    return cftime.datetime.timetuple(cftime_datetime_object).tm_yday
jday = np.array([get_jday(d) for d in this_ds.indexes["time"]])
def get_year(cftime_datetime_object):
    return cftime.datetime.timetuple(cftime_datetime_object).tm_year
year = np.array([get_year(d) for d in this_ds.indexes["time"]])
year_jday = np.stack((year, jday), axis=1)

# Find sowing and harvest dates in dataset
cphase_da = utils.get_thisVar_da("CPHASE", this_ds)
false_1xNpft = np.full((1,np.size(this_ds.patches1d_itype_veg_str.values)), fill_value=False)
is_sdate = np.bitwise_and( \
    cphase_da.values[:-1,:]==4, \
    cphase_da.values[1:,:]<4)
is_hdate = np.bitwise_and( \
    cphase_da.values[:-1,:]<4, \
    cphase_da.values[1:,:]==4)

# Add False to beginning or end of is_date arrays to ensure correct alignment
is_sdate = np.concatenate((is_sdate, false_1xNpft))
is_hdate = np.concatenate((false_1xNpft, is_hdate))

# Define function for extracting an array of sowing or harvest dates (each row: year, DOY) for a given crop
def get_dates(thisCrop, vegtype_str, is_somedate, year_jday):
    is_somedate_thiscrop = is_somedate[:,[d==thisCrop for d in vegtype_str]]
    is_somedate_thiscrop = np.squeeze(is_somedate_thiscrop)
    result = year_jday[is_somedate_thiscrop,:]
    if result.size == 0:
        raise ValueError("No dates found")
    return result

# Loop through crops and print their sowing and harvest dates
for thisCrop in this_ds.patches1d_itype_veg_str.values:
    
    # Get dates
    this_sdates = get_dates(thisCrop, this_ds.patches1d_itype_veg_str.values, is_sdate, year_jday)
    this_hdates = get_dates(thisCrop, this_ds.patches1d_itype_veg_str.values, is_hdate, year_jday)
    
    # The first event in a dataset could be a harvest. If so, discard.
    if this_sdates[0,1] > this_hdates[0,1]:
        this_hdates = this_hdates[1:,:]
    
    # There should be at least as many sowings as harvests
    nsow = np.shape(this_sdates)[0]
    nhar = np.shape(this_hdates)[0]
    if nsow < nhar:
        raise ValueError("%d harvests but only %d sowings" % \
            (nhar, nsow))

    # If there are more sowings than harvests, append NaN for last growing season
    if nsow > nhar:
        if nsow > nhar + 1:
            raise ValueError("%d sowings but only %d harvests" % \
            (nsow, nhar))
        this_hdates = np.concatenate(( \
            this_hdates[1:,:], 
            np.array([[this_sdates[-1,0], np.nan]])))
    
    # Ensure harvests occurred either the same year as sowing or the next year
    if any(this_hdates[:,0] > this_sdates[:,0] + 1):
        raise ValueError("Some harvest does not occur in either the same year as or year after corresponding sowing")
    
    # Print dates. Each row: sowing year, sowing DOY, harvest DOY
    this_dates = np.concatenate((this_sdates, this_hdates[:,1:]), axis=1)
    print(thisCrop)
    print(this_dates)


# %% Get read-in sowing dates for this cell

sdate_file = "/Volumes/Reacher/CESM_work/crop_dates/sdates_ggcmi_crop_calendar_phase3_v1.01.2000-2000.nc"


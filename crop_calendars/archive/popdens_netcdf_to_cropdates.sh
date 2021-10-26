#!/bin/bash
set -e

infile="/Volumes/Reacher/CESM_inputdata/lnd/clm2/firedata/clmforc.Li_2018_SSP1_CMIP6_hdm_0.5x0.5_AVHRR_simyr1850-2100_c181205.nc"

outdir="/Volumes/Reacher/CESM_work/crop_dates"
cd "${outdir}"

# Get one time step (first in file is 182.5)
echo "Extracting one time step..."
ncks -O -d time,182.5,182.5 "${infile}" ggcmi3_crop_calendars.nc

# Copy over crop calendar variables
cp step01.nc step02.nc
file_cropcal="/Volumes/Reacher/GGCMI/AgMIP.input/phase3/ISIMIP3/crop_calendar/mai_ir_ggcmi_crop_calendar_phase3_v1.01.nc4"
#nccopy -V planting_day "${file_cropcal}" step02.nc # just replaces step02.nc
#ncks -A -v planting_day "${file_cropcal}" step02.nc # comes out upside-down
ncpdq -A -v planting_day -a -lat "${file_cropcal}" step02.nc


echo "Done."

exit 0
# What system is the script running on?
import socket
hostname = socket.gethostname()

# Import the CTSM Python utilities
import sys
if hostname == "Sams-2021-MacBook-Pro.local":
    sys.path.append("/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/")
    import utils
else:
    # Only possible because I have export PYTHONPATH=$HOME in my .bash_profile
    from ctsm_python_gallery_myfork.ctsm_py import utils

# Import everything else
import numpy as np
import xarray as xr
import cftime
import getopt
import os

def main(argv):

    help_string = "repeat_gdds_over_years.py -i <input-file> -1 <first-year> -N <last-year> OPTIONAL: [(-d <output-dir> OR -o <output-file>)]"

    # Get arguments
    y1 = None
    yN = None
    infile = None
    outfile = None
    outdir = None
    overwrite = 0
    try:
        opts, args = getopt.getopt(argv, "hi:o:1:N:n:d", ["input-file", "output-file", "first-year", "last-year", "output-dir", "overwrite", "no-overwrite"])
    except getopt.GetoptError:
        print(help_string)
        print("Error parsing arguments. Probably an incorrect option specified?")
        sys.exit(1)
    for opt, arg in opts:
        if opt == "-h":
            print(help_string)
            print("   If neither -o/--output-file nor -d/--output-dir specified, will save to input-file-parent-dir/inputfilenamenoext.firstyear-lastyear.ext")
            print("   If -d/--output-dir specified, will save to output-dir/inputfilenamenoext.firstyear-lastyear.ext")
            sys.exit()
        elif opt in ("-i", "--input-file"):
            infile = arg
        elif opt in ("-o", "--output-file"):
            outfile = arg
        elif opt in ("-d", "--output-dir"):
            outdir = arg
        elif opt in ("-1", "--first-year"):
            y1 = int(arg)
        elif opt in ("-N", "-n", "--last-year"):
            yN = int(arg)
        elif opt == "--overwrite":
            overwrite = 1
        elif opt == "--no-overwrite":
            overwrite = -1

    # Check arguments
    if not infile:
        print(help_string)
        print("You must provide an input file with -i/--input-file.")
        sys.exit(2)
    elif outfile and outdir:
        print(help_string)
        print("Provide only one of -o/--output-file and -d/--output-dir")
        sys.exit(2)
    elif y1 == None or yN == None:
        print(help_string)
        print("You must provide both -y1/--first-year and -yN/--last-year")
        sys.exit(2)

    # Parse remaining info
    if not outfile:
        if not outdir:
            outdir = os.path.dirname(infile)
        outname, outext = os.path.splitext(os.path.basename(infile))
        outfile = os.path.join(outdir, f"{outname}.{y1}-{yN}{outext}")
        print(f"Saving to {outfile}")
    yearList = np.arange(y1, yN+1)
    Nyears = len(yearList)

    # Handle existing output file
    if os.path.exists(outfile):
        print("Output file already exists.")
        while overwrite == 0:
            answer = input("Overwrite? (yes or no) ")
            if any(answer.lower() == f for f in ["yes", 'y', '1', 'ye']):
                overwrite = 1
            elif any(answer.lower() == f for f in ['no', 'n', '0']):
                overwrite = -1
            else:
                print('   Please enter yes or no')
        if overwrite == -1:
            print("Exiting.")
            sys.exit()
        else:
            print("Overwriting.")

    # Import
    ds_in = xr.open_dataset(infile)
    
    # Repeat for each year in specified list
    ds_out = utils.tile_over_time(ds_in, years=yearList)
    
    # Save
    ds_out.to_netcdf(outfile, mode='w', format='NETCDF3_CLASSIC')

if __name__ == "__main__":
   main(sys.argv[1:])


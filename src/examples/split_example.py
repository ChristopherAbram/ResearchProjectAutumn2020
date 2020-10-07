import sys
import os
import numpy as np

from utils.definitions import get_project_path


def main(argc, argv):
    country = 'NGA'
    wp_file = os.path.join(get_project_path(), "data", "worldpop", '%s' % country, '%s_ppp_2015.tif' % country.lower())

    patch_size = 8192 # [px]
    output_dirpath = os.path.join(get_project_path(), "data", "worldpop", '%s' % country, "patches")

    # Create directory if doesn't exist:
    os.system('mkdir -p %s' % output_dirpath)
    # Run splitting with gdal:
    os.system('gdal_retile.py -ps %d %d -targetDir %s %s' % (patch_size, patch_size, output_dirpath, wp_file))

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))

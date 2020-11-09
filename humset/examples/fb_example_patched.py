import sys
import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import geopandas as gpd
import rasterio.mask as mask
from shapely.geometry import box
import json

from humset.utils.definitions import get_project_path


def main(argc, argv):
    # First run split_example.py

    country = 'NGA'
    wp_dir = os.path.join(get_project_path(), "data", "humdata", '%s' % country, 'patches', '*.tif')
    
    # Iterates over smaller patches and plots entire patch:
    for filepath in glob.iglob(wp_dir):
        print(filepath)
        with rasterio.open(filepath) as pop:
            X = pop.read(1)

            fig, ax1 = plt.subplots()
            cmap = plt.cm.magma

            im1 = ax1.imshow(X, cmap=cmap, norm=LogNorm())
            cbar = fig.colorbar(im1, ax=ax1, pad=0.005, fraction=0.1)
            cbar.set_label('Population density', rotation=90)
            ax1.axis('on')

            plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))

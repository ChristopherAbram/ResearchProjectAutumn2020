import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import LogNorm

from utils.definitions import get_project_path
from utils.raster import RasterWindow, RasterWindowSize


def main(argc, argv):

    country = 'NGA'
    in_file = os.path.join(
        get_project_path(), "data", "humdata", '%s' % country, 'population_%s_2018-10-01.tif' % country.lower())

    cmap = plt.cm.magma

    # Read by window, define horizontal and vertical split.
    # e.g. RasterWindow(in_file, 8, 4) splits underlying data into matrix of 8 columns and 4 rows.
    for (window, (row, col), (width, height)) in RasterWindow(in_file, 8, 4):
        fig, ax = plt.subplots()
        ax.imshow(window, cmap=cmap, norm=LogNorm())
        plt.show()

    # Read by window size, 5000 X 3000 px
    # Note: The sizes around edge will differ from given size. You can get these values from (width, height)
    # for (window, (row, col), (width, height)) in RasterWindowSize(in_file, 5000, 3000):
    #     fig, ax = plt.subplots()
    #     ax.imshow(window, cmap=cmap, norm=LogNorm())
    #     plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import LogNorm

from utils.definitions import get_project_path
from utils.raster import RasterWindow


def main(argc, argv):

    country = 'NGA'
    in_file = os.path.join(
        get_project_path(), "data", "humdata", '%s' % country, 'population_%s_2018-10-01.tif' % country.lower())

    cmap = plt.cm.magma
    for window in RasterWindow(in_file, 100, 100):
        fig, ax = plt.subplots()
        ax.imshow(window, cmap=cmap, norm=LogNorm())
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import LogNorm
import rasterio
from rasterio.windows import Window

from utils.definitions import get_project_path
from utils.raster import RasterWindow, RasterWindowSize


def main(argc, argv):

    country = 'NGA'
    in_file = os.path.join(
        get_project_path(), "data", "humdata", '%s' % country, 'population_%s_2018-10-01.tif' % country.lower())

    cmap = plt.cm.magma

    # Generate samples tiffs
    # it = RasterWindow(in_file, 10, 10)
    # y = (1000 - 9*100) / 8
    # image = np.ones((100, 100), dtype=rasterio.ubyte) * 127
    # with rasterio.open(
    #         'example_%s_%s.tif' % (0,1), 'w',
    #         driver='GTiff', width=1000, height=1000, count=1,
    #         dtype=image.dtype) as dst:
    #     for i, j in itertools.product(range(9), range(9)):
    #         dst.write(image, window=Window(j * (100+y), i * (100+y), 100, 100), indexes=1)

    # mean = [500, 500]
    # cov = [[10000, 0], [0, 10000]]
    # x, y = np.random.multivariate_normal(mean, cov, 100000).T
    # x, y = x.astype(np.uint32), y.astype(np.uint32)

    # image = np.zeros((1000, 1000), dtype=rasterio.ubyte)
    # image[x, y] = 255
    # with rasterio.open(
    #     'example_%s_%s.tif' % (0,0), 'w',
    #     driver='GTiff', width=1000, height=1000, count=1,
    #     dtype=image.dtype) as dst:

    #     dst.write(image, indexes=1)



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

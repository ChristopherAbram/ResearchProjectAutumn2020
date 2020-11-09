import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import LogNorm
import rasterio
from rasterio.windows import Window

from humset.utils.definitions import get_dataset_paths, get_project_path
from humset.utils.raster import RasterTable, RasterTableSize, RasterTableAligned


def main(argc, argv):

    in_files = get_dataset_paths('NGA')
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
    # e.g. RasterTable(in_file, 8, 4) splits underlying data into matrix of 8 columns and 4 rows.
    # table = RasterTable(in_file, 8, 4)
    # for window in table:
    #     fig, ax = plt.subplots()
    #     ax.imshow(window.data, cmap=cmap, norm=LogNorm())
    #     plt.show()

    # Read by window size, 5000 X 3000 px
    # Note: The sizes around edge will differ from given size. You can get these values from (width, height)
    table = RasterTableSize(in_files['humdata'], 2000, 1000)
    for window in table:
        fig, ax = plt.subplots()
        ax.imshow(window.data, cmap=cmap, norm=LogNorm())
        plt.show()

    table = RasterTableAligned(in_files['humdata'], in_files['grid3'], 10, 10)
    for w_hum, w_grid3 in table:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15 ,10))
        ax1.imshow(w_hum.data, cmap=cmap, norm=LogNorm())
        w_grid3.data = w_grid3.data * 255
        ax2.imshow(w_grid3.data, cmap='gray')
        plt.show()

    table = RasterTable(os.path.join(get_project_path(), 'data', 'out', 'nga_metrics.tif'), 1, 1, index=1)
    for window in table:
        fig, ax = plt.subplots()
        ax.imshow(window.data, cmap=cmap, norm=LogNorm())
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))

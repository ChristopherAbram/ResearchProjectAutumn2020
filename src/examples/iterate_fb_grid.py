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


def upsize(img, n):
    return np.repeat(np.repeat(img,n,axis=1),n,axis=0)


def main(argc, argv):

    country = 'NGA'
    fb_infile = os.path.join(
        get_project_path(), "data", "humdata", '%s' % country, 'population_%s_2018-10-01.tif' % country.lower())

    grid_infile = os.path.join(get_project_path(), "data", "grid", '%s' % country, '%s_population.tif' % country.lower())

    xsplit, ysplit = 200, 200
    fb = RasterWindow(fb_infile, xsplit, ysplit)
    grid = RasterWindow(grid_infile, xsplit, ysplit)

    metrics = np.zeros((xsplit, ysplit, 8))
    for (fb_win, (row, col), (width, height)) in fb:
        grid_win = grid.get(row, col)[0]
        
        # Process windows:
        w, h = min(fb_win.shape[1], grid_win.shape[1]), min(fb_win.shape[0], grid_win.shape[0])
        fb_win = np.nan_to_num(fb_win)
        fb_win[np.where(fb_win > 0)] = 255
        fb_win = fb_win[:h,:w]
        grid_win = grid_win * 255
        grid_win = upsize(grid_win, 3)
        grid_win = grid_win[:h,:w]

        diff_win = fb_win - grid_win

        FP = (diff_win > 0).astype(int)
        FN = (diff_win < 0).astype(int)

        metrics[row, col, 0] = FP.mean()
        metrics[row, col, 1] = FN.mean()
        coo_grid = np.array(grid.get_coords(0, 0))
        coo_fb = np.array(fb.get_coords(0, 0))
        shift = coo_grid - coo_fb
        metrics[row, col, 2] = coo_grid[0]
        metrics[row, col, 3] = coo_grid[1]
        metrics[row, col, 4] = coo_fb[0]
        metrics[row, col, 5] = coo_fb[1]
        metrics[row, col, 6] = shift[0]
        metrics[row, col, 7] = shift[1]

        # if fb_win.mean() > 50:
        #     # coo_grid = np.array(grid.get_coords(0, 0))
        #     # coo_fb = np.array(fb.get_coords(0, 0))
        #     # shift = coo_grid - coo_fb
        #     print("GRID3: {}".format((coo_grid[0], coo_grid[1])))
        #     print("FB: {}".format((coo_fb[0], coo_fb[1])))
        #     print("Shift: {}".format((shift[0], shift[1])))

        #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        #     ax1.imshow(fb_win, cmap='gray')
        #     ax1.set_title('Facebook')
        #     ax2.imshow(grid_win, cmap='gray')
        #     ax2.set_title('GRID3')
        #     ax3.imshow(FP, cmap='gray')
        #     ax3.set_title('FP')
        #     ax4.imshow(FN, cmap='gray')
        #     ax4.set_title('FN')
        #     plt.show()

    plt.imshow(metrics[:,:,0])
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))

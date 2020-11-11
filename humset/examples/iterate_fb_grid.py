import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.colors import LogNorm
import rasterio
from rasterio.windows import Window
from salem import GoogleVisibleMap

from humset.utils.definitions import get_project_path
from humset.utils.raster import RasterTable, RasterTableSize


def upsize(img, x, y):
    return np.repeat(np.repeat(img,x,axis=1),y,axis=0)


def main(argc, argv):

    country = 'NGA'
    fb_infile = os.path.join(
        get_project_path(), "data", "humdata", 'population_%s_2018-10-01.tif' % country.lower())

    grid_infile = os.path.join(get_project_path(), "data", "grid3", '%s - population - v1.2 - mastergrid.tif' % country)

    xsplit, ysplit = 200, 200
    fb = RasterTable(fb_infile, xsplit, ysplit)
    grid = RasterTable(grid_infile, xsplit, ysplit)

    metrics = np.zeros((xsplit, ysplit, 8))
    for fb_win in fb:
        # (fb_win, (row, col), (width, height))
        row, col = fb_win.pos
        width, height = fb_win.size
        grid_win = grid.get(row, col)
        
        # Process windows:
        w, h = min(fb_win.size[1], grid_win.size[1]), min(fb_win.size[0], grid_win.size[0])
        fb_win.data = np.nan_to_num(fb_win.data)
        fb_win.data[np.where(fb_win.data > 0)] = 255
        fb_win.data = fb_win.data[:h,:w]
        grid_win.data = grid_win.data * 255
        grid_win.data = upsize(grid_win.data, 3, 3) # fb-dataset has 1-arcsec resolution, grid has 3-arcsec
        grid_win.data = grid_win.data[:h,:w]

        diff_win = fb_win.data - grid_win.data

        FP = (diff_win > 0).astype(int)
        FN = (diff_win < 0).astype(int)

        metrics[row, col, 0] = FP.mean()
        metrics[row, col, 1] = FN.mean()
        coo_grid = np.array(grid.find_geo_coords(row, col, 0, 0))
        coo_fb = np.array(fb.find_geo_coords(row, col, 0, 0))
        shift = coo_grid - coo_fb
        metrics[row, col, 2] = coo_grid[0]
        metrics[row, col, 3] = coo_grid[1]
        metrics[row, col, 4] = coo_fb[0]
        metrics[row, col, 5] = coo_fb[1]
        metrics[row, col, 6] = shift[0]
        metrics[row, col, 7] = shift[1]

        # if fb_win.data.mean() > 50:
        #     # coo_grid = np.array(grid.get_coords(0, 0))
        #     # coo_fb = np.array(fb.get_coords(0, 0))
        #     # shift = coo_grid - coo_fb
        #     print("GRID3: {}".format((coo_grid[0], coo_grid[1])))
        #     print("FB: {}".format((coo_fb[0], coo_fb[1])))
        #     print("Shift: {}".format((shift[0], shift[1])))

        #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        #     ax1.imshow(fb_win.data, cmap='gray')
        #     ax1.set_title('Facebook')
        #     ax2.imshow(grid_win.data, cmap='gray')
        #     ax2.set_title('GRID3')
        #     ax3.imshow(FP, cmap='gray')
        #     ax3.set_title('FP')
        #     ax4.imshow(FN, cmap='gray')
        #     ax4.set_title('FN')
        #     plt.show()

    c = plt.imshow(metrics[:,:,0])
    # plt.colorobar(c)
    plt.suptitle('HRSL-Grid3 FP rate Nigeria')
    plt.show()

    #np.savetxt('NGA-FP.csv', metrics[:,:,0], delimiter=',')
    #np.savetxt('NGA-FN.csv', metrics[:,:,1], delimiter=',')

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))

import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import gdal
import osr
from matplotlib.colors import LogNorm

import geopandas as gpd
import rasterio.mask as mask
from shapely.geometry import box
import json

from humset.utils.definitions import get_project_path


def main(argc, argv):

    country = 'NGA'
    in_file = os.path.join(get_project_path(), "data", "humdata", '%s/population_%s_2018-10-01.tif' % (country, country.lower()))
    out_file = os.path.join(get_project_path(), "data", "humdata", '%s' % country, 'population_%s_out.tif' % country.lower())

    ds = gdal.Open(in_file)
    geo_s = ds.GetGeoTransform()
    x_size = ds.RasterXSize # Raster xsize
    y_size = ds.RasterYSize # Raster ysize
    # ds = gdal.Warp('', in_file, dstSRS='EPSG:4326')

    # Read this!!!: https://gdal.org/user/virtual_file_systems.html
    # mem_drv = gdal.GetDriverByName('MEM')

    # ds = gdal.Warp(out_file, in_file, dstSRS='EPSG:4326', format='GTiff')
    ds = gdal.Translate(out_file, ds, projWin=[3.35-0.1, 6.5+0.1, 3.35+0.1, 6.5-0.1], projWinSRS='EPSG:4326')
    ds = None

    with rasterio.open(out_file) as pop:
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

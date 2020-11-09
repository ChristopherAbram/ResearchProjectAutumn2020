import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import geopandas as gpd
import rasterio.mask as mask
from shapely.geometry import box
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map
import json

from humset.utils.definitions import get_project_path


def main(argc, argv):
    country = 'NGA'
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 10))
    cmap = plt.cm.magma

    bbox = box(3.35-0.1, 6.5-0.1, 3.35+0.1, 6.5+0.1)
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs='EPSG:4326')
    coords = geo['geometry']

    # Worldpop:
    wp_file = os.path.join(get_project_path(), "data/worldpop", '%s/%s_ppp_2015.tif' % (country, country.lower()))
    with rasterio.open(wp_file) as pop:
        X = pop.read(1)
        out_img, out_transform = mask.mask(pop, coords, crop=True)
        pop_wp = np.float32(out_img[0].copy())
        pop_wp[np.where(pop_wp==-99999)] = 0
        im1 = ax1.imshow(pop_wp, cmap=cmap, norm=LogNorm())

    # Humdata:
    fb_file = os.path.join(get_project_path(), "data/humdata", '%s/population_%s_2018-10-01.tif' % (country, country.lower()))
    with rasterio.open(fb_file) as pop:
        X = pop.read(1)
        out_img, out_transform = mask.mask(pop, coords, crop=True)
        pop_fb = np.float32(out_img[0].copy())
        im2 = ax2.imshow(pop_fb, cmap=cmap, norm=LogNorm())

    # GRID3:
    grid_file = os.path.join(get_project_path(), "data/grid", '%s/%s_population.tif' % (country, country.lower()))
    with rasterio.open(grid_file) as pop:
        X = pop.read(1)
        sum = np.mean(X)
        out_img, out_transform = mask.mask(pop, coords, crop=True)
        pop_fb = np.float32(out_img[0].copy())
        im3 = ax3.imshow(pop_fb, cmap=cmap, norm=LogNorm())

    # Satelite:
    g = GoogleVisibleMap(x=[3.35-0.1, 3.35+0.1], y=[6.5-0.1, 6.5+0.1],
            size_x = 500, size_y = 500,
            #size_x=img_arr1.shape[0], size_y=img_arr1.shape[1],
            scale=4,  # scale is for more details
            maptype='satellite'
        )  # try out also: 'terrain'

    ggl_img = g.get_vardata()
    ax4.imshow(ggl_img)

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))

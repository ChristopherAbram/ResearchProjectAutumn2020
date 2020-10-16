import sys
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import geopandas as gpd
import rasterio.mask as mask
from shapely.geometry import box
import json

from utils.definitions import get_project_path


def main(argc, argv):

    country = 'NGA'
    wp_file = os.path.join(get_project_path(), "data/worldpop", '%s_ppp_2015.tif' % country.lower())

    with rasterio.open(wp_file) as pop:
        X = pop.read(1)

        bbox = box(3.35-0.1, 6.5-0.1, 3.35+0.1, 6.5+0.1)
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs='EPSG:4326')

        coords = geo['geometry']
        out_img, out_transform = mask.mask(pop, coords, crop=True)
        pop_wp = np.float32(out_img[0].copy())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        cmap = plt.cm.magma

        im1 = ax1.imshow(pop_wp, cmap=cmap, norm=LogNorm())
        cbar = fig.colorbar(im1, ax=ax1, pad=0.005, fraction=0.1)
        cbar.set_label('Population density', rotation=90)
        ax1.axis('on')

        ax2.imshow(X, cmap=cmap, norm=LogNorm())
        
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))

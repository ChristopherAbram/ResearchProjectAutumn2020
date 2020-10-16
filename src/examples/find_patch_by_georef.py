import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import rasterio
from shapely.geometry import box
from matplotlib.colors import LogNorm
from utils.location import GeoLocation

from utils.definitions import get_project_path
from utils.raster import RasterTable, RasterTableSize, get_window_px, get_window_geo

### PARAMS ###
country = 'NGA'
in_files = [
    os.path.join(get_project_path(), "data", "humdata", 'population_%s_2018-10-01.tif' % country.lower()),
    os.path.join(get_project_path(), "data", "worldpop", '%s_ppp_2015.tif' % country.lower())
    #os.path.join(get_project_path(), "data", "grid3", '%s - population - v1.2 - mastergrid.tif' % country)
    ]
lon, lat = (3.35, 6.5)  # better to hardcode now so no dep on third-party
box_spread = 0.1
ROI = 'Lagos, Nigeria'
patch_shape = (64,64)
##############

def get_grid_shape(img, patch_shape):
    return (np.floor(img.shape[0] / patch_shape[0]), np.floor(img.shape[1] / patch_shape[1]))

def crop_to_fit(img, patch_shape):
    grid_shape = get_grid_shape(img, patch_shape)
    x_lim = int(grid_shape[0] * patch_shape[0])
    y_lim = int(grid_shape[1] * patch_shape[1])
    return img[:x_lim, :y_lim]

def get_coords(query):
    """returns longtitude, latitude for query (e.g. 'Lagos, Nigeria')"""
    geo = GeoLocation()
    coords = geo.get_coordinates(query)
    return (coords['lon'], coords['lat'])

def rasterio_display_info(src):
    print('Displaying properties of geospatial raster file:')
    print(f'width: {src.width}, height: {src.height}')
    print(f'crs: {src.crs}')
    print(f'transform: {src.transform}')
    print(f'count: {src.count}')
    print(f'indexes: {src.indexes}\n')

def upsize(img, n):
    return np.repeat(np.repeat(img,n,axis=1),n,axis=0)

def main(argc, argv):

    cmap = plt.cm.magma
    fig, ax = plt.subplots(1,3)

    for i, in_file in enumerate(in_files):
        with rasterio.open(in_file) as dataset:
            print(in_file)
            rasterio_display_info(dataset)

            #lon, lat = get_coords(ROI)
            #print(f'longtitutde: {lon}\nlatitude: {lat}')

            px, py = dataset.index(lon, lat)
            #print('Pixel Y, X coords: {}, {}'.format(py, px))

            # window = get_window_px(dataset, px - 500, py - 500, 1000, 1000)
            #rows, cols = rasterio.transform.rowcol(dataset.transform, [lon-0.1, lon+0.1], [lat-0.1, lat+0.1])

            window = get_window_geo(dataset, box(lon-box_spread, lat-box_spread, lon+box_spread, lat+box_spread))
            if i>0:
                window = upsize(window,3)
            print(f'window shape after box: {window.shape}')

            #window = crop_to_fit(window, patch_shape)
            #print(f'window shape after crop: {window.shape}')

            ax[i].imshow(window, cmap=cmap, norm=LogNorm())
            #ax[i].
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import rasterio
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map, GoogleCenterMap
import salem
from shapely.geometry import box
from matplotlib.colors import LogNorm
from utils.location import GeoLocation

from humset.utils.definitions import get_project_path
from humset.utils.raster import RasterTable, RasterTableSize, get_window_px, get_window_geo
from humset.visualization import AlignMapsEditor

### PARAMS ###
country = 'NGA'
in_files = [
    os.path.join(get_project_path(), "data", "humdata", 'population_%s_2018-10-01.tif' % country.lower()),
    os.path.join(get_project_path(), "data", "worldpop", '%s_ppp_2015.tif' % country.lower()),
    os.path.join(get_project_path(), "data", "grid3", '%s - population - v1.2 - mastergrid.tif' % country)
    ]

# lon, lat = (3.36, 6.49)  # Lagos 1
# lat, lon = (6.541456, 3.312719)  # Lagos 2
lat, lon = (8.499714, 3.423570) # Ago-Are
# lat, lon = (7.382932, 3.929635) # Ibadan
# lat, lon = (4.850891, 6.993961) # Port Harcourt

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

    editor = AlignMapsEditor(in_files[0], in_files[2], (lat, lon))
    editor.wait()

    # cmap = plt.cm.magma
    # with rasterio.open(in_files[0]) as humdata, rasterio.open(in_files[2]) as grid3:
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))
        # print(in_files[1])
        # rasterio_display_info(humdata)
        # print(in_files[2])
        # rasterio_display_info(grid3)

        # h_data, h_window = get_window_geo(humdata, box(lon-box_spread, lat-box_spread, lon+box_spread, lat+box_spread))
        # bounds = humdata.window_bounds(h_window)
        # g_window = grid3.window(*bounds)
        # g_data = grid3.read(1, window=g_window)

        # # Preprocess data:
        # h_data = np.nan_to_num(h_data)
        # h_data[np.where(h_data > 0)] = 255
        # h_data = h_data.astype(np.uint8)
        # g_data = g_data * 255
        # h_data = cv2.merge((h_data, h_data, h_data))
        # g_data = cv2.merge((g_data, g_data, g_data))

        # ax1.imshow(h_data, cmap='gray')
        # ax2.imshow(g_data, cmap='gray')
        # plt.show()

        # g = GoogleVisibleMap(x=[bounds[0], bounds[2]], y=[bounds[1], bounds[3]],
        #         size_x = 485, size_y = 485, 
        #         crs=salem.wgs84,
        #         #size_x=img_arr1.shape[0], size_y=img_arr1.shape[1],
        #         scale=4,
        #         maptype='satellite'
        #     )

        # ggl_img = g.get_vardata()
        # ggl_img = ggl_img * 255
        # ggl_img = ggl_img.astype(np.uint8)

        # h_data = cv2.resize(h_data, (ggl_img.shape[1], ggl_img.shape[0]), interpolation=cv2.INTER_AREA)
        # g_data = cv2.resize(g_data, (ggl_img.shape[1], ggl_img.shape[0]), interpolation=cv2.INTER_AREA)

        # alpha = 0.8
        # gg_h_img = cv2.addWeighted(ggl_img, alpha, h_data, 1.-alpha, 0.0)
        # gg_g_img = cv2.addWeighted(ggl_img, alpha, g_data, 1.-alpha, 0.0)
        
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))
        # ax1.imshow(gg_h_img)
        # ax2.imshow(gg_g_img)
        # plt.show()


    # for i, in_file in enumerate(in_files):
    #     with rasterio.open(in_file) as dataset:
    #         print(in_file)
    #         rasterio_display_info(dataset)

    #         #lon, lat = get_coords(ROI)
    #         #print(f'longtitutde: {lon}\nlatitude: {lat}')

    #         px, py = dataset.index(lon, lat)
    #         #print('Pixel Y, X coords: {}, {}'.format(py, px))

    #         # window = get_window_px(dataset, px - 500, py - 500, 1000, 1000)
    #         #rows, cols = rasterio.transform.rowcol(dataset.transform, [lon-0.1, lon+0.1], [lat-0.1, lat+0.1])

    #         window = get_window_geo(dataset, box(lon-box_spread, lat-box_spread, lon+box_spread, lat+box_spread))
    #         if i>0:
    #             window = upsize(window,3)
    #         print(f'window shape after box: {window.shape}')

    #         #window = crop_to_fit(window, patch_shape)
    #         #print(f'window shape after crop: {window.shape}')

    #         ax[i].imshow(window, cmap=cmap, norm=LogNorm())
    #         #ax[i].
    # plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
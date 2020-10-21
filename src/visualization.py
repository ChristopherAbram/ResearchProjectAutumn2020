import numpy as np
import cv2, rasterio, salem
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map, GoogleCenterMap
from shapely.geometry import box

from utils.raster import get_window_geo


class AlignMapsEditor:
    """
    Opens an interactive window which allows to visualize Humdata and GRID3 aligned with Google Static Maps.

    Attributes:
        humdata_path (string): A path to humdata GeoTiff file,
        grid3_path (string): A path to GRID3 GeoTiff file,
        location (tuple): A pair (lat, lon), location on the map.
    """

    def __init__(self, humdata_path, grid3_path, location):

        self.humdata_path = humdata_path
        self.grid3_path = grid3_path
        self.window_name = "align_humdata_and_grid3"
        self.lat = location[0]
        self.lon = location[1]

        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("size", self.window_name, 1, 600, self.update)
        self.update(3)

    def make_grid(self, out_shape, size_vertical, size_horizontal, thicc=1):
        grid = np.zeros(out_shape).astype(np.uint8)
        height, width, channels = out_shape
        for x in range(0, width - 1, size_vertical):
            cv2.line(grid, (x, 0), (x, height), (255, 255, 255), thicc, 1)
        for x in range(0, width - 1, size_horizontal):
            cv2.line(grid, (0, x), (width, x), (255, 255, 255), thicc, 1)

        return grid

    def convolve(self, img_in, img_out, kernel, stride):
        for i in range(img_in.shape[0] // stride):
            for j in range(img_in.shape[1] // stride):
                img_out[i,j] = np.sum(\
                                img_in[i*stride:i*stride+stride, j*stride:j*stride+stride] *\
                                kernel)
        return img_out

    def update(self, val):
        box_spread = val / 1000.
        with rasterio.open(self.humdata_path) as humdata, rasterio.open(self.grid3_path) as grid3:

            h_data, h_window = get_window_geo(
                humdata, box(self.lon - box_spread, self.lat - box_spread, 
                             self.lon + box_spread, self.lat + box_spread))
            bounds = humdata.window_bounds(h_window)
            g_window = grid3.window(*bounds)
            g_data = grid3.read(1, window=g_window)

            # Preprocess data:
            h_data = np.nan_to_num(h_data)
            h_data[np.where(h_data > 0)] = 255
            h_data = h_data.astype(np.uint8)
            g_data = g_data * 255
            self.h_data = cv2.merge((h_data, np.zeros(h_data.shape, dtype=np.uint8), h_data))
            self.g_data = cv2.merge((g_data, g_data, np.zeros(g_data.shape, dtype=np.uint8)))

        crs = salem.gis.check_crs('epsg:4326')
        g = GoogleVisibleMap(x=[self.lon-box_spread, self.lon+box_spread], y=[self.lat-box_spread, self.lat+box_spread],
                size_x = 640, size_y = 640, 
                crs='epsg:4326',
                scale=1,
                maptype='satellite'
            )

        ggl_img = g.get_vardata()
        ggl_img = ggl_img * 255
        ggl_img = ggl_img.astype(np.uint8)

        # TODO: Perhaps more sophasticated method should be used...
        # It doesn't align google maps well... It seems that google api doesn't return different bboxes depending on box_spread value.
        # that is, for some cases it returns the same image for slightly different bbox regardless change in box_spread value.
        # Also, google maps doesn't use the same projection:
        # https://gis.stackexchange.com/questions/48949/epsg-3857-or-4326-for-googlemaps-openstreetmap-and-leaflet 
        h_d = cv2.resize(self.h_data, (ggl_img.shape[1], ggl_img.shape[0]), interpolation=cv2.INTER_AREA)
        g_d = cv2.resize(self.g_data, (ggl_img.shape[1], ggl_img.shape[0]), interpolation=cv2.INTER_AREA)

        # add alpha and overlap
        alpha = 0.8
        gg_h_img = cv2.addWeighted(ggl_img, alpha, h_d, 1.-alpha, 0.0)
        gg_g_img = cv2.addWeighted(ggl_img, alpha, g_d, 1.-alpha, 0.0)

        # add grid
        if val<5:
            ratio_g_h = 3
            grid_h_outer = self.make_grid(gg_h_img.shape,\
                                            ggl_img.shape[0] // self.h_data.shape[0] * ratio_g_h,\
                                            ggl_img.shape[1] // self.h_data.shape[1] * ratio_g_h,\
                                            thicc=2)
            grid_h_inner = self.make_grid(gg_h_img.shape,\
                                            ggl_img.shape[0] // self.h_data.shape[0],\
                                            ggl_img.shape[1] // self.h_data.shape[1],\
                                            thicc = 2)
            grid_g = self.make_grid(gg_g_img.shape,\
                                            ggl_img.shape[0] // self.g_data.shape[0],\
                                            ggl_img.shape[1] // self.g_data.shape[1],\
                                            thicc = 2)

            alpha = 0.8
            gg_h_img = cv2.addWeighted(gg_h_img, alpha, grid_h_inner, 1-alpha, 0.0)
            gg_h_img = cv2.addWeighted(gg_h_img, alpha, grid_h_outer, 1-alpha, 0.0)
            gg_g_img = cv2.addWeighted(gg_g_img, alpha, grid_g, 1-alpha, 0.0)

        # horizontally stack two images
        hg_img = np.hstack((gg_h_img, gg_g_img))

        cv2.putText(hg_img, 'HUMDATA', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(hg_img, 'HUMDATA', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1)
        cv2.putText(hg_img, 'GRID3', (gg_h_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(hg_img, 'GRID3', (gg_h_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)

        cv2.imshow(self.window_name, hg_img)
import numpy as np
import cv2
import rasterio
import salem
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map, GoogleCenterMap
from shapely.geometry import box
import pyperclip
import time

from utils.raster import get_window_geo
from metrics import confusion_matrix
from utils.image import *
import matplotlib.pyplot as plt


class AlignMapsEditor:
    """
    Opens an interactive window which allows to visualize 2 different datasets(e.g. Humdata and GRID3)
    aligned with Google Static Maps.

    Attributes:
        data1_path (string): A path to the first GeoTiff file,
        data2_path (string): A path to second GeoTiff file,
        location (tuple): A pair (lat, lon), location on the map.

    Interactions:
        Press 'Esc' to quit,
        Press 's' to enable search mode,
        Press 'v' to paste coordinates (first copy it from google maps to the clipboard), input in form: 'lat, lon'
                    For now only pasting works, no other input is handled.
        Press left mouse button and drag to change location,
        Release left mouse button to stop dragging.
    """

    def __init__(self, data1_path, data2_path, data1_name, data2_name, location):

        self.humdata_path = humdata_path
        self.grid3_path = grid3_path
        self.window_name = "align_humdata_and_grid3"
        self.window_name_1 = "align_humdata_and_grid3_1"
        self.lat = location[0]
        self.lon = location[1]
        self.box_spread = 0.05
        self.zoom = 1
        self.is_dragging = False
        self.b = (0, 0)
        self.map_size = None
        self.hg_img = None
        self.ra12 = None

        cv2.namedWindow(self.window_name)
        cv2.namedWindow(self.window_name_1)
        cv2.createTrackbar("zoom out", self.window_name, 1, 100, self.update_zoom)
        cv2.setMouseCallback(self.window_name, self.drag_update)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.update_zoom(3)

    def make_grid(self, out_shape, size_vertical, size_horizontal, thicc=1):
        grid = np.zeros(out_shape).astype(np.uint8)
        height, width, channels = out_shape
        for x in range(0, height - 1, size_vertical):
            cv2.line(grid, (x, 0), (x, height), (255, 255, 255), thicc, 1)
        for x in range(0, width - 1, size_horizontal):
            cv2.line(grid, (0, x), (width, x), (255, 255, 255), thicc, 1)
        return grid

    def drag_update(self, event, x, y, flags, param):
        dv = 10
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging = True
            self.b = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
            self.b = (0, 0)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging:
                dx, dy = self.b[0] - x, y - self.b[1]
                if abs(dx) >= dv or abs(dy) >= dv:
                    self.b = (x, y)
                    plon, plat = \
                        (2 * self.box_spread) / self.map_size[1], \
                        (2 * self.box_spread) / self.map_size[0]
                    dlon, dlat = dx * plon, dy * plat
                    self.lat = self.lat + dlat
                    self.lon = self.lon + dlon
                    self.update()

    def input_mode(self):
        err_msg = "Wrong input! Give input in following format: [lat], [lon]"
        text = pyperclip.paste()
        print("Paste: '{}'".format(text))
        l = text.split(', ')
        if (len(l) != 2):
            print(err_msg)
            return 1
        lat, lon = None, None
        try:
            lat, lon = float(l[0]), float(l[1])
        except ValueError as ve:
            print(err_msg)
            return 1
        if lat is None or lon is None:
            return 1

        self.lat = lat
        self.lon = lon
        size = self.final_img.shape
        cv2.putText(self.final_img, 'lat: {:.6f} lon: {:.6f}'.format(self.lat, self.lon),
                    (10, self.final_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow(self.window_name, self.final_img)
        return 0

    def enable_search(self):
        size = self.final_img.shape
        cv2.rectangle(
            self.final_img, (0, size[0] - 50), (size[0], size[0]), (255, 255, 255), -1)
        cv2.imshow(self.window_name, self.final_img)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13:  # on enter escape the input mode
                self.update()
                break
            elif key == 27:
                return 1
            elif key == ord('v'):  # allows to paste the text..
                self.input_mode()
        return 0

    def wait(self):
        # plt.show()
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
            elif key == ord('s'):
                if self.enable_search() == 1:
                    break

    def update_zoom(self, val):
        if val == 0:
            return
        self.zoom = val
        self.box_spread = val / 1000.
        self.update()

    def update(self):
        box_spread = self.box_spread
        with rasterio.open(self.data1_path) as data1, rasterio.open(self.data2_path) as data2:

            data_first_win, first_window = get_window_geo(
                data1, box(self.lon - box_spread, self.lat - box_spread,
                             self.lon + box_spread, self.lat + box_spread))
            bounds = data1.window_bounds(first_window)
            second_window = data2.window(*bounds)
            data_second_win = data2.read(1, window=second_window)
            data_second_win = data_second_win.astype(np.uint8)
            # Preprocess data:
            h_data = humdata2visualization(h_data)
            g_data = grid2visualization(g_data)

            ra1, ra2 = confusion_matrix(h_data, g_data, 3)
            self.ra12 = np.hstack((ra1, ra2))
            cv2.imshow(self.window_name_1, self.ra12)

            self.h_data = cv2.merge((h_data, np.zeros(h_data.shape, dtype=np.uint8), h_data))
            self.g_data = cv2.merge((g_data, g_data, np.zeros(g_data.shape, dtype=np.uint8)))

        crs = salem.gis.check_crs('epsg:4326')
        g = GoogleVisibleMap(x=[self.lon-box_spread, self.lon+box_spread], y=[self.lat-box_spread, self.lat+box_spread],
                             size_x=640, size_y=640,
                             crs='epsg:4326',
                             scale=1,
                             maptype='satellite'
                             )

        ggl_img = googlemaps2visualization(g)
        self.map_size = ggl_img.shape

        # TODO: Perhaps more sophasticated method should be used...
        # It doesn't align google maps well... It seems that google api doesn't return different bboxes depending on box_spread value.
        # that is, for some cases it returns the same image for slightly different bbox regardless change in box_spread value.
        # Also, google maps doesn't use the same projection:
        # https://gis.stackexchange.com/questions/48949/epsg-3857-or-4326-for-googlemaps-openstreetmap-and-leaflet 
        h_d = resize(self.h_data, (ggl_img.shape[1], ggl_img.shape[0]), interpolation=cv2.INTER_AREA)
        g_d = resize(self.g_data, (ggl_img.shape[1], ggl_img.shape[0]), interpolation=cv2.INTER_AREA)

        # add alpha and overlap
        alpha = 0.8
        gg_d1_img = cv2.addWeighted(ggl_img, alpha, d1, 1.-alpha, 0.0)
        gg_d2_img = cv2.addWeighted(ggl_img, alpha, d2, 1.-alpha, 0.0)

        # add grid
        if self.zoom < 5:
            ratio_d1_d2 = 3
            grid_d1_outer = self.make_grid(gg_d1_img.shape,
                                          ggl_img.shape[0] // self.data_first_win.shape[0] *
                                          ratio_d1_d2,
                                          ggl_img.shape[1] // self.data_first_win.shape[1] *
                                          ratio_d1_d2,
                                          thicc=1)
            grid_d1_inner = self.make_grid(gg_d1_img.shape,
                                          ggl_img.shape[0] // self.data_first_win.shape[0],
                                          ggl_img.shape[1] // self.data_first_win.shape[1],
                                          thicc=1)
            grid_d2 = self.make_grid(gg_d2_img.shape,
                                    ggl_img.shape[0] // self.data_second_win.shape[0],
                                    ggl_img.shape[1] // self.data_second_win.shape[1],
                                    thicc=1)

            alpha = 0.8
            gg_d1_img = cv2.addWeighted(
                gg_d1_img, alpha, grid_d1_inner, 1-alpha, 0.0)
            gg_d1_img = cv2.addWeighted(
                gg_d1_img, alpha, grid_d1_outer, 1-alpha, 0.0)
            gg_d2_img = cv2.addWeighted(gg_d2_img, alpha, grid_d2, 1-alpha, 0.0)

        # horizontally stack two images
        self.final_img = np.hstack((gg_d1_img, gg_d2_img))

        cv2.putText(self.final_img, self.data1_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(self.final_img, self.data1_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
        cv2.putText(self.final_img, self.data2_name,
                    (gg_d1_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(self.final_img, self.data2_name,
                    (gg_d1_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
        cv2.putText(self.final_img, 'lat: {:.6f} lon: {:.6f}'.format(self.lat, self.lon),
                    (10, self.final_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.window_name, self.final_img)

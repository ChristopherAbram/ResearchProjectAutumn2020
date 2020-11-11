import numpy as np
import cv2, rasterio, salem
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map, GoogleCenterMap
from shapely.geometry import box
import pyperclip, time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider
from sklearn.metrics import ConfusionMatrixDisplay

from humset.utils.raster import get_window_geo
from humset.metrics import AggregatedMetrics, SimpleMetrics
from humset.utils.image import *


class AlignMapsEditor:
    """
    Opens an interactive window which allows to visualize Humdata and GRID3 aligned with Google Static Maps.

    Attributes:
        hrsl_path (string): A path to humdata GeoTiff file,
        grid3_path (string): A path to GRID3 GeoTiff file,
        location (tuple): A pair (lat, lon), location on the map.

    Interactions:
        Press 'Esc' to quit,
        Press 's' to enable search mode,
        Press 'v' to paste coordinates (first copy it from google maps to the clipboard), input in form: 'lat, lon'
                    For now only pasting works, no other input is handled.
        Press left mouse button and drag to change location,
        Release left mouse button to stop dragging.
    """

    def __init__(self, hrsl_path, grid3_path, location, name1='HUMDATA', name2='GRID3', index1=1, index2=1):

        self.hrsl_path = hrsl_path # high resoultion settlement layer from FB
        self.grid3_path = grid3_path
        self.window_name = "align_humdata_and_grid3"
        self.window_name_1 = "Visualize metrics"
        self.lat = location[0]
        self.lon = location[1]
        self.zoom = location[2] if len(location) >= 3 else 3
        self.box_spread = 0.05
        self.is_dragging = False
        self.b = (0,0)
        self.map_size = None
        self.images_combined = None
        self.name1, self.name2 = name1, name2
        self.index1, self.index2 = index1, index2

        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("zoom out", self.window_name, self.zoom, 2000, self.update_zoom)
        cv2.setMouseCallback(self.window_name, self.drag_update)

        if self.name2 == 'GRID3':
            self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(3, 2, figsize=(6, 9))
            axcolor = 'lightgoldenrodyellow'
            self.thresh_ax = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)
            self.thresh_slider = Slider(self.thresh_ax, 'Threshold', 0.0, 1.0, valinit=0.5, valstep=0.1)
            self.fig.canvas.set_window_title(self.window_name_1)
            self.fig.canvas.draw_idle()
            self.thresh_slider.on_changed(self.on_threshold_update)

        elif self.name2 == 'NORM':
            self.fig1, self.ax1 = plt.subplots(2, 2)
            self.fig2, self.ax2 = plt.subplots(2, 2)
            self.plot_colorbars = False

        self.raw_hrsl_data = None
        self.raw_grid3_data = None
        self.update_zoom(self.zoom)

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
            self.b = (0,0)
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
        """Used for pasting the coordinates to be searched in the window."""
        err_msg = "Wrong input! Give input in following format: [lat], [lon]"
        text = pyperclip.paste()
        print("Paste: '{}'".format(text))
        l = text.split(', ')
        if (len(l) < 2):
            print(err_msg)
            return 1
        lat, lon, zoom = None, None, self.zoom
        try:
            lat, lon = float(l[0]), float(l[1])
            if len(l) >= 3:
                zoom = int(l[2])
        except ValueError as ve:
            print(err_msg)
            return 1
        if lat is None or lon is None:
            return 1
        
        self.lat = lat
        self.lon = lon
        self.zoom = zoom
        size = self.images_combined.shape
        cv2.putText(self.images_combined, 'lat: {:.6f} lon: {:.6f}'.format(self.lat, self.lon),
                    (10, self.images_combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow(self.window_name, self.images_combined)
        return 0

    def enable_search(self):
        size = self.images_combined.shape
        cv2.rectangle(self.images_combined, (0, size[0] - 50), (size[0], size[0]), (255, 255, 255), -1)
        cv2.imshow(self.window_name, self.images_combined)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13: # on enter escape the input mode
                self.update_zoom(self.zoom)
                break
            elif key == 27:
                return 1
            elif key == ord('v'): # allows to paste the text..
                self.input_mode()
        return 0

    def wait(self):
        plt.ion()
        plt.show()
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
        with rasterio.open(self.hrsl_path) as hrsl_file, rasterio.open(self.grid3_path) as grid3_file:
            self.raw_hrsl_data, hrsl_window = get_window_geo(
                hrsl_file, box(self.lon - self.box_spread, self.lat - self.box_spread,
                             self.lon + self.box_spread, self.lat + self.box_spread), self.index1)
            bounds = hrsl_file.window_bounds(hrsl_window)
            grid3_window = grid3_file.window(*bounds)
            self.raw_grid3_data = grid3_file.read(self.index2, window=grid3_window)
            self.visualize_maps(self.raw_hrsl_data, self.raw_grid3_data)
            
    def visualize_maps(self, hrsl_data, grid3_data):
        # compare the two datasets and get confusion matrix
        if self.name1 == 'HUMDATA' and self.name2 == 'GRID3':
            self.plot_products(hrsl_data, grid3_data, 0.5)

        # make map {0,1} to {0,255}
        if self.name1 == 'HUMDATA':
            hrsl_data = humdata2visualization(hrsl_data)
        if self.name2 == 'GRID3':
            grid3_data = grid2visualization(grid3_data)
        if self.name2 == 'NORM':
            self.plot_norm_products()
            grid3_data = normalized2visualization(grid3_data)

        # update the images to be displayed overlayed on satellite image
        self.hrsl_data = cv2.merge((hrsl_data, np.zeros(hrsl_data.shape, dtype=np.uint8), hrsl_data))
        if self.name2 == 'GRID3':
            self.grid3_data = cv2.merge(
                (grid3_data, grid3_data, np.zeros(grid3_data.shape, dtype=np.uint8)))
        elif self.name2 == 'NORM':
            self.grid3_data = cv2.applyColorMap(grid3_data, cv2.COLORMAP_JET)

        crs = salem.gis.check_crs('epsg:4326')
        g = GoogleVisibleMap(
            x=[self.lon-self.box_spread, self.lon+self.box_spread],
            y=[self.lat-self.box_spread, self.lat+self.box_spread],
            size_x = 640, size_y = 640,
            crs='epsg:4326',
            scale=1,
            maptype='satellite'
        )

        googlemaps_image = googlemaps2visualization(g)
        self.map_size = googlemaps_image.shape

        # TODO: Perhaps more sophasticated method should be used...
        # It doesn't align google maps well... It seems that google api doesn't return different bboxes depending on self.box_spread value.
        # that is, for some cases it returns the same image for slightly different bbox regardless change in self.box_spread value.
        # Also, google maps doesn't use the same projection:
        # https://gis.stackexchange.com/questions/48949/epsg-3857-or-4326-for-googlemaps-openstreetmap-and-leaflet 
        h_d = resize(self.hrsl_data, (googlemaps_image.shape[1], googlemaps_image.shape[0]), interpolation=cv2.INTER_AREA)
        g_d = resize(self.grid3_data, (googlemaps_image.shape[1], googlemaps_image.shape[0]), interpolation=cv2.INTER_AREA)

        # add alpha and overlap
        alpha = 0.8
        gg_h_img = cv2.addWeighted(googlemaps_image, alpha, h_d, 1.-alpha, 0.0)
        gg_g_img = cv2.addWeighted(googlemaps_image, alpha, g_d, 1.-alpha, 0.0)

        # add grid
        if self.zoom < 5:
            ratio_g_h = 3
            grid_h_outer = self.make_grid(gg_h_img.shape, \
                                          googlemaps_image.shape[0] // self.hrsl_data.shape[0] * ratio_g_h, \
                                          googlemaps_image.shape[1] // self.hrsl_data.shape[1] * ratio_g_h, \
                                          thicc=1)
            grid_h_inner = self.make_grid(gg_h_img.shape, \
                                          googlemaps_image.shape[0] // self.hrsl_data.shape[0], \
                                          googlemaps_image.shape[1] // self.hrsl_data.shape[1], \
                                          thicc = 1)
            grid_g = self.make_grid(gg_g_img.shape, \
                                    googlemaps_image.shape[0] // self.grid3_data.shape[0], \
                                    googlemaps_image.shape[1] // self.grid3_data.shape[1], \
                                    thicc = 1)
            alpha = 0.9
            gg_h_img = cv2.addWeighted(gg_h_img, alpha, grid_h_inner, 1-alpha, 0.0)
            gg_h_img = cv2.addWeighted(gg_h_img, alpha, grid_h_outer, 1-alpha, 0.0)
            gg_g_img = cv2.addWeighted(gg_g_img, alpha, grid_g, 1-alpha, 0.0)

        # horizontally stack two images
        self.images_combined = np.hstack((gg_h_img, gg_g_img))

        cv2.putText(self.images_combined, self.name1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(self.images_combined, self.name1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
        cv2.putText(self.images_combined, self.name2, (gg_h_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(self.images_combined, self.name2, (gg_h_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
        cv2.putText(self.images_combined, 'lat: {:.6f} lon: {:.6f}'.format(self.lat, self.lon),
                    (10, self.images_combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(self.window_name, self.images_combined)

    def plot_products(self, hrsl_data, grid3_data, threshold):
        hrsl_binary = humdata2binary(hrsl_data)
        grid3_binary = grid2binary(grid3_data)

        impl = AggregatedMetrics()
        cm, accuracy, recall, precision, f1, convolution_product, (hrsl_thresholded, grid3_resized) = \
            impl.compute_metrics_view(hrsl_binary, grid3_binary, threshold)

        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()
        self.ax6.cla()

        self.ax1.imshow((hrsl_thresholded * 255).astype(np.uint8), cmap='gray')
        self.ax1.set_title('HRSL resized and thresholded')
        self.ax2.imshow(grid3_data, cmap='gray')
        self.ax2.set_title('GRID3 original')
        
        self.ax3.imshow((convolution_product * 255).astype(np.uint8), cmap='gray')
        self.ax3.set_title('HRSL processing product')

        cmd = ConfusionMatrixDisplay(cm, display_labels=['t', 'f'])
        cmd = cmd.plot(ax=self.ax4)
        cmd.im_.colorbar.remove()

        # Draw a table:
        columns = ('Metric', 'Value')
        rows = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
        cell_text = [[l, '%1.4f' % m] for (l, m) in zip(rows, [accuracy, recall, precision, f1])]
        t = self.ax5.table(cellText=cell_text, rowLabels=None, colLabels=columns, loc='center', fontsize=14)
        t.scale(1., 1.3)
        self.ax5.get_xaxis().set_visible(False)
        self.ax5.get_yaxis().set_visible(False)
        plt.draw()

    def on_threshold_update(self, threshold):
        self.plot_products(self.raw_hrsl_data, self.raw_grid3_data, threshold)
        self.fig.canvas.draw_idle()

    def plot_norm_products(self):
        self.names = ['True positive', 'False positive', 'False negative', 
                'True negative', 'Accuracy', 'Recall', 'Precision', 'F1 Score']

        for ax1, ax2 in zip(self.ax1.ravel(), self.ax2.ravel()):
                ax1.cla()
                ax2.cla()

        with rasterio.open(self.grid3_path) as file:
            for layer, ax in enumerate(self.ax1.ravel()):
                data, window = get_window_geo(
                    file, box(self.lon - self.box_spread, self.lat - self.box_spread,
                                self.lon + self.box_spread, self.lat + self.box_spread), layer + 1)
                im = ax.imshow(data, cmap='magma', norm=LogNorm())
                if not self.plot_colorbars:
                    self.fig1.colorbar(im, ax=ax)
                ax.set_title(self.names[layer])

            for layer, ax in enumerate(self.ax2.ravel()):
                data, window = get_window_geo(
                    file, box(self.lon - self.box_spread, self.lat - self.box_spread,
                                self.lon + self.box_spread, self.lat + self.box_spread), 4 + layer + 1)
                im = ax.imshow(data, cmap='jet')
                if not self.plot_colorbars:
                    self.fig2.colorbar(im, ax=ax)
                ax.set_title(self.names[4 + layer])
        
        if not self.plot_colorbars:
            self.plot_colorbars = True

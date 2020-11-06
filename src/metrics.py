import numpy as np
import cv2, os, sys
from utils.image import convolve2D
from sklearn import metrics

import threading
import logging
import rasterio

from utils.definitions import get_project_path, get_dataset_paths
from utils.raster import RasterTableSizeAligned
from utils.image import *


logging.basicConfig(
    level=logging.INFO,
    format='(%(threadName)-10s) %(message)s',)


def pad_to_square(array):
    """
    Pads 2D matrix with zeros if it has different number of cols and rows. 
    Resulting matrix is square.
    """
    height, width = array.shape
    if height > width:
        padded = np.zeros((height, height), dtype=array.dtype)
        padded[:,:width] = array
    elif height < width:
        padded = np.zeros((width, width), dtype=array.dtype)
        padded[:height,:] = array
    else:
        padded = array
    return padded


def make_comparable(predicted, truth):
    """
    Bring two images to same dimension by scaling each dimension by corresponding least-common-multiple and
    get shape of eventual kernel.
    :param predicted: 2D ndarray, predicted binary values
    :param truth: 2D ndarray, ground truth binary values
    :return: rescaled predicted, rescaled truth, kernel shape
    """
    # Account for small differences in sizes
    predicted = pad_to_square(predicted)
    truth = pad_to_square(truth)
    predicted_height, predicted_width = predicted.shape
    truth_height, truth_width = truth.shape
    common_height = np.lcm(predicted_height, truth_height)
    common_width = np.lcm(predicted_width, truth_width)
    # common_height, common_width = max(predicted_height, truth_height), max(predicted_width, truth_width)
    predicted_resized = cv2.resize(predicted, (common_width, common_height), interpolation=cv2.INTER_AREA)
    truth_resized = cv2.resize(truth, (common_width, common_height), interpolation=cv2.INTER_AREA)
    kernel_shape = (int(common_height / truth_height), int(common_width / truth_width))
    return predicted_resized, truth_resized, kernel_shape


def make_kernel(kernel_shape):
    """
    Create kernel as (identity matrix * (1 / matrix size)).
    :param kernel_shape: tuple, (height, width)
    :return: ndarray with shape kernel_shape
    """
    kernel_height, kernel_width = kernel_shape
    kernel = 1. / (kernel_height * kernel_width) * np.ones((kernel_width, kernel_height), dtype=np.float32)
    return kernel


def confusion_matrix(hrsl_binary, grid3_binary, threshold, products=True):
    hrsl_resized, grid3_resized, kernel_shape = make_comparable(hrsl_binary, grid3_binary)
    kernel = make_kernel(kernel_shape)
    convolved = convolve2D(hrsl_resized, kernel, strides=kernel_shape)
    hrsl_resized_thresholded = cv2.threshold(\
        convolved, thresh=threshold, maxval=1.0, type=cv2.THRESH_BINARY)[1]
    cm = metrics.confusion_matrix(grid3_binary.ravel(), hrsl_resized_thresholded.ravel(), labels=[1, 0])
    if products:
        return cm, convolved, (hrsl_resized_thresholded, grid3_resized)
    else:
        return cm

def compute_metrics(hrsl_binary, grid3_binary, threshold):
    hrsl_resized, grid3_resized, kernel_shape = make_comparable(hrsl_binary, grid3_binary)
    kernel = make_kernel(kernel_shape)
    convolved = convolve2D(hrsl_resized, kernel, strides=kernel_shape)
    hrsl_resized_thresholded = cv2.threshold(\
        convolved, thresh=threshold, maxval=1.0, type=cv2.THRESH_BINARY)[1]
    grid3_binary = pad_to_square(grid3_binary) # in most cases doesn't change anything
    g2r, h2r = grid3_binary.ravel(), hrsl_resized_thresholded.ravel()
    cm = metrics.confusion_matrix(g2r, h2r, labels=[1, 0])
    accuracy = metrics.accuracy_score(g2r, h2r)
    recall = metrics.recall_score(g2r, h2r)
    precision = metrics.precision_score(g2r, h2r)
    f1 = metrics.f1_score(g2r, h2r)
    return cm, accuracy, recall, precision, f1


class MetricsWorker(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
        args=(), kwargs=None, verbose=None, result_storage=None):

        super(MetricsWorker, self).__init__(
            group=group, target=target, name='MetricsWorker' + str(name))

        self.args = args
        self.kwargs = kwargs
        self.verbose = verbose
        self.threshold = kwargs['threshold'] if 'threshold' in kwargs else None
        self.patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else None
        self.hrsl_path = kwargs['hrsl_path'] if 'hrsl_path' in kwargs else None
        self.grid3_path = kwargs['grid3_path'] if 'grid3_path' in kwargs else None
        self.fake = kwargs['fake'] if 'fake' in kwargs else False
        self.result_storage = result_storage
        self.table = RasterTableSizeAligned(
            self.hrsl_path, self.grid3_path, self.patch_size, self.patch_size)
    
    def run(self):
        if self.table is None:
            return
        
        (row_start, col_start), (row_end, col_end) = self.args
        n_patches = (row_end - row_start + 1) * (col_end + 1)
        i = 0
        last_progress = 0.
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                # Log progress:
                i += 1
                progress = float(i) / float(n_patches)
                if progress - last_progress > 0.01:
                    last_progress = progress
                    logging.info("Progress: {}%".format(int(progress * 100)))

                if self.fake:
                    self.result_storage[row, col, :] = np.array(
                        [row, col, 0, 0, 0, 0, 0, 0])
                    continue

                # Retrieve data:
                hrsl, grid3 = self.table.get(row, col)
                hrsl_binary = humdata2binary(hrsl.data)
                grid3_binary = grid2binary(grid3.data)

                # Skip computation if both are empty:
                if hrsl_binary.sum() == 0 and grid3_binary.sum() == 0:
                    h, w = grid3_binary.shape
                    self.result_storage[row, col, :] = np.array(
                        [0, 0, 0, h * w, 1, 0, 0, 0])
                    continue

                cm, accuracy, recall, precision, f1 = compute_metrics(
                    hrsl_binary, grid3_binary, self.threshold)
                
                # Store results in shared matrics:
                tp, fp, fn, tn = cm.ravel()
                self.result_storage[row, col, :] = np.array(
                    [tp, fp, fn, tn, accuracy, recall, precision, f1])

        logging.info("Complete!")


class RasterTableScheduler:

    def __init__(self, hrsl_path, grid3_path, patch_size, threshold, n_threads=12, fake=False):
        self.table = RasterTableSizeAligned(hrsl_path, grid3_path, patch_size, patch_size)
        self.hrsl_path = hrsl_path
        self.grid3_path = grid3_path
        self.n_threads = n_threads
        self.threshold = threshold
        self.patch_size = patch_size
        self.metrics = None
        self.hrsl_path_thread = []
        self.grid3_path_thread = []
        self.fake = fake

    def split_thread_indexes(self):
        """
        TODO: ...
        """
        # Compute how many rows assign to a one thread:
        n_table_rows, n_table_cols = self.table.height_slices, self.table.width_slices
        rows_per_thread = int(np.ceil(n_table_rows / self.n_threads))
        rows_per_thread_last = n_table_rows - (self.n_threads - 1) * rows_per_thread
        indexes = []
        for tix in range(self.n_threads - 1):
            indexes.append(((tix * rows_per_thread, 0), ((tix + 1) * rows_per_thread - 1, n_table_cols - 1)))
        # The last thread will have less rows to compute:
        _, (l_row_inx, _) = indexes[-1]
        indexes.append(((l_row_inx + 1, 0), (n_table_rows - 1, n_table_cols - 1)))
        return indexes

    def __before(self):
        # Create a copy of dataset per thread:
        dest_path = os.path.join(get_project_path(), "data", "tmp")
        os.system('mkdir -p %s' % dest_path)
        for tix in range(self.n_threads):
            # Copy hrsl:
            dest_filepath = os.path.join(dest_path, 'hrsl_%d.tif' % (tix + 1))
            self.hrsl_path_thread.append(dest_filepath)
            os.system('cp "%s" "%s"' % (self.hrsl_path, dest_filepath))
            # Copy grid3:
            dest_filepath = os.path.join(dest_path, 'grid3_%d.tif' % (tix + 1))
            self.grid3_path_thread.append(dest_filepath)
            os.system('cp "%s" "%s"' % (self.grid3_path, dest_filepath))

    def __after(self):
        # Remove all copies of datasets:
        dest_path = os.path.join(get_project_path(), "data", "tmp")
        os.system('rm -r %s' % dest_path)

    def run(self):
        # Metrics storage:
        # tp, fp, fn, tn, accuracy, recall, precision, f1
        n_table_rows, n_table_cols = self.table.height_slices, self.table.width_slices
        self.metrics = np.zeros((n_table_rows, n_table_cols, 8))
        
        indexes = self.split_thread_indexes()
        self.__before()

        # Init, start and join threads:
        threads = []
        for tix in range(self.n_threads):
            threads.append(MetricsWorker(
                result_storage=self.metrics,
                args=indexes[tix],
                kwargs={
                    'threshold': self.threshold,
                    'patch_size': self.patch_size,
                    'hrsl_path': self.hrsl_path_thread[tix],
                    'grid3_path': self.grid3_path_thread[tix],
                    'fake': self.fake
                },
                name=(tix + 1)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.__after()

    def save(self, dirpath, filename):
        logging.info("Create directory '%s'" % dirpath)
        os.system('mkdir -p %s' % dirpath)
        outpath = os.path.join(dirpath, filename)
        height, width, layers = self.metrics.shape

        rio = self.table.get_raster()[0]
        transform = rio.transform * rio.transform.scale(
            (rio.width / width),
            (rio.height / height)
        )

        meta = {
            'driver': 'GTiff',
            'dtype': rasterio.float64,
            'crs': rio.crs,
            'width': width,
            'height': height, 
            'count': layers,
            'transform': transform
        }

        logging.info("Write GTiff file to '%s'" % outpath)
        with rasterio.open(outpath, 'w', **meta) as out:
            for layer in range(layers):
                out.write(self.metrics[:,:,layer], indexes=(layer + 1))

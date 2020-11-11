import unittest
import threading, logging
import os, rasterio
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from humset.utils.definitions import get_project_path
from humset.metrics import RasterTableScheduler, SimpleMetrics
from humset.utils.raster import *
from humset.utils.image import *


logging.basicConfig(
    level=logging.INFO,
    format='(%(threadName)-10s) %(message)s',)


class WorkerMock(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
        args=(), kwargs=None, verbose=None, result_storage=None):

        super(WorkerMock, self).__init__(
            group=group, target=target, name='WorkerMock' + str(name))

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

            hrsl, grid3 = self.table.get(row, col)
            srow, scol, swidth, sheight = \
                int(grid3.window.row_off), \
                int(grid3.window.col_off), \
                int(grid3.window.width), \
                int(grid3.window.height)
            self.result_storage[srow:srow+sheight, scol:scol+swidth, 0] = grid3.data

        logging.info("Complete!")        


# class RasterTableSchedulerTest(unittest.TestCase):

#     def setUp(self):
#         return super().setUp()

#     @classmethod
#     def tearDownClass(cls):
#         return super().tearDownClass()

#     def test_table_scheduler_exhaustivness(self):
#         country = 'NGA'
#         in_files = {
#             'humdata': os.path.join(get_project_path(), 'test', 'data', 'align', 'example_humdata.tif'),
#             'grid3': os.path.join(get_project_path(), 'test', 'data', 'align', 'example_grid3.tif')
#         }

#         patch_size = 60
#         threshold = 0.2
#         threads = 12

#         # Build output filename:
#         dirpath = os.path.join(get_project_path(), "data", "out")
#         filename = '%s_metrics_p%d_t%d.tif' % (country.lower(), patch_size, int(threshold * 100))

#         scheduler = RasterTableScheduler(
#             in_files['humdata'], in_files['grid3'], 
#             patch_size, threshold, threads, fake=False, worker_class=WorkerMock)

#         scheduler.run()
#         scheduler.save(dirpath, filename)

#         with rasterio.open(in_files['grid3']) as dataset, \
#             rasterio.open(os.path.join(dirpath, filename)) as bypatch:
#             X = dataset.read(1)
#             Y = bypatch.read(1)
#             X = np.nan_to_num(X)
#             Y = np.nan_to_num(Y)
#             self.assertTrue((X == Y).astype(np.uint8).mean() >= 0.97)


class SimpleMetricsTest(unittest.TestCase):

    def setUp(self):
        self.cities = ['lagos1', 'lagos2', 'ago_are', 'ibadan', 'port_harcourt']
        self.spreads = ['003', '01', '05']

        self.small_grid3_path = [os.path.join(get_project_path(), 'test', 'data', 'metrics', '%s_%s_grid3.tif' % (city, self.spreads[0])) for city in self.cities]
        self.middle_grid3_path = [os.path.join(get_project_path(), 'test', 'data', 'metrics', '%s_%s_grid3.tif' % (city, self.spreads[1])) for city in self.cities]
        self.large_grid3_path = [os.path.join(get_project_path(), 'test', 'data', 'metrics', '%s_%s_grid3.tif' % (city, self.spreads[2])) for city in self.cities]

        self.small_humdata_path = [os.path.join(get_project_path(), 'test', 'data', 'metrics', '%s_%s_humdata.tif' % (city, self.spreads[0])) for city in self.cities]
        self.middle_humdata_path = [os.path.join(get_project_path(), 'test', 'data', 'metrics', '%s_%s_humdata.tif' % (city, self.spreads[1])) for city in self.cities]
        self.large_humdata_path = [os.path.join(get_project_path(), 'test', 'data', 'metrics', '%s_%s_humdata.tif' % (city, self.spreads[2])) for city in self.cities]

    def test_small(self):
        for inx, (grid3_path, humdata_path) in enumerate(zip(self.small_grid3_path, self.small_humdata_path)):
            with rasterio.open(humdata_path) as humdata, rasterio.open(grid3_path) as grid3:
                hrsl_data = humdata.read(1)
                grid3_data = grid3.read(1)

                hrsl_binary = humdata2binary(hrsl_data)
                grid3_binary = grid2binary(grid3_data)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15 ,10))
                ax1.imshow(hrsl_binary, cmap='gray')
                gc_data = grid3_binary * 255
                ax2.imshow(gc_data, cmap='gray')
                plt.show()

                #  TODO
                a = 0


def get_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(RasterTableSchedulerTest))
    return suite


if __name__ == "__main__":
    unittest.main()

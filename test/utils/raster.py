import unittest
from unittest import mock
from unittest.mock import patch
import os, glob, itertools, rasterio
from os import path
from rasterio.windows import Window as rWindow
import numpy as np

from src.utils.definitions import get_project_path
from src.utils.raster import RasterTable, RasterTableSize


class RasterTableTest(unittest.TestCase):

    def setUp(self):
        self.infile = path.join(get_project_path(), "test", "data", "example1.tif")
        self.infiles = ["example0.tif", "example1.tif", "example2.tif", "example3.tif", "example4.tif"]
        self.infiles = [path.join(get_project_path(), "test", "data", filename) for filename in self.infiles]

    @classmethod
    def tearDownClass(cls):
        tmpdir = path.join(get_project_path(), "test", "data", "tmp", "*.tif")
        for tif in glob.glob(tmpdir):
            os.remove(tif)

    def test_wrong_path(self):
        with self.assertRaises(Exception):
            RasterTable('this is a wrong path')

    def test_open_tiff(self):
        RasterTable(self.infile)

    def test_iteration_simple(self):
        with self.assertRaises(StopIteration):
            it = RasterTable(self.infile, 8, 4).iterator()
            for i in range(8*4 + 4):
                next(it)

    def test_iteration_rows_and_cols(self):
        pairs = list(itertools.product(range(4), range(8)))
        table = RasterTable(self.infile, 8, 4)
        for (window, pair) in zip(table.iterator(), pairs):
            self.assertEqual(window.pos, pair)

    # def get_window_size(self, xslices, yslices):
    #     w, h = 0, 0
    #     with rasterio.open(self.infile) as raster:
    #         w, h = raster.width, raster.height
    #     # Window sizes:
    #     return w // xslices, h // yslices

    def test_get_out_of_range(self):
        table = RasterTable(self.infile, 8, 4)
        with self.assertRaises(IndexError):
            table.get(8, 2)

    def test_get_by_index(self):
        table = RasterTable(self.infile, 8, 4)
        window = table.get(3, 1)

    def test_raster_table_iterator_exhaustiveness(self):
        pairs = [(2, 2), (3, 11), (103, 10), (7, 4)]
        for j, filename in enumerate(self.infiles):
            for i, pair in enumerate(pairs):
                outfile = path.join(get_project_path(), "test", "data", "tmp", "example0_%s_%s.tif" % (j, i))
                table = RasterTable(filename, pair[0], pair[1])

                with rasterio.open(outfile, 'w',
                    driver='GTiff', width=table.get_raster().width, height=table.get_raster().height, count=1,
                    dtype=rasterio.ubyte) as dst:

                    for window in table.iterator():
                        data = np.array(window.data, dtype=rasterio.ubyte)
                        width, height = window.size
                        row, col = window.pos
                        dst.write(data, window=rWindow(
                            col * table.si_width, row * table.si_height, width, height), indexes=1)

                with rasterio.open(filename) as dataset, rasterio.open(outfile) as bypatch:
                    X = dataset.read()
                    Y = bypatch.read()
                    self.assertTrue((X == Y).all())

    def test_raster_table_size_iterator_exhaustiveness(self):
        pairs = [(23, 20), (103, 160), (303, 254), (55, 11)]
        for j, filename in enumerate(self.infiles):
            for i, pair in enumerate(pairs):
                outfile = path.join(get_project_path(), "test", "data", "tmp", "example1_%s_%s.tif" % (j, i))
                table = RasterTableSize(filename, pair[0], pair[1])
                
                with rasterio.open(outfile, 'w',
                    driver='GTiff', width=table.get_raster().width, height=table.get_raster().height, count=1,
                    dtype=rasterio.ubyte) as dst:

                    for window in table.iterator():
                        data = np.array(window.data, dtype=rasterio.ubyte)
                        width, height = window.size
                        row, col = window.pos
                        dst.write(data, window=rWindow(
                            col * table.si_width, row * table.si_height, width, height), indexes=1)

                with rasterio.open(filename) as dataset, rasterio.open(outfile) as bypatch:
                    X = dataset.read()
                    Y = bypatch.read()
                    self.assertTrue((X == Y).all())

    def test_raster_table_compare_iterator_and_get_patches(self):
        table = RasterTable(self.infile, 8, 4)
        for window in table.iterator():
            self.assertTrue((window.data == table.get(window.pos[0], window.pos[1]).data).all())

    def test_raster_table_size_compare_iterator_and_get_patches(self):
        table = RasterTableSize(self.infile, 13, 14)
        for window in table.iterator():
            self.assertTrue((window.data == table.get(window.pos[0], window.pos[1]).data).all())


if __name__ == "__main__":
    unittest.main()

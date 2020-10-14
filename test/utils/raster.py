import unittest
from unittest import mock
from unittest.mock import patch
import os, glob, itertools, rasterio
from os import path
from rasterio.windows import Window
import numpy as np

from src.utils.definitions import get_project_path
from src.utils.raster import RasterWindow, RasterWindowSize


class RasterWindowTest(unittest.TestCase):

    def setUp(self):
        self.infile = path.join(get_project_path(), "test", "data", "example.tif")

    @classmethod
    def tearDownClass(cls):
        tmpdir = path.join(get_project_path(), "test", "data", "tmp", "*.tif")
        for tif in glob.glob(tmpdir):
            os.remove(tif)

    def test_wrong_path(self):
        with self.assertRaises(Exception):
            RasterWindow('this is a wrong path')

    def test_open_tiff(self):
        RasterWindow(self.infile)

    def test_iteration_simple(self):
        with self.assertRaises(StopIteration):
            it = RasterWindow(self.infile, 8, 4)
            for i in range(8*4 + 4):
                next(it)

    def test_iteration_rows_and_cols(self):
        pairs = list(itertools.product(range(4), range(8)))
        for ((window, (row, col), size), pair) in zip(RasterWindow(self.infile, 8, 4), pairs):
            self.assertEqual((row, col), pair)

    def get_window_size(self, xslices, yslices):
        w, h = 0, 0
        with rasterio.open(self.infile) as raster:
            w, h = raster.width, raster.height
        # Window sizes:
        return w // xslices, h // yslices

    def test_size(self):
        # Window sizes:
        ww, wh = self.get_window_size(8, 4)
        # All windows should have same size:
        for window, (row, col), (width, height) in RasterWindow(self.infile, 8, 4):
            self.assertEqual((width, height), (ww, wh))

    def test_get_out_of_range(self):
        it = RasterWindow(self.infile, 8, 4)
        with self.assertRaises(IndexError):
            it.get(8, 2)

    def test_get_by_index(self):
        it = RasterWindow(self.infile, 8, 4)
        window, (w, h) = it.get(3, 1)

    def test_read_by_slice_consistence(self):
        pairs = [(2, 2), (3, 10), (100, 150), (8, 4)]
        for i, pair in enumerate(pairs):
            outfile = path.join(get_project_path(), "test", "data", "tmp", "example1_%s.tif" % i)
            with rasterio.open(outfile, 'w',
                    driver='GTiff', width=500, height=300, count=1,
                    dtype=rasterio.ubyte) as dst:
                for (win, (row, col), (width, height)) in RasterWindow(self.infile, pair[0], pair[1]):
                    win = np.array(win, dtype=rasterio.ubyte)
                    dst.write(win, window=Window(    
                        col * width, row * height, width, height), indexes=1)

            with rasterio.open(self.infile) as dataset, rasterio.open(outfile) as bypatch:
                X = dataset.read()
                Y = bypatch.read()
                self.assertTrue((X == Y).all())

    def test_read_by_slice_consistence_size(self):
        pairs = [(20, 20), (100, 100), (500, 250), (5, 5)]
        for i, pair in enumerate(pairs):
            outfile = path.join(get_project_path(), "test", "data", "tmp", "example2_%s.tif" % i)
            with rasterio.open(outfile, 'w',
                    driver='GTiff', width=500, height=300, count=1,
                    dtype=rasterio.ubyte) as dst:
                for (win, (row, col), (width, height)) in RasterWindowSize(self.infile, pair[0], pair[1]):
                    win = np.array(win, dtype=rasterio.ubyte)
                    dst.write(win, window=Window(    
                        col * width, row * height, width, height), indexes=1)

            with rasterio.open(self.infile) as dataset, rasterio.open(outfile) as bypatch:
                X = dataset.read()
                Y = bypatch.read()
                self.assertTrue((X == Y).all())

    def test_compare_byindex_with_bysize(self):
        pairs = [(2, 2), (3, 10), (100, 150), (8, 4)]
        for i, pair in enumerate(pairs): 
            w, h = self.get_window_size(pair[0], pair[1])
            its = RasterWindowSize(self.infile, w, h)
            it = RasterWindow(self.infile, pair[0], pair[1])

            self.assertEqual(its.num_xslices, it.num_xslices)
            self.assertEqual(its.num_yslices, it.num_yslices)
            for r1, r2 in zip(it, its):
                self.assertTrue((r1[0] == r2[0]).all())

    def test_get_patch(self):
        it = RasterWindow(self.infile, 8, 4)
        for win, (row, col), (width, height) in RasterWindow(self.infile, 8, 4):
            self.assertTrue((win == it.get(row, col)[0]).all())

    def test_get_patch(self):
        it = RasterWindowSize(self.infile, 13, 14)
        for win, (row, col), (width, height) in RasterWindowSize(self.infile, 13, 14):
            self.assertTrue((win == it.get(row, col)[0]).all())


if __name__ == "__main__":
    unittest.main()

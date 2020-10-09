import sys
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import itertools


class RasterWindow:
    def __init__(self, filepath, x_slices = 3, y_slices = 3):
        self.filepath = filepath
        self.num_xslices = x_slices
        self.num_yslices = y_slices
        self._raster = rasterio.open(self.filepath)

    def __iter__(self):
        self._w_pairs = itertools.product(
            range(self.num_yslices),
            range(self.num_xslices))
        return self

    def __next__(self):
        try:
            xi, yi = next(self._w_pairs)
            si_width, si_height = \
                int(self._raster.width / self.num_xslices), \
                int(self._raster.height / self.num_yslices)
            w = Window.from_slices(
                (xi * si_height, xi * si_height + si_height), 
                (yi * si_width, yi * si_width + si_width))
            X = self._raster.read(1, window=w)
            return X, (xi, yi), (X.shape[1], X.shape[0])
        except StopIteration:
            self.__raster.close()
            raise


class RasterWindowSize(RasterWindow):
    def __init__(self, filepath, width=1024, height=1024):
        super(RasterWindowSize, self).__init__(filepath)
        self.width = width
        self.height = height
        self.num_xslices = int(np.ceil(self._raster.width / self.width))
        self.num_yslices = int(np.ceil(self._raster.height / self.height))

    def __next__(self):
        try:
            xi, yi = next(self._w_pairs)
            si_width, si_height = self.width, self.height
            w = Window.from_slices(
                (xi * si_height, xi * si_height + si_height), 
                (yi * si_width, yi * si_width + si_width))
            X = self._raster.read(1, window=w)
            return X, (xi, yi), (X.shape[1], X.shape[0])
        except StopIteration:
            self._raster.close()
            raise
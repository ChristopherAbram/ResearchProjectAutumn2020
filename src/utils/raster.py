import sys
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import itertools
from shapely.geometry import box


class RasterWindow:
    def __init__(self, filepath, x_slices = 3, y_slices = 3):
        self.filepath = filepath
        self.num_xslices = x_slices
        self.num_yslices = y_slices
        self._raster = rasterio.open(self.filepath)
        self._w_pairs = itertools.product(
            range(self.num_yslices),
            range(self.num_xslices))
        self.si_width, self.si_height = \
            int(np.ceil(self._raster.width / self.num_xslices)), \
            int(np.ceil(self._raster.height / self.num_yslices))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            xi, yi = next(self._w_pairs)
            w = Window.from_slices(
                (xi * self.si_height, xi * self.si_height + self.si_height), 
                (yi * self.si_width, yi * self.si_width + self.si_width))
            # w = Window(yi * self.si_width, xi * self.si_height, self.si_width, self.si_height)
            X = self._raster.read(1, window=w)
            return X, (xi, yi), (X.shape[1], X.shape[0])
        except StopIteration:
            self._raster.close()
            raise

    def get(self, x, y):
        """
        Get (x, y) window from (N, M) possible windows.
        Remeber that windows are indexed the same as arrays, that is starting form 0.
        """
        if x >= self.num_yslices or y >= self.num_xslices:
            raise IndexError(
                "Window indexes out of range. Given ({}, {}) and maximum are ({}, {})".format(
                    x, y, self.num_yslices - 1, self.num_xslices - 1))

        w = Window.from_slices(
            (x * self.si_height, x * self.si_height + self.si_height), 
            (y * self.si_width, y * self.si_width + self.si_width))
        X = self._raster.read(1, window=w)
        return X, (X.shape[1], X.shape[0])

    def get_raster(self):
        return self._raster


class RasterWindowSize(RasterWindow):
    def __init__(self, filepath, width=1024, height=1024):
        super(RasterWindowSize, self).__init__(filepath)
        self.width = width
        self.height = height
        self.num_xslices = self._raster.width // self.width
        self.num_yslices = self._raster.height // self.height
        self._w_pairs = itertools.product(
            range(self.num_yslices),
            range(self.num_xslices))
        self.si_width, self.si_height = self.width, self.height


def get_window_px(raster, x, y, width, height):
    return raster.read(1, window=Window.from_slices((x, x + height), (y, y + width)))


def get_window_geo(raster, bbox):
    [lonl, latb, lonr, latu] = bbox.bounds
    rx, ry = raster.index(lonr, latb)
    lx, ly = raster.index(lonl, latu)
    w, h = np.abs(ry - ly), np.abs(lx - rx)
    return get_window_px(raster, lx, ly, w, h)
